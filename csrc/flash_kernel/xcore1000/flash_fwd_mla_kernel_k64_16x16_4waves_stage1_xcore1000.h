// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#pragma once

#include <cute/algorithm/copy.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

namespace flash {

using namespace cute;
template<typename Kernel_traits, bool Split, bool Is_even_MN, bool Is_even_K, typename AccO, typename Softmax, typename Params>
__forceinline__ __device__ void store_16x16(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, char* smem_, AccO acc_o, Softmax softmax ) {
  using ElementAccum = typename Kernel_traits::ElementAccum;
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    // The thread index.
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 64;
    const int lane_idx = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Return_lse*/true, Split>(acc_o, params.scale_softmax);
    Tensor acc_o_view = make_tensor(acc_o.data(), make_layout(Shape<_4, Shape<_4, _2>>{},
                                                                Stride<_1, Shape<_4, _16>>{}));
    Tensor acc_o_copy = make_fragment_like(acc_o_view);
    #pragma unroll
    for (int k = 0; k < size<1, 1>(acc_o_view); k++) {
        #pragma unroll
        for (int idx = 0; idx < 16; idx++) {
            int row = idx / 4;
            int col = idx % 4;
            acc_o_copy(row, make_coord(col, k)) = acc_o_view(col, make_coord(row, k));
        }
    }
    if constexpr (!Split) {
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomO;
        CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o_copy, rO)
        int warp_offset = warp_idx * 16 * 64;
        int thread_offset = lane_idx % 16 * 64 + lane_idx / 16 * 16;
        ElementO *Osmem_ptr_sts = reinterpret_cast<ElementO *>(smem_) + warp_offset + thread_offset;
        Tensor tOsO = make_tensor(make_smem_ptr(Osmem_ptr_sts), make_layout(Shape<_16, _2>{},
                                                                          Stride<_1, Int<16*64*kNWarps>>{}));
        Tensor tOrO = make_tensor(rO.data(), make_layout(Shape<_16, _2>{},
                                                                Stride<_1, _16>{}));


        if constexpr (Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }

        cute::copy(tOrO, tOsO);

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        flash::sync_threads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            tOrOaccum, tOgOaccum, tOcO, params.d_v, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        const int split_offset = params.num_splits_ptr[bidb];
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q
                                            + m_block * kBlockM) * params.d_v;
        const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.oaccum_ptr) + row_offset_oaccum),
                                    Shape<Int<kBlockM>, Int<kHeadDimV/2>>{},
                                    make_stride(kHeadDimV, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
        Tensor taccOrOaccum = make_tensor(acc_o_copy.data(), acc_o_copy.layout());

        int warp_offset = warp_idx * 16 * 64;
        int thread_offset = lane_idx % 16 * 64 + lane_idx / 16 * 16;
        ElementO *accOsmem_ptr_sts = reinterpret_cast<ElementO *>(smem_) + warp_offset + thread_offset;
        Tensor taccOsOaccum = make_tensor(make_smem_ptr(accOsmem_ptr_sts), make_layout(Shape<_4, _4>{},
                                                                          Stride<_1, _4>{}));

        if constexpr (Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }
        int O_swizzle_row_sts = tidx % 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cute::copy(taccOrOaccum(_, make_coord(i, 0)), taccOsOaccum(_, O_swizzle_row_sts ^ i));
        }
        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        // Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        int O_swizzle_row_lds = tidx / 16 % 4;
        int O_swizzle_col_lds = tidx % 16 % 4;
        int O_swizzle_col_lds_new = O_swizzle_col_lds ^ O_swizzle_row_lds;
        ElementO *accOsmem_ptr_lds = reinterpret_cast<ElementO *>(smem_) + (tidx + O_swizzle_col_lds_new - O_swizzle_col_lds) * 4;

        Tensor tOsOaccum = make_tensor(make_smem_ptr(accOsmem_ptr_lds), make_layout(Shape<_4, _1, Int<kHeadDimV/2/64>>{},
                                                                          Stride<_1, _0, Int<16*64>>{}));

        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        flash::sync_threads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);
        flash::sync_threads();
        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcaccO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            tOrOaccum, tOgOaccum, tOcaccO, params.d_v, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            cute::copy(taccOrOaccum(_, make_coord(i, 1)), taccOsOaccum(_, O_swizzle_row_sts ^ i));
        }
        flash::sync_threads();
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);
        tOgOaccum.data() = tOgOaccum.data() + (kHeadDimV/2);


        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            tOrOaccum, tOgOaccum, tOcaccO, params.d_v, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}
template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_enable_dcp, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla_k64_16x16_4waves_xcore1000(
        const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int n_block_min, const int n_block_max, const bool NoSplit) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 64;
    const int lane_idx = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockKSmem = Kernel_traits::kBlockKSmem;
    constexpr int kAtomLayoutMS = Kernel_traits::kAtomLayoutMS;
    constexpr int kAtomLayoutMO = Kernel_traits::kAtomLayoutMO;
    constexpr int Num_Stages = Kernel_traits::Num_Stages;
    constexpr int kSmemSize = Kernel_traits::kSmemSize;
    static_assert(kBlockKSmem == 64);


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int actual_seqlen_k = params.cu_seqlens_k[bidb];

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ424{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutK424{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed424{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyB64 gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::GmemTiledCopyB64 gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    // S is only 16x16 size, so all 4 waves compute the same S
    int tidx_mma_s = tidx & 0x3F;
    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx_mma_s);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK(_, _, 0));                           // (MMA,MMA_N,MMA_K)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    // Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrVt = make_tensor<Element>(Shape<_4, Shape<_4, _2>, _1>{});

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx_mma_s);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx_mma_s);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    int warp_offset = warp_idx / kAtomLayoutMO * 16 * 64;
    int thread_offset = lane_idx / 16 * 4 * 64;
    Element *Vtsmem_ptr_lds = reinterpret_cast<Element *>(sVt.data().get()) + warp_offset + thread_offset;
    Tensor tOsVt = make_tensor(make_smem_ptr(Vtsmem_ptr_lds), make_layout(Shape<_4, _2, Int<Num_Stages>>{},                  // MMA  MMA_N  NUM_STAGES
                                                                          Stride<_1, Int<16*256>, Int<kBlockN*kHeadDim>>{}));

    // PREDICATES

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)


    // Prologue


    // Read Q from gmem to smem, optionally apply rotary embedding.
    Tensor tQrQ = make_fragment_like(tQgQ);
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy_b64<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    cute::copy(tQrQ, tQsQ);

    if constexpr (Kernel_traits::Is_Q_in_regs) {
        flash::sync_threads();
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);
        flash::sync_threads();
    }


    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor tKrK = make_fragment_like(tKgK);
    int row_offset = tidx / 16  + n_block * kBlockN;
    int virtual_page_idx = row_offset / params.page_block_size;
    int page_offset = row_offset - virtual_page_idx * params.page_block_size;
    int32_t page_idx = block_table[virtual_page_idx];
    flash::copy_b64_page_one<Kernel_traits, Is_even_MN, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block,
                                                                        block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, page_idx, page_offset, actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<Is_causal, Is_enable_dcp> mask(actual_seqlen_k, binfo.actual_seqlen_q, params.ngroups, binfo.tot_seqlen_k, params.cp_world_size, params.cp_rank);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        if (n_block > n_block_min) {
            // prefetch load page index
            row_offset -= kBlockN;
            virtual_page_idx = row_offset / params.page_block_size;
            page_offset = row_offset - virtual_page_idx * params.page_block_size;
            page_idx = block_table[virtual_page_idx];
        }
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        flash::sync_threads();
        cute::copy(tKrK, tKsK(_, _, _, 0));
        clear(acc_s);

        flash::sync_threads();
        if (n_block > n_block_min) {
            flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, page_idx, page_offset);
        }
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, 0), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % kAtomLayoutMS * 16 + (tidx & 0xf), kAtomLayoutMS * 16
        );

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2);
        // Convert acc_s from fp32 to fp16/bf16
        //Tensor rP = flash::convert_type<Element>(acc_s);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        // flash::sync_threads();
        lds4x4_with_swizzle424(tOsVt(_, _, 0), tOrVt);
        CUTE_STATIC_ASSERT_V(size<2>(tOrVt) == _1{}); // only support MMA_K = 1
        Tensor tOrVt_permute_view = make_tensor(tOrVt.data(), make_layout(make_shape(size<0>(tOrVt), size<1, 0>(tOrVt), size<1, 1>(tOrVt))));
        permute_4x4_b16(tOrVt_permute_view);
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        if (n_block > n_block_min) {
            // prefetch load page index
            row_offset -= kBlockN;
            virtual_page_idx = row_offset / params.page_block_size;
            page_offset = row_offset - virtual_page_idx * params.page_block_size;
            page_idx = block_table[virtual_page_idx];
        }
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        flash::sync_threads();
        cute::copy(tKrK, tKsK(_, _, _, 0));
        clear(acc_s);
        flash::sync_threads();
        if (n_block > n_block_min) {
            // Advance gK
            flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, page_idx, page_offset);
        }

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, 0), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % kAtomLayoutMS * 16 + (tidx & 0xf), kAtomLayoutMS * 16
        );
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        //Tensor rP = flash::convert_type<Element>(acc_s);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // flash::sync_threads();
        lds4x4_with_swizzle424(tOsVt(_, _, 0), tOrVt);
        CUTE_STATIC_ASSERT_V(size<2>(tOrVt) == _1{}); // only support MMA_K = 1
        Tensor tOrVt_permute_view = make_tensor(tOrVt.data(), make_layout(make_shape(size<0>(tOrVt), size<1, 0>(tOrVt), size<1, 1>(tOrVt))));
        permute_4x4_b16(tOrVt_permute_view);
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);
    }
    // Epilogue
    if (NoSplit) {
        store_16x16<Kernel_traits, false, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }else{
        store_16x16<Kernel_traits, true, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

} // namespace flash
