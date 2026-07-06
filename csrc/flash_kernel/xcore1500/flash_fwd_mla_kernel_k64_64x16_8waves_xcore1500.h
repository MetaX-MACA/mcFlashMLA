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
__forceinline__ __device__ void store_64x16_xcore1500(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, char* smem_, AccO acc_o, Softmax softmax){
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

    const int split_offset = params.num_splits_ptr[bidb];
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o, rO)
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)


    if constexpr (Split || Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q
                                        + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

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
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_enable_dcp, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla_k64_64x16_8waves_xcore1500(
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
    constexpr int kHeadDimNope = kHeadDimV;
    constexpr int kHeadDimRope = kHeadDim - kHeadDimV;

    static_assert(kBlockKSmem == 64);
    static_assert(Kernel_traits::Share_Q_K_smem && Kernel_traits::Is_Q_in_regs);


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

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

    Tensor gNopeQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDimNope>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gRopeQ = make_tensor(make_gmem_ptr(gNopeQ.data().get() + kHeadDimNope),
                            Shape<Int<kBlockM>, Int<kHeadDimRope>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sNopeQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutNopeQ{});
    Tensor sRopeQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutRopeQ{});

    Tensor sK = make_tensor(sNopeQ.data(), typename Kernel_traits::SmemLayoutK242{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed242{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyB128 gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tNopeQgNopeQ = gmem_thr_copy_Q.partition_S(gNopeQ);
    Tensor tNopeQsNopeQ = gmem_thr_copy_Q.partition_D(sNopeQ);
    Tensor tRopeQgRopeQ = gmem_thr_copy_Q.partition_S(gRopeQ);
    Tensor tRopeQsRopeQ = gmem_thr_copy_Q.partition_D(sRopeQ);

    typename Kernel_traits::GmemTiledCopyB32 gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    // gemm S is 4x1 wave layout, wave(n) compute the same S with wave(n + 4)
    int tidx_mma_s = tidx & 0xFF;
    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx_mma_s);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrNopeQ  = thr_mma_s.partition_fragment_A(sNopeQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrRopeQ  = thr_mma_s.partition_fragment_A(sRopeQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK(_, _, 0));                           // (MMA,MMA_N,MMA_K)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx_mma_s);
    Tensor tSsNopeQ = smem_thr_copy_Q.partition_S(sNopeQ);
    Tensor tSsRopeQ = smem_thr_copy_Q.partition_S(sRopeQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::UniversalCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx_mma_s);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::LDSB64Trans4x16Atom{}, tiled_mma_o);
    int warp_offset = warp_idx / 4 * 16;
    int thread_offset = lane_idx % 4 * 4
                        + lane_idx % 16 / 4 * 64
                        + lane_idx / 16 * 4 * 64;

    Element *Vtsmem_ptr_lds = reinterpret_cast<Element *>(sVt.data().get()) + warp_offset + thread_offset;
    Tensor tOsVt = make_tensor(make_smem_ptr(Vtsmem_ptr_lds), make_layout(Shape<_4, Shape<_2, _8>,  Int<kBlockN / 16>, Int<Num_Stages>>{},                  // MMA  MMA_N  NUM_STAGES
                                                                            Stride<_1, Stride<_32, Int<kBlockN*64>>,Int<16*64>, Int<kBlockN*kHeadDim>>{}));

    // PREDICATES

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)


    // Prologue


    // Read Q from gmem to smem, optionally apply rotary embedding.
    Tensor tNopeQrNopeQ = make_fragment_like(tNopeQgNopeQ);
    Tensor tRopeQrRopeQ = make_fragment_like(tRopeQgRopeQ);
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy_b128<Is_even_MN, Is_even_K>(tNopeQgNopeQ, tNopeQrNopeQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy_b128<Is_even_MN, Is_even_K>(tRopeQgRopeQ, tRopeQrRopeQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

    cute::copy(tNopeQrNopeQ, tNopeQsNopeQ);
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsNopeQ, tSrNopeQ);
    flash::sync_threads();

    cute::copy(tRopeQrRopeQ, tRopeQsRopeQ);
    flash::sync_threads();
    cute::copy(smem_tiled_copy_Q, tSsRopeQ, tSrRopeQ);
    flash::sync_threads();
    flash::concat(tSrNopeQ, tSrRopeQ, tSrQ);


    int n_block = n_block_max - 1;
    int Ksmem_read_index = 0;
    int Ksmem_write_index = 0;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor tKrK = make_fragment_like(tKgK);
    int row_offset = tidx / 32  + n_block * kBlockN;
    int virtual_page_idx = row_offset / params.page_block_size;
    int page_offset = row_offset - virtual_page_idx * params.page_block_size;
    int32_t page_idx = block_table[virtual_page_idx];
    flash::copy_b32_page_one<Kernel_traits, Is_even_MN, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block,
                                                                        block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, page_idx, page_offset, binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<Is_causal, Is_enable_dcp> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.ngroups, binfo.tot_seqlen_k, params.cp_world_size, params.cp_rank);

    for (; n_block >= n_block_min; --n_block) {
        if (n_block > n_block_min) {
            // prefetch load page index
            row_offset -= kBlockN;
            virtual_page_idx = row_offset / params.page_block_size;
            page_offset = row_offset - virtual_page_idx * params.page_block_size;
            page_idx = block_table[virtual_page_idx];
        }
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(tKrK, tKsK(_, _, _, Ksmem_write_index));
        Ksmem_write_index ^= 1;
        clear(acc_s);
        flash::sync_threads();
        if (n_block > n_block_min) {
            // Advance gK
           flash::copy_b32_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, page_idx, page_offset);
        }

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsNopeQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % kAtomLayoutMS * 16 + (tidx & 0xf), kAtomLayoutMS * 16
        );
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || !Is_even_MN, false, false>(acc_s, acc_o, params.scale_softmax_log2);

        //Tensor rP = flash::convert_type<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, _, Ksmem_read_index), tOrVt);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);
        Ksmem_read_index ^= 1;
    }

    // Epilogue
    if (NoSplit) {
        store_64x16_xcore1500<Kernel_traits, false, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }else{
        store_64x16_xcore1500<Kernel_traits, true, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }

}
} // namespace flash
