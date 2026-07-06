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

template<typename Kernel_traits, bool Is_causal, bool Is_even_TopK, typename Params>
__forceinline__ __device__ void sparse_attn_fwd_kernel(const Params &params) {
    constexpr bool Is_local = false;

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
    constexpr int kBlockTopK = Kernel_traits::kBlockTopK;
    constexpr int kHeadDimNope = kHeadDimV;
    constexpr int kHeadDimRope = kHeadDim - kHeadDimV;

    static_assert(kBlockKSmem == 64);
    static_assert(Kernel_traits::Share_Q_K_smem && Kernel_traits::Is_Q_in_regs);

    using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyO;

    using ElementO =  Element;

    const int bidb = 0;
    const int h_q_idx = blockIdx.x % (params.h_q / kBlockM); //q_h_idx
    const int s_q_idx = blockIdx.x / (params.h_q / kBlockM); //s_q_idx
    const int q_block_idx = h_q_idx * kBlockM;
    // const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (q_block_idx >= params.h_q) return;

    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(params.topk, kBlockN);
    // if (Is_causal || Is_local) {
    //     n_block_max = std::min(n_block_max,
    //                            cute::ceil_div((m_block + 1) * kBlockM + params.s_kv - params.s_q / params.ngroups + params.window_size_right, kBlockN));
    //

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const index_t row_offset_q = q_block_idx  * params.q_head_stride + s_q_idx  * params.q_row_stride;
    const index_t row_offset_k = (n_block_max - 1) * kBlockN * params.k_row_stride;
    const index_t offset_indices = s_q_idx * params.stride_indices_s_q;
    // We move K and V to the last block.
    // const int bidb_cache = bidb;
    // const int *block_table = nullptr;
    const int* gIndices = params.indices_ptr + offset_indices;
    const bool indices_all_valid_per_q = params.indices_all_valid_per_q_ptr[s_q_idx];

    Tensor gNopeQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDimNope>>{},
                            make_stride(params.q_head_stride, _1{}));
    Tensor gRopeQ = make_tensor(make_gmem_ptr(gNopeQ.data().get() + kHeadDimNope),
                            Shape<Int<kBlockM>, Int<kHeadDimRope>>{},
                            make_stride(params.q_head_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.kv_ptr)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sNopeQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutNopeQ{});
    Tensor sRopeQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutRopeQ{});
    Tensor sK = make_tensor(sNopeQ.data(),
                            typename Kernel_traits::SmemLayoutK424{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed424{});

    int32_t* indices_smem_ptr = reinterpret_cast<int32_t *>(reinterpret_cast<char *>(smem_) + (size(sK) * sizeof(Element) + 3) / 4 * 4);

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
    // Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrVt = make_tensor<Element>(Shape<_4, Shape<_4, _4>, _1>{});

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
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    int warp_offset = warp_idx / kAtomLayoutMO * 16 * 64;
    int thread_offset = lane_idx / 16 * 4 * 64;
    Element *Vtsmem_ptr_lds = reinterpret_cast<Element *>(sVt.data().get()) + warp_offset + thread_offset;
    Tensor tOsVt = make_tensor(make_smem_ptr(Vtsmem_ptr_lds), make_layout(Shape<_4, _4, Int<Num_Stages>>{},                  // MMA  MMA_N  NUM_STAGES
                                                                          Stride<_1, Int<16*128>, Int<kBlockN*kHeadDim>>{}));

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
    flash::copy_b128</*Is_even_MN*/ true, true>(tNopeQgNopeQ, tNopeQrNopeQ, tQcQ, params.d_qk, params.h_q - q_block_idx);
    flash::copy_b128</*Is_even_MN*/ true, true>(tRopeQgRopeQ, tRopeQrRopeQ, tQcQ, params.d_qk, params.h_q - q_block_idx);

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

    int indices_start_block = (n_block * kBlockN) / kBlockTopK * kBlockTopK;
    int offset_indices_per_q = indices_start_block + (tidx << 2);
    int4 indices_vec = {-1, -1, -1, -1};

    if (offset_indices_per_q + 4 <= params.topk) {
        indices_vec = __ldg(reinterpret_cast<const int4*>(&gIndices[offset_indices_per_q]));
    }
    else if (offset_indices_per_q < params.topk) {
        indices_vec.x = offset_indices_per_q + 0 < params.topk ? gIndices[offset_indices_per_q + 0] : -1;
        indices_vec.y = offset_indices_per_q + 1 < params.topk ? gIndices[offset_indices_per_q + 1] : -1;
        indices_vec.z = offset_indices_per_q + 2 < params.topk ? gIndices[offset_indices_per_q + 2] : -1;
        indices_vec.w = offset_indices_per_q + 3 < params.topk ? gIndices[offset_indices_per_q + 3] : -1;
    }

    *((int4*)(&indices_smem_ptr[offset_indices_per_q % kBlockTopK])) = indices_vec;
    flash::sync_threads();

    uint32_t row_offset = tidx / 32  + n_block * kBlockN;
    int32_t topk_sparse_idx = indices_smem_ptr[row_offset % kBlockTopK];
    topk_sparse_idx = topk_sparse_idx < 0 ? 0 : topk_sparse_idx;

    flash::copy_b32_sparse<Kernel_traits, /*Is_even_MN=*/false, /*Is_even_K=*/true>(gK, tKgK, tKrK, tKVcKV, params.d_qk, n_block, params.k_row_stride, topk_sparse_idx, params.topk - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<Is_causal> mask(params.topk, params.s_q, params.h_q);
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(tKrK, tKsK(_, _, _, Ksmem_write_index));
        Ksmem_write_index ^= 1;
        clear(acc_s);
        flash::sync_threads();
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsNopeQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        mask.template apply_sparse_attn_mask<kBlockTopK, /*Is_even_MN=*/true>(acc_s, n_block * kBlockN, indices_smem_ptr, indices_all_valid_per_q);

        if ((n_block * kBlockN) % kBlockTopK == 0 && n_block >  n_block_min) {
            flash::barrier();
            indices_start_block = ((n_block - 1) * kBlockN) / kBlockTopK * kBlockTopK;
            offset_indices_per_q = indices_start_block + (tidx << 2);
            int4 indices_vec = __ldg(reinterpret_cast<const int4*>(&gIndices[offset_indices_per_q]));
            *((int4*)(&indices_smem_ptr[offset_indices_per_q % kBlockTopK])) = indices_vec;
            flash::sync_threads();
        }
        n_block == n_block_max - 1
            ? softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true, true, true>(acc_s, acc_o, params.sm_scale_div_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/true, true, true>(acc_s, acc_o, params.sm_scale_div_log2);

        if (n_block > n_block_min) {
            row_offset -= kBlockN;
            topk_sparse_idx = indices_smem_ptr[row_offset % kBlockTopK];
            topk_sparse_idx = topk_sparse_idx < 0 ? 0 : topk_sparse_idx;
            flash::copy_b32_sparse<Kernel_traits, /*Is_even_MN=*/true, true>(gK, tKgK, tKrK, tKVcKV, params.d_qk, n_block - 1, params.k_row_stride, topk_sparse_idx);
        }
        //Tensor rP = flash::convert_type<Element>(acc_s);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        lds4x4_with_swizzle424(tOsVt(_, _, Ksmem_read_index), tOrVt);
        CUTE_STATIC_ASSERT_V(size<2>(tOrVt) == _1{}); // only support MMA_K = 1
        Tensor tOrVt_permute_view = make_tensor(tOrVt.data(), make_layout(make_shape(size<0>(tOrVt), size<1, 0>(tOrVt), size<1, 1>(tOrVt))));
        permute_4x4_b16(tOrVt_permute_view);
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);
        Ksmem_read_index ^= 1;
    }
    // if(thread0()){print(acc_o);}
    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Return_lse*/true, false>(acc_o, params.sm_scale);
    Tensor acc_o_view = make_tensor(acc_o.data(), make_layout(Shape<_4, Shape<_4, _4>>{},
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
    // if (cute::thread0()) { print(lse); }

    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomO;
    CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o_copy, rO)
    warp_offset = warp_idx * 16 * 64;
    thread_offset = lane_idx % 16 * 64 + lane_idx / 16 * 16;
    Element *Osmem_ptr_sts = reinterpret_cast<ElementO *>(smem_) + warp_offset + thread_offset;
    Tensor tOsO = make_tensor(make_smem_ptr(Osmem_ptr_sts), make_layout(Shape<_16, _4>{},
                                                                        Stride<_1, Int<16*64*kNWarps>>{}));
    Tensor tOrO = make_tensor(rO.data(), make_layout(Shape<_16, _4>{},
                                                            Stride<_1, _16>{}));


    if constexpr (Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }

    cute::copy(tOrO, tOsO);

    const index_t row_offset_o = q_block_idx * params.o_head_stride + s_q_idx * params.o_row_stride;
    const index_t row_offset_lseaccum = s_q_idx * params.h_q + q_block_idx;
    const index_t row_offset_max_logits = row_offset_lseaccum;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.out_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_head_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gMaxLogits = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.max_logits) + row_offset_max_logits),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, s_q_idx = %d, gOaccum = %p\n", row_offset_o, s_q_idx, gOaccum.data()); }
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
            if (row < params.h_q - q_block_idx) {
                gMaxLogits(row) = softmax.row_max(mi) * params.sm_scale * M_LOG2E;
                gLSEaccum(row) = lse(mi);
            }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy_reg_to_global</*Is_even_MN*/ true, true>(
        tOrOaccum, tOgOaccum, tOcO, params.d_v, params.h_q - q_block_idx
    );
}

} // namespace flash