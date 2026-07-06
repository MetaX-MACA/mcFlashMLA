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
#include "flash_fwd_mla_kernel_k64_64x16_8waves_xcore1000.h"

namespace flash {

using namespace cute;


template<typename Kernel_traits, bool Is_causal, bool Is_even_TopK, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_sparse_mla_k64_64x16_8waves_xcore1000(
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
    constexpr int kBlockTopK = Kernel_traits::kBlockTopK;
    constexpr int kHeadDimNope = kHeadDimV;
    constexpr int kHeadDimRope = kHeadDim - kHeadDimV;

    static_assert(kBlockKSmem == 64);
    static_assert(Kernel_traits::Share_Q_K_smem && Kernel_traits::Is_Q_in_regs);

    const BlockInfo<true> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];

    const int s_q_idx = params.ngroups >= kBlockM ? m_block / (params.ngroups / kBlockM) : 0;  //s_q_idx, ngroups = head_q_ori
    const index_t offset_indices = bidb * params.indices_batch_stride + s_q_idx * params.indices_row_stride;

    const int* gIndices = params.indices_ptr + offset_indices; // top_k values
    const bool indices_all_valid_per_q = params.indices_all_valid_per_q_ptr[bidb * params.indices_all_valid_per_q_batch_stride + s_q_idx];

    // will calculate row_offset_k later

    const index_t row_offset_k = (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gNopeQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDimNope>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gRopeQ = make_tensor(make_gmem_ptr(gNopeQ.data().get() + kHeadDimNope),
                            Shape<Int<kBlockM>, Int<kHeadDimRope>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sNopeQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutNopeQ{});
    Tensor sRopeQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutRopeQ{});
    Tensor sK = make_tensor(sNopeQ.data(), typename Kernel_traits::SmemLayoutK424{});           // kBlockN * kheadDim * NumStages
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});        // kBlockN * kHeadDimV
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed424{});            // kBlockN * kheadDimV * NumStages
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{}); // kBlockN * kheadDimV * NumStages

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
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                                   // (MMA,MMA_M,MMA_K)
    Tensor tSrNopeQ  = thr_mma_s.partition_fragment_A(sNopeQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrRopeQ  = thr_mma_s.partition_fragment_A(sRopeQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK(_, _, 0));                          // (MMA,MMA_N,MMA_K)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
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
    flash::copy_b128</*Is_even_MN=*/false, /*Is_even_K*/true>(tNopeQgNopeQ, tNopeQrNopeQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy_b128</*Is_even_MN=*/false, /*Is_even_K*/true>(tRopeQgRopeQ, tRopeQrRopeQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

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

    uint32_t row_offset = tidx / (kNWarps * 64 / kBlockN) + n_block * kBlockN;
    int32_t topk_sparse_idx = indices_smem_ptr[row_offset % kBlockTopK];
    topk_sparse_idx = topk_sparse_idx < 0 ? 0 : topk_sparse_idx;

    flash::copy_b32_sparse<Kernel_traits, /*Is_even_MN=*/false, /*Is_even_K=*/true>(gK, tKgK, tKrK, tKVcKV, params.d, n_block,
                                                                 params.d, topk_sparse_idx,
                                                                 params.topk - n_block * kBlockN);

    flash::cp_async_wait<0>();

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<Is_causal> mask(params.topk, binfo.actual_seqlen_q, params.ngroups);

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

        mask.template apply_sparse_attn_mask<kBlockTopK, false>(acc_s, n_block * kBlockN, indices_smem_ptr, indices_all_valid_per_q);

        if ((n_block * kBlockN) % kBlockTopK == 0 && n_block >  n_block_min) {
            flash::barrier();
            indices_start_block = ((n_block - 1) * kBlockN) / kBlockTopK * kBlockTopK;
            offset_indices_per_q = indices_start_block + (tidx << 2);
            indices_vec = __ldg(reinterpret_cast<const int4*>(&gIndices[offset_indices_per_q]));
            *((int4*)(&indices_smem_ptr[offset_indices_per_q % kBlockTopK])) = indices_vec;
            flash::sync_threads();
        }

        n_block == n_block_max - 1
            ? softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || true, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || true, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        if (n_block > n_block_min) {
            row_offset -= kBlockN;
            topk_sparse_idx = indices_smem_ptr[row_offset % kBlockTopK];
            topk_sparse_idx = topk_sparse_idx < 0 ? 0 : topk_sparse_idx;
            flash::copy_b32_sparse<Kernel_traits, /*Is_even_MN=*/true, true>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1, params.d, topk_sparse_idx);
        }

        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        lds4x4_with_swizzle424(tOsVt(_, _, Ksmem_read_index), tOrVt);
        CUTE_STATIC_ASSERT_V(size<2>(tOrVt) == _1{}); // only support MMA_K = 1
        Tensor tOrVt_permute_view = make_tensor(tOrVt.data(), make_layout(make_shape(size<0>(tOrVt), size<1, 0>(tOrVt), size<1, 1>(tOrVt))));
        permute_4x4_b16(tOrVt_permute_view);
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);

        Ksmem_read_index ^= 1;
    }

    // Epilogue
    if (NoSplit) {
        store_64x16_xcore1000<Kernel_traits, /*Split*/false, /*Is_even_MN=*/false, /*Is_even_K=*/true>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }else{
        store_64x16_xcore1000<Kernel_traits, /*Split*/true, /*Is_even_MN=*/false, /*Is_even_K=*/true>(params, bidb, bidh, m_block, n_split_idx, smem_, acc_o, softmax);
    }
}

} // namespace flash