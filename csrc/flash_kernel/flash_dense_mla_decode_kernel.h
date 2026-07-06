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


#include "xcore1000/flash_fwd_mla_kernel_k64_16x16_4waves_stage1_xcore1000.h"
// #include "xcore1000/flash_fwd_mla_kernel_k64_16x16_4waves_xcore1000.h"
#include "xcore1000/flash_fwd_mla_kernel_k64_32x16_4waves_xcore1000.h"
#include "xcore1000/flash_fwd_mla_kernel_k64_64x16_8waves_xcore1000.h"
#include "xcore1500/flash_fwd_mla_kernel_k64_64x16_8waves_xcore1500.h"
#include "xcore1500/flash_fwd_mla_kernel_k64_64x32_8waves_xcore1500.h"

#include "static_switch.h"

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_enable_dcp, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const Params &params, const int m_block_max) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    // int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int4 tile_scheduler_metadata = *reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr);
    int begin_idx = tile_scheduler_metadata.x;
    int sched_begin_block_idx = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int sched_end_block_idx = tile_scheduler_metadata.w;
    if (begin_idx >= params.b || begin_idx < 0) return;
    // int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);
    int begin_n_split_idx = tile_scheduler_metadata_ptr[4];

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = params.cu_seqlens_k[batch_id];
        const int n_block_min = batch_id == begin_idx ? sched_begin_block_idx : 0;
        const int n_block_max = batch_id == end_idx ? sched_end_block_idx : cute::ceil_div(seqlen_k, kBlockN);
        // [n_block_min, n_block_max) need be calculated in kernel
        if (n_block_max <= n_block_min) continue;
        const bool NoSplit = __ldg(params.num_splits_ptr + batch_id + 1) - __ldg(params.num_splits_ptr + batch_id) == 1;
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        #if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1500)
        if constexpr (Kernel_traits::kBlockM == 64 && Kernel_traits::kBlockN == 16 && Kernel_traits::kNWarps == 8) {
            compute_attn_1rowblock_splitkv_mla_k64_64x16_8waves_xcore1500<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_enable_dcp>(
                params, batch_id, bidh, m_block, n_split_idx, n_block_min, n_block_max, NoSplit);
        }else if constexpr (Kernel_traits::kBlockM == 64 && Kernel_traits::kBlockN == 32 && Kernel_traits::kNWarps == 8) {
            compute_attn_1rowblock_splitkv_mla_k64_64x32_8waves_xcore1500<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_enable_dcp>(
                params, batch_id, bidh, m_block, n_split_idx, n_block_min, n_block_max, NoSplit);
        }
        #else defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000)
        if constexpr (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 16 && Kernel_traits::kNWarps == 4) {
            compute_attn_1rowblock_splitkv_mla_k64_32x16_4waves_xcore1000<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_enable_dcp>(
                params, batch_id, bidh, m_block, n_split_idx, n_block_min, n_block_max, NoSplit);
        } else if constexpr (Kernel_traits::kBlockM == 64 && Kernel_traits::kBlockN == 16 && Kernel_traits::kNWarps == 8) {
            compute_attn_1rowblock_splitkv_mla_k64_64x16_8waves_xcore1000<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_enable_dcp>(
                params, batch_id, bidh, m_block, n_split_idx, n_block_min, n_block_max, NoSplit);
        } else if constexpr (Kernel_traits::kBlockM == 16 && Kernel_traits::kBlockN == 16 && Kernel_traits::kNWarps == 4) {
            compute_attn_1rowblock_splitkv_mla_k64_16x16_4waves_xcore1000<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_enable_dcp>(
                params, batch_id, bidh, m_block, n_split_idx, n_block_min, n_block_max, NoSplit);
        }
        #endif

    }
}

}// namespace flash
