// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include "utils.h"

namespace flash {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0, const int warp_col_stride = 16) {
    // tensor has shape (nrow=(1, MMA_M), ncol=(4, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    static_assert(decltype(size<0, 0>(tensor))::value == 1);
    static_assert(decltype(size<1, 0>(tensor))::value == 4);
    const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * warp_col_stride;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}


template <bool Is_causal, bool Is_enable_dcp = false>
struct Mask {

    const int max_seqlen_k, max_seqlen_q, ngroups;
    // CP (Context Parallelism) parameters
    const int tot_seqlen_k, cp_world_size, cp_rank;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q, const int ngroups, const int tot_seqlen_k = 0, const int cp_world_size = 1, const int cp_rank = 0)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , ngroups(ngroups)
        , tot_seqlen_k(tot_seqlen_k)
        , cp_world_size(cp_world_size)
        , cp_rank(cp_rank) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, int Elem_per_thread = 4, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == Elem_per_thread, "The tensor_ first dimension not match the Elem_per_thread");
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;

        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !Causal_mask;
            const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 16;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 16;
                        const int col_idx_limit_right = std::min(tot_seqlen_k, row_idx / ngroups + 1 + tot_seqlen_k - max_seqlen_q / ngroups);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 16;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Is_enable_dcp) {
                                    // casusal mask with dcp
                                    const int actual_col_idx = col_idx * cp_world_size + cp_rank + 1;
                                    // actual_col_idx start from 1 to tot_seqlen_k
                                    if (actual_col_idx > col_idx_limit_right || col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                } else {
                                    // casusal mask without dcp
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    template <int kBlockTopK, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_sparse_attn_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               int32_t* indices_smem_ptr,
                                               bool is_indices_all_valid) {
        Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
        const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
            const int col_idx_base = col_idx_offset + nj * 16;
            #pragma unroll
            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                // const bool invalid_flag = !is_indices_all_valid && (CHECK_BIT(is_valid_indices[(col_idx % kBlockN) >> 5], col_idx % kBlockN) == false);
                const bool invalid_flag = !is_indices_all_valid && indices_smem_ptr[col_idx % kBlockTopK] < 0;
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    if constexpr (!Is_even_MN) {
                        if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                    }
                    if (invalid_flag) tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    };

};

} // namespace flash
