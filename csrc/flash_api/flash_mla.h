// Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>
#include "mctlass/bfloat16.h"
#include "static_switch.h"
constexpr int maxValidBlockSizeM = 128;

namespace mcFlashAttn {

struct Qkv_params {
    using index_t = int64_t;

    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t indices_batch_stride;
    index_t indices_row_stride;
    index_t indices_all_valid_per_q_batch_stride;
    index_t indices_all_valid_per_q_row_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, d_v, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;
    int ngroups;
    bool is_sparse_attn = false;
    int topk;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;
    int *__restrict__ indices_ptr;   // [batch, s_q, topk]

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    // when page attn is not enable, page_block_size will has default value 0.
    int page_block_size;

    // KV Cache dequant
    int dequant_group;
    void *__restrict__ k_scale_ptr;
    void *__restrict__ v_scale_ptr;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;

    // the RNG seed and offset .
    uint64_t rng_state_seed = 0;
    uint64_t rng_state_offset = 0;

    bool is_bf16;
    bool is_fp8 = false;
    bool is_causal;
    bool* indices_all_valid_per_q_ptr;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    int num_splits;  // For split-KV version

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    int d_value;
    int d_value_rounded;
    int arch;

    int *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    // fp8 params
    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;

    // CP (Context Parallelism) parameters
    int cp_world_size;
    int cp_rank;
    int *__restrict__ cp_tot_seqused_k;

    cudaStream_t stream;
};


struct Flash_launch_params {
    bool is_balance;
    int rowblock_parallel;
    int block_type;

    bool performance_mode; // from offline

    Flash_launch_params():
        is_balance(false),rowblock_parallel(0),block_type(0),performance_mode(false){}
};
}

struct SparsePrefillParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale, sm_scale_div_log2;
    int arch;

    // Input tensors
    mctlass::bfloat16_t* __restrict__ q_ptr;    // [s_q, h_q, d_qk]
    mctlass::bfloat16_t* __restrict__ kv_ptr;   // [s_kv, h_kv, d_qk]
    int* __restrict__ indices_ptr;   // [s_q, h_kv, topk]
    bool* indices_all_valid_per_q_ptr; // [1]

    // int stride_q_s_q; int stride_q_h_q;
    // int stride_kv_s_kv; int stride_kv_h_kv;
    int q_row_stride;int q_head_stride;
    int k_row_stride;int k_head_stride;
    int stride_indices_s_q; int stride_indices_h_kv;
    int o_row_stride;int o_head_stride;
    // Output tensors
    mctlass::bfloat16_t* __restrict__ out_ptr;   // [s_q, h_q, d_v]
    float* __restrict__ max_logits; // [s_q, h_q]
    float* __restrict__ lse_ptr; // [s_q, h_q]

    cudaStream_t stream;
};

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

struct GetDecodingMetadataParams {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
    int topk;
};
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};

void run_get_mla_metadata_kernel(GetDecodingMetadataParams &params, cudaStream_t stream);
