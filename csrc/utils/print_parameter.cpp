#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "flash_mla.h"
#include "logger.h"
#include "process_str.h"
#include "print_parameter.h"

std::stringstream process_params(mcFlashAttn::Flash_fwd_mla_params params, bool Is_causal, const std::string& debug_flag) {
    /*

        ==============================Parts that require special handling（cu_seqlens）==============================

    */
    std::stringstream cu_seqlen_q;
    std::stringstream cu_seqlen_k;
    if (params.cu_seqlens_q != nullptr) {
        std::vector<int> host_cuseq_q(params.b + 1);
        cudaMemcpy(host_cuseq_q.data(), params.cu_seqlens_q, (sizeof(int) * (params.b + 1)), cudaMemcpyDeviceToHost);
        for(int i = 0; i < (params.b + 1); ++i) {
            if(cu_seqlen_q.str().size() == 0)
            {
                cu_seqlen_q << "[";
            } else if(cu_seqlen_q.str().size() > 1) {
                cu_seqlen_q << "-";
            }
            cu_seqlen_q << std::to_string(host_cuseq_q[i]);
        }
        cu_seqlen_q << "]";
    } else {
        cu_seqlen_q << "[nil]";
    }

    if (params.cu_seqlens_k != nullptr) {
        // when debug_flag is kvcache, cu_seqlens_k_size == batch_size
        // when debug_flag is fwd & bwd, cu_seqlens_k_size == batch_size + 1
        const int seq_k_size = debug_flag == "kvcache" ? params.b : (params.b + 1);//?
        std::vector<int> host_cuseq_k(seq_k_size);
        cudaMemcpy(host_cuseq_k.data(), params.cu_seqlens_k, (sizeof(int) * (seq_k_size)), cudaMemcpyDeviceToHost);
        for(int i = 0; i < seq_k_size; ++i) {
            if(cu_seqlen_k.str().size() == 0)
            {
                cu_seqlen_k << "[";
            } else if(cu_seqlen_k.str().size() > 1) {
                cu_seqlen_k << "-";
            }
            cu_seqlen_k << std::to_string(host_cuseq_k[i]);
        }
        cu_seqlen_k << "]";
    } else {
        cu_seqlen_k << "[nil]";
    }

    /*

        ==============================The part where the Bool_switch is printed==============================

    */

    bool Split = params.num_splits > 1;

    std::vector<std::string> bool_info{
                                /*================
                                The unique part of the fwd and shared parts.
                                ================*/
                                std::to_string(params.is_bf16),
                                std::to_string(Is_causal),
                                std::to_string(params.is_seqlens_k_cumulative),
                                std::to_string(params.unpadded_lse),
                                std::to_string(Split),
                                std::to_string(params.is_sparse_attn),
    };
    /*

        ==============================The part where the Dim_info is printed==============================

    */
    std::vector<std::string> dim_info{
                                std::to_string(params.b),
                                std::to_string(params.seqlen_q),
                                std::to_string(params.seqlen_k),
                                std::to_string(params.ngroups),
                                std::to_string(params.topk),
                                std::to_string(params.h),
                                std::to_string(params.h_k),
                                std::to_string(params.d),
                                std::to_string(params.d_value),
                                std::to_string(params.d_value_rounded),
                                std::to_string(params.num_splits),
                                std::to_string(params.page_block_size),
                                std::to_string(params.scale_softmax),
                                std::to_string(params.seqlen_knew),
                                std::to_string(params.seqlen_q_rounded),
                                std::to_string(params.seqlen_k_rounded),
                                std::to_string(params.d_rounded),
                                std::to_string(params.rotary_dim),
                                /*================
                                cu_seqlen
                                ================*/
                                cu_seqlen_q.str(),
                                cu_seqlen_k.str(),
    };

    return concat_total_strs("MLA", debug_flag, bool_info, dim_info);
}

void shape_print(mcFlashAttn::Flash_fwd_mla_params params, bool Is_causal, const std::string& debug_flag) {
    auto total_strs = process_params(params, Is_causal, debug_flag);
    /*
    Bool Switch:
                is_bf16, is_causal, is_seqlens_k_cumulative, unpadded_lse, Split, is_sparse_attn
    Dim_info:
                b, seqlen_q, seqlen_k, ngroups, topk, h, h_k, d, d_value, d_value_rounded, num_splits, page_block_size
                scale_softmax, seqlen_knew, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim,
    cu_seqlens:
                cu_seqlens_q, cu_seqlens_k,
    */
   LOG_SHAPE("%s\n", total_strs.str().c_str());
}

void shape_print(SparsePrefillParams params, bool Is_causal, const std::string& debug_flag) {
    /*
    Bool Switch:
                is_causal
    Dim_info:
                s_q, s_kv, h_q, h_kv, d_qk, d_v, topk
                sm_scale, sm_scale_div_log2
    */
   std::vector<std::string> bool_info{
                                std::to_string(Is_causal),
    };
    /*

        ==============================The part where the Dim_info is printed==============================

    */
    std::vector<std::string> dim_info{
                                std::to_string(params.s_q),
                                std::to_string(params.s_kv),
                                std::to_string(params.h_q),
                                std::to_string(params.h_kv),
                                std::to_string(params.d_qk),
                                std::to_string(params.d_v),
                                std::to_string(params.topk),
                                std::to_string(params.sm_scale),
                                std::to_string(params.sm_scale_div_log2),
    };
   auto total_strs = concat_total_strs("MLA", debug_flag, bool_info, dim_info);
   LOG_SHAPE("%s\n", total_strs.str().c_str());
}


void debug_print(mcFlashAttn::Flash_fwd_mla_params params, const std::string& debug_flag) {
    printf("==============%s-debug parameters recored start...\n", debug_flag.c_str());

    printf("----rng_state_seed=%d\n",params.rng_state_seed);
    printf("----rng_state_offset=%d\n",params.rng_state_offset);
    printf("----p_ptr=%p\n", params.p_ptr);
    printf("----rp_dropout=%f\n",params.rp_dropout);

    printf("----rotary_cos_ptr=%p\n", params.rotary_cos_ptr);
    printf("----rotary_sin_ptr=%p\n", params.rotary_sin_ptr);
    printf("----cache_batch_idx=%p\n", params.cache_batch_idx);
    printf("----block_table=%p\n", params.block_table);
    printf("----block_table_batch_stride=%ld\n", params.block_table_batch_stride);
    printf("----page_block_size=%d\n", params.page_block_size);
    printf("----knew_ptr=%p\n",params.knew_ptr);
    printf("----vnew_ptr=%p\n",params.vnew_ptr);
    printf("----oaccum_ptr=%p\n", params.oaccum_ptr);
    printf("----num_splits=%d\n",params.num_splits);
    printf("----softmax_lse_ptr=%p\n", params.softmax_lse_ptr);
    printf("----softmax_lseaccum_ptr=%p\n", params.softmax_lseaccum_ptr);

    printf("----softmax_lse_ptr=%p\n", params.softmax_lse_ptr);
    printf("----seqused_k=%p\n",params.seqused_k);
    printf("----q_ptr=%p\n", params.q_ptr);
    printf("----k_ptr=%p\n", params.k_ptr);
    printf("----v_ptr=%p\n", params.v_ptr);
    printf("----o_ptr=%p\n", params.o_ptr);
    printf("----scale_softmax=%f\n",params.scale_softmax);
    printf("----scale_softmax_log2=%f\n",params.scale_softmax_log2);
    printf("----o_batch_stride=%ld\n", params.o_batch_stride);
    printf("----o_row_stride=%ld\n",params.o_row_stride);
    printf("----o_head_stride=%ld\n",params.o_head_stride);
    printf("----q_batch_stride=%ld\n",params.q_batch_stride);
    printf("----k_batch_stride=%ld\n",params.k_batch_stride);
    printf("----v_batch_stride=%ld\n",params.v_batch_stride);
    printf("----q_row_stride=%ld\n",params.q_row_stride);
    printf("----k_row_stride=%ld\n",params.k_row_stride);
    printf("----v_row_stride=%ld\n",params.v_row_stride);
    printf("----q_head_stride=%ld\n",params.q_head_stride);
    printf("----k_head_stride=%ld\n",params.k_head_stride);
    printf("----v_head_stride=%ld\n",params.v_head_stride);
    printf("----unpadded lse=%d\n", params.unpadded_lse);

    printf("----d_value=%d\n", params.d_value);
    printf("----d_value_rounded=%d\n", params.d_value_rounded);
    printf("----num_sm_parts=%d\n", params.num_sm_parts);
    printf("==============%s-debug parameters recored end...\n", debug_flag.c_str());
}

void debug_print(SparsePrefillParams params, const std::string& debug_flag) {
    printf("==============%s-debug parameters recored start...\n", debug_flag.c_str());

    printf("----s_q=%d\n",params.s_q);
    printf("----s_kv=%d\n",params.s_kv);
    printf("----h_q=%d\n",params.h_q);
    printf("----h_kv=%d\n",params.h_kv);
    printf("----d_qk=%d\n",params.d_qk);
    printf("----d_v=%d\n",params.d_v);
    printf("----topk=%d\n",params.topk);

    printf("----sm_scale=%f\n",params.sm_scale);
    printf("----sm_scale_div_log2=%f\n",params.sm_scale_div_log2);

    printf("==============%s-debug parameters recored end...\n", debug_flag.c_str());
}
