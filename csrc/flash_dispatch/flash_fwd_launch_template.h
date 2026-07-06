// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#pragma once

#include <cuda.h>

#include "flash_mla.h"
#include "static_switch.h"
#include "flash_dense_mla_decode_kernel.h"
#include "flash_sparse_mla_decode_kernel.h"
#include "flash_fwd_splitkv_mla_combine_kernel.h"

#include "xcore1000/sparse_prefill_kernel_64x16_8waves_xcore1000.h"
#include "print_parameter.h"
using namespace mcFlashAttn;

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_enable_dcp>
__global__ void flash_fwd_splitkv_mla_kernel(const Flash_fwd_mla_params params, const int num_m_block) {
    flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_enable_dcp>(params, num_m_block);
}
template<typename Kernel_traits, bool Is_causal, bool Is_even_TopK>
__global__ void flash_fwd_splitkv_sparse_mla_kernel(const Flash_fwd_mla_params params, const int num_m_block) {
    flash::compute_attn_1rowblock_splitkv_sparse_mla<Kernel_traits, Is_causal, Is_even_TopK>(params, num_m_block);
}
template<typename Kernel_traits, bool Is_causal, bool Is_even_TopK>
__global__ void sparse_attn_global_fwd_kernel(const SparsePrefillParams params) {
    flash::sparse_attn_fwd_kernel<Kernel_traits, Is_causal, Is_even_TopK>(params);
}
//flash-meta combine kernel
template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K>
__global__ void flash_fwd_splitkv_mla_combine_kernel(const Flash_fwd_mla_params params) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel_splitkv_mla<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_causal, Arch arch>
void run_flash_splitkv_fwd_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    constexpr int max_smem_size = arch == Arch::xcore1000 ? 64 * 1024 : 128 * 1024;
    constexpr int smem_size = std::min(Kernel_traits::kSmemSize, max_smem_size);
    // const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    // dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    dim3 grid(num_m_block, params.h, params.num_sm_parts);
    static_assert(Kernel_traits::kHeadDim == 576 && Kernel_traits::kHeadDimV == 512);

    const bool is_even_K = params.d == Kernel_traits::kHeadDim && params.d_v == Kernel_traits::kHeadDimV;
    BOOL_SWITCH(params.cp_world_size > 1 && params.cp_tot_seqused_k != nullptr, IsEnableDcp, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (std::getenv("MHA_PRINT_PARA")) {
                shape_print(params, Is_causal, "mla_dense_decode");
            }
            if (std::getenv("MHA_DEBUG_PARA")){
                debug_print(params, "mla_dense_decode");
            }
            auto kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, false, IsEvenKConst, IsEnableDcp>;
            if (smem_size >= 32 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block);
            CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
    // We want kBlockM to be as small as possible for more parallelism.
    // In MLA case head_dim_vo = 512, we will switch different kBlockM for different case to get better performance
    COMBINE_BLOCKM_SWITCH(params.b, params.h, params.seqlen_q, kBlockM, [&] {
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            NUMSPLITS_SWITCH(params.num_sm_parts, kLogMaxSplits, [&] {
                const int kNThreads = kBlockM == 1 ? 64 : (kBlockM == 2 ? 128 : 256);
                flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kBlockM, kLogMaxSplits, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
                CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });

}

template<typename Kernel_traits, bool Is_causal, Arch arch>
void run_flash_splitkv_fwd_sparse_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    constexpr int max_smem_size = arch == Arch::xcore1000 ? 64 * 1024 : 128 * 1024;
    constexpr int smem_size = std::min(Kernel_traits::kSmemSize, max_smem_size);
    // const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    // dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    dim3 grid(num_m_block, params.h, params.num_sm_parts);
    static_assert(Kernel_traits::kHeadDim == 576 && Kernel_traits::kHeadDimV == 512);

    // const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = true; // is_even_k is always true in mla case;
    const bool is_even_topK = params.topk % Kernel_traits::kBlockN == 0;
    BOOL_SWITCH(is_even_topK, IsEvenTopKConst, [&] {
        if (std::getenv("MHA_PRINT_PARA")) {
            shape_print(params, Is_causal, "mla_sparse_decode");
        }
        if (std::getenv("MHA_DEBUG_PARA")){
            debug_print(params, "mla_sparse_decode");
        }
        auto kernel = &flash_fwd_splitkv_sparse_mla_kernel<Kernel_traits, Is_causal, IsEvenTopKConst>;
        if (smem_size >= 32 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block);
        CUDA_KERNEL_LAUNCH_CHECK();
    });
    // We want kBlockM to be as small as possible for more parallelism.
    // In MLA case head_dim_vo = 512, we will switch different kBlockM for different case to get better performance
    COMBINE_BLOCKM_SWITCH(params.b, params.h, params.seqlen_q, kBlockM, [&] {
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            NUMSPLITS_SWITCH(params.num_sm_parts, kLogMaxSplits, [&] {
                const int kNThreads = kBlockM == 1 ? 64 : (kBlockM == 2 ? 128 : 256);
                flash_fwd_splitkv_mla_combine_kernel<Kernel_traits, kBlockM, kLogMaxSplits, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
                CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal, Arch arch>
void run_sparse_prefill(SparsePrefillParams &params, cudaStream_t stream) {
    constexpr int max_smem_size = arch == Arch::xcore1000 ? 64 * 1024 : 128 * 1024;
    constexpr int smem_size = std::min(Kernel_traits::kSmemSize, max_smem_size);

    dim3 grid((params.h_q/Kernel_traits::kBlockM)*params.s_q, 1, 1);
    static_assert(Kernel_traits::kHeadDim == 576 && Kernel_traits::kHeadDimV == 512);

    const bool is_even_topK = params.topk % Kernel_traits::kBlockN == 0;
    // WARNING: Be aware of the correctness of this condition
    BOOL_SWITCH(is_even_topK, IsEvenTopKConst, [&] {
        if (std::getenv("MHA_PRINT_PARA")) {
            shape_print(params, Is_causal, "mla_sparse_prefill");
        }
        if (std::getenv("MHA_DEBUG_PARA")){
            debug_print(params, "mla_sparse_prefill");
        }
        auto kernel = &sparse_attn_global_fwd_kernel<Kernel_traits, /*is_causal*/false, IsEvenTopKConst>;
        if (smem_size >= 32 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
        CUDA_KERNEL_LAUNCH_CHECK();
    });


}