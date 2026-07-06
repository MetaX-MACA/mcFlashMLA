// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#pragma once

#include <cuda.h>

#include "flash_fwd_launch_template.h"
#include "flash_mla.h"
#include "static_switch.h"

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    bool Is_splits = false,
    int kHeadDimV = kHeadDim,
    int Num_Stages = 1,
    Arch arch = Arch::xcore1000
>
void run_flash_splitkv_fwd_mla_template(mcFlashAttn::Flash_fwd_mla_params &params, cudaStream_t stream){
    using Kernel_traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type, Is_splits, kHeadDimV, Num_Stages>;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_splitkv_fwd_mla<Kernel_traits, Is_causal, arch>(params, stream);
    });
}

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    bool Is_splits = false,
    int kHeadDimV = kHeadDim,
    int Num_Stages = 1,
    Arch arch = Arch::xcore1000
>
void run_flash_splitkv_fwd_sparse_mla_template(mcFlashAttn::Flash_fwd_mla_params &params, cudaStream_t stream){
    using Kernel_traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type, Is_splits, kHeadDimV, Num_Stages>;
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_flash_splitkv_fwd_sparse_mla<Kernel_traits, Is_causal, arch>(params, stream);
    });
}

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    bool Is_splits = false,
    int kHeadDimV = kHeadDim,
    int Num_Stages = 1,
    Arch arch = Arch::xcore1000
>
void run_flash_mla_sparse_prefill_template(SparsePrefillParams &params, cudaStream_t stream){
    using Kernel_traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kNWarps, Is_Q_in_regs, Share_Q_K_smem, elem_type, Is_splits, kHeadDimV, Num_Stages>;

    run_sparse_prefill<Kernel_traits, false, arch>(params, stream);

}