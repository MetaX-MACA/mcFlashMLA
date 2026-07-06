// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#include <ATen/cuda/CUDAContext.h>
#include <mctlass/numeric_types.h>
#include "run_mla.h"
#include "static_switch.h"
#include "flash_fwd_dispatch_template.h"

void run_mla_fwd(mcFlashAttn::Flash_fwd_mla_params &params, cudaStream_t stream) {

    constexpr int kHeadDim = 576;
    ARCH_SWITCH(params.arch, kArch, [&] {
        mcFlashAttn::run_mla_fwd_splitkv_dispatch<kHeadDim, kArch>(params, stream);
    });
}


void run_mla_fwd(SparsePrefillParams &params) {

    constexpr int kHeadDim = 576;
    ARCH_SWITCH(params.arch, kArch, [&] {
        mcFlashAttn::run_flash_mla_sparse_prefill_dispatch<kHeadDim, kArch>(params, params.stream);
    });
}
