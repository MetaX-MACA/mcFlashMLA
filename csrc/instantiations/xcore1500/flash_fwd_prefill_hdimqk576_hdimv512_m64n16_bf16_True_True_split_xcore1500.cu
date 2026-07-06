// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#include "flash_mla.h"
#include "flash_fwd_run_template.h"
#include <mctlass/numeric_types.h>

template void run_flash_mla_sparse_prefill_template<
                576,
                64,
                16,
                8,
                true,
                true,
                mctlass::bfloat16_t,
                true,
                512,
                2,
                Arch::xcore1500
            >(SparsePrefillParams &params, cudaStream_t stream);
