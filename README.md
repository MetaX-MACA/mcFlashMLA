# FlashMLA on MXMACA
We provide the implementation of FlashMLA from FlashAttention-2(version 2.6.3), based on MACA toolkit and C500 chips.

FlashAttention-2 currently supports:
1. Datatype fp16 and bf16.
2. Multi-Token Prediction greater or equal to 1.
3. Paged kvcache with block size equal to 2^n (n >= 0).

## How to run on MXMACA Device
## Installation

Requirements:
- MXMACA GPUs.
- MACA development toolkit.
- Mctlass source code.
- Pytorch2.1 and Triton2.1 from maca toolkit wheel package and above.

To install in conda env:
1. Make sure that maca pyTorch2.1 and Triton2.1 are installed.
2. Download mctlass source code from: https://sw-download.metax-tech.com/

### Set environment variables
```bash
export MACA_PATH=/your/maca/path
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH
```