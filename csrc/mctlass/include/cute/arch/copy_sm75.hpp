/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>

// Config
#if defined(__clang__) && defined(__MACA__)
  // ldmatrix PTX instructions added in Clang 14: https://reviews.llvm.org/D107046
  // ... but will not work until Clang 15:
  //   * https://reviews.llvm.org/D121666
  //   * https://reviews.llvm.org/D126846
  #define CUTE_ARCH_CLANG_SUPPORTS_LDSM_SM75 (__clang_major__ >= 15)
#endif

#if defined(__MXCC__) || defined(__MACACC_RTC__)
  // ldmatrix PTX instruction added in CUDA 10.2+
  #define CUTE_ARCH_NVCC_SUPPORTS_LDSM_SM75 ((__CUDACC_VER_MAJOR__  == 10 && __CUDACC_VER_MINOR__ >= 2) || __CUDACC_VER_MAJOR__ >= 11)
#endif

#if ! defined(CUTE_ARCH_LDSM_SM75_SUPPORTED)
  #define CUTE_ARCH_LDSM_SM75_SUPPORTED (CUTE_ARCH_NVCC_SUPPORTS_LDSM_SM75 || CUTE_ARCH_CLANG_SUPPORTS_LDSM_SM75)
#endif

#if ! defined(CUTE_ARCH_LDSM_SM75_ENABLED)
  #define CUTE_ARCH_LDSM_SM75_ENABLED (CUTE_ARCH_LDSM_SM75_SUPPORTED)
#endif

#if 0
  #define CUTE_ARCH_LDSM_SM75_ACTIVATED 1
#endif

namespace cute
{

struct SM75_U32x1_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U32x2_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U32x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));

#elif defined(__MACA_ARCH__)
    const int lane_id = __lane_id();
    if (lane_id >= 32) return;
    uint64_t sm_ptr = reinterpret_cast<uint64_t>(&smem_src);
    uint64_t row_ptr[32];
    for (int i = 0; i < 32; ++i) {
      row_ptr[i] = __shfl_down_sync(ULONG_MAX, sm_ptr, i - lane_id);
    }

    const int row_id = lane_id / 4;
    const int col_offset = lane_id % 4;
    dst0 = *(reinterpret_cast<uint32_t *>(row_ptr[0 + row_id]) + col_offset);
    dst1 = *(reinterpret_cast<uint32_t *>(row_ptr[8 + row_id]) + col_offset);
    dst2 = *(reinterpret_cast<uint32_t *>(row_ptr[16 + row_id]) + col_offset);
    dst3 = *(reinterpret_cast<uint32_t *>(row_ptr[24 + row_id]) + col_offset);
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U32x4_LDSM_N_B
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)

    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#elif defined(__MACA_ARCH__)
    const int lane_id = __lane_id();
    if (lane_id >= 32) return;
    uint64_t sm_ptr = reinterpret_cast<uint64_t>(&smem_src);
    uint64_t row_ptr[32];
    for (int i = 0; i < 32; ++i) {
      row_ptr[i] = __shfl_down_sync(ULONG_MAX, sm_ptr, i - lane_id);
    }

    const int row_id = lane_id / 4;
    const int col_offset = lane_id % 4;
    dst0 = *(reinterpret_cast<uint32_t *>(row_ptr[0 + row_id]) + col_offset);
    dst1 = *(reinterpret_cast<uint32_t *>(row_ptr[8 + row_id]) + col_offset);
    dst2 = *(reinterpret_cast<uint32_t *>(row_ptr[16 + row_id]) + col_offset);
    dst3 = *(reinterpret_cast<uint32_t *>(row_ptr[24 + row_id]) + col_offset);
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U16x2_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(dst)
        :  "r"(smem_int_ptr));
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U16x4_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst0), "=r"(dst1)
        :  "r"(smem_int_ptr));
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

struct SM75_U16x8_LDSM_T
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM75_ACTIVATED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#elif defined(__MACA_ARCH__)
    const int lane_id = __lane_id();
    if (lane_id >= 32) return;
    uint64_t sm_ptr = reinterpret_cast<uint64_t>(&smem_src);
    uint64_t row_ptr[32];
    for (int i = 0; i < 32; ++i) {
      row_ptr[i] = __shfl_down_sync(ULONG_MAX, sm_ptr, i - lane_id);
    }

    const int row_offset = lane_id % 4 * 2;
    const int col_offset = lane_id / 4;
    auto low_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[0 + row_offset]) + col_offset;
    auto high_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[0 + row_offset + 1]) + col_offset;
    auto dst_b16 = reinterpret_cast<uint16_t *>(&dst0);
    *dst_b16 = *low_b16_addr;
    *(dst_b16 + 1) = *high_b16_addr;

    low_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[8 + row_offset]) + col_offset;
    high_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[8 + row_offset + 1]) + col_offset;
    dst_b16 = reinterpret_cast<uint16_t *>(&dst1);
    *dst_b16 = *low_b16_addr;
    *(dst_b16 + 1) = *high_b16_addr;

    low_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[16 + row_offset]) + col_offset;
    high_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[16 + row_offset + 1]) + col_offset;
    dst_b16 = reinterpret_cast<uint16_t *>(&dst2);
    *dst_b16 = *low_b16_addr;
    *(dst_b16 + 1) = *high_b16_addr;

    low_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[24 + row_offset]) + col_offset;
    high_b16_addr = reinterpret_cast<uint16_t *>(row_ptr[24 + row_offset + 1]) + col_offset;
    dst_b16 = reinterpret_cast<uint16_t *>(&dst3);
    *dst_b16 = *low_b16_addr;
    *(dst_b16 + 1) = *high_b16_addr;
#else
    CUTE_RUNTIME_ASSERT("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM75_ACTIVATED.");
#endif
  }
};

//
// Legacy LDSM interfaces that aren't very useful
//

template <class T>
CUTE_HOST_DEVICE
void
copy_ldsm(uint128_t const* const smem_ptr,
          T* rmem_ptr)
{
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    SM75_U32x1_LDSM_N::copy(smem_ptr[0], reg_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    SM75_U32x2_LDSM_N::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1]);
  }
  else if (sizeof(T) == 16) {
    SM75_U32x4_LDSM_N::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

template <class T>
CUTE_HOST_DEVICE
void
copy_ldsm_trans(uint128_t const* const smem_ptr,
                T* rmem_ptr)
{
  uint32_t* reg_ptr = reinterpret_cast<uint32_t*>(rmem_ptr);

  // if constexpr
  if (sizeof(T) == 4) {
    SM75_U16x2_LDSM_T::copy(smem_ptr[0], reg_ptr[0]);
  }
  else if (sizeof(T) == 8) {
    SM75_U16x4_LDSM_T::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1]);
  }
  else if (sizeof(T) == 16) {
    SM75_U16x8_LDSM_T::copy(smem_ptr[0], reg_ptr[0], reg_ptr[1], reg_ptr[2], reg_ptr[3]);
  }
  else {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, "sizeof(T) is not supported");
  }
}

} // end namespace cute
