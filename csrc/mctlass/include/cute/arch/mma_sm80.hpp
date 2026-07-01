 /**************************************************************************************************
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
#include <cute/arch/mma.hpp>
#include <maca_bfloat16.h>

// Config
#if defined(__MACA_ARCH__)
// #  define CUTE_ARCH_MMA_SM80_ENABLED
#endif

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct SM80_16x8x8_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},"
      "{%2, %3},"
      "{%4},"
      "{%5, %6};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct SM80_16x8x8_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x8_F32F16F16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TN
struct MACA_16x16x16_F32F16F16F32
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using VectorType = __NATIVE_VECTOR__(2, uint32_t);

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {

    VectorType a = {a0, a1};
    VectorType b = {b0, b1};

    auto result = __builtin_mxc_mma_16x16x16f16(b, a, {c0, c1, c2, c3});
    d0 = result[0];
    d1 = result[1];
    d2 = result[2];
    d3 = result[3];

  }
};

// MMA 16x16x32 TN
struct MACA_16x16x32_F32F16F16F32
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];
  using VectorType = __NATIVE_VECTOR__(2, uint32_t);

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, // A, k0
      uint32_t const& a2, uint32_t const& a3, // A, k1
      uint32_t const& b0, uint32_t const& b1, // B, k0
      uint32_t const& b2, uint32_t const& b3, // B, k1
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {

    VectorType a = {a0, a1};
    VectorType b = {b0, b1};

    // first mma , A(k0) * B(k0)
    auto acc = __builtin_mxc_mma_16x16x16f16(b, a, {c0, c1, c2, c3});

    // second mma , A(k1) * B(k1), accum
    a = {a2, a3};
    b = {b2, b3};
    acc = __builtin_mxc_mma_16x16x16f16(b, a, {acc[0], acc[1], acc[2], acc[3]});

    d0 = acc[0];
    d1 = acc[1];
    d2 = acc[2];
    d3 = acc[3];
  }
};

// MMA 16x64x16 TN
// use for lds4x4 + perm4x4
struct MACA_16x64x16_F32F16F16F32
{
  using DRegisters = float[16];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[8];
  using CRegisters = float[16];
  using VectorType = __NATIVE_VECTOR__(2, uint32_t);

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      float         & d4, float         & d5, float         & d6, float         & d7,
      float         & d8, float         & d9, float         & d10, float         & d11,
      float         & d12, float         & d13, float         & d14, float         & d15,
      uint32_t const& a0, uint32_t const& a1, // A, k0
      uint32_t const& b0, uint32_t const& b1, // B, m0
      uint32_t const& b2, uint32_t const& b3, // B, m1
      uint32_t const& b4, uint32_t const& b5, // B, m2
      uint32_t const& b6, uint32_t const& b7, // B, m3
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      float const   & c4, float const   & c5, float const   & c6, float const   & c7,
      float const   & c8, float const   & c9, float const   & c10, float const   & c11,
      float const   & c12, float const   & c13, float const   & c14, float const   & c15)
  {

    VectorType a = {a0, a1};
    VectorType b = {b0, b1};

    auto acc0 = __builtin_mxc_mma_16x16x16f16(b, a, {c0, c1, c2, c3});
    d0 = acc0[0];
    d1 = acc0[1];
    d2 = acc0[2];
    d3 = acc0[3];

    b = {b2, b3};
    auto acc1 = __builtin_mxc_mma_16x16x16f16(b, a, {c4, c5, c6, c7});
    d4 = acc1[0];
    d5 = acc1[1];
    d6 = acc1[2];
    d7 = acc1[3];

    b = {b4, b5};
    auto acc2 = __builtin_mxc_mma_16x16x16f16(b, a, {c8, c9, c10, c11});
    d8 = acc2[0];
    d9 = acc2[1];
    d10 = acc2[2];
    d11 = acc2[3];

    b = {b6, b7};
    auto acc3 = __builtin_mxc_mma_16x16x16f16(b, a, {c12, c13, c14, c15});
    d12 = acc3[0];
    d13 = acc3[1];
    d14 = acc3[2];
    d15 = acc3[3];

  }
};

// MMA 16x16x16 TN
struct MACA_16x16x16_I32I8I8I32 {
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[1];
  using BRegisters = int32_t[1];
  using CRegisters = int32_t[4];

  CUTE_HOST_DEVICE static void fma(
            int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3,
            int32_t const &a0,
            int32_t const &b0,
            int32_t const &c0, int32_t const &c1, int32_t const &c2, int32_t const &c3) {

#if defined(CUTE_MACA_XCORE1000_ENABLED)
    auto result = __builtin_mxc_mma_16x16x16i8(b0, a0, {c0, c1, c2, c3});
    d0 = result[0];
    d1 = result[1];
    d2 = result[2];
    d3 = result[3];
#else
    CUTE_RUNTIME_ASSERT("Attempting to use MACA_16x16x16_I32I8I8I32 without CUTE_MACA_XCORE1000_ENABLED");
#endif

  }
};

// MMA 16x16x32 TN
struct MACA_16x16x32_I32I8I8I32 {
  using DRegisters = int32_t[4];
  using ARegisters = int32_t[2];
  using BRegisters = int32_t[2];
  using CRegisters = int32_t[4];

  CUTE_HOST_DEVICE static void fma(
            int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3,
            int32_t const &a0, int32_t const &a1,
            int32_t const &b0, int32_t const &b1,
            int32_t const &c0, int32_t const &c1, int32_t const &c2, int32_t const &c3) {

#if defined(CUTE_MACA_XCORE1500_ENABLED)
    auto result = __builtin_mxc_mma_16x16x32i8({b0, b1}, {a0, a1}, {c0, c1, c2, c3});
    d0 = result[0];
    d1 = result[1];
    d2 = result[2];
    d3 = result[3];
#else
    CUTE_RUNTIME_ASSERT("Attempting to use MACA_16x16x32_I32I8I8I32 without CUTE_MACA_XCORE1500_ENABLED");
#endif
  }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_F32F16F16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#elif defined(__MACA_ARCH__)

    uint32_t aa[2];
    uint32_t bb[2];
    const int lane_id = __lane_id();
    int delta = (lane_id % 32 % 8 * 4 + lane_id % 32 / 16 * 2) - lane_id;

    // shuffle A
    uint32_t tmp0 = __shfl_down_sync(ULONG_MAX, a0, delta);
    uint32_t tmp0_1 = __shfl_down_sync(ULONG_MAX, a0, delta + 1);
    uint32_t tmp1 = __shfl_down_sync(ULONG_MAX, a1, delta);
    uint32_t tmp1_1 = __shfl_down_sync(ULONG_MAX, a1, delta + 1);
    uint32_t tmp2 = __shfl_down_sync(ULONG_MAX, a2, delta);
    uint32_t tmp2_1 = __shfl_down_sync(ULONG_MAX, a2, delta + 1);
    uint32_t tmp3 = __shfl_down_sync(ULONG_MAX, a3, delta);
    uint32_t tmp3_1 = __shfl_down_sync(ULONG_MAX, a3, delta + 1);
    bool up_down_flag = lane_id % 16 / 8;
    if (lane_id < 32) {
      aa[0] = up_down_flag ? tmp1 : tmp0;
      aa[1] = up_down_flag ? tmp1_1 : tmp0_1;
    } else if (lane_id < 64) {
      aa[0] = up_down_flag ? tmp3 : tmp2;
      aa[1] = up_down_flag ? tmp3_1 : tmp2_1;
    }

    // shuffle B
    tmp0 = __shfl_down_sync(ULONG_MAX, b0, delta);
    tmp0_1 = __shfl_down_sync(ULONG_MAX, b0, delta + 1);
    tmp1 = __shfl_down_sync(ULONG_MAX, b1, delta);
    tmp1_1 = __shfl_down_sync(ULONG_MAX, b1, delta + 1);
    if (lane_id < 32) {
      bb[0] = tmp0;
      bb[1] = tmp0_1;
    } else if (lane_id < 64) {
      bb[0] = tmp1;
      bb[1] = tmp1_1;
    }
    __half2 const *ha0 = reinterpret_cast<__half2 const *>(&a0);
    __half2 const *ha1 = reinterpret_cast<__half2 const *>(&a1);
    __half2 const *ha2 = reinterpret_cast<__half2 const *>(&a2);
    __half2 const *ha3 = reinterpret_cast<__half2 const *>(&a3);
    __half2 const *hb0 = reinterpret_cast<__half2 const *>(&b0);
    __half2 const *hb1 = reinterpret_cast<__half2 const *>(&b1);
    __half2 const *haa0 = reinterpret_cast<__half2 const *>(&aa[0]);
    __half2 const *haa1 = reinterpret_cast<__half2 const *>(&aa[1]);
    __half2 const *hbb0 = reinterpret_cast<__half2 const *>(&bb[0]);
    __half2 const *hbb1 = reinterpret_cast<__half2 const *>(&bb[1]);

    // shuffle C
    delta = (lane_id % 8 / 2 + lane_id % 32 / 16 * 16) - lane_id;
    float tmp_c0 = __shfl_down_sync(ULONG_MAX, c0, delta);
    float tmp_c0_4 = __shfl_down_sync(ULONG_MAX, c0, delta + 4);
    float tmp_c0_8 = __shfl_down_sync(ULONG_MAX, c0, delta + 8);
    float tmp_c0_12 = __shfl_down_sync(ULONG_MAX, c0, delta + 12);
    float tmp_c1 = __shfl_down_sync(ULONG_MAX, c1, delta);
    float tmp_c1_4 = __shfl_down_sync(ULONG_MAX, c1, delta + 4);
    float tmp_c1_8 = __shfl_down_sync(ULONG_MAX, c1, delta + 8);
    float tmp_c1_12 = __shfl_down_sync(ULONG_MAX, c1, delta + 12);
    float tmp_c2 = __shfl_down_sync(ULONG_MAX, c2, delta);
    float tmp_c2_4 = __shfl_down_sync(ULONG_MAX, c2, delta + 4);
    float tmp_c2_8 = __shfl_down_sync(ULONG_MAX, c2, delta + 8);
    float tmp_c2_12 = __shfl_down_sync(ULONG_MAX, c2, delta + 12);
    float tmp_c3 = __shfl_down_sync(ULONG_MAX, c3, delta);
    float tmp_c3_4 = __shfl_down_sync(ULONG_MAX, c3, delta + 4);
    float tmp_c3_8 = __shfl_down_sync(ULONG_MAX, c3, delta + 8);
    float tmp_c3_12 = __shfl_down_sync(ULONG_MAX, c3, delta + 12);
    float cc0, cc1, cc2, cc3;
    if (lane_id < 32) {
      if (lane_id % 2) {
        cc0 = tmp_c1;
        cc1 = tmp_c1_4;
        cc2 = tmp_c1_8;
        cc3 = tmp_c1_12;
      } else {
        cc0 = tmp_c0;
        cc1 = tmp_c0_4;
        cc2 = tmp_c0_8;
        cc3 = tmp_c0_12;
      }
    } else if (lane_id < 64) {
      if (lane_id % 2) {
        cc0 = tmp_c3;
        cc1 = tmp_c3_4;
        cc2 = tmp_c3_8;
        cc3 = tmp_c3_12;
      } else {
        cc0 = tmp_c2;
        cc1 = tmp_c2_4;
        cc2 = tmp_c2_8;
        cc3 = tmp_c2_12;
      }
    }

    auto result = __builtin_mxc_mma_16x16x16f16({static_cast<__fp16>(float(haa0->x)), static_cast<__fp16>(float(haa0->y)),
                                                 static_cast<__fp16>(float(haa1->x)), static_cast<__fp16>(float(haa1->y))},
                                                {static_cast<__fp16>(float(hbb0->x)), static_cast<__fp16>(float(hbb0->y)),
                                                 static_cast<__fp16>(float(hbb1->x)), static_cast<__fp16>(float(hbb1->y))},
                                                {cc0, cc1, cc2, cc3});

    delta = (lane_id % 4 * 2 + lane_id / 16 * 16) - lane_id;
    float tmp_d0 = __shfl_down_sync(ULONG_MAX, result[0], delta);
    float tmp_d1 = __shfl_down_sync(ULONG_MAX, result[1], delta);
    float tmp_d2 = __shfl_down_sync(ULONG_MAX, result[2], delta);
    float tmp_d3 = __shfl_down_sync(ULONG_MAX, result[3], delta);
    float tmp_d0_1 = __shfl_down_sync(ULONG_MAX, result[0], delta + 1);
    float tmp_d1_1 = __shfl_down_sync(ULONG_MAX, result[1], delta + 1);
    float tmp_d2_1 = __shfl_down_sync(ULONG_MAX, result[2], delta + 1);
    float tmp_d3_1 = __shfl_down_sync(ULONG_MAX, result[3], delta + 1);
    float tmp_d0_32 = __shfl_down_sync(ULONG_MAX, result[0], delta + 32);
    float tmp_d1_32 = __shfl_down_sync(ULONG_MAX, result[1], delta + 32);
    float tmp_d2_32 = __shfl_down_sync(ULONG_MAX, result[2], delta + 32);
    float tmp_d3_32 = __shfl_down_sync(ULONG_MAX, result[3], delta + 32);
    float tmp_d0_32_1 = __shfl_down_sync(ULONG_MAX, result[0], delta + 32 + 1);
    float tmp_d1_32_1 = __shfl_down_sync(ULONG_MAX, result[1], delta + 32 + 1);
    float tmp_d2_32_1 = __shfl_down_sync(ULONG_MAX, result[2], delta + 32 + 1);
    float tmp_d3_32_1 = __shfl_down_sync(ULONG_MAX, result[3], delta + 32 + 1);

    if (lane_id < 32) {
      if (lane_id % 16 / 4 == 0) {
        d0 = tmp_d0;
        d1 = tmp_d0_1;
        d2 = tmp_d0_32;
        d3 = tmp_d0_32_1;
      } else if (lane_id % 16 / 4 == 1) {
        d0 = tmp_d1;
        d1 = tmp_d1_1;
        d2 = tmp_d1_32;
        d3 = tmp_d1_32_1;
      } else if (lane_id % 16 / 4 == 2) {
        d0 = tmp_d2;
        d1 = tmp_d2_1;
        d2 = tmp_d2_32;
        d3 = tmp_d2_32_1;
      } else if (lane_id % 16 / 4 == 3) {
        d0 = tmp_d3;
        d1 = tmp_d3_1;
        d2 = tmp_d3_32;
        d3 = tmp_d3_32_1;
      }
    }

#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_F32F16F16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct SM80_16x8x8_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x8_F32BF16BF16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x16x16 TN
struct MACA_16x16x16_F32BF16BF16F32
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using VectorType = __NATIVE_VECTOR__(2, uint32_t);

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {

    VectorType a = {a0, a1};
    VectorType b = {b0, b1};

    auto result = __builtin_mxc_mma_16x16x16bf16(b, a, {c0, c1, c2, c3});

    d0 = result[0];
    d1 = result[1];
    d2 = result[2];
    d3 = result[3];
  }
};

// MMA 16x16x32 TN
struct MACA_16x16x32_F32BF16BF16F32
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];
  using VectorType = __NATIVE_VECTOR__(2, uint32_t);

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, // A, k0
      uint32_t const& a2, uint32_t const& a3, // A, k1
      uint32_t const& b0, uint32_t const& b1, // B, k0
      uint32_t const& b2, uint32_t const& b3, // B, k1
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {

    VectorType a = {a0, a1};
    VectorType b = {b0, b1};

    // first mma , A(k0) * B(k0)
    auto acc = __builtin_mxc_mma_16x16x16bf16(b, a, {c0, c1, c2, c3});

    // second mma , A(k1) * B(k1), accum
    a = {a2, a3};
    b = {b2, b3};
    acc = __builtin_mxc_mma_16x16x16bf16(b, a, {acc[0], acc[1], acc[2], acc[3]});

    d0 = acc[0];
    d1 = acc[1];
    d2 = acc[2];
    d3 = acc[3];
  }
};

// MMA 16x64x16 TN
// use for lds4x4 + perm4x4
struct MACA_16x64x16_F32BF16BF16F32
{
  using DRegisters = float[16];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[8];
  using CRegisters = float[16];
  using VectorType = __NATIVE_VECTOR__(2, uint32_t);

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      float         & d4, float         & d5, float         & d6, float         & d7,
      float         & d8, float         & d9, float         & d10, float         & d11,
      float         & d12, float         & d13, float         & d14, float         & d15,
      uint32_t const& a0, uint32_t const& a1, // A, k0
      uint32_t const& b0, uint32_t const& b1, // B, m0
      uint32_t const& b2, uint32_t const& b3, // B, m1
      uint32_t const& b4, uint32_t const& b5, // B, m2
      uint32_t const& b6, uint32_t const& b7, // B, m3
      float const   & c0, float const   & c1, float const   & c2, float const   & c3,
      float const   & c4, float const   & c5, float const   & c6, float const   & c7,
      float const   & c8, float const   & c9, float const   & c10, float const   & c11,
      float const   & c12, float const   & c13, float const   & c14, float const   & c15)
  {

    VectorType a = {a0, a1};
    VectorType b = {b0, b1};

    auto acc0 = __builtin_mxc_mma_16x16x16bf16(b, a, {c0, c1, c2, c3});
    d0 = acc0[0];
    d1 = acc0[1];
    d2 = acc0[2];
    d3 = acc0[3];

    b = {b2, b3};
    auto acc1 = __builtin_mxc_mma_16x16x16bf16(b, a, {c4, c5, c6, c7});
    d4 = acc1[0];
    d5 = acc1[1];
    d6 = acc1[2];
    d7 = acc1[3];

    b = {b4, b5};
    auto acc2 = __builtin_mxc_mma_16x16x16bf16(b, a, {c8, c9, c10, c11});
    d8 = acc2[0];
    d9 = acc2[1];
    d10 = acc2[2];
    d11 = acc2[3];

    b = {b6, b7};
    auto acc3 = __builtin_mxc_mma_16x16x16bf16(b, a, {c12, c13, c14, c15});
    d12 = acc3[0];
    d13 = acc3[1];
    d14 = acc3[2];
    d15 = acc3[3];

  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_F32BF16BF16F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));

#elif defined(__MACA_ARCH__)

    uint32_t aa[2];
    uint32_t bb[2];
    const int lane_id = __lane_id();
    int delta = (lane_id % 32 % 8 * 4 + lane_id % 32 / 16 * 2) - lane_id;

    // shuffle A
    uint32_t tmp0 = __shfl_down_sync(ULONG_MAX, a0, delta);
    uint32_t tmp0_1 = __shfl_down_sync(ULONG_MAX, a0, delta + 1);
    uint32_t tmp1 = __shfl_down_sync(ULONG_MAX, a1, delta);
    uint32_t tmp1_1 = __shfl_down_sync(ULONG_MAX, a1, delta + 1);
    uint32_t tmp2 = __shfl_down_sync(ULONG_MAX, a2, delta);
    uint32_t tmp2_1 = __shfl_down_sync(ULONG_MAX, a2, delta + 1);
    uint32_t tmp3 = __shfl_down_sync(ULONG_MAX, a3, delta);
    uint32_t tmp3_1 = __shfl_down_sync(ULONG_MAX, a3, delta + 1);
    bool up_down_flag = lane_id % 16 / 8;
    if (lane_id < 32) {
      aa[0] = up_down_flag ? tmp1 : tmp0;
      aa[1] = up_down_flag ? tmp1_1 : tmp0_1;
    } else if (lane_id < 64) {
      aa[0] = up_down_flag ? tmp3 : tmp2;
      aa[1] = up_down_flag ? tmp3_1 : tmp2_1;
    }

    // shuffle B
    tmp0 = __shfl_down_sync(ULONG_MAX, b0, delta);
    tmp0_1 = __shfl_down_sync(ULONG_MAX, b0, delta + 1);
    tmp1 = __shfl_down_sync(ULONG_MAX, b1, delta);
    tmp1_1 = __shfl_down_sync(ULONG_MAX, b1, delta + 1);
    if (lane_id < 32) {
      bb[0] = tmp0;
      bb[1] = tmp0_1;
    } else if (lane_id < 64) {
      bb[0] = tmp1;
      bb[1] = tmp1_1;
    }
    __maca_bfloat162 const *ha0 = reinterpret_cast<__maca_bfloat162 const *>(&a0);
    __maca_bfloat162 const *ha1 = reinterpret_cast<__maca_bfloat162 const *>(&a1);
    __maca_bfloat162 const *ha2 = reinterpret_cast<__maca_bfloat162 const *>(&a2);
    __maca_bfloat162 const *ha3 = reinterpret_cast<__maca_bfloat162 const *>(&a3);
    __maca_bfloat162 const *hb0 = reinterpret_cast<__maca_bfloat162 const *>(&b0);
    __maca_bfloat162 const *hb1 = reinterpret_cast<__maca_bfloat162 const *>(&b1);
    __maca_bfloat162 const *haa0 = reinterpret_cast<__maca_bfloat162 const *>(&aa[0]);
    __maca_bfloat162 const *haa1 = reinterpret_cast<__maca_bfloat162 const *>(&aa[1]);
    __maca_bfloat162 const *hbb0 = reinterpret_cast<__maca_bfloat162 const *>(&bb[0]);
    __maca_bfloat162 const *hbb1 = reinterpret_cast<__maca_bfloat162 const *>(&bb[1]);

    // shuffle C
    delta = (lane_id % 8 / 2 + lane_id % 32 / 16 * 16) - lane_id;
    float tmp_c0 = __shfl_down_sync(ULONG_MAX, c0, delta);
    float tmp_c0_4 = __shfl_down_sync(ULONG_MAX, c0, delta + 4);
    float tmp_c0_8 = __shfl_down_sync(ULONG_MAX, c0, delta + 8);
    float tmp_c0_12 = __shfl_down_sync(ULONG_MAX, c0, delta + 12);
    float tmp_c1 = __shfl_down_sync(ULONG_MAX, c1, delta);
    float tmp_c1_4 = __shfl_down_sync(ULONG_MAX, c1, delta + 4);
    float tmp_c1_8 = __shfl_down_sync(ULONG_MAX, c1, delta + 8);
    float tmp_c1_12 = __shfl_down_sync(ULONG_MAX, c1, delta + 12);
    float tmp_c2 = __shfl_down_sync(ULONG_MAX, c2, delta);
    float tmp_c2_4 = __shfl_down_sync(ULONG_MAX, c2, delta + 4);
    float tmp_c2_8 = __shfl_down_sync(ULONG_MAX, c2, delta + 8);
    float tmp_c2_12 = __shfl_down_sync(ULONG_MAX, c2, delta + 12);
    float tmp_c3 = __shfl_down_sync(ULONG_MAX, c3, delta);
    float tmp_c3_4 = __shfl_down_sync(ULONG_MAX, c3, delta + 4);
    float tmp_c3_8 = __shfl_down_sync(ULONG_MAX, c3, delta + 8);
    float tmp_c3_12 = __shfl_down_sync(ULONG_MAX, c3, delta + 12);
    float cc0, cc1, cc2, cc3;
    if (lane_id < 32) {
      if (lane_id % 2) {
        cc0 = tmp_c1;
        cc1 = tmp_c1_4;
        cc2 = tmp_c1_8;
        cc3 = tmp_c1_12;
      } else {
        cc0 = tmp_c0;
        cc1 = tmp_c0_4;
        cc2 = tmp_c0_8;
        cc3 = tmp_c0_12;
      }
    } else if (lane_id < 64) {
      if (lane_id % 2) {
        cc0 = tmp_c3;
        cc1 = tmp_c3_4;
        cc2 = tmp_c3_8;
        cc3 = tmp_c3_12;
      } else {
        cc0 = tmp_c2;
        cc1 = tmp_c2_4;
        cc2 = tmp_c2_8;
        cc3 = tmp_c2_12;
      }
    }

    auto result = __builtin_mxc_mma_16x16x16bf16({*reinterpret_cast<const _Float16*>(&(haa0->x)), *reinterpret_cast<const _Float16*>(&(haa0->y)),
                                                 *reinterpret_cast<const _Float16*>(&(haa1->x)), *reinterpret_cast<const _Float16*>(&(haa1->y))},
                                                {*reinterpret_cast<const _Float16*>(&(hbb0->x)), *reinterpret_cast<const _Float16*>(&(hbb0->y)),
                                                 *reinterpret_cast<const _Float16*>(&(hbb1->x)), *reinterpret_cast<const _Float16*>(&(hbb1->y))},
                                                {cc0, cc1, cc2, cc3});

    delta = (lane_id % 4 * 2 + lane_id / 16 * 16) - lane_id;
    float tmp_d0 = __shfl_down_sync(ULONG_MAX, result[0], delta);
    float tmp_d1 = __shfl_down_sync(ULONG_MAX, result[1], delta);
    float tmp_d2 = __shfl_down_sync(ULONG_MAX, result[2], delta);
    float tmp_d3 = __shfl_down_sync(ULONG_MAX, result[3], delta);
    float tmp_d0_1 = __shfl_down_sync(ULONG_MAX, result[0], delta + 1);
    float tmp_d1_1 = __shfl_down_sync(ULONG_MAX, result[1], delta + 1);
    float tmp_d2_1 = __shfl_down_sync(ULONG_MAX, result[2], delta + 1);
    float tmp_d3_1 = __shfl_down_sync(ULONG_MAX, result[3], delta + 1);
    float tmp_d0_32 = __shfl_down_sync(ULONG_MAX, result[0], delta + 32);
    float tmp_d1_32 = __shfl_down_sync(ULONG_MAX, result[1], delta + 32);
    float tmp_d2_32 = __shfl_down_sync(ULONG_MAX, result[2], delta + 32);
    float tmp_d3_32 = __shfl_down_sync(ULONG_MAX, result[3], delta + 32);
    float tmp_d0_32_1 = __shfl_down_sync(ULONG_MAX, result[0], delta + 32 + 1);
    float tmp_d1_32_1 = __shfl_down_sync(ULONG_MAX, result[1], delta + 32 + 1);
    float tmp_d2_32_1 = __shfl_down_sync(ULONG_MAX, result[2], delta + 32 + 1);
    float tmp_d3_32_1 = __shfl_down_sync(ULONG_MAX, result[3], delta + 32 + 1);

    if (lane_id < 32) {
      if (lane_id % 16 / 4 == 0) {
        d0 = tmp_d0;
        d1 = tmp_d0_1;
        d2 = tmp_d0_32;
        d3 = tmp_d0_32_1;
      } else if (lane_id % 16 / 4 == 1) {
        d0 = tmp_d1;
        d1 = tmp_d1_1;
        d2 = tmp_d1_32;
        d3 = tmp_d1_32_1;
      } else if (lane_id % 16 / 4 == 2) {
        d0 = tmp_d2;
        d1 = tmp_d2_1;
        d2 = tmp_d2_32;
        d3 = tmp_d2_32_1;
      } else if (lane_id % 16 / 4 == 3) {
        d0 = tmp_d3;
        d1 = tmp_d3_1;
        d2 = tmp_d3_32;
        d3 = tmp_d3_32_1;
      }
    }

#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_F32BF16BF16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x4 TN
struct SM80_16x8x4_F32TF32TF32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x4_F32TF32TF32F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x8 TN
struct SM80_16x8x8_F32TF32TF32F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x8_F32TF32TF32F32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x4 TN
struct SM80_8x8x4_F64F64F64F64_TN
{
  using DRegisters = double[2];
  using ARegisters = double[1];
  using BRegisters = double[1];
  using CRegisters = double[2];

  CUTE_HOST_DEVICE static void
  fma(double      & d0, double      & d1,
      double const& a0,
      double const& b0,
      double const& c0, double const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=d"(d0), "=d"(d1)
      :  "d"(a0),
         "d"(b0),
         "d"(c0),  "d"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x4_F64F64F64F64_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

// MMA 8x8x4 TN with Planar Complex multiplication
struct SM80_8x8x4_C64C64C64C64_TN
{
  using DRegisters = complex<double>[2];
  using ARegisters = complex<double>[1];
  using BRegisters = complex<double>[1];
  using CRegisters = complex<double>[2];

  CUTE_HOST_DEVICE static void
  fma(complex<double>      & d0, complex<double>      & d1,
      complex<double> const& a0,
      complex<double> const& b0,
      complex<double> const& c0, complex<double> const& c1)
  {
    // Because thrust::complex does not provide a mutable ref
    double& rd0 = reinterpret_cast<double(&)[2]>(d0)[0];
    double& id0 = reinterpret_cast<double(&)[2]>(d0)[1];
    double& rd1 = reinterpret_cast<double(&)[2]>(d1)[0];
    double& id1 = reinterpret_cast<double(&)[2]>(d1)[1];

    // d.real() =  a.real() * b.real() + c.real();
    SM80_8x8x4_F64F64F64F64_TN::fma(
      rd0, rd1,
      a0.real(),
      b0.real(),
      c0.real(), c1.real());

    // d.imag() =  a.imag() * b.real() + c.imag();
    SM80_8x8x4_F64F64F64F64_TN::fma(
      id0, id1,
      a0.imag(),
      b0.real(),
      c0.imag(), c1.imag());

    // d.real() = -a.imag() * b.imag() + d.real();
    SM80_8x8x4_F64F64F64F64_TN::fma(
      rd0, rd1,
      -a0.imag(),
      b0.imag(),
      d0.real(), d1.real());

    // d.imag() =  a.real() * b.imag() + d.imag();
    SM80_8x8x4_F64F64F64F64_TN::fma(
      id0, id1,
      a0.real(),
      b0.imag(),
      d0.imag(), d1.imag());
  }
};

// MMA 8x8x4 TN with Gaussian Complex multiplication:
//    (a + bi)*(c + di)
//  yields
//    t0 += a*c
//    t1 += b*d
//    t2 += (a+b)*(c+d)
//  then
//    re = t0 - t1
//    im = t2 - t0 - t1
struct SM80_8x8x4_GC64C64C64GC64_TN
{
  struct GaussComplex {
    double t0, t1, t2;

    CUTE_HOST_DEVICE //constexpr
    operator complex<double>() const { return complex<double>(t0 - t1, t2 - t0 - t1); }

    CUTE_HOST_DEVICE friend //constexpr
    complex<double> operator*(GaussComplex const& a, complex<double> const& b) { return static_cast<complex<double>>(a) * b; }
    CUTE_HOST_DEVICE friend //constexpr
    complex<double> operator*(complex<double> const& a, GaussComplex const& b) { return b * a; }

    CUTE_HOST_DEVICE friend //constexpr
    complex<double> operator+(GaussComplex const& a, complex<double> const& b) { return static_cast<complex<double>>(a) + b; }
    CUTE_HOST_DEVICE friend //constexpr
    complex<double> operator+(complex<double> const& a, GaussComplex const& b) { return b + a; }
  };

  using DRegisters = GaussComplex[2];
  using ARegisters = complex<double>[1];
  using BRegisters = complex<double>[1];
  using CRegisters = GaussComplex[2];

  CUTE_HOST_DEVICE static void
  fma(GaussComplex         & d0, GaussComplex         & d1,
      complex<double> const& a0,
      complex<double> const& b0,
      GaussComplex    const& c0, GaussComplex    const& c1)
  {
    SM80_8x8x4_F64F64F64F64_TN::fma(d0.t0, d1.t0,
                                    a0.real(),
                                    b0.real(),
                                    c0.t0, c1.t0);
    SM80_8x8x4_F64F64F64F64_TN::fma(d0.t1, d1.t1,
                                    a0.imag(),
                                    b0.imag(),
                                    c0.t1, c1.t1);
    SM80_8x8x4_F64F64F64F64_TN::fma(d0.t2, d1.t2,
                                    a0.real() + a0.imag(),
                                    b0.real() + b0.imag(),
                                    c0.t2, c1.t2);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32S8S8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32S8S8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32S8S8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32S8S8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32S8S8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S8S8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S8S8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32S8U8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32S8U8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32S8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32S8U8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32S8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32S8U8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32S8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32S8U8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S8U8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S8U8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32U8S8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32U8S8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32U8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32U8S8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32U8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32U8S8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32U8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32U8S8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U8S8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U8S8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U8S8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U8S8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32U8U8S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32U8U8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x16 TN
struct SM80_8x8x16_S32U8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x16_S32U8U8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32U8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32U8U8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x16 TN
struct SM80_16x8x16_S32U8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_S32U8U8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U8U8S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U8U8S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U8U8S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U8U8S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32S4S4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32S4S4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32S4S4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S4S4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S4S4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32S4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32S4S4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32S4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32S4S4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32S4U4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32S4U4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32S4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32S4U4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S4U4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32S4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32S4U4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32S4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32S4U4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32S4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32S4U4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32U4S4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32U4S4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32U4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32U4S4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U4S4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U4S4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32U4S4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32U4S4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32U4S4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32U4S4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32U4U4S32_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32U4U4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x32 TN
struct SM80_8x8x32_S32U4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32.satfinite "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x32_S32U4U4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U4U4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x32 TN
struct SM80_16x8x32_S32U4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x32_S32U4U4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32U4U4S32_TN
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32U4U4S32_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x64 TN
struct SM80_16x8x64_S32U4U4S32_TN_SATURATE
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x64_S32U4U4S32_TN_SATURATE without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 8x8x128 TN
struct SM80_8x8x128_S32U1U1S32_TN_XORPOPC
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
      "{%0, %1},"
      "{%2},"
      "{%3},"
      "{%4, %5};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_8x8x128_S32U1U1S32_TN_XORPOPC without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x128 TN
struct SM80_16x8x128_S32U1U1S32_TN_XORPOPC
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5},"
      "{%6},"
      "{%7,  %8,  %9,  %10};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x128_S32U1U1S32_TN_XORPOPC without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 16x8x256 TN
struct SM80_16x8x256_S32U1U1S32_TN_XORPOPC
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x256_S32U1U1S32_TN_XORPOPC without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cute
