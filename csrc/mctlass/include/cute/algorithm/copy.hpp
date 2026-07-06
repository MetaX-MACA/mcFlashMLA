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

#include <cute/tensor.hpp>
#include <cute/tensor_predicate.hpp>

#include <cute/atom/copy_atom.hpp>

namespace cute
{

//
// Accept mutable temporaries
//

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_if(pred, src, dst);
}

template <class... CopyArgs,
          class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(Copy_Atom<CopyArgs...>       const& copy_atom,
        PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_if(copy_atom, pred, src, dst);
}

template <class VecType,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
         Tensor<DstEngine, DstLayout>     && dst)
{
  return copy_vec<VecType>(src, dst);
}

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy(src, dst);
}

template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>     && dst)
{
  return copy(copy_atom, src, dst);
}

//
// copy_if -- Predicated Copy
//

template <class PrdTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(PrdTensor                    const& pred,
        Tensor<SrcEngine, SrcLayout> const& src,
        Tensor<DstEngine, DstLayout>      & dst)
{
  auto copy_op = select_elementwise_copy(src, dst);

  CUTE_UNROLL
  for (int i = 0; i < size(src); ++i) {
    if (pred(i)) {
      copy_op.copy(src(i), dst(i));
    }
  }
}

//
// copy_if -- Predicated CopyAtom
//

template <class... CopyArgs,
          class PredTensor,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_if(Copy_Atom<CopyArgs...>       const& copy_atom,
        PredTensor                   const& pred,      // (Rest...)
        Tensor<SrcEngine, SrcLayout> const& src,       // (V,Rest...)
        Tensor<DstEngine, DstLayout>      & dst)       // (V,Rest...)
{
  static_assert(SrcLayout::rank == DstLayout::rank, "CopyAtom rank-mismatch.");
  if constexpr (SrcLayout::rank == 1) {   // Dispatch the copy
    copy_atom.call(src, dst);
  } else {                                // Loop over all but the first mode
    constexpr int R = SrcLayout::rank;
    auto src_v = group_modes<1,R>(src);
    auto dst_v = group_modes<1,R>(dst);
    CUTE_UNROLL
    for (int i = 0; i < size<1>(src_v); ++i) {
      if (pred(i)) {
        copy_atom.call(src_v(_,i), dst_v(_,i));
      }
    }
  }
}

//
// copy_vec -- attempt vectorized copy with VecType
//

template <class VecType,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_vec(Tensor<SrcEngine, SrcLayout> const& src,
         Tensor<DstEngine, DstLayout>      & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;
  if constexpr (sizeof(SrcType) == sizeof(DstType) && sizeof(VecType) > sizeof(DstType))
  {
    /* @pre  is_aligned<N>(src.data()) &&
     *       is_aligned<N>(dst.data())
     */
    auto src_v = recast<VecType const>(src);
    auto dst_v = recast<VecType      >(dst);

#if 0
    if (thread0()) {
      print("copy_vec -- vectorizing copy from %3db to %3db\n", int(8*sizeof(SrcType)), int(8*sizeof(VecType)));
      print("   "); print(layout(src)); print(" => "); print(layout(src_v)); print("\n");
      print("   "); print(layout(dst)); print(" => "); print(layout(dst_v)); print("\n");
    }
#endif

    return copy_if(TrivialPredTensor{}, src_v, dst_v);
  } else {
#if 0
  if (thread0()) {
    print("copy_vec -- not vectorizing, copy with %3db and %3db\n", int(8*sizeof(SrcType)), int(8*sizeof(DstType)));
    print("   "); print(layout(src)); print("\n");
    print("   "); print(layout(dst)); print("\n");
  }
#endif

    return copy_if(TrivialPredTensor{}, src, dst);
  }
}

//
// copy -- auto-vectorizing copy
//

template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  constexpr int N = decltype(max_common_vector(src, dst))::value;

#if 0
  if (thread0()) {
    print("copy -- found a max_common_vector of %d\n", N);
    print("   "); print(src.data()); print(" o "); print(layout(src)); print("\n");
    print("   "); print(dst.data()); print(" o "); print(layout(dst)); print("\n");
  }
#endif

  if constexpr (N <= 1) {
    return copy_if(TrivialPredTensor{}, src, dst);
  } else {
    constexpr int vec_bits = N * sizeof_bits<typename SrcEngine::value_type>::value;
    using VecType = uint_bit_t<cute::min(128, vec_bits)>;
    return copy_vec<VecType>(src, dst);
  }
}

//
// copy -- CopyAtom
//

template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst)
{
  return copy_if(copy_atom, TrivialPredTensor{}, src, dst);
}

template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<DefaultCopy, CopyArgs...> const&,
     Tensor<SrcEngine, SrcLayout>        const& src,
     Tensor<DstEngine, DstLayout>             & dst)
{
  return copy(src, dst);
}

#if defined(__MERGE_LDS_B32)

CUTE_DEVICE
void reg_trans(uint32_t &a) {

  /* ************************************************************
  ** tmp_0[n]=a[(n&0x3c)+shfl[n%4]]
  ** 0x0b1 means shfl[0]=1, shfl[1]=0, shfl[2]=3, shfl[3]=2
  ** and tmp_0[n]/a[n] means the value of tmp_0/a while lane_id=n
  * ************************************************************/
  auto tmp_0 = __builtin_mxc_mov_raw_shfl(a, 0x0b1, 0xf, 0xf, false);
  auto tmp_1 = __builtin_mxc_byte_perm(a, tmp_0, 0x07060302);
  a = __builtin_mxc_byte_perm(tmp_0, a, 0x05040100);

  if (__lane_id() & 0x1) {
    a = tmp_1;
  }

}

CUTE_DEVICE
void reg_trans(uint32_t &a, uint32_t &b) {

  reg_trans(a);
  reg_trans(b);

}

template <
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_DEVICE
void copy_trans(Tensor<SrcEngine, SrcLayout>       const && src,
     Tensor<DstEngine, DstLayout>             && dst,
     const uint32_t &src_stride,
     const uint32_t &dst_stride,
     const uint32_t *cpy_offset)
{

  auto dst_ptr = reinterpret_cast<uint32_t *>(dst.data());
  auto src_addr = reinterpret_cast<uint64_t>(src.data().ptr_);
  src_addr = src_addr - cpy_offset[8];

  /* *************************************************
  ** The address attribute of src_addr has benn destoried,
  ** So we need  to use __attribute__((address_space (3)))
  * *************************************************/
  uint32_t __attribute__((address_space(3))) *src_ptr[8];
  CUTE_UNROLL
  for (uint32_t i = 0; i < 4; ++i) {
    src_ptr[2 * i] = (uint32_t __attribute__((address_space(3))) *)(src_addr) + cpy_offset[2 * i];
    src_ptr[2 * i + 1] = (uint32_t __attribute__((address_space(3))) *)(src_addr) + cpy_offset[2 * i + 1];
  }

  CUTE_UNROLL
  for (uint32_t i = 0; i < size(dst) / 4; ++i) {
    dst_ptr[i] = src_ptr[i][0];
  }

  CUTE_UNROLL
  for (uint32_t i = 0; i < 8; ++i) {
    src_ptr[i] = src_ptr[i] + src_stride;
  }
  dst_ptr = dst_ptr + dst_stride;

  CUTE_UNROLL
  for (uint32_t i = 0; i < size(dst) / 2 - size(dst) / 4; ++i) {
    dst_ptr[i] = src_ptr[i][0];
  }

}


#elif defined(__MERGE_LDS_B64)

CUTE_DEVICE
void reg_trans(uint32_t &a, uint32_t &b) {

  const int laneId = __lane_id();

  /* ************************************************************
  ** tmp_0[n]=a[(n&0x3c)+shfl[n%4]]
  ** 0x0b1 means shfl[0]=1, shfl[1]=0, shfl[2]=3, shfl[3]=2
  ** and tmp_0[n]/a[n] means the value of tmp_0/a while lane_id=n
  * ************************************************************/
  auto tmp_0 = __builtin_mxc_mov_raw_shfl(a, 0x0b1, 0xf, 0xf, false);
  auto tmp_1 = __builtin_mxc_byte_perm(a, tmp_0, 0x07060302);
  a = __builtin_mxc_byte_perm(tmp_0, a, 0x05040100);

  auto tmp_2 = __builtin_mxc_mov_raw_shfl(b, 0x0b1, 0xf, 0xf, false);
  auto tmp_3 = __builtin_mxc_byte_perm(b, tmp_2, 0x07060302);
  b = __builtin_mxc_byte_perm(tmp_2, b, 0x05040100);

  if (laneId & 0x1) {
    a = tmp_1;
    b = tmp_3;
  }

  /* ************************************************************
  ** tmp_0[n]=a[(n&0x3c)+shfl[n%4]]
  ** 0x04e means shfl[0]=2, shfl[1]=3, shfl[2]=0, shfl[3]=1
  ** and tmp_0[n]/a[n] means the value of tmp_0/a while lane_id=n
  * ************************************************************/
  tmp_0 = __builtin_mxc_mov_raw_shfl(a, 0x04e, 0xf, 0xf, false);
  tmp_1 = __builtin_mxc_mov_raw_shfl(b, 0x04e, 0xf, 0xf, false);

  if ((laneId & 0x3) >> 1) {
    a = tmp_1;
  }
  else {
    b = tmp_0;
  }
}

template <
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_DEVICE
void
copy_trans(
     Tensor<SrcEngine, SrcLayout>        const&& src,
     Tensor<DstEngine, DstLayout>             && dst,
     const int src_stride,
     const int dst_stride,
     const uint32_t *cpy_offset)
{

  auto dst_ptr = reinterpret_cast<uint32_t *>(dst.data());
  auto src_addr = reinterpret_cast<uint64_t const>(src.data().ptr_);
  src_addr = src_addr - cpy_offset[4];

  /* ************************************************
  ** The address attribute of src_addr has benn destoried
  ** So we need  to use __attribute__((address_space (3)))
  * *************************************************/
  uint32_t __attribute__((address_space (3))) *src_ptr[4];
  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    src_ptr[i] = (uint32_t __attribute__((address_space(3))) *)(src_addr) + cpy_offset[i];
  }
  CUTE_UNROLL
  for (int i = 0; i < size(dst) / 8; ++i) {
    dst_ptr[2 * i] = src_ptr[i][0];
    dst_ptr[2 * i + 1] = src_ptr[i][1];
  }

  CUTE_UNROLL
  for (int i = 0; i < 4; ++i) {
    src_ptr[i] = src_ptr[i] + src_stride;
  }

  dst_ptr = dst_ptr + dst_stride;
  CUTE_UNROLL
  for (int i = 0; i < size(dst) / 4 - size(dst) / 8; ++i) {
    dst_ptr[2 * i] = src_ptr[i][0];
    dst_ptr[2 * i + 1] = src_ptr[i][1];
  }

}

#endif

#if defined(__MERGE_LDS_B32) || defined(__MERGE_LDS_B64)
template<class DstEngine, class DstLayout>
CUTE_DEVICE
void tensor_trans(Tensor<DstEngine, DstLayout> && dst,
              const uint32_t stride) {

  auto dst_ptr = reinterpret_cast<uint32_t *>(dst.data());
  CUTE_UNROLL
  for (uint32_t i = 0; i < size(dst) / 8; ++i) {
    reg_trans(dst_ptr[2 * i], dst_ptr[2 * i + 1]);
  }

  dst_ptr = dst_ptr + stride;
  CUTE_UNROLL
  for (uint32_t i = 0; i < size(dst) / 4 - size(dst) / 8; ++i) {
    reg_trans(dst_ptr[2 * i], dst_ptr[2 * i + 1]);
  }

}
#endif

template <class SrcEngine, class SrcLayout>
CUTE_HOST_DEVICE
void
copy_global_to_reg(
     Tensor<SrcEngine, SrcLayout> const&& src,
     uint32_t *dst)
{

  typedef __NATIVE_VECTOR__(4, int) VecType;
  auto src_ptr = (VecType *)(src.data().ptr_);
  auto dst_ptr = (VecType *)(dst);
  dst_ptr[0] = __builtin_mxc_load_global_async128(src_ptr);

}

template <class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy_reg_to_share(
     uint32_t *src_ptr,
     Tensor<DstEngine, DstLayout>     && dst)
{

  auto dst_ptr = reinterpret_cast<uint32_t *>(dst.data().ptr_);
  dst_ptr[0] = src_ptr[0];
  dst_ptr[1] = src_ptr[1];
  dst_ptr[2] = src_ptr[2];
  dst_ptr[3] = src_ptr[3];

}

//////////////////////////////////////////
// Special Auto-Vectorizing Overloads
//////////////////////////////////////////

#if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)
template <class... CT_Args, class... CA_Args,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<SM90_BULK_COPY_AUTO, CT_Args...>, CA_Args...> const& atom,
     Tensor<SrcEngine, SrcLayout>                const& src,
     Tensor<DstEngine, DstLayout>                     & dst)
{
  using SrcType = typename SrcEngine::value_type;
  using DstType = typename DstEngine::value_type;
  static_assert(sizeof_bits<SrcType>::value == sizeof_bits<DstType>::value);
  static_assert((is_gmem<SrcEngine>::value && is_smem<DstEngine>::value) ||
                (is_smem<SrcEngine>::value && is_gmem<DstEngine>::value),
                "Bulk Copy only supports gmem -> smem or smem -> gmem movement.");
  // Do BulkCopy dispatch
  using BULK_COPY_OP = conditional_t<is_gmem<SrcEngine>::value,
                                          SM90_BULK_COPY_G2S,
                                          SM90_BULK_COPY_S2G>;

  constexpr int N = decltype(max_common_vector(src, dst))::value;

  // Construct a new concrete Atom of the vector size
  using N_BITS    = Int<N*sizeof_bits<SrcType>::value>;
  using COPY_ATOM = Copy_Atom<Copy_Traits<BULK_COPY_OP, N_BITS, CT_Args...>, SrcType>;
  auto bulk_atom = apply(atom.opargs_, [&](auto const&... args) { return COPY_ATOM{args...}; });

  // Tile the src and dst to the Atom
  auto tiler = right_inverse(dst.layout()).compose(Int<N>{});

#if 0
  if (thread0()) {
    print("copy -- found a max_common_vector of %d\n", N);
    print("   "); print(src.data()); print(" o "); print(layout(src)); print("\n");
    print("   "); print(dst.data()); print(" o "); print(layout(dst)); print("\n");
  }
#endif

  return copy(bulk_atom, logical_divide(src, tiler), logical_divide(dst, tiler));
}
#endif // #if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)

} // end namespace cute
