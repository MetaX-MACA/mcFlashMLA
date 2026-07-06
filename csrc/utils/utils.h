// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <mcr/mc_runtime_api.h>

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <mctlass/array.h>
#include <mctlass/mctlass.h>
#include <mctlass/numeric_conversion.h>
#include <mctlass/numeric_types.h>
#define SET_BIT(var, pos) ((var) |= (1U << (pos)))
#define CLEAR_BIT(var, pos) ((var) &= ~(1U << (pos)))
#define TOGGLE_BIT(var, pos) ((var) ^= (1U << (pos)))
#define CHECK_BIT(var, pos) (((var) >> (pos)) & 1U)
#define WRITE_BIT(var, pos, val) \
    ((val) ? SET_BIT(var, pos) : CLEAR_BIT(var, pos))

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint64_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// reduce val(tidx) val(tidx+16) val(tidx+32) val(tidx+48)
struct Partialreduce {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        #if 0
        constexpr int OFFSET = 32;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET / 2));
        return x;
        #endif

        /**********************************************
         ** Using one addtional __shfl_xor_sync can
         ** reduce the time of waiting arrive inst
        **********************************************/
        auto x1 = __shfl_xor_sync(uint64_t(-1), x, 48);
        auto x2 = __shfl_xor_sync(uint64_t(-1), x, 32);
        auto x3 = __shfl_xor_sync(uint64_t(-1), x, 16);
        return op(op(op(x, x1), x2), x3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<bool A_in_regs=false, bool B_in_regs=false, int prefetch_lds_num=1, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_prefetch_lds(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    static_assert(decltype(size<2>(tCrA))::value >= prefetch_lds_num);

    #pragma unroll
    for (int i = 0; i < prefetch_lds_num + 1; ++i) {
        if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i), tCrA_copy_view(_, _, i)); }
        if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i), tCrB_copy_view(_, _, i)); }
    }
    // do first gemm outside the loop, for compiler obey the sequence
    cute::gemm(tiled_mma, tCrA(_, _, _0{}), tCrB(_, _, _0{}), acc);

    #pragma unroll
    for (int i = 1; i < size<2>(tCrA); ++i) {
        if (i + prefetch_lds_num < size<2>(tCrA)) {
            if constexpr (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + prefetch_lds_num), tCrA_copy_view(_, _, i + prefetch_lds_num)); }
            if constexpr (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + prefetch_lds_num), tCrB_copy_view(_, _, i + prefetch_lds_num)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_rr(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    cute::gemm(tiled_mma, tCrA, tCrB, acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// cu: Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// mc: Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(1, MMA_M), ncol=(4, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    // static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    //auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    //return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)), make_layout(get<0>(acc_layout), get<2>(acc_layout)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    mctlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const mctlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

#define CONVERT_TENSOR_TYPE(type_s, type_d, tensor_s, tensor_d)                                                                         \
    constexpr int tensor_d##_numel = decltype(size(tensor_s))::value;                                                                   \
    mctlass::NumericArrayConverter<type_d, type_s, tensor_d##_numel > tensor_d##_convert_op;                                            \
    auto tensor_d##_frag = tensor_d##_convert_op(*reinterpret_cast<const mctlass::Array<type_s, tensor_d##_numel> *>(tensor_s.data())); \
    Tensor tensor_d = make_tensor(make_rmem_ptr<type_d>(&tensor_d##_frag), tensor_s.layout());

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
__forceinline__ __device__ void cp_async_wait() {
    __builtin_mxc_arrive_gvmcnt(N);
}

// barrier_ex(2) == barrier_inst()
template <int N = 2>
__forceinline__ __device__ void sync_threads() {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_ex(N);
}

template <int N = 2>
__forceinline__ __device__ void barrier() {
    __builtin_mxc_barrier_ex(N);
}

template <int N, int M = 2>
__forceinline__ __device__ void barrier_gvm() {
    __builtin_mxc_arrive_gvmcnt(N);
    __builtin_mxc_barrier_ex(M);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int& d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < d) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_reg_to_global(Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int &d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(D(_, m, k).data().ptr_));
            auto S_ptr = (VecType const *)(reinterpret_cast<int32_t const *>(S(_, m, k).data()));
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, S_ptr[0], true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// for tensor shape is (cols=8, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b128(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

// for tensor shape is (cols=8, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void copy_b128_bsm_async(Tensor0 const &S,
                                                    Tensor1 &&D,
                                                    Tensor2 const &identity_MN,
                                                    const int d,
                                                    const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem pointer
            auto dst_ptr = (VecType *)(D(_, m, k).data().get());    // smem pointer
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            if constexpr (Is_even_K && Is_even_MN) {
                __builtin_mxc_ldg_b128_bsm(dst_ptr, src_ptr, 0, -1, true, true, false, true);
            } else {
                __builtin_mxc_ldg_b128_bsm_predicator(
                    dst_ptr, // shared memory pointer
                    src_ptr, // global memory pointer
                    0,       // Immediate value,use the default value 0.
                    true,    // bool
                    true,    // bool
                    false,   // bool
                    true,    // bool,If it is true, the compiler will not insert arrive.
                    col_mask && row_mask,
                    1,
                    MACA_ICMP_EQ
                );
            }
        }
    }
}

// for tensor shape is (cols=4, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b64(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                            row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void swap_fragment(Tensor<Engine, Layout> &S) {
    using data_type = typename Engine::value_type;
    static_assert(decltype(size<0>(S))::value == 8);
    static_assert(std::is_same_v<data_type, mctlass::half_t> || std::is_same_v<data_type, mctlass::bfloat16_t>);

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(S); ++n) {
            uint64_t *first = reinterpret_cast<uint64_t *>(S(_, m, n).data());
            uint64_t *second = first + 1;
            uint64_t tmp = *first;
            *first = *second;
            *second = tmp;
        }
    }
}

#define SWIZZLE_STORE_QDO(smem_s, reg, smem_d)      \
    cute::copy(smem_s, reg);                        \
    if (tidx / 8 % 2 == 1) {                        \
        flash::swap_fragment(reg);                  \
    }                                               \
    cute::copy(reg, smem_d);

////////////////////////////////////////////////////////////////////////////////////////////////////

// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int page_block_size, const int* block_table, const int page_stride, const int row_stride, const int row_offset, const int col_offset) {
    const int virtual_page_idx = row_offset / page_block_size;
    const int page_offset = row_offset - virtual_page_idx * page_block_size;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
}

// when prefetch ldg page_idx, use the follow function
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int page_block_size, const int page_idx, const int page_offset, const int page_stride, const int row_stride, const int col_offset) {

    return ((int64_t) page_idx) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 8;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b64_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int page_idx,
                                          const int page_offset,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 4;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x4 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, page_idx, page_offset, page_stride, row_stride, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}


template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b64_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 4;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x4 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

// when prefetch ldg page_idx and ldgbsm, use the follow function
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_page_bsm_async(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &&D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const uint32_t *page_idx,
                                          const uint32_t *page_offset,
                                          const int swz_offset,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 8;
    constexpr int kGmemRowsPerThread = 1;

    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * 8;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, page_idx[m], page_offset[m], page_stride, row_stride, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + swz_offset + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data().get());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_K && Is_even_MN) {
                __builtin_mxc_ldg_b128_bsm(dst_ptr, src_ptr, 0, -1, true, true, false, true);
            } else {
                __builtin_mxc_ldg_b128_bsm_predicator(
                    dst_ptr, // shared memory pointer
                    src_ptr, // global memory pointer
                    0,       // Immediate value,use the default value 0.
                    true,    // bool
                    true,    // bool
                    false,   // bool
                    true,    // bool,If it is true, the compiler will not insert arrive.
                    col_mask && row_mask,
                    1,
                    MACA_ICMP_EQ
                );
            }
        }
    }
}

template <typename Kernel_traits, typename Engine0, typename Layout0>
__forceinline__ __device__ void copy_page(Tensor<Engine0, Layout0> &S,
                                          uint32_t *page_idx,
                                          uint32_t *page_offset,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_block_size) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / 8;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;
    const int log2_page_size = __builtin_ctz(page_block_size);

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        int virtual_page_idx = row_offset >> log2_page_size;
        page_offset[m] = row_offset - virtual_page_idx * page_block_size;
        page_idx[m] = block_table[virtual_page_idx];
    }
}

template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b32(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(1, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false,
                                                            row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}
template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b32_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int page_idx,
                                          const int page_offset,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = Kernel_traits::kGmemElemsPerLoadB32;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 32 bit per thread, 1x2 fp16
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(1, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, page_idx, page_offset, page_stride, row_stride, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b32_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = Kernel_traits::kGmemElemsPerLoadB32;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 32 bit per thread, 1x2 fp16
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(1, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int64_t global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b32_sparse(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int row_stride,
                                          const int topk_sparse_idx,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 2;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x2 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(1, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int64_t global_kv_sparse_offset = topk_sparse_idx * row_stride + col_offset;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_sparse_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
        }
    }
}

template<typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void concat(Tensor0 &lhs, Tensor1 &rhs, Tensor2 &out) {
    CUTE_STATIC_ASSERT_V(rank(lhs) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(rhs) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(out) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(lhs) == size<0>(rhs));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(lhs) == size<0>(out));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(lhs) == size<1>(rhs));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(lhs) == size<1>(out));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(lhs) + size<2>(rhs) == size<2>(out));      // MMA_K
    #pragma unroll
    for (int k = 0; k < size<2>(lhs); k++) {
        #pragma unroll
        for (int m = 0; m < size<1>(out); m++) {
            #pragma unroll
            for (int i = 0; i < size<0>(out); i++) {
                out(i, m, k) = lhs(i, m, k);
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < size<2>(rhs); k++) {
        #pragma unroll
        for (int m = 0; m < size<1>(out); m++) {
            #pragma unroll
            for (int i = 0; i < size<0>(out); i++) {
                out(i, m, k + size<2>(lhs)) = rhs(i, m, k);
            }
        }
    }

}

template<typename Tensor0, typename Tensor1>
__forceinline__ __device__ void lds4x4_with_swizzle424(Tensor0 const& tCsA, Tensor1& tCrA) {
    CUTE_STATIC_ASSERT_V(size<0>(tCsA) == size<0>(tCrA));
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == (size<1, 1>(tCrA)));
    CUTE_STATIC_ASSERT_V((size<1, 0>(tCrA)) == _4{});
    const int lane_idx = threadIdx.x % 64;
    const int Vt_swizzle_row = lane_idx / 16 * 4;
    const int Vt_swizzle_col = lane_idx % 16;

    #pragma unroll
    for (int m = 0; m < size<1>(tCsA); m++) {
        uint64_t* src_ptr = reinterpret_cast<uint64_t *>(&tCsA(0, m));
        #pragma unroll
        for (int row = 0; row < 4; row++) {
            int col_idx = Vt_swizzle_col ^ (Vt_swizzle_row + row);
            uint64_t* dst_ptr = reinterpret_cast<uint64_t *>(&tCrA(0, make_coord(row, m), 0));
            *dst_ptr = *(src_ptr + row * 16 + col_idx);
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_4x4_b16(Tensor<Engine, Layout> &t) {
    using data_type = typename Engine::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _4>{});
    uint32_t v1, v2;
    uint32_t *dest;

    #pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        cute::copy(tPerm, t(_, _, i));
    }
    return t;
}

#define CHECK_MSG(x, ...) do { if((x) == false) {throw std::invalid_argument(__VA_ARGS__);} }while(0)
#define CUDA_CHECK(expr) {auto x = (expr); CHECK_MSG(x == cudaSuccess, #expr + std::string(" check failed!"));}
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

}  // namespace flash
