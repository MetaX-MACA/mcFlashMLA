// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"

#include "mctlass/mctlass.h"
#include "mctlass/layout/layout.h"
#include <mctlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=mctlass::half_t>
struct Flash_kernel_traits {

    using Element = elem_type;
    static constexpr bool Has_cp_async = false;

    using ElementAccum = float;
    using index_t = int64_t;

    using MMA_Atom_Arch_16x16x16_fp16 = std::conditional_t<std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x16_F32F16F16F32>,
        MMA_Atom<MACA_16x16x16_F32BF16BF16F32>
    >;
    using MMA_Atom_Arch_16x16x32_fp16 = std::conditional_t<std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x32_F32F16F16F32>,
        MMA_Atom<MACA_16x16x32_F32BF16BF16F32>
    >;
    using MMA_Atom_Arch_16x16x32_i8 = MMA_Atom<MACA_16x16x32_I32I8I8I32>;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;

    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
    using UniversalCopyAtomB32 = Copy_Atom<UniversalCopy<uint32_t>, elem_type>;
    using UniversalCopyAtomB64 = Copy_Atom<UniversalCopy<uint64_t>, elem_type>;
    using UniversalCopyAtomB128 = Copy_Atom<UniversalCopy<uint128_t>, elem_type>;
    using LDSB64Trans4x16Atom = Copy_Atom<Copy_Traits<MACA_LDS_TRANS_4X16>, elem_type>;
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=mctlass::half_t, bool Is_Splits_=false,
         int kHeadDimV_=kHeadDim_, int Num_Stages_ = 1, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;

    using UniversalCopyAtomB32 = typename Base::UniversalCopyAtomB32;
    using UniversalCopyAtomB64 = typename Base::UniversalCopyAtomB64;
    using UniversalCopyAtomB128 = typename Base::UniversalCopyAtomB128;
    using LDSB64Trans4x16Atom = typename Base::LDSB64Trans4x16Atom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    using MMA_Atom_Arch_16x16x16_fp16 = typename Base::MMA_Atom_Arch_16x16x16_fp16;
    using MMA_Atom_Arch_16x16x32_fp16 = typename Base::MMA_Atom_Arch_16x16x32_fp16;
    using MMA_Atom_Arch_16x16x32_i8 = typename Base::MMA_Atom_Arch_16x16x32_i8;

    using ElementO = Element;
    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;
    static constexpr int Num_Stages = Num_Stages_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kHeadDimNope = kHeadDimV;
    static constexpr int kHeadDimRope = kHeadDim - kHeadDimV;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int MBase = 3;
    static constexpr int SShift = 3;
    static constexpr int SShift_OPT = kBlockKSmem == 32 ? 3 : 4;    // for bank conflict free
    static constexpr int kAtomLayoutMS = std::min(kBlockM / 16, kNWarps);
    static constexpr int kAtomLayoutMO = kAtomLayoutMS;
    static constexpr int kBlockTopK = kNThreads * (16 / sizeof(int32_t));

    using MMA_Atom_QK = MMA_Atom_Arch_16x16x16_fp16;
    using MMA_Atom_PV = MMA_Atom_Arch_16x16x16_fp16;

    using TiledMmaS = TiledMMA<
        MMA_Atom_QK,
        Layout<Shape<Int<kAtomLayoutMS>,_1,_1>>,  // 2x1x1 or 4x1x1
        typename Base::ValLayoutMNK>;

    using TiledMmaS_16x16x32 = TiledMMA<
        MMA_Atom_Arch_16x16x32_fp16,
        Layout<Shape<Int<kAtomLayoutMS>,_1,_1>>,  // 2x1x1 or 4x1x1
        typename Base::ValLayoutMNK>;

    using TiledMmaS_16x16x32_4x2 = TiledMMA<
        MMA_Atom_Arch_16x16x32_fp16,
        Layout<Shape<Int<kAtomLayoutMS>,Int<kNWarps / kAtomLayoutMS>,_1>>,  // 4x2x1
        typename Base::ValLayoutMNK>;

    using TiledMmaO = TiledMMA<
        MMA_Atom_PV,
        Layout<Shape<Int<kAtomLayoutMO>,Int<kNWarps / kAtomLayoutMO>,_1>>,  // 2x2x1 or 4x2x1
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomRowMax = decltype(
        composition(Swizzle<0, 0, 0>{},
                    Layout<Shape<_1, Int<kBlockM>>,
                           Stride<Int<kBlockM>, _1>>{}));
    using SmemLayoutRowMax = decltype(tile_to_shape(
        SmemLayoutAtomRowMax{},
        Shape<Int<kNWarps / kAtomLayoutMS>, Int<kBlockM>>{})); //rowmax 16 value per wave

    using SmemLayoutAtomRowSum = decltype(
        composition(Swizzle<0, 0, 0>{},
                    Layout<Shape<_1, Int<kBlockM>>,
                           Stride<Int<kBlockM>, _1>>{}));
    using SmemLayoutRowSum = decltype(tile_to_shape(
        SmemLayoutAtomRowSum{},
        Shape<Int<kNWarps / kAtomLayoutMS>, Int<kBlockM>>{})); //rowsum 64 value per wave

    using SmemLayoutAtomP = decltype(
        composition(Swizzle<3, 2, 4>{},
                    Layout<Shape<Int<kBlockM>,Int<kBlockN>>,
                           Stride<Int<kBlockN>, _1>>{}));
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<kBlockM>,Int<kBlockN>>{})); //rowmax_wg0

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomQ424 = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQNoSwizzle = decltype(tile_to_shape(
        Layout<Shape<_16, Int<kBlockKSmem>>,
            Stride<Int<kBlockKSmem>, _1>>{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutQ424 = decltype(tile_to_shape(
        SmemLayoutAtomQ424{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutNopeQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));

    using SmemLayoutRopeQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim - kHeadDimV>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomKNoswizzle = Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>;
    using SmemLayoutAtomK424 = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomK242 = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomK333 = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutK424 = decltype(tile_to_shape(
        SmemLayoutAtomK424{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{}));
    using SmemLayoutK242 = decltype(tile_to_shape(
        SmemLayoutAtomK242{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{}));
    using SmemLayoutK333 = decltype(tile_to_shape(
        SmemLayoutAtomK333{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{}));
    using SmemLayoutKNoswizzle = decltype(tile_to_shape(
        SmemLayoutAtomKNoswizzle{},
        Shape<Int<kBlockN>, Int<kHeadDim>, Int<Num_Stages>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemV>, Int<kBlockN>, Int<Num_Stages>>,
                                                      Stride<_1, Int<kBlockKSmemV>, Int<kBlockN*kHeadDim>>>;
    using SmemLayoutAtomVtransposed424 = decltype(
        composition(Swizzle<4, 2, 4>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed424 = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed424{},
        Shape<Int<kHeadDimV>, Int<kBlockN>, Int<Num_Stages>>{}));

    using SmemLayoutAtomVtransposed242 = decltype(
        composition(Swizzle<2, 4, 2>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed242 = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed242{},
        Shape<Int<kHeadDimV>, Int<kBlockN>, Int<Num_Stages>>{}));
    // Maybe the VtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{},
        Shape<Int<kHeadDimV>, Int<kBlockN>, Int<Num_Stages>>{}));

    using SmemLayoutVtNoSwizzle = decltype(tile_to_shape(
        Layout<Shape<_16, Int<kBlockKSmemV>>,
               Stride<Int<kBlockKSmemV>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<0, 0, 0>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmemV>>,
                           Stride<Int<kBlockKSmemV>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomOb128 = Copy_Atom<UniversalCopy<uint128_t>, ElementO>;
    using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint64_t>, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<UniversalCopy<uint128_t>, ElementAccum>;

    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(ElementAccum);
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutK424{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKSize + kSmemVSize;
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(std::max(kSmemQSize, kSmemKSize), kSmemOSize) : std::max(kSmemQSize + kSmemKSize, kSmemOSize);

    static constexpr int kGmemElemsPerLoadB128 = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemElemsPerLoadB64 = sizeof(cute::uint64_t) / sizeof(Element);
    static constexpr int kGmemElemsPerLoadB32 = sizeof(cute::uint32_t) / sizeof(Element);

    static_assert(kHeadDim % kGmemElemsPerLoadB128 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB128");
    static_assert(kHeadDim % kGmemElemsPerLoadB64 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB64");
    static_assert(kHeadDim % kGmemElemsPerLoadB32 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB32");

    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRowB128 = kBlockKSmem / kGmemElemsPerLoadB128;
    static constexpr int kGmemThreadsPerRowB64 = kBlockKSmem / kGmemElemsPerLoadB64;
    static constexpr int kGmemThreadsPerRowB32 = kBlockKSmem / kGmemElemsPerLoadB32;
    static_assert(kNThreads % kGmemThreadsPerRowB128 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB128");
    static_assert(kNThreads % kGmemThreadsPerRowB64 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB64");
    static_assert(kNThreads % kGmemThreadsPerRowB32 == 0, "kNThreads must be a multiple of kGmemThreadsPerRowB32");
    using GmemLayoutAtomB128 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB128>, Int<kGmemThreadsPerRowB128>>,
                                  Stride<Int<kGmemThreadsPerRowB128>, _1>>;
    using GmemLayoutAtomB64 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB64>, Int<kGmemThreadsPerRowB64>>,
                                  Stride<Int<kGmemThreadsPerRowB64>, _1>>;
    using GmemLayoutAtomB32 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB32>, Int<kGmemThreadsPerRowB32>>,
                                  Stride<Int<kGmemThreadsPerRowB32>, _1>>;

    static constexpr int kGmemElemsPerLoadB128O = sizeof(cute::uint128_t) / sizeof(ElementO);
    static constexpr int kGmemThreadsPerRowO = kBlockKSmemV / kGmemElemsPerLoadB128O;
    static_assert(kNThreads % kGmemThreadsPerRowO == 0, "kNThreads must be a multiple of kGmemThreadsPerRowO");
    static constexpr bool UseWarpsNx1 = kBlockM % (kNThreads / kGmemThreadsPerRowO) == 0;
    using GmemLayoutAtomO = std::conditional_t<
        UseWarpsNx1,
        Layout<Shape <Int<kNThreads / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
                                   Stride<Int<kGmemThreadsPerRowO>, _1>>,
        Layout<Shape<Int<kBlockM>, Shape<Int<kGmemThreadsPerRowO>, Int<kNThreads / kGmemThreadsPerRowO / kBlockM>>>,
               Stride<Int<kGmemThreadsPerRowO>, Stride<_1, Int<kBlockM * kGmemThreadsPerRowO>>>>
    >;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemTiledCopyB128 = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomB128{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadB128>>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyB64 = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomB64{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadB64>>>{}));  // Val layout, 4 vals per read
    using GmemTiledCopyB32 = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint32_t>, Element>{},
                        GmemLayoutAtomB32{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadB32>>>{}));  // Val layout, 2 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementO>{},
                        GmemLayoutAtomO{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoadB128O>>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <Int<kNThreads / 8>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreads / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
};
////////////////////////////////////////////////////////////////////////////////////////////////////
