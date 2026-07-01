/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

  The shared memory resource is time-sliced across warps.
*/

#pragma once

#if defined(__MACACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/array.h"
#include "mctlass/layout/vector.h"
#include "mctlass/layout/tensor.h"
#include "mctlass/tensor_coord.h"
#include "mctlass/aligned_buffer.h"
#include "mctlass/functional.h"

#include "mctlass/gemm/gemm.h"
#include "mctlass/arch/mma.h"

#include "mctlass/transform/pitch_linear_thread_map.h"
#include "mctlass/transform/threadblock/regular_tile_iterator.h"
#include "mctlass/epilogue/threadblock/epilogue_base.h"
#include "mctlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "mctlass/epilogue/threadblock/predicated_tile_iterator.h"

////////////////////////////////////////////////////////////////////////////////
template <class T>
MCTLASS_DEVICE
T shfl_down_sync(unsigned long mask, T var, unsigned int laneDelta, int width = 64) {
  return var;
}

template <class T>
MCTLASS_DEVICE
T shfl_up_sync(unsigned long mask, T var, unsigned int laneDelta, int width = 64) {
  return var;
}

MCTLASS_DEVICE
int32_t shfl_down_sync(unsigned long mask, int32_t var, unsigned int laneDelta, int width = 64) {
  return __shfl_down_sync(mask, var, laneDelta, width);
}

MCTLASS_DEVICE
int32_t shfl_up_sync(unsigned long mask, int32_t var, unsigned int laneDelta, int width = 64) {
  return __shfl_up_sync(mask, var, laneDelta, width);
}

MCTLASS_DEVICE
float shfl_down_sync(unsigned long mask, float var, unsigned int laneDelta, int width = 64) {
  return __shfl_down_sync(mask, var, laneDelta, width);
}

MCTLASS_DEVICE
float shfl_up_sync(unsigned long mask, float var, unsigned int laneDelta, int width = 64) {
  return __shfl_up_sync(mask, var, laneDelta, width);
}

MCTLASS_DEVICE
mctlass::half_t shfl_down_sync(unsigned long mask, mctlass::half_t var, unsigned int laneDelta, int width = 64) {
  auto temp_value = __shfl_down_sync(mask, var, laneDelta, width);
  return mctlass::half_t(temp_value);
}

MCTLASS_DEVICE
mctlass::half_t shfl_up_sync(unsigned long mask, mctlass::half_t var, unsigned int laneDelta, int width = 64) {
  auto temp_value = __shfl_up_sync(mask, var, laneDelta, width);
  return mctlass::half_t(temp_value);
}

namespace mctlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class Epilogue :
  public EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>,
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{

public:

  using Base = EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::Shape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>;

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Number of warps per block
  using WarpCount = typename Base::WarpCount;

  /// Number of threads per block
  static int const kBlockThreads = 64 * WarpCount::kCount;

  /// Per-thread accumulator tile type
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Numerical accumulation element type
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename mctlass::TensorRef<int, mctlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Vector type used by the global output iterator
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;
  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;


public:

  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess),
    "Divisibility");

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");

public:

  /// Aspect for when epilogue source is not needed
  struct SourceAspectNotNeeded
  {
    /// Constructor
    MCTLASS_DEVICE
    SourceAspectNotNeeded()
    {}

    /// Invoke the output functor over each vector of output
    MCTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      MCTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
      }
    }
  };


  /// Aspect for when epilogue source is needed
  struct SourceAspectNeeded
  {
    OutputTileIterator source_iterator;

    typename OutputTileIterator::Fragment source_fragment;

    /// Invoke the output functor over each vector of output
    MCTLASS_DEVICE
    static void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename OutputTileIterator::Fragment const &source_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr =
        reinterpret_cast<OutputAccessType const *>(&source_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      MCTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
      }
    }

    /// Constructor
    MCTLASS_DEVICE
    SourceAspectNeeded(OutputTileIterator source_iterator) :
      source_iterator(source_iterator)
    {
      source_fragment.clear();
    }

    /// Invoke the output functor over each vector of output
    MCTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      // Load addend source fragment from global memory
      source_iterator.load(source_fragment);
      ++source_iterator;

      apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
    }
  };

private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Thread index in the threadblock
  int thread_idx;

  /// Warp index in the threadblock
  int warp_idx;

public:

  /// Constructor
  MCTLASS_DEVICE
  Epilogue(
      typename Base::SharedStorage &shared_storage,   ///< Shared storage object
      int thread_idx,                                 ///< ID of a thread within the threadblock
      int warp_idx,                                   ///< ID of warp within threadblock
      int lane_idx)                                   ///< Id of thread within warp
  :
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      BaseStreamK(thread_idx),
      shared_load_iterator_(shared_storage.reference(), thread_idx),
      thread_idx(thread_idx),
      warp_idx(warp_idx)
  {}

  /// Aggregates the accumulator sets shared by peer blocks in the global workspace,
  /// performing epilogue computations, writing to output
  MCTLASS_DEVICE
  void reduce(
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *element_workspace,
      OutputOp const &output_op,                      ///< Output operator
      OutputTileIterator destination_iterator,        ///< Tile iterator for destination
      OutputTileIterator source_iterator)             ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
  {
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    // Initialize/load source-fragment data
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    if (output_op.is_source_needed())
    {
      source_iterator += reduce_fragment_idx;
      source_iterator.load(source_fragment);
    }

    // Load fragment from shared memory
    typename SharedLoadIterator::Fragment aligned_accum_fragment;
    shared_load_iterator_.load(aligned_accum_fragment);

    // Add fragments shared by other k partitions
    if (kPartitionsK > 1)
    {
      plus <typename SharedLoadIterator::Fragment> add_fragments;

      MCTLASS_PRAGMA_UNROLL
      for ( int i = 1; i < kPartitionsK; ++i) {
        typename SharedLoadIterator::Fragment aligned_addend_fragment;
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        shared_load_iterator_.load(aligned_addend_fragment);
        aligned_accum_fragment = add_fragments(aligned_accum_fragment, aligned_addend_fragment);
      }
    }

    // Compute the output result
    typename OutputTileIterator::Fragment output_fragment;

    // Apply the output operator
    SourceAspectNeeded::apply_output_operator(
        output_fragment,
        output_op,
        aligned_accum_fragment,
        source_fragment);

    // Store the final result
    destination_iterator += reduce_fragment_idx;
    destination_iterator.store(output_fragment);
  }

 /// Perform the epilogue computations and stream the result to global memory.
  MCTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators)            ///< Complete warp-level accumulator tile
  {
    operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
  }

  /// Streams the result to global memory
  MCTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile &accumulators,                ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator) {         ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)

    // Special process for m16n8k32 int32_t cases
    // For type mctlass::complex<float>,when in AccumulatorTile,the real part and the imag part are stored separately.
    if ((mctlass::platform::is_same<ElementAccumulator, int32_t>::value == true && WarpTileIterator::WarpShape::kK >= 64) ||
	(mctlass::platform::is_same<ElementAccumulator, mctlass::complex<float>>::value && mctlass::platform::is_same<typename WarpMmaOperator::ArchTag, arch::Sm80>::value) ||
  (mctlass::platform::is_same<ElementAccumulator, float>::value && (mctlass::platform::is_same<typename WarpMmaOperator::ArchTag, arch::Sm80>::value || mctlass::platform::is_same<typename WarpMmaOperator::ArchTag, arch::Sm75>::value) &&
  mctlass::platform::is_same<typename WarpMmaOperator::OperatorClass, mctlass::arch::OpClassTensorOp>::value &&
  WarpTileIterator::WarpShape::kN >= 64) ||
  (mctlass::platform::is_same<ElementAccumulator, mctlass::half_t>::value && mctlass::platform::is_same<typename WarpMmaOperator::ArchTag, arch::Sm80>::value)) {
      using Element = typename AccumulatorTile::Element;
      for (int i = 0; i < accumulators.size() / 2; i += 2) {

        Element x_real = accumulators[2 * i + 0];
        Element y_real = accumulators[2 * i + 1];
        Element z_real = accumulators[2 * (i + 1) + 0];
        Element w_real = accumulators[2 * (i + 1) + 1];

        Element x0 = shfl_down_sync(UINT64_MAX, x_real, 32);
        Element y0 = shfl_down_sync(UINT64_MAX, y_real, 32);
        Element z0 = shfl_up_sync(UINT64_MAX, z_real, 32);
        Element w0 = shfl_up_sync(UINT64_MAX, w_real, 32);
        if (__lane_id() < 32) {

          accumulators[2 * (i + 1) + 0] = x0;
          accumulators[2 * (i + 1) + 1] = y0;
        }
        else {
          accumulators[2 * i + 0] = z0;
          accumulators[2 * i + 1] = w0;
        }
      }
    }

    if (output_op.is_source_needed()) {
      operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
    }
    else {
      operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
    }
  }

  /// Perform the epilogue computations and stream the result to global memory.  Implements
  /// two alternative codepaths, depending on whether the output op requires addend data to be loaded.
  // MCTLASS_DEVICE
  // void operator()(
  //   OutputOp const &output_op,                      ///< Output operator
  //   OutputTileIterator destination_iterator,        ///< Tile iterator for destination
  //   AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
  //   OutputTileIterator source_iterator )            ///< Tile iterator for addend source
  // {
  //   if (output_op.is_source_needed())
  //   {
  //     operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
  //   }
  //   else
  //   {
  //     operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
  //   }
  // }


  /// Perform the epilogue computations and stream the result to global memory.  Implements a
  /// single codepath, regardless of whether the output op requires addend data to be loaded
  MCTLASS_DEVICE
  void unified(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator )            ///< Tile iterator for addend source
  {
    if (!output_op.is_source_needed())
    {
      source_iterator.clear_mask();
      __syncthreads();  // Dummy (CUDA 11.0)
    }

    operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
  }

private:

  template <class Seq>
  struct acc2smem;

  template <size_t... Seq>
  struct acc2smem<mctlass::index_sequence<Seq...>> {
    template <int Advance>
    MCTLASS_DEVICE static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                                      WarpTileIterator &warp_tile_iterator) {
      MCTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      MCTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
        typename AccumulatorFragmentIterator::Fragment accum_fragment;

        accum_fragment_iterator.load(accum_fragment);
        ++accum_fragment_iterator;

        warp_tile_iterator.store(accum_fragment);
        if (p < Base::kFragmentsPerIteration - 1) {
          warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
        }
      }

      if (Base::kFragmentsPerIteration > 1) {
        warp_tile_iterator.add_pointer_offset(kSmemPointerOffset *
                                              (1 - Base::kFragmentsPerIteration));
      }
    }

    MCTLASS_DEVICE
    static void push(size_t pos,
                    AccumulatorFragmentIterator const &iterator_begin,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }
  };

  /// Streams the result to global memory
  template <typename SourceAspect>
  MCTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                    ///< Output operator
    OutputTileIterator destination_iterator,      ///< Tile iterator for destination
    AccumulatorTile const &accumulators,         ///< Complete warp-level accumulator tile
    SourceAspect source)
  {

    //
    // Iterator over warp-level accumulator fragment
    //

    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    //#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations / Base::kFragmentsPerIteration : 1)
    //mxcc not support using unroll with parentheses,
    //and we would support with unroll with OutputTileIterator::kIterations / Base::kFragmentsPerIteration someday
    #pragma unroll 1
    for (int iter = 0; iter < OutputTileIterator::kIterations; iter += Base::kFragmentsPerIteration) {

      //
      // Convert and store fragment
      //

      __syncthreads();


    acc2smem<mctlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, accum_fragment_iterator, this->warp_tile_iterator_);


      //
      // Load fragments from shared memory
      //

      __syncthreads();

      MCTLASS_PRAGMA_UNROLL
      for (int p = 0; p < Base::kFragmentsPerIteration; ++p)
      {
        typename SharedLoadIterator::Fragment aligned_accum_fragment;
        shared_load_iterator_.load(aligned_accum_fragment);

        if (p < Base::kFragmentsPerIteration - 1)
        {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        }
        else if (kPartitionsK > 1)
        {
          plus <typename SharedLoadIterator::Fragment> add_fragments;

          MCTLASS_PRAGMA_UNROLL
          for ( int i = 1; i < kPartitionsK; ++i) {
            typename SharedLoadIterator::Fragment aligned_accum_fragment_addend;
            shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
            shared_load_iterator_.load(aligned_accum_fragment_addend);
            aligned_accum_fragment = add_fragments(aligned_accum_fragment, aligned_accum_fragment_addend);
          }

          shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
        }

        //
        // Compute the output result
        //

        typename OutputTileIterator::Fragment output_fragment;

        source.apply_output_operator(output_fragment, output_op, aligned_accum_fragment);


        //
        // Store the final result
        //

        destination_iterator.store(output_fragment);
        ++destination_iterator;
      }

      if (Base::kFragmentsPerIteration > 1) {
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace mctlass

////////////////////////////////////////////////////////////////////////////////
