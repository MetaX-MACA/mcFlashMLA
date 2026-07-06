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
/*!
  \file
  \brief The universal GEMM accommodates streamk, batched strided, and batched array variants.
*/


#pragma once

#if defined(__MACACC_RTC__)
#include <mc/std/limits>
#else
#include <limits>
#endif

#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/arch/arch.h"
#include "mctlass/device_kernel.h"

#include "mctlass/gemm/gemm.h"
#include "mctlass/gemm/kernel/gemm_universal.h"

#include "mctlass/gemm/kernel/default_gemm_universal.h"
#include "mctlass/gemm/device/default_gemm_configuration.h"

#include "mctlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////


template <typename GemmKernel_>
class GemmUniversalBase {
public:

  using GemmKernel = GemmKernel_;
  using ThreadblockShape = typename GemmKernel::Mma::Shape;

  using ElementA = typename GemmKernel::ElementA;
  using LayoutA = typename GemmKernel::LayoutA;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  static ComplexTransform const kTransformA = GemmKernel::kTransformA;

  using ElementB = typename GemmKernel::ElementB;
  using LayoutB = typename GemmKernel::LayoutB;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  static ComplexTransform const kTransformB = GemmKernel::kTransformB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;

  /// Numerical accumulation element type
  using ElementAccumulator = typename GemmKernel::Mma::ElementC;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using Operator = typename GemmKernel::Operator;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;

protected:

  //
  // Device properties (uniform across all instances of the current thread)
  //

  // Device ordinal
  thread_local static int device_ordinal_;

  /// Device SM count
  thread_local static int device_sms_;

  /// Kernel SM occupancy (in thread blocks)
  thread_local static int sm_occupancy_;

  /// Kernel dynamic shared memory allocation requirement
  thread_local static int smem_size_;

  /// Initialize static thread-local members for the thread's current device,
  /// if necessary.
  static Status init_device_props()
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::init_device_props()");

    mcError_t mcrt_result;

    // Get current device ordinal
    int current_ordinal;
    mcrt_result = mcGetDevice(&current_ordinal);
    if (mcrt_result != mcSuccess) {
      MCTLASS_TRACE_HOST("  mcGetDevice() returned error " << mcGetErrorString(mcrt_result));
      return Status::kErrorInternal;
    }

    // Done if matches the current static member
    if (current_ordinal == device_ordinal_) {
      // Already initialized
      return Status::kSuccess;
    }

    // Update SM count member
    mcrt_result = mcDeviceGetAttribute (&device_sms_, mcDeviceAttributeMultiProcessorCount, current_ordinal);
    if (mcrt_result != mcSuccess) {
      MCTLASS_TRACE_HOST("  mcDeviceGetAttribute() returned error " << mcGetErrorString(mcrt_result));
      return Status::kErrorInternal;
    }

    // Update the kernel function's shared memory configuration for the current device
    smem_size_ = int(sizeof(typename GemmKernel::SharedStorage));

    // If requires more than 48KB: configure for extended, dynamic shared memory
    if (smem_size_ >= (48 << 10))
    {
      mcrt_result = mcFuncSetAttribute(
        Kernel2<GemmKernel>,
        mcFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_);
      if (mcrt_result != mcSuccess) {
        MCTLASS_TRACE_HOST("  mcFuncSetAttribute() returned error " << mcGetErrorString(mcrt_result));
        return Status::kErrorInternal;
      }

      mcrt_result = mcFuncSetAttribute(
          Kernel2<GemmKernel>,
          mcFuncAttributePreferredSharedMemoryCarveout, 100); // 100% shared memory
      if (mcrt_result != mcSuccess) {
        MCTLASS_TRACE_HOST("  mcFuncSetAttribute() returned error " << mcGetErrorString(mcrt_result));
        return Status::kErrorInternal;
      }
    }

    // Update SM occupancy member
    mcrt_result = mcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &sm_occupancy_,
      Kernel2<GemmKernel>,
      GemmKernel::kThreadCount,
      smem_size_,
      mcOccupancyDisableCachingOverride);
    if (mcrt_result != mcSuccess) {
      MCTLASS_TRACE_HOST("  mcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() returned error " << mcGetErrorString(mcrt_result));
      return Status::kErrorInternal;
    }

    // Update device ordinal member on success
    device_ordinal_ = current_ordinal;

    MCTLASS_TRACE_HOST("  "
      "device_ordinal: (" << device_ordinal_ << "), "
      "device_sms: (" << device_sms_ << "), "
      "sm_occupancy: (" << sm_occupancy_ << ") "
      "smem_size: (" << smem_size_ << ") "
      "GemmKernel::kThreadCount: (" << GemmKernel::kThreadCount << ")");

    return Status::kSuccess;
  }


protected:

  //
  // Instance data members
  //

  /// Kernel parameters
  typename GemmKernel::Params params_;


  /// Initialize params member
  Status init_params(Arguments const &args)
  {
    // Initialize static device properties, if necessary
    Status result = init_device_props();
    if (result != Status::kSuccess) {
      return result;
    }

    // Initialize params member
    params_ = typename GemmKernel::Params(args, device_sms_, sm_occupancy_);
    return Status::kSuccess;
  }

public:

  //---------------------------------------------------------------------------------------------
  // Stateless API
  //---------------------------------------------------------------------------------------------

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args)
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::can_implement()");

    // Initialize static kernel and device properties, if necessary.
    Status result = init_device_props();
    if (result != Status::kSuccess) {
      return result;
    }

    dim3 grid = get_grid_shape(args);

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max()))
    {
      return Status::kErrorInvalidProblem;
    }

    return GemmKernel::can_implement(args);
  }


  /// Returns the workspace size (in bytes) needed for the problem
  /// geometry expressed by these arguments
  static size_t get_workspace_size(Arguments const &args)
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::get_workspace_size()");

    // Initialize parameters from args
    GemmUniversalBase base;
    if (base.init_params(args) != Status::kSuccess) {
      return 0;
    }

    // Get size from parameters
    size_t workspace_bytes = base.params_.get_workspace_size();

    MCTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);
    return workspace_bytes;
  }


  /// Returns the grid extents in thread blocks to launch
  static dim3 get_grid_shape(Arguments const &args)
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::get_grid_shape()");

    // Initialize parameters from args
    GemmUniversalBase base;
    if (base.init_params(args) != Status::kSuccess) {
      return dim3(0,0,0);
    }

    // Get dims from parameters
    dim3 grid_dims = base.params_.get_grid_dims();

    MCTLASS_TRACE_HOST(
         "  tiled_shape: " << base.params_.get_tiled_shape()  << "\n"
      << "  grid_dims: {" << grid_dims << "}");

    return grid_dims;
  }


  /// Returns the maximum number of active thread blocks per multiprocessor
  static int maximum_active_blocks()
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

    // Initialize static device properties, if necessary
    if (init_device_props() != Status::kSuccess) {
      return -1;
    }

    MCTLASS_TRACE_HOST("  max_active_blocks: " << sm_occupancy_);
    return sm_occupancy_;
  }


  //---------------------------------------------------------------------------------------------
  // Stateful API
  //---------------------------------------------------------------------------------------------

  /// Initializes GEMM state from arguments and workspace memory
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    mcStream_t stream = nullptr)
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize parameters from args
    Status result = init_params(args);
    if (result != Status::kSuccess) {
      return result;
    }

    // Assign and prepare workspace memory
    return params_.init_workspace(workspace, stream);
  }


  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args)
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase()::update()");
    params_.update(args);
    return Status::kSuccess;
  }


  /// Runs the kernel using initialized state.
  Status run(mcStream_t stream = nullptr)
  {
    MCTLASS_TRACE_HOST("GemmUniversalBase::run()");

    // Configure grid and block dimensions
    dim3 block(GemmKernel::kThreadCount, 1, 1);
    dim3 grid = params_.get_grid_dims();

    // Launch kernel
    MCTLASS_TRACE_HOST("  "
      "grid: (" << grid << "), "
      "block: (" << block << "), "
      "SMEM: (" << smem_size_ << ")");

    Kernel2<GemmKernel><<<grid, block, smem_size_, stream>>>(params_);

    // Query for errors
    mcError_t result = mcGetLastError();
    if (result != mcSuccess) {
      MCTLASS_TRACE_HOST("  grid launch failed with error " << mcGetErrorString(result));
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }


  /// Runs the kernel using initialized state.
  Status operator()(mcStream_t stream = nullptr)
  {
    return run(stream);
  }


  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    mcStream_t stream = nullptr)
  {
    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Static initializers
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Device ordinal
template <typename GemmKernel_>
thread_local int GemmUniversalBase<GemmKernel_>::device_ordinal_ = -1;

/// Device SM count
template <typename GemmKernel_>
thread_local int GemmUniversalBase<GemmKernel_>::device_sms_ = -1;

/// Kernel SM occupancy (in thread blocks)
template <typename GemmKernel_>
thread_local int GemmUniversalBase<GemmKernel_>::sm_occupancy_ = -1;

/// Kernel dynamic shared memory allocation requirement
template <typename GemmKernel_>
thread_local int GemmUniversalBase<GemmKernel_>::smem_size_ = -1;



/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace mctlass

/////////////////////////////////////////////////////////////////////////////////////////////////
