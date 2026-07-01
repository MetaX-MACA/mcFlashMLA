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

#include "mctlass/mctlass.h"
#include "cute/numeric/integral_constant.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "mctlass/arch/barrier.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace mctlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

enum class BarrierStatus : uint32_t {
  WaitAgain = 0u,
  WaitDone  = 1u
};

class ArrivalToken {
public:
  MCTLASS_HOST_DEVICE
  ArrivalToken(BarrierStatus barrier_status) : barrier_status_(barrier_status) {}

  MCTLASS_HOST_DEVICE
  ArrivalToken() = delete;

  MCTLASS_HOST_DEVICE
  BarrierStatus get() const {
    return barrier_status_;;
  }

  MCTLASS_HOST_DEVICE
  bool operator==(ArrivalToken const& other) const {
    return barrier_status_ == other.get();
  }

private:
  BarrierStatus barrier_status_;

  MCTLASS_HOST_DEVICE
  friend bool operator==(const ArrivalToken& left, const BarrierStatus& right) {
    return left.get() == right;
  }

  MCTLASS_HOST_DEVICE
  friend bool operator==(const BarrierStatus& left, const ArrivalToken& right) {
    return left == right.get();
  }
};

class ProducerToken : public ArrivalToken {
  using ArrivalToken::ArrivalToken;
};

class ConsumerToken : public ArrivalToken {
  using ArrivalToken::ArrivalToken;
};

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct PipelineState {

  static constexpr uint32_t Stages = Stages_;

private:
  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t phase_count_ = 0;

public:
  MCTLASS_DEVICE
  PipelineState(): index_{}, phase_{}, phase_count_{} {}

  MCTLASS_DEVICE
  PipelineState(int index, uint32_t phase, uint32_t phase_count)
    : index_(index)
    , phase_(phase)
    , phase_count_(phase_count) {}

  MCTLASS_DEVICE
  int index() const {
    return index_;
  }

  MCTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  MCTLASS_DEVICE
  uint32_t phase_count() const {
    return phase_count_;
  }

  MCTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
        ++phase_count_;
      }
    }
  }

  MCTLASS_DEVICE
  PipelineState& operator=(const PipelineState& other) {
    index_ = other.index();
    phase_ = other.phase();
    phase_count_ = other.phase_count();
    return *this;
  }

  MCTLASS_DEVICE
  PipelineState advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      phase_count_ += (index_ + num_iterations) / Stages;
      index_ = (index_ + num_iterations) % Stages;
    }
    return *this;
  }

  MCTLASS_DEVICE
  static PipelineState make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

template<class Pipeline>
MCTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerPhaseCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerPhaseCount};
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA load (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
// Assumptions : Constructor is visible Cluster-wide (as it needs a Cluster-Sync)
// We have exactly one thread elected in the Producer as the "leader"
// Currently, it is optional to elect a leader for the Consumers
template <
  int Stages_,
  class ClusterShape_
>
class PipelineTmaAsync {
public :
  using ClusterShape = ClusterShape_;
  using FullBarrier = mctlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = mctlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
  };

  // Constructor
  MCTLASS_DEVICE
  PipelineTmaAsync(SharedStorage& storage, Params params)
      : params_(params)
      , full_barrier_ptr_(&storage.full_barrier_[0])
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    if (warp_idx == 0 && lane_predicate == 1) {
      // Barrier FULL init
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[i].init(1);
      }
      // Barrier EMPTY init
      uint32_t const num_consumer_warpgroups_per_cluster = params_.num_consumers / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
          num_consumer_warpgroups_per_cluster;
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr_[i].init(multicast_consumer_arrival_count);
      }
    }

    // Logic to optimally schedule Empty Arrives
    // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
    dim3 block_id = cute::block_id_in_cluster();
    auto cluster_size = cute::size(cluster_shape);
    static constexpr int MaxClusterSize = 16;
    static_assert(cluster_size <= MaxClusterSize, "ERROR : Cluster size too large !" );

    // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
    if (params_.num_consumers % NumThreadsPerWarpGroup == 0) {
      int thread_idx = threadIdx.x % NumThreadsPerWarpGroup;
      is_signalling_thread_ = (thread_idx % (NumThreadsPerWarpGroup / MaxClusterSize)) == 0;
      auto layout = cute::composition(Swizzle<2,0,-2>{},
                                      Layout<Shape<_4,_4>,Stride<_4,_1>>{});
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else if (params_.num_consumers == 32) {
      int thread_idx = threadIdx.x % 32;
      is_signalling_thread_ = (thread_idx % (32 / MaxClusterSize)) == 0;
      auto layout = Layout<Shape<_4,_4>,Stride<_4, _1>>{};
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else {
      is_signalling_thread_ = 0;
      #ifndef NDEBUG
        asm volatile ("brkpt;\n" ::);
      #endif
    }

    // STEP 2: Find if this dst block-id needs an arrival for this problem
    is_signalling_thread_ &= dst_blockid_ < cluster_size;
    is_signalling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);

    mctlass::arch::fence_barrier_init();
  }

  MCTLASS_DEVICE
  bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return ((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x ||
            (dst_block_id / cute::size<0>(cluster_shape)) == block_id.y);
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.

  MCTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  MCTLASS_DEVICE
  void producer_acquire(PipelineState<Stages> state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  MCTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  MCTLASS_DEVICE
  void producer_tail(PipelineState<Stages> state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  MCTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState<Stages> state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  MCTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  MCTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state) {
    consumer_wait(state.index(), state.phase());
  }

  MCTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state, ConsumerToken barrier_token) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  MCTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state) {
    consumer_release(state.index());
  }

private :
  uint32_t dst_blockid_ = 0;
  uint32_t is_signalling_thread_ = 0;
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;

  MCTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  MCTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_reset_bytes(params_.transaction_bytes);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  // NOP for TMA based mainloop
  MCTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    // Below code is used only for unit-testing (in the absence of TMA commit)
    #if MCTLASS_UNIT_TEST_PIPELINE
      if (params_.is_leader) {
        // STEP 1 : Commit to self
        full_barrier_ptr_[stage].commit(bytes);

        // STEP 2 : Commit to other blocks in our cluster
        auto cluster_shape = ClusterShape{};
        Layout block_layout_in_cluster = make_layout(cluster_shape);
        dim3 local_block_id = cute::block_id_in_cluster();

        MCTLASS_PRAGMA_UNROLL
        for(int n = 0; n < size<1>(block_layout_in_cluster); ++n) {
          uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x,n,Int<0>{});
          full_barrier_ptr_[stage].commit(dst_block_id, bytes, n!=local_block_id.y);
        }

        MCTLASS_PRAGMA_UNROLL
        for(int m = 0; m < size<0>(block_layout_in_cluster); ++m) {
          uint32_t dst_block_id = block_layout_in_cluster(m,local_block_id.y,Int<0>{});
          full_barrier_ptr_[stage].commit(dst_block_id, bytes, m!=local_block_id.x);
        }
      }
    #endif
  }

  MCTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  // Wait for producer to commit transactions (done by TMA)
  MCTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t done = full_barrier_ptr_[stage].test_wait(phase);
    if (not done) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  // Wait for producer to commit transactions (done by TMA)
  MCTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  MCTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signalling_thread_ & (!skip));
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  MCTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA store (consumer) pipeline class
// producer-only class, no async barriers between threads because consumer is TMA unit
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <
  int Stages_
>
class PipelineTmaStore {
public:
  static constexpr uint32_t Stages = Stages_;

  struct Params {
    bool always_wait = false;
  };

  MCTLASS_DEVICE
  PipelineTmaStore(Params params = {}) : params_(params) {}

  ////////////////////
  // Producer APIs
  ////////////////////
  // Wait for the least recently committed batch of TMA stores to complete
  MCTLASS_DEVICE
  void producer_acquire(PipelineState<Stages> state) {
    producer_acquire(state.index(), state.phase_count());
  }

  // Commit the most recently issued batch of TMA stores
  MCTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state) {
    producer_commit(state.index(), state.phase_count());
  }

  // Wait for all TMA stores to complete
  MCTLASS_DEVICE
  void producer_tail([[maybe_unused]] PipelineState<Stages> state) {
    tma_store_wait<0>();
  }

private:
  Params params_;

  // Wait for the least recently committed batch of TMA stores to complete
  MCTLASS_DEVICE
  void producer_acquire([[maybe_unused]] uint32_t stage, uint32_t phase_count) {
    if (params_.always_wait || phase_count > 0) {
      tma_store_wait<Stages-1>();
    }
  }

  // Commit the most recently issued batch of TMA stores
  MCTLASS_DEVICE
  void producer_commit([[maybe_unused]] uint32_t stage, [[maybe_unused]] uint32_t phase_count) {
    tma_store_arrive();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class using producer transaction barriers
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_>
class PipelineTransactionAsync {
public :
  using FullBarrier = mctlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = mctlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t transaction_bytes = 0;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
  };

  // Constructor
  MCTLASS_DEVICE
  PipelineTransactionAsync(SharedStorage& storage, Params const& params)
    : params_(params)
    , full_barrier_ptr_(&storage.full_barrier_[0])
    , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by thread 0 of the block
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[i].init(params.producer_arv_count);
        empty_barrier_ptr_[i].init(params.consumer_arv_count);
      }
    }

    mctlass::arch::fence_barrier_init();
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  MCTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  MCTLASS_DEVICE
  void producer_acquire(PipelineState<Stages> state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  MCTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state) {
    producer_commit(state.index());
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  MCTLASS_DEVICE
  void producer_tail(PipelineState<Stages> state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  MCTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState<Stages> state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  MCTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  MCTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  MCTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state) {
    consumer_release(state.index());
  }

private:
  FullBarrier *full_barrier_ptr_ = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  Params params_;

  MCTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  MCTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    full_barrier_ptr_[stage].arrive_and_reset_bytes(params_.transaction_bytes, params_.dst_blockid);
  }

  MCTLASS_DEVICE
  void producer_commit([[maybe_unused]] uint32_t stage) {
  }

  MCTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

  MCTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  MCTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  MCTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid, (not skip));
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <int Stages_>
class PipelineAsync {
public :
  using FullBarrier = mctlass::arch::ClusterBarrier;
  using EmptyBarrier = mctlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;

  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
  };

  // Default assumption when only storage is passed is :
  // => single producer, single consumer & they are in the same block (within the Cluster)
  MCTLASS_DEVICE
  PipelineAsync(SharedStorage& storage)
    : PipelineAsync(storage, {}) {}

  MCTLASS_DEVICE
  PipelineAsync(
    SharedStorage& storage,
    Params const& params) :
      params_(params),
      full_barrier_ptr_(&storage.full_barrier_[0]),
      empty_barrier_ptr_(&storage.empty_barrier_[0]) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by thread 0 of the block
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[i].init(params.producer_arv_count);
        empty_barrier_ptr_[i].init(params.consumer_arv_count);
      }
    }

    mctlass::arch::fence_barrier_init();
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  MCTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  MCTLASS_DEVICE
  void producer_acquire(PipelineState<Stages> state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  MCTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state) {
    producer_commit(state.index());
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  MCTLASS_DEVICE
  void producer_tail(PipelineState<Stages> state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  MCTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState<Stages> state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  MCTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  MCTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  MCTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state) {
    consumer_release(state.index());
  }

private:
  Params params_;
  FullBarrier *full_barrier_ptr_;
  EmptyBarrier *empty_barrier_ptr_;

  MCTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  MCTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  MCTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    full_barrier_ptr_[stage].arrive();
  }

  MCTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

  MCTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  MCTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  MCTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  MCTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Barrier to ensure an Ordered Sequence between
// SequenceLength number of groups (each with group_size participants) executing SequenceDepth Stages
// i.e., for all i < j - only after id "i" arrives at a particular stage "m"
// will the wait() for id "j" succeed for the same stage
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<int SequenceDepth, int SequenceLength>
class OrderedSequenceBarrier {
public :
  using Barrier = mctlass::arch::ClusterBarrier;

  struct SharedStorage {
    Barrier barrier_[SequenceDepth][SequenceLength];
  };

  struct Params {
    uint32_t group_id;
    uint32_t group_size;
  };

private :
  // In future this Params object can be replaced easily with a CG object
  Params params_;
  Barrier *barrier_ptr_;
  PipelineState<SequenceDepth> stage_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  OrderedSequenceBarrier() = delete;
  OrderedSequenceBarrier(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier(OrderedSequenceBarrier&&) = delete;
  OrderedSequenceBarrier& operator=(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier& operator=(OrderedSequenceBarrier&&) = delete;
  ~OrderedSequenceBarrier() = default;

  MCTLASS_DEVICE
  OrderedSequenceBarrier(SharedStorage& storage, Params const& params) :
      params_(params),
      barrier_ptr_(&storage.barrier_[0][0]),
      // Group 0 - starts with an opposite phase
      stage_({0, params.group_id == 0, 0}) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();

    // Barrier FULL, EMPTY init
    // Init is done only by the one elected thread of the block
    if (warp_idx == 0 && lane_predicate == 1) {
      for (int d = 0; d < Depth; ++d) {
        for (int l = 0; l < Length; ++l) {
          barrier_ptr_[d * Length + l].init(params.group_size);
        }
      }
    }

    mctlass::arch::fence_barrier_init();
  }

  // Wait on a stage to be unlocked
  MCTLASS_DEVICE
  void wait() {
    get_barrier_for_current_stage(params_.group_id).wait(stage_.phase());
  }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  MCTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

  MCTLASS_DEVICE
  void advance() {
    ++stage_;
  }

private:

  MCTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // end namespace mctlass
