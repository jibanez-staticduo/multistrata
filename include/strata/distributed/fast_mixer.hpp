#pragma once

#include "strata/core/geometry.hpp"
#include "strata/core/tensor_parallel.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {
class WeightTable;
}

namespace strata::distributed {

// `max_rows` bounds a prompt chunk or decode batch. `snapshot_depth` is the
// maximum number of speculative positions that can be rolled back per slot.
struct FastMixerLimits {
    int max_rows = 0;
    int max_slots = 0;
    int snapshot_depth = 0;
};

uint64_t fast_mixer_state_bytes_per_slot(const core::ModelGeometry&,
                                         const FastMixerLimits&);

// Rank-local GDN/QSA mixer. It owns sliced output-projection weights and scratch,
// borrows per-request cache state, and never performs communication.
class FastMixer {
public:
    FastMixer(const core::WeightTable&, const core::ModelGeometry&, int rank);
    ~FastMixer();
    FastMixer(const FastMixer&) = delete;
    FastMixer& operator=(const FastMixer&) = delete;

    bool initialize(const FastMixerLimits&, void* stream, std::string& error);
    bool bind_sessions(core::SessionState* const* sessions, int slot_count,
                       std::string& error);
    bool reset_slots(const int32_t* host_slot_ids, int count, void* stream,
                     std::string& error);
    bool begin_slots(const int32_t*, const int32_t*, int, bool, void*, std::string&);
    bool commit_slots(const int32_t* host_slot_ids, const int32_t* host_keep,
                      int count, void* stream, std::string& error);
    bool rollback_slots(const int32_t* host_slot_ids, const int64_t* host_to_pos,
                       int count, void* stream, std::string& error);
    bool validate_execution_status(void* stream, std::string& error);
    bool has_sharded_shared() const noexcept;
    bool shared_expert_local_add(int64_t layer, const core::TensorParallelRows& rows,
                                 const float* mixed, float* routed_local_sum,
                                 void* stream, std::string& error);
    bool has_fast_hc() const;
    bool use_fast_hc(int rows) const;
    bool hyper_connection(const kernels::FusedGrArgs*, int, void*, std::string&);

    // `mixed` and `output_local_partial` are [rows, hidden]. This enqueues
    // local computation only; the caller performs the one output AllReduce.
    bool run(int64_t layer, const core::TensorParallelRows& rows,
             const float* mixed, float* output_local_partial,
             void* stream, std::string& error);

    int64_t owned_weight_bytes() const;
    bool initialized() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::distributed
