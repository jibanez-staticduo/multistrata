#pragma once

#include "strata/distributed/model_worker.hpp"

#include <functional>

namespace strata::core { class WeightTable; class NativeHead; struct SessionState; }
namespace strata::kernels { class PleTable; }

namespace strata::distributed {
class RoutedExperts;

struct SlotPoolConfig {
    uint32_t slots = 1;
    uint32_t context = 256;
    uint32_t timeout_ms = 60000;
    std::string session_uuid;
    std::string model_hash;
    uint32_t rank = 0;
    bool batched_expert_collectives = false;
    uint64_t checkpoint_max_bytes = 0;
    bool preagreed_collectives = false;
    ExpertOperandSchedule expert_operands;
};
struct SlotPoolResources {
    const core::WeightTable& weights;
    const core::ModelGeometry& geometry;
    const core::NativeHead& head;
    kernels::PleTable& ple_table;
    SessionCollectives& collectives;
    RoutedExperts& experts;
    Session& session;
    GuardedTransport& transport;
    void* stream;
    std::function<bool()> external_requires_process_exit = {};
};

// Serialized CUDA slots: immutable weights/PLE and expert scratch are shared; every mutable model,
// SessionState, logits buffer, checkpoint and Worker belongs to one slot. No concurrent execute calls.
// Declare PendingDmaExitGuard immediately after construction, before initialize or any CUDA owner
// that could unwind first. On requires_process_exit(), terminate before releasing borrowed resources.
class SlotPool final {
public:
    SlotPool(SlotPoolConfig, SlotPoolResources);
    ~SlotPool();
    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;
    Failure initialize();
    StepResult execute(uint32_t slot, StepPhase, std::string_view request,
                       int32_t token = -1, uint64_t position = 0);
    void abort() noexcept;
    bool requires_process_exit() const noexcept;
    uint32_t slots() const;
    uint64_t allocated_session_bytes() const; // CUDA session arenas only; excludes shared weights/scratch.
    uint64_t collective_sequence() const;
    const float* committed_logits(uint32_t slot) const;
    size_t logits_size(uint32_t slot) const;
    ModelWorkerBackend& execution(uint32_t slot);
    const core::SessionState& session_state(uint32_t slot) const;
    const ModelForward& model(uint32_t slot) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::distributed
