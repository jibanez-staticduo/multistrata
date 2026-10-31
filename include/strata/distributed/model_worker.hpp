#pragma once

#include "strata/distributed/worker.hpp"
#include "strata/core/geometry.hpp"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::distributed {

class ModelForward;
class SessionCollectives;
namespace nccl { class NcclApi; }

struct ExpertOperandSchedule {
    bool compact = false;
    std::array<uint64_t, 3> valid_counts{256, 192, 192};
    // One GGML down codec per model layer. Required by compact scheduling.
    std::vector<int> down_codecs;
};

// Canonical exact bytes, bounded to one MiB. This is an agreement payload, not a hash and not a wire
// decoder. It covers every StepCommand field including route labels and complete collective shapes.
Failure canonical_step_command(const StepCommand&, std::string& out);
Failure model_forward_schedule(const StepCommand&, const core::ModelGeometry&, uint64_t first_sequence,
                               std::vector<ScheduledCollective>& out, bool batched_expert_collectives = false,
                               const ExpertOperandSchedule& expert_operands = {},
                               bool layer_major_prefill_batch = false,
                               bool prefill_expert_batch = false);

struct ModelWorkerConfig {
    WorkerConfig worker;
    void* stream = nullptr;
    int32_t position_base = 0;
    bool batched_expert_collectives = false;
    // Checkpoints allocate only on the explicit phase. Zero byte cap disables capture.
    uint64_t checkpoint_max_bytes = uint64_t(1) << 30;
    uint64_t checkpoint_max_context = 0;
    bool preagreed_collectives = false;
    ExpertOperandSchedule expert_operands;
    bool prefill_batch_exactnative = false;
    bool prefill_expert_batch = false;
    // Borrowed approved NCCL API. Non-null enables experimental C1 decode capture.
    nccl::NcclApi* decode_graph_api = nullptr;
    // When true, prefill runs eagerly and only one-token decode steps are captured.
    bool decode_graph_only = false;
    std::function<Failure(const StepCommand&, uint64_t position, bool before)> memory_phase;
};

// CUDA single-slot bridge with opt-in fixed-buffer graph replay. Serialized slots may share collectives.
// Borrowed model, collectives, session, transport and stream outlive this
// object. Init initializes the model and zeros request state. Candidate forward mutates private model
// state, but no logits escape until explicit group commit. Failure permanently invalidates the model.
class ModelWorkerBackend final : public ExecutionBackend {
 public:
    ModelWorkerBackend(ModelWorkerConfig, ModelForward&, SessionCollectives&, Session&, GuardedTransport&);
    ~ModelWorkerBackend();
    ModelWorkerBackend(const ModelWorkerBackend&) = delete;
    ModelWorkerBackend& operator=(const ModelWorkerBackend&) = delete;
    bool model_driven_collectives() const override { return true; }
    Failure plan(const StepCommand&, std::vector<ScheduledCollective>&) const override;
    Failure agree(const StepCommand&) override;
    Failure begin(const StepCommand&) override;
    Failure collective(const ScheduledCollective&) override;
    Failure finish(const StepCommand&, BackendCompletion&) override;
    void abort() noexcept override;
    void discard_session() noexcept override;

    // Host logits, rank zero only, valid until the next accepted command or destruction. Null before
    // commit, after rollback/cancellation/reset/failure, or on followers. The coordinator alone samples/writes.
    const float* committed_logits() const;
    size_t logits_size() const;
    bool requires_process_exit() const noexcept;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::distributed
