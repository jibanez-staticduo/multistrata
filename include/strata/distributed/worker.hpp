#pragma once

#include "strata/distributed/session.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::distributed {

inline constexpr uint32_t kWorkerMaxSlots = 8;

enum class StepPhase : uint32_t { init, prefill, decode, commit, reset, cancel, abort, checkpoint, rollback };
enum class WorkerState : uint32_t { fresh, idle, active, pending, failed, aborted };
enum class StepStatus : uint32_t { completed, rejected, failed, cancelled, aborted };

struct ScheduledCollective {
    CollectiveDescriptor descriptor;
    // Canonical global route decisions, including expert order; never rank-local routing.
    std::string route;
};

struct StepCommand {
    std::string session;
    std::string request;
    uint32_t slot = 0;  // Must name the receiving Worker's configured slot.
    uint64_t step = 0;  // Starts at one; advances across requests and checkpoint/rollback/reset/cancel.
    StepPhase phase = StepPhase::init;
    std::vector<int32_t> tokens;
    std::vector<uint64_t> positions;
    std::string manifest_hash;
    std::vector<ScheduledCollective> expected_collective_schedule;
};

struct StepResult {
    StepStatus status = StepStatus::rejected;
    Failure failure;
    // Diagnostic equality token only, explicitly not a cryptographic state checksum.
    std::string state_digest;
    // No logits/tokens on this channel. Only rank zero may publish/sample after this gate.
    bool output_ready = false;
};

struct BackendCompletion {
    bool complete = false;
    bool outputs_finite = false;
    bool all_ranks_committed = false;
    // Backend must cover its KV, GDN, QSA/indexer and PLE state, not just token positions.
    std::string state_digest;
};

// This is an execution boundary, not a model implementation. All calls are synchronous and bounded by
// the backend's agreed session deadlines. The future CUDA backend must enforce these contracts.
class ExecutionBackend {
 public:
    virtual ~ExecutionBackend() = default;
    // Real model execution may issue collectives at its own compute seams. Such a backend validates
    // the declared schedule against observed operations; the worker must not replay those operations.
    virtual bool model_driven_collectives() const { return false; }
    // Read-only: derive the exact global schedule from loaded model/policy and command.
    virtual Failure plan(const StepCommand&, std::vector<ScheduledCollective>& out) const = 0;
    // All peers must agree the ENTIRE command (identity/tokens/positions/routes/schedule) before begin.
    virtual Failure agree(const StepCommand&) = 0;
    virtual Failure begin(const StepCommand&) = 0;
    // Return success only after this collective completed, including asynchronous error polling.
    virtual Failure collective(const ScheduledCollective&) = 0;
    // Complete all compute and state operations. Commit success includes agreement from every rank;
    // finite validation covers every candidate output. No backend may publish output itself.
    virtual Failure finish(const StepCommand&, BackendCompletion&) = 0;
    virtual void abort() noexcept = 0;
    virtual void discard_session() noexcept = 0;
};

struct WorkerConfig {
    std::string session;
    std::string manifest_hash;
    uint32_t rank = 0;
    uint32_t slot = 0;
    // Optional borrowed cursor shared by serialized Workers. It is independent of the backend's
    // execution counter: only Worker advances this cursor after a successful compute command.
    uint64_t* collective_sequence = nullptr;
};

// Single thread, one synchronous command at a time across all Workers sharing a cursor. Cancellation is BETWEEN
// execute calls: it cannot interrupt a collective or skip any remaining operations of a prior command.
// One checkpoint belongs to the active request; rollback restores committed state and drops pending output.
// The backend outlives the worker. A failed/aborted worker can never initialize or reuse state again.
class Worker {
 public:
    Worker(WorkerConfig config, ExecutionBackend& backend);
    ~Worker();
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    Worker(Worker&&) = delete;
    Worker& operator=(Worker&&) = delete;

    StepResult execute(const StepCommand& command);
    WorkerState state() const { return state_; }
    uint64_t next_step() const { return next_step_; }

 private:
    Failure validate(const StepCommand&) const;
    StepResult fail(Failure failure);
    std::string digest() const;
    uint64_t collective_sequence() const;

    WorkerConfig config_;
    ExecutionBackend& backend_;
    WorkerState state_ = WorkerState::fresh;
    Failure failure_;
    uint64_t next_step_ = 1;
    uint64_t next_collective_ = 1;
    uint64_t committed_positions_ = 0;
    uint64_t pending_positions_ = 0;
    std::string request_;
    std::vector<std::string> retired_requests_;
    std::string backend_digest_;
    bool decoding_ = false;
    bool pending_decode_ = false;
    uint64_t checkpoint_positions_ = 0;
    uint64_t checkpoint_step_ = 0;
    bool checkpoint_decoding_ = false;
};

}  // namespace strata::distributed
