#include "strata/distributed/worker.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace strata::distributed {
namespace {
Failure reject(std::string_view detail) {
    return Failure::fail(FailureCode::bad_operation, detail);
}

bool same_collective(const ScheduledCollective& a, const ScheduledCollective& b) {
    const auto& x = a.descriptor;
    const auto& y = b.descriptor;
    return x.operation.sequence == y.operation.sequence && x.operation.layer == y.operation.layer &&
           x.operation.step == y.operation.step && x.kind == y.kind && x.type == y.type &&
           x.elements == y.elements && x.valid_counts == y.valid_counts && a.route == b.route;
}

Failure validate_schedule(const StepCommand& command, uint64_t next) {
    for (const auto& item : command.expected_collective_schedule) {
        const auto& d = item.descriptor;
        if (next == std::numeric_limits<uint64_t>::max() || d.operation.sequence != next ||
            d.operation.step != command.step) return reject("collective sequence or step mismatch");
        ++next;
        uint64_t bytes = 0;
        if (!element_bytes(d.elements, d.type, bytes)) return reject("invalid collective dtype or extent");
        switch (d.kind) {
            case CollectiveKind::all_reduce_sum:
                if (!d.valid_counts.empty()) return reject("reduction carries gather counts");
                break;
            case CollectiveKind::all_gather_padded:
                if (d.valid_counts.size() != kTpWorld ||
                    d.elements > std::numeric_limits<uint64_t>::max() / kTpWorld ||
                    !element_bytes(d.elements * kTpWorld, d.type, bytes) ||
                    std::any_of(d.valid_counts.begin(), d.valid_counts.end(),
                                [&](uint64_t n) { return n > d.elements; })) {
                    return reject("invalid padded gather counts or extent");
                }
                break;
            default: return reject("unknown collective kind");
        }
    }
    return {};
}
}  // namespace

Worker::Worker(WorkerConfig config, ExecutionBackend& backend)
    : config_(std::move(config)), backend_(backend) {}

Worker::~Worker() {
    if (state_ != WorkerState::fresh && state_ != WorkerState::failed && state_ != WorkerState::aborted) {
        backend_.abort();
        backend_.discard_session();
    }
}

Failure Worker::validate(const StepCommand& command) const {
    if (config_.session.empty() || config_.manifest_hash.empty() || config_.rank >= kTpWorld ||
        config_.slot >= kWorkerMaxSlots || collective_sequence() == 0) {
        return Failure::fail(FailureCode::invalid_config, "worker identity/manifest/rank/slot/cursor is invalid");
    }
    if (command.session != config_.session) return reject("foreign session");
    if (command.manifest_hash != config_.manifest_hash) {
        return Failure::fail(FailureCode::model_mismatch, "manifest mismatch");
    }
    if (command.slot != config_.slot) return reject("command belongs to another worker slot");
    if (command.step != next_step_ || next_step_ == std::numeric_limits<uint64_t>::max()) {
        return reject("stale, duplicate, skipped or exhausted step");
    }
    const bool compute = command.phase == StepPhase::prefill || command.phase == StepPhase::decode;
    if (compute) {
        if (command.tokens.empty() || command.tokens.size() != command.positions.size() ||
            command.tokens.size() > std::numeric_limits<uint64_t>::max() - committed_positions_) {
            return reject("invalid token/position extent");
        }
        for (size_t i = 0; i < command.tokens.size(); ++i) {
            if (command.tokens[i] < 0 || command.positions[i] != committed_positions_ + i) {
                return reject("negative token or noncontiguous position");
            }
        }
    } else if (!command.tokens.empty() || !command.positions.empty()) {
        return reject("only prefill/decode may carry tokens and positions");
    }

    switch (command.phase) {
        case StepPhase::init:
            if (state_ != WorkerState::fresh || !command.request.empty()) return reject("init order/identity");
            break;
        case StepPhase::prefill:
            if ((state_ != WorkerState::idle && state_ != WorkerState::active) || decoding_) {
                return reject("prefill requires idle or committed prefill state");
            }
            break;
        case StepPhase::decode:
            if (state_ != WorkerState::active || committed_positions_ == 0 || command.tokens.size() != 1) {
                return reject("decode requires committed input and exactly one token");
            }
            break;
        case StepPhase::commit:
            if (state_ != WorkerState::pending) return reject("commit requires a completed candidate");
            break;
        case StepPhase::checkpoint:
            if (state_ != WorkerState::active || committed_positions_ == 0)
                return reject("checkpoint requires a committed active request");
            break;
        case StepPhase::rollback:
            if ((state_ != WorkerState::active && state_ != WorkerState::pending) || checkpoint_step_ == 0)
                return reject("rollback requires an active request with a committed checkpoint");
            break;
        case StepPhase::reset:
        case StepPhase::cancel:
            if (state_ != WorkerState::active && state_ != WorkerState::pending) {
                return reject("reset/cancel requires an active request");
            }
            break;
        case StepPhase::abort: break;
        default: return reject("unknown step phase");
    }
    if (command.phase != StepPhase::init && command.phase != StepPhase::abort) {
        if (command.request.empty()) return reject("request identity is empty");
        if (state_ == WorkerState::idle) {
            if (std::find(retired_requests_.begin(), retired_requests_.end(), command.request) !=
                retired_requests_.end()) return reject("request was already retired");
        } else if (command.request != request_) {
            return reject("command belongs to another request");
        }
    }
    if (!compute && !command.expected_collective_schedule.empty()) {
        return reject("only prefill/decode may carry a collective schedule");
    }
    return validate_schedule(command, collective_sequence());
}

uint64_t Worker::collective_sequence() const {
    return config_.collective_sequence ? *config_.collective_sequence : next_collective_;
}

std::string Worker::digest() const {
    // Explicit lengths keep opaque backend digest bytes unambiguous. Rank is omitted intentionally:
    // backend implementations supply their declared global/canonical state summary for peer comparison.
    std::string value = "noncrypto:worker-v1:" + std::to_string(static_cast<uint32_t>(state_)) + ":" +
           std::to_string(next_step_) + ":" + std::to_string(collective_sequence()) + ":" +
           std::to_string(committed_positions_) + ":" + std::to_string(pending_positions_) + ":" +
           std::to_string(decoding_) + ":" + std::to_string(pending_decode_) + ":" +
           std::to_string(config_.session.size()) + ":" + config_.session + ":" +
           std::to_string(config_.manifest_hash.size()) + ":" + config_.manifest_hash + ":" +
           std::to_string(request_.size()) + ":" + request_ + ":" +
           std::to_string(backend_digest_.size()) + ":" + backend_digest_;
    if (config_.slot != 0) value += ":slot=" + std::to_string(config_.slot);
    if (checkpoint_step_ != 0)
        value += ":checkpoint=" + std::to_string(checkpoint_step_) + ":" +
                 std::to_string(checkpoint_positions_) + ":" + std::to_string(checkpoint_decoding_);
    value += ":" + std::to_string(retired_requests_.size());
    for (const auto& request : retired_requests_) value += ":" + std::to_string(request.size()) + ":" + request;
    return value;
}

StepResult Worker::fail(Failure failure) {
    state_ = WorkerState::failed;
    failure_ = std::move(failure);
    backend_.abort();
    backend_.discard_session();
    request_.clear();
    retired_requests_.clear();
    backend_digest_.clear();
    pending_positions_ = committed_positions_ = 0;
    checkpoint_step_ = checkpoint_positions_ = 0;
    checkpoint_decoding_ = false;
    return {StepStatus::failed, failure_, {}, false};
}

StepResult Worker::execute(const StepCommand& command) {
    if (state_ == WorkerState::failed) return {StepStatus::failed, failure_, {}, false};
    if (state_ == WorkerState::aborted) return {StepStatus::aborted, failure_, {}, false};
    if (auto f = validate(command)) return {StepStatus::rejected, std::move(f), digest(), false};

    if (command.phase == StepPhase::abort) {
        state_ = WorkerState::aborted;
        failure_ = Failure::fail(FailureCode::aborted, "scheduled group abort");
        backend_.abort();
        backend_.discard_session();
        request_.clear();
        retired_requests_.clear();
        backend_digest_.clear();
        pending_positions_ = committed_positions_ = 0;
        checkpoint_step_ = checkpoint_positions_ = 0;
        checkpoint_decoding_ = false;
        return {StepStatus::aborted, failure_, {}, false};
    }

    try {
        const uint64_t first_collective = collective_sequence();
        std::vector<ScheduledCollective> planned;
        if (auto f = backend_.plan(command, planned)) return fail(std::move(f));
        if (planned.size() != command.expected_collective_schedule.size() ||
            !std::equal(planned.begin(), planned.end(), command.expected_collective_schedule.begin(), same_collective)) {
            return {StepStatus::rejected, reject("loaded policy disagrees with route/collective schedule"), digest(), false};
        }
        if (auto f = backend_.agree(command)) return fail(std::move(f));
        if (auto f = backend_.begin(command)) return fail(std::move(f));
        if (!backend_.model_driven_collectives()) {
            for (const auto& collective : command.expected_collective_schedule) {
                if (auto f = backend_.collective(collective)) return fail(std::move(f));
            }
        }
        BackendCompletion completed;
        if (auto f = backend_.finish(command, completed)) return fail(std::move(f));
        if (!completed.complete) return fail(reject("backend did not complete the scheduled command"));
        const bool compute = command.phase == StepPhase::prefill || command.phase == StepPhase::decode;
        if ((compute || command.phase == StepPhase::commit) && !completed.outputs_finite) {
            return fail(reject("backend reported nonfinite candidate output"));
        }
        if (!compute && !completed.all_ranks_committed) {
            return fail(reject("state transition was not committed on all ranks"));
        }
        if (completed.state_digest.empty()) return fail(reject("backend supplied no state summary"));
        if (collective_sequence() != first_collective)
            return fail(reject("shared worker cursor changed during backend execution"));

        bool output_ready = false;
        switch (command.phase) {
            case StepPhase::init: state_ = WorkerState::idle; break;
            case StepPhase::prefill:
            case StepPhase::decode:
                if (state_ == WorkerState::idle) {
                    checkpoint_step_ = checkpoint_positions_ = 0;
                    checkpoint_decoding_ = false;
                }
                request_ = command.request;
                pending_positions_ = committed_positions_ + command.tokens.size();
                pending_decode_ = command.phase == StepPhase::decode;
                state_ = WorkerState::pending;
                break;
            case StepPhase::commit:
                committed_positions_ = pending_positions_;
                pending_positions_ = 0;
                decoding_ = decoding_ || pending_decode_;
                pending_decode_ = false;
                state_ = WorkerState::active;
                output_ready = config_.rank == 0;
                break;
            case StepPhase::checkpoint:
                checkpoint_positions_ = committed_positions_;
                checkpoint_decoding_ = decoding_;
                checkpoint_step_ = command.step;
                break;
            case StepPhase::rollback:
                committed_positions_ = checkpoint_positions_;
                decoding_ = checkpoint_decoding_;
                pending_positions_ = 0;
                pending_decode_ = false;
                state_ = WorkerState::active;
                break;
            case StepPhase::reset:
            case StepPhase::cancel:
                retired_requests_.push_back(request_);
                request_.clear();
                committed_positions_ = pending_positions_ = 0;
                decoding_ = pending_decode_ = false;
                checkpoint_step_ = checkpoint_positions_ = 0;
                checkpoint_decoding_ = false;
                state_ = WorkerState::idle;
                break;
            case StepPhase::abort: break;  // Handled before backend execution.
        }
        ++next_step_;
        if (compute) {
            const auto next = first_collective + command.expected_collective_schedule.size();
            if (config_.collective_sequence) *config_.collective_sequence = next;
            else next_collective_ = next;
        }
        backend_digest_ = std::move(completed.state_digest);
        return {command.phase == StepPhase::cancel ? StepStatus::cancelled : StepStatus::completed,
                {}, digest(), output_ready};
    } catch (...) {
        // A backend may have enqueued device work before throwing. No local retry is safe.
        return fail(Failure::fail(FailureCode::comm_failed, "execution backend threw; session discarded"));
    }
}
}  // namespace strata::distributed
