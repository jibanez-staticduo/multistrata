#pragma once

#include "strata/distributed/serve_protocol.hpp"
#include "strata/distributed/worker.hpp"

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace strata::distributed {

inline constexpr uint32_t kMultiSlotMaximum = 8;

struct MultiSlotConfig {
    uint32_t rank = 0;
    uint32_t slots = 1;
    int32_t n_vocab = 0;
    uint32_t context = 0;
    // Remember every admitted identity, including retired ones, without unbounded growth.
    uint32_t request_limit = 4096;
};

struct MultiSlotCallbacks {
    // Synchronous all-rank agreement. Only root supplies bytes; followers supply an empty string.
    std::function<Failure(std::string_view label, std::string& bytes)> broadcast;
    // Each slot owns its Worker, ModelForward and complete mutable session state. The adapter owns
    // one GLOBAL collective sequence; a slot-local Worker counter must never restart that sequence.
    // Successful commit means all ranks committed and checked finite outputs. Reset/cancel clears
    // only this slot. The scheduler never runs more than one forward at once.
    std::function<Failure(uint32_t slot, StepPhase, std::string_view request,
                          int32_t token, uint64_t position)> execute;
    std::function<Failure(uint32_t slot, const ServeRequest&, const std::vector<int32_t>& history,
                          uint64_t draw, int32_t& token)> sample;
    std::function<void(std::string_view type, uint32_t slot, std::string_view request, int32_t token)> emit;
    // Required, idempotent, non-throwing: invalidate ALL slot state and the shared transport.
    std::function<void()> abort_all;
};

enum class MultiSlotOutcome { accepted, rejected, progressed, idle, failed };
struct MultiSlotResult {
    MultiSlotOutcome outcome = MultiSlotOutcome::idle;
    uint32_t slot = kMultiSlotMaximum;
    Failure failure;
    std::string reason;
};

// Single-threaded deterministic scalar scheduler, not a batched model implementation. Call the SAME
// submit/cancel/tick operations on all ranks in the SAME order. submit takes the exact public generate
// JSON, preserving sampling parameters without a second serialization. Invalid/full/duplicate admission
// is a nonterminal rejection. Divergent actions, stale envelopes or callback failures abort the group.
// The caller broadcasts its input protocol before dispatching these methods. These methods additionally
// agree their complete decisions before mutating state. They must not be called reentrantly.
class MultiSlotScheduler {
 public:
    MultiSlotScheduler(MultiSlotConfig config, MultiSlotCallbacks callbacks);
    MultiSlotScheduler(const MultiSlotScheduler&) = delete;
    MultiSlotScheduler& operator=(const MultiSlotScheduler&) = delete;

    MultiSlotResult submit(std::string_view exact_generate_json);
    MultiSlotResult cancel(std::string_view request_id);
    // Quantum one: at most one forward + commit; cancellation takes effect at this slot's next turn.
    MultiSlotResult tick();
    // Emergency local fail-stop; no peer agreement is possible after a transport failure.
    MultiSlotResult abort(Failure failure);

    bool failed() const { return bool(failure_); }
    size_t active() const;
    uint64_t global_tick() const { return tick_; }
    std::optional<std::string> request(uint32_t slot) const;

 private:
    struct Slot {
        std::optional<ServeRequest> request;
        std::vector<int32_t> history;
        size_t prompt = 0;
        uint64_t generated = 0;
        int32_t feedback = -1;
        bool cancel = false;
    };
    Failure agree(std::string_view kind, const std::string& expected);
    Failure sampled(std::string_view identity, int32_t& token);
    Failure execute(uint32_t slot, StepPhase phase, int32_t token, uint64_t position);
    std::string identity(uint32_t slot) const;
    MultiSlotResult reject(std::string reason) const;
    MultiSlotResult terminal() const;

    MultiSlotConfig config_;
    MultiSlotCallbacks callbacks_;
    std::vector<Slot> slots_;
    std::set<std::string> admitted_;
    uint32_t cursor_ = 0;
    uint64_t tick_ = 0;
    uint64_t action_ = 1;
    Failure failure_;
};

}  // namespace strata::distributed
