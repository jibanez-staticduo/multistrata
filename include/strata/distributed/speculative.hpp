#pragma once

#include "strata/distributed/worker.hpp"

#include <functional>

namespace strata::distributed {

struct SpeculativeWindow {
    std::string request_id;
    uint64_t window_id = 0;  // Strictly increasing for the verifier's lifetime.
    uint64_t position = 0;
    uint64_t draw_base = 0;
    // The already emitted feedback token followed by zero to four proposed tokens.
    std::vector<int32_t> inputs;
    std::vector<int32_t> history;  // Consumed baseline, excluding feedback; size == position.
    uint64_t output_budget = 0;
};

struct SpeculativeCallbacks {
    // Synchronous all-rank operations. The caller owns monotonic Worker steps and schedules.
    std::function<Failure(StepPhase, int32_t token, uint64_t position)> execute;
    // Root only. Must be a pure logical draw: rejected suffix attempts cannot advance sampler state.
    std::function<Failure(const std::vector<int32_t>& history, uint64_t draw, int32_t& token)> sample;
    std::function<Failure(std::string_view label, std::string& bytes)> broadcast;
    // Root only, after a completed row, before the next forward overwrites its final residual.
    std::function<Failure(size_t row, int32_t input, int32_t next_token)> record_target_row;
};

struct SpeculativeResult {
    std::vector<int32_t> tokens;
    std::vector<int32_t> consumed_inputs;
    size_t accepted_rows = 0;  // Includes feedback; accepted_drafts == accepted_rows - 1.
    size_t accepted_drafts = 0;
    int32_t next_feedback = -1;
    uint64_t next_position = 0;
};

// Exact target sampling plus equality admission. This serial correctness foundation does not claim
// batched verification speedups or rejection-sampling acceptance. It never emits output itself.
class SpeculativeVerifier final {
public:
    SpeculativeVerifier(uint32_t rank, int32_t n_vocab, uint32_t context)
        : rank_(rank), n_vocab_(n_vocab), context_(context) {}
    // Any failure is terminal: caller must abort the session. Result stays empty until all replay
    // commits and final agreement succeed. Only committed rows may update history or draft state.
    Failure verify(const SpeculativeWindow&, const SpeculativeCallbacks&, SpeculativeResult&);
private:
    uint32_t rank_;
    int32_t n_vocab_;
    uint32_t context_;
    uint64_t last_window_ = 0;
    bool failed_ = false;
};

} // namespace strata::distributed
