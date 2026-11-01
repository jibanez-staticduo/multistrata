#pragma once

#include "strata/distributed/serve_protocol.hpp"
#include "strata/distributed/worker.hpp"
#include "strata/distributed/speculative.hpp"

#include <functional>
#include <span>
#include <string>

namespace strata::distributed {

inline constexpr size_t kAutoregressiveMaxLine = kDistributedMaxLineBytes;
inline constexpr size_t kAutoregressiveChunk = 24u << 10;

// Synchronous host wall time at service callback boundaries, not device kernel time.
struct RequestTiming {
    uint64_t queue_ns = 0;
    uint64_t prefill_ns = 0;
    uint64_t decode_ns = 0;
    uint64_t sampling_ns = 0;
    uint64_t mtp_draft_ns = 0;
    uint64_t mtp_checkpoint_ns = 0;
    uint64_t mtp_rollback_ns = 0;
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
};

inline std::string request_timing_json(const RequestTiming& timing) {
    return "{\"clock\":\"host_steady_wall\",\"queue_ns\":" + std::to_string(timing.queue_ns) +
        ",\"prefill_ns\":" + std::to_string(timing.prefill_ns) +
        ",\"decode_ns\":" + std::to_string(timing.decode_ns) +
        ",\"sampling_ns\":" + std::to_string(timing.sampling_ns) +
        ",\"mtp_draft_ns\":" + std::to_string(timing.mtp_draft_ns) +
        ",\"mtp_checkpoint_ns\":" + std::to_string(timing.mtp_checkpoint_ns) +
        ",\"mtp_rollback_ns\":" + std::to_string(timing.mtp_rollback_ns) +
        ",\"prompt_tokens\":" + std::to_string(timing.prompt_tokens) +
        ",\"generated_tokens\":" + std::to_string(timing.generated_tokens) + "}";
}

// One synchronous coordinator loop. Only rank zero polls/samples/emits. Broadcast and execute callbacks
// finish on every rank before returning; failures are terminal and leave no output ready for publication.
struct AutoregressiveCallbacks {
    std::function<Failure(uint32_t wait_ms, std::string& line, bool& eof)> poll;
    std::function<Failure(std::string_view label, std::string& bytes)> broadcast;
    std::function<Failure()> health;
    std::function<Failure(StepPhase, std::string_view request, int32_t token, uint64_t position)> execute;
    // Optional exact multi-token prompt forward. The coordinator commits the returned rows once.
    std::function<Failure(std::string_view request, std::span<const int32_t> tokens,
                          uint64_t first_position)> execute_prefill_batch;
    uint32_t prefill_batch_size = 1;
    std::function<Failure(const ServeRequest&, const std::vector<int32_t>& history, uint64_t draw, int32_t&)> sample;
    std::function<void(std::string_view type, std::string_view request, int32_t token, std::string_view state)> emit;
    // Opt-in replacement for the ordinary done event; other event types use emit.
    std::function<void(std::string_view type, std::string_view request, int32_t token,
                       std::string_view state, const RequestTiming&)> emit_timing;
    // Optional monotonic clock in nanoseconds; defaults to std::chrono::steady_clock.
    std::function<uint64_t()> now_ns;
    bool mtp_enabled = false;
    // Optional root-only drafter adapter. Followers enable the same policy but supply no hooks.
    std::function<Failure(const ServeRequest&)> begin_request;
    std::function<Failure(int32_t next_token, uint64_t position)> prompt_row;
    std::function<Failure(const std::vector<int32_t>& history, int32_t feedback, uint64_t draw,
                          std::vector<int32_t>& drafts)> propose;
    std::function<Failure(size_t row, int32_t input, int32_t next)> record_target_row;
    std::function<Failure(const SpeculativeWindow&, const SpeculativeResult&)> accepted_window;
};

Failure run_autoregressive(uint32_t rank, int32_t n_vocab, uint32_t context,
                           const AutoregressiveCallbacks& callbacks, uint32_t operation_budget_ms = 60000);

}  // namespace strata::distributed
