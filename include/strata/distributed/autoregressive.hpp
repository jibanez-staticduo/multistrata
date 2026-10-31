#pragma once

#include "strata/distributed/serve_protocol.hpp"
#include "strata/distributed/worker.hpp"
#include "strata/distributed/speculative.hpp"

#include <functional>
#include <string>

namespace strata::distributed {

inline constexpr size_t kAutoregressiveMaxLine = 1u << 20;
inline constexpr size_t kAutoregressiveChunk = 24u << 10;

// One synchronous coordinator loop. Only rank zero polls/samples/emits. Broadcast and execute callbacks
// finish on every rank before returning; failures are terminal and leave no output ready for publication.
struct AutoregressiveCallbacks {
    std::function<Failure(uint32_t wait_ms, std::string& line, bool& eof)> poll;
    std::function<Failure(std::string_view label, std::string& bytes)> broadcast;
    std::function<Failure()> health;
    std::function<Failure(StepPhase, std::string_view request, int32_t token, uint64_t position)> execute;
    std::function<Failure(const ServeRequest&, const std::vector<int32_t>& history, uint64_t draw, int32_t&)> sample;
    std::function<void(std::string_view type, std::string_view request, int32_t token, std::string_view state)> emit;
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
