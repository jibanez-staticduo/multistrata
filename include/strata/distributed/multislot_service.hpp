#pragma once

#include "strata/distributed/autoregressive.hpp"
#include "strata/distributed/multislot.hpp"

namespace strata::distributed {
struct MultiSlotServiceCallbacks {
    std::function<Failure(uint32_t wait_ms, std::string& line, bool& eof)> poll;
    std::function<Failure()> health;
    MultiSlotCallbacks scheduler;
    // Root only. ready/health use global state; done/cancelled refer to one retired request.
    std::function<void(std::string_view type, std::string_view request, int32_t token,
                       std::string_view state, std::string_view reason)> emit;
    // Opt-in callback for the final done event. Times are synchronous host callback wall time.
    std::function<void(std::string_view request, const RequestTiming&)> emit_timing;
    std::function<uint64_t()> now_ns;
};

// JSONL service over the real deterministic scheduler. Pool initialization and sampler allocation
// must finish before entry. Root input is agreed in bounded chunks before every scheduling quantum.
Failure run_multislot_service(MultiSlotConfig, const MultiSlotServiceCallbacks&,
                              uint32_t operation_budget_ms = 60000);
} // namespace strata::distributed
