#pragma once
#include "strata/distributed/fast_assets.hpp"
#include "strata/distributed/fast_execution.hpp"
#include <cstdint>

namespace strata::distributed {
struct FastRuntimeConfig {
    uint32_t rank = 0, port = 30420, timeout_ms = 120000;
    uint32_t slots = 8, context = 262144, prompt_chunk = 2048;
    uint32_t drafts = 0;
    std::string mtp_path;
    int protocol_output_fd = -1; // preserved by entry point before asset diagnostics
    std::string rank0_address, session_uuid, model_hash, config_hash;
    std::string nccl_library, nccl_version, nccl_provenance;
};
// Providers must already be initialized with the same assets and stream.
// State hooks bind the sessions allocated here before the first captured step.
int run_fast_runtime(const FastRuntimeConfig&, FastAssets&, FastTensorParallel::Mixer,
                     FastTensorParallel::Experts, core::TensorParallelHooks* state_provider,
                     void* stream, std::function<bool(void*, std::string&)> validate = {});
}
