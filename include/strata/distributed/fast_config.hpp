#pragma once
#include "strata/distributed/fast_runtime.hpp"

namespace strata::distributed {
struct FastLaunchConfig {
    FastRuntimeConfig runtime;
    std::string pack, dense, embedding, ple, mtp;
    uint32_t drafts = 0;
};
FastLaunchConfig parse_fast_launch(int argc, char** argv);
}
