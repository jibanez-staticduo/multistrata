#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::distributed {

// CPU-only mirror of the compact CUDA payload contract, used by schedule planning.
inline int compact_down_payload_width(int ggml_type, int input_width,
                                      const std::array<uint64_t, 3>& valid_input) {
    const int block = ggml_type == 20 ? 32 : ggml_type == 42 ? 64 : 0;
    if (!block || input_width <= 0 || input_width > 1024 || input_width % block)
        throw std::invalid_argument("unsupported compact expert down codec or width");

    uint64_t sum = 0;
    uint64_t largest = 0;
    for (const auto count : valid_input) {
        if (!count || count % uint64_t(block) || sum > uint64_t(input_width) ||
            count > uint64_t(input_width) - sum)
            throw std::invalid_argument("invalid compact expert ownership counts");
        sum += count;
        largest = std::max(largest, count);
    }
    if (sum != uint64_t(input_width) || largest / uint64_t(block) * 4 > uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("incomplete compact expert ownership");
    return int(largest / uint64_t(block) * 4);
}

} // namespace strata::distributed
