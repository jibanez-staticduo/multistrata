#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::distributed {

// Diagnostic views borrow device memory on the supplied stream. An observer
// must consume or enqueue its copy before returning; the next stage reuses it.
struct ParityTraceView {
    const char* stage;
    int64_t layer;
    int64_t position;
    const float* device_data;
    std::size_t elements;
};
using ParityObserver = std::function<bool(const ParityTraceView&, void* stream, std::string& err)>;

} // namespace strata::distributed
