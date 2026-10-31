#pragma once

#include "strata/distributed/parity_trace.hpp"

#include <memory>

namespace strata::distributed {

// Construct only when diagnostic tracing is requested. Capture is bounded by
// timeout_ms and never synchronizes the device or an entire CUDA stream.
class ParityTraceWriter final {
public:
    explicit ParityTraceWriter(std::string directory, uint32_t timeout_ms = 60000,
                               std::size_t max_elements = 25600);
    ~ParityTraceWriter();
    ParityTraceWriter(const ParityTraceWriter&) = delete;
    ParityTraceWriter& operator=(const ParityTraceWriter&) = delete;
    // Creates a fresh request directory. Existing evidence is never replaced.
    bool request(const std::string& name, std::string& err);
    bool capture(const ParityTraceView&, void* stream, std::string& err);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::distributed
