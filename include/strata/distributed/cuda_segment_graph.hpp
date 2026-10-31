#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace strata::distributed {

// A fixed-address GPU compute segment. The caller owns all captured buffers,
// keeps them alive, and orders all replays on its request stream. No transport,
// allocation, host callback or mutable graph parameters belong in enqueue.
// Capture, warmup and upload happen before READY. Any failure is terminal.
class CudaSegmentGraph final {
public:
    using Enqueue = std::function<void(void* stream)>;
    CudaSegmentGraph();
    ~CudaSegmentGraph();
    CudaSegmentGraph(const CudaSegmentGraph&) = delete;
    CudaSegmentGraph& operator=(const CudaSegmentGraph&) = delete;
    bool initialize(const Enqueue& enqueue, uint32_t timeout_ms, std::string& err);
    bool launch(void* stream, std::string& err);
    void poison() noexcept;
    bool ready() const noexcept;
    // True after an uncompleted warmup/replay or failed completion query.
    // The owner must terminate the worker process without freeing its buffers.
    bool requires_process_exit() const noexcept;
    std::size_t node_count() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::distributed
