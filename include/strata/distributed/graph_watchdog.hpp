#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace strata::distributed {

// CUDA entry points can block inside the driver after a peer disappears.
// This independent host deadline exits without touching CUDA or borrowed buffers.
class GraphDeadlineWatchdog final {
public:
    explicit GraphDeadlineWatchdog(std::chrono::steady_clock::time_point deadline)
        : thread_([this, deadline] {
            std::unique_lock lock(mutex_);
            if (changed_.wait_until(lock, deadline, [this] { return completed_; })) return;
            constexpr char message[] = "MODEL_GRAPH fatal deadline: process exit with possible pending DMA\n";
            (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
            std::_Exit(86);
        }) {}
    ~GraphDeadlineWatchdog() {
        if (!succeeded_) {
            constexpr char message[] = "MODEL_GRAPH failed operation: process exit before CUDA cleanup\n";
            (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
            std::_Exit(86);
        }
        { std::lock_guard lock(mutex_); completed_ = true; }
        changed_.notify_one();
        thread_.join();
    }
    void complete() noexcept { succeeded_ = true; }
    GraphDeadlineWatchdog(const GraphDeadlineWatchdog&) = delete;
    GraphDeadlineWatchdog& operator=(const GraphDeadlineWatchdog&) = delete;
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool completed_ = false;
    bool succeeded_ = false;
    std::thread thread_;
};

} // namespace strata::distributed
