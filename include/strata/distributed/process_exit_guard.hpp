#pragma once

#include <cstdlib>

namespace strata::distributed {

// A failed enqueue may report an earlier asynchronous error. Retain the
// borrowed buffers until completion is confirmed, including that failure case.
template<class Enqueue, class Wait>
auto bounded_readback(bool& pending, Enqueue enqueue, Wait wait) {
    pending = true;
    if (auto failure = enqueue()) return failure;
    auto failure = wait();
    if (!failure) pending = false;
    return failure;
}

// Declare after CUDA-owning objects. Pending DMA must outlive all borrowed
// buffers, so process teardown replaces normal destruction on that failure.
template<class Owner> class PendingDmaExitGuard final {
public:
    explicit PendingDmaExitGuard(Owner& owner) : owner_(owner) {}
    ~PendingDmaExitGuard() { if (owner_.requires_process_exit()) std::_Exit(2); }
    PendingDmaExitGuard(const PendingDmaExitGuard&) = delete;
    PendingDmaExitGuard& operator=(const PendingDmaExitGuard&) = delete;
private:
    Owner& owner_;
};

} // namespace strata::distributed
