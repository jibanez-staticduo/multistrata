#pragma once

#include <cstdlib>

namespace strata::distributed {

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
