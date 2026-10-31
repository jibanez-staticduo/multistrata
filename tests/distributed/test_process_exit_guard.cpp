#include "strata/distributed/process_exit_guard.hpp"
#include "strata/distributed/graph_watchdog.hpp"

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>

namespace {
struct Owner { bool pending; bool requires_process_exit() const { return pending; } };
struct Cleanup {
    int fd;
    ~Cleanup() { const char marker = 'x'; (void)::write(fd, &marker, 1); }
};
bool run(bool enqueue_failed, bool wait_failed) {
    int descriptors[2];
    if (::pipe(descriptors)) return false;
    const auto pid = ::fork();
    if (pid == 0) {
        ::close(descriptors[0]);
        Owner owner{false};
        try {
            Cleanup cuda_owner{descriptors[1]};
            strata::distributed::PendingDmaExitGuard guard(owner);
            (void)strata::distributed::bounded_readback(owner.pending,
                [&] { return enqueue_failed; }, [&] { return wait_failed; });
            throw 1;
        } catch (...) { ::close(descriptors[1]); std::_Exit(0); }
    }
    ::close(descriptors[1]);
    if (pid < 0) { ::close(descriptors[0]); return false; }
    char marker = 0;
    const auto count = ::read(descriptors[0], &marker, 1);
    ::close(descriptors[0]);
    int status = 0;
    return ::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
           WEXITSTATUS(status) == (enqueue_failed || wait_failed ? 2 : 0) &&
           count == (enqueue_failed || wait_failed ? 0 : 1);
}
bool driver_stall_exits_on_deadline() {
    const auto start = std::chrono::steady_clock::now();
    const auto pid = ::fork();
    if (pid == 0) {
        strata::distributed::GraphDeadlineWatchdog watchdog(start + std::chrono::milliseconds(30));
        for (;;) ::pause();
    }
    if (pid < 0) return false;
    int status = 0;
    return ::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 86 &&
           std::chrono::steady_clock::now() - start < std::chrono::seconds(2);
}
}
int main() {
    {
        strata::distributed::GraphDeadlineWatchdog completed(std::chrono::steady_clock::now() + std::chrono::seconds(1));
        completed.complete();
    }
    if (!run(false, false) || !run(false, true) || !run(true, false) || !driver_stall_exits_on_deadline()) {
        std::puts("exit guard failed"); return 1;
    }
    std::puts("pending DMA exits before cleanup; completed DMA unwinds normally");
}
