#include "strata/distributed/process_exit_guard.hpp"

#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>

namespace {
struct Owner { bool pending; bool requires_process_exit() const { return pending; } };
struct Cleanup {
    int fd;
    ~Cleanup() { const char marker = 'x'; (void)::write(fd, &marker, 1); }
};
bool run(bool pending) {
    int descriptors[2];
    if (::pipe(descriptors)) return false;
    const auto pid = ::fork();
    if (pid == 0) {
        ::close(descriptors[0]);
        Owner owner{pending};
        try {
            Cleanup cuda_owner{descriptors[1]};
            strata::distributed::PendingDmaExitGuard guard(owner);
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
           WEXITSTATUS(status) == (pending ? 2 : 0) && count == (pending ? 0 : 1);
}
}
int main() {
    if (!run(false) || !run(true)) { std::puts("exit guard failed"); return 1; }
    std::puts("pending DMA exits before cleanup; completed DMA unwinds normally");
}
