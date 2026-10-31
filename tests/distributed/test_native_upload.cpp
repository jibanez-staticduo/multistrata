#include "strata/core/native_upload.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
struct Backend {
    using Error = int;
    static constexpr Error success = 0;
    void* stage = nullptr;
    size_t allocated = 0, allocations = 0, copies = 0, releases = 0;
    size_t fail_copy = 0;
    bool fail_allocate = false;
    std::vector<size_t> lengths;
    Error allocate(void** p, size_t bytes) {
        ++allocations;
        if (fail_allocate) return 11;
        allocated = bytes;
        *p = stage = std::malloc(bytes);
        return *p ? 0 : 12;
    }
    void release(void* p) { ++releases; std::free(p); }
    Error copy(void* dst, const void* src, size_t bytes) {
        ++copies;
        if (src != stage || bytes > allocated) return 13;
        lengths.push_back(bytes);
        if (copies == fail_copy) return 14;
        std::memcpy(dst, src, bytes);
        return 0;
    }
};
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
}

int main() {
    using Upload = strata::core::NativeUpload<Backend>;
    const size_t bytes = Upload::capacity * 2 + 37;
    std::vector<unsigned char> source(bytes + 19), destination(bytes + 22, 0xa5);
    for (size_t i = 0; i < source.size(); ++i) source[i] = (unsigned char) ((i * 131 + 17) % 251);
    Backend backend;
    {
        Upload upload(backend);
        check(upload.copy(nullptr, nullptr, 0) == 0 && backend.allocations == 0, "zero bytes allocates nothing");
        check(upload.copy(destination.data() + 11, source.data() + 7, bytes) == 0, "unaligned multi-chunk upload");
        check(std::memcmp(destination.data() + 11, source.data() + 7, bytes) == 0, "every original source byte preserved");
        check(destination[10] == 0xa5 && destination[bytes + 11] == 0xa5, "destination guards preserved");
        check(backend.allocated == Upload::capacity && backend.allocations == 1, "pinned allocation bounded to 8 MiB");
        check(backend.lengths == std::vector<size_t>{Upload::capacity, Upload::capacity, 37}, "last partial chunk copied exactly");
        check(upload.copy(destination.data(), source.data(), 1) == 0 && backend.allocations == 1, "stage reused across tensors");
    }
    check(backend.releases == 1, "stage released once");
    Backend failed;
    failed.fail_copy = 2;
    std::fill(destination.begin(), destination.end(), 0xa5);
    {
        Upload upload(failed);
        check(upload.copy(destination.data(), source.data(), bytes) == 14, "copy failure returned without fallback");
        check(failed.copies == 2 && destination[Upload::capacity] == 0xa5, "failure stops before tail or publication");
    }
    check(failed.releases == 1, "failed copy releases stage");
    Backend no_memory;
    no_memory.fail_allocate = true;
    {
        Upload upload(no_memory);
        check(upload.copy(destination.data(), source.data(), bytes) == 11 && no_memory.copies == 0,
              "allocation failure returned without any transfer");
    }
    check(no_memory.releases == 0, "failed allocation has no release");
    std::puts(failures ? "test_native_upload FAILED" : "test_native_upload PASSED");
    return failures ? 1 : 0;
}
