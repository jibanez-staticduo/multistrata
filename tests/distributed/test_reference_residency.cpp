#include "strata/distributed/reference_residency.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {
using namespace strata::distributed;
struct Backend {
    using Error = int;
    static constexpr Error success = 0;
    void* stage = nullptr;
    size_t allocations = 0, releases = 0, copies = 0, fail_copy = 0;
    size_t fail_allocate = 0;
    Error allocate(void** p, size_t bytes) {
        ++allocations;
        if (fail_allocate) return 31;
        *p = stage = std::malloc(bytes);
        return *p ? 0 : 32;
    }
    void release(void* p) { ++releases; std::free(p); stage = nullptr; }
    Error copy(void* dst, const void* src, size_t bytes) {
        ++copies;
        if (src != stage || bytes > strata::core::NativeUpload<Backend>::capacity) return 33;
        if (copies == fail_copy) return 34;
        std::memcpy(dst, src, bytes);
        return 0;
    }
};
bool check(bool ok, const char* message) {
    if (!ok) std::fprintf(stderr, "reference residency failed: %s\n", message);
    return ok;
}
template<class F> bool rejects(F&& f) {
    try { f(); } catch (const std::runtime_error&) { return true; }
    return false;
}
}

int main() {
    bool ok = true;
    const auto plan = ReferenceResidencyPlan::all({37, 61, 83}, 512);
    ok &= check(plan.layer_bytes == std::vector<size_t>{18944, 31232, 42496} &&
                plan.owned_weight_bytes == 92672 && plan.required_bytes(1024) == 93696,
                "actual per-layer original slot sizes, with no selected-expert multiplier or padding");
    const auto max = std::numeric_limits<size_t>::max();
    ok &= check(rejects([&] { (void)ReferenceResidencyPlan::all({max}, 2); }) &&
                rejects([&] { (void)ReferenceResidencyPlan::all({max, 1}, 1); }) &&
                rejects([&] { (void)plan.required_bytes(max); }), "overflow rejects before allocation");
    ok &= check(rejects([] { (void)ReferenceResidencyPlan::all({}, 512); }) &&
                rejects([] { (void)ReferenceResidencyPlan::all({0}, 512); }) &&
                rejects([] { (void)ReferenceResidencyPlan::all({37}, 0); }), "empty plans reject");

    // Different row widths represent mixed original GU/down codecs. Literal
    // role sentinels establish the blob contract independently of upload order.
    const size_t role_bytes[3] = {2 * 13, 2 * 13, 5 * 7};
    constexpr size_t slot = 87, experts = 4;
    std::array<std::vector<uint8_t>, 3> source;
    const uint8_t* roles[3];
    for (size_t role = 0; role < source.size(); ++role) {
        source[role].resize(experts * role_bytes[role]);
        for (size_t e = 0; e < experts; ++e)
            for (size_t b = 0; b < role_bytes[role]; ++b)
                source[role][e * role_bytes[role] + b] = uint8_t(17 + role * 67 + e * 9 + b);
        roles[role] = source[role].data();
    }
    const auto original = source;
    std::vector<uint8_t> resident(experts * slot + 2, 0xcd);
    Backend backend;
    {
        strata::core::NativeUpload<Backend> upload(backend);
        ok &= check(upload_reference_experts(upload, resident.data() + 1, experts, slot, roles, role_bytes) == 0,
                    "original role upload");
        for (size_t e = 0; e < experts; ++e) {
            const auto* blob = resident.data() + 1 + e * slot;
            for (size_t b = 0; b < 26; ++b) {
                ok &= check(blob[b] == uint8_t(17 + e * 9 + b), "gate original bytes");
                ok &= check(blob[26 + b] == uint8_t(84 + e * 9 + b), "up original bytes");
            }
            for (size_t b = 0; b < 35; ++b)
                ok &= check(blob[52 + b] == uint8_t(151 + e * 9 + b), "down original bytes");
        }
        // Repeated and reordered tokens read the immutable expert slots.
        for (const size_t id : {3u, 0u, 2u, 3u, 1u, 0u}) {
            const auto* blob = resident.data() + 1 + id * slot;
            ok &= check(blob[0] == uint8_t(17 + id * 9) && blob[26] == uint8_t(84 + id * 9) &&
                        blob[52] == uint8_t(151 + id * 9), "stateful routed reads preserve expert identity");
        }
        ok &= check(source == original && resident.front() == 0xcd && resident.back() == 0xcd,
                    "source and destination guards preserved");
        ok &= check(backend.allocations == 1 && backend.copies == 12, "one pinned stage reused for all experts");
    }
    ok &= check(backend.releases == 1, "pinned stage released once");
    Backend failed;
    failed.fail_copy = 5;
    std::fill(resident.begin(), resident.end(), 0xcd);
    {
        strata::core::NativeUpload<Backend> upload(failed);
        ok &= check(upload_reference_experts(upload, resident.data() + 1, experts, slot, roles, role_bytes) == 34 &&
                    failed.copies == 5 && resident[1 + slot + 26] == 0xcd && resident[1 + 2 * slot] == 0xcd,
                    "partial failure stops, with no selected-expert fallback or tail copy");
    }
    ok &= check(failed.releases == 1, "failed transfer releases pinned stage");
    Backend no_memory;
    no_memory.fail_allocate = 1;
    {
        strata::core::NativeUpload<Backend> upload(no_memory);
        ok &= check(upload_reference_experts(upload, resident.data() + 1, experts, slot, roles, role_bytes) == 31 &&
                    no_memory.copies == 0, "pinned allocation failure returned before any copy");
        ok &= check(rejects([&] {
            (void)upload_reference_experts(upload, resident.data() + 1, experts, slot - 1, roles, role_bytes);
        }), "incompatible original role boundaries reject");
    }
    if (!ok) return 1;
    std::puts("reference residency: sizing, immutable original blobs, routed reads and failures passed");
}
