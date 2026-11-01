#include "strata/core/session.hpp"
#include "strata/core/vmm.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main() {
    using namespace strata::core;
    if (cudaFree(nullptr) != cudaSuccess || !vmm_available()) return 77;
    ModelGeometry geometry;
    qsa_set_kv_int8(true);
    qsa_set_kv_elastic(true, 2048);
    void *arena_a = nullptr, *arena_b = nullptr;
    const auto bytes = session_bytes(geometry, 8192, 10);
    if (cudaMalloc(&arena_a, bytes) != cudaSuccess || cudaMalloc(&arena_b, bytes) != cudaSuccess) return 1;
    SessionState a, b;
    if (!session_init(geometry, 8192, 10, arena_a, a) || !session_init(geometry, 8192, 10, arena_b, b)) return 2;
    auto& qa = a.qsa_states[a.qsa_primary()];
    auto& qb = b.qsa_states[b.qsa_primary()];
    std::vector<int8_t> pattern(4096, 37), got(pattern.size());
    if (cudaMemcpy(qb.k_q, pattern.data(), pattern.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    const auto before = qsa_kv_elastic_mapped_bytes();
    if (!qsa_kv_elastic_grow_state(qa, 8192) || qsa_kv_elastic_mapped_bytes() <= before) return 4;
    if (cudaDeviceSynchronize() != cudaSuccess || !qsa_kv_elastic_shrink_state(qa, 2048)) return 5;
    if (cudaMemcpy(got.data(), qb.k_q, got.size(), cudaMemcpyDeviceToHost) != cudaSuccess || got != pattern) return 6;
    if (qsa_kv_elastic_grow_state(qa, 8193) || qsa_kv_elastic_shrink_state(qa, -1)) return 7;
    session_release(a); session_release(b);
    cudaFree(arena_a); cudaFree(arena_b);
    std::puts("per-slot elastic KV growth preserves the other request's mapped data");
}
