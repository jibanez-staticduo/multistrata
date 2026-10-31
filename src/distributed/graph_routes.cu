#include "strata/distributed/graph_routes.hpp"

#include <cuda_runtime.h>

namespace strata::distributed {
namespace {
constexpr int kRouteWidth = 10;
constexpr int kWorldSize = 3;
constexpr int kMaxExperts = 512;

__global__ void compare_routes(const int* gathered, int* sticky_status) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *sticky_status != 0) return;
    for (int rank = 0; rank < kWorldSize; ++rank) {
        const int* ids = gathered + rank * kRouteWidth;
        for (int slot = 0; slot < kRouteWidth; ++slot) {
            const int id = ids[slot];
            if (id < 0 || id >= kMaxExperts) {
                atomicCAS(sticky_status, 0, 1);
                return;
            }
            for (int prior = 0; prior < slot; ++prior) {
                if (ids[prior] == id) {
                    atomicCAS(sticky_status, 0, 2);
                    return;
                }
            }
        }
    }
    const int* reference = gathered;
    for (int rank = 1; rank < kWorldSize; ++rank) {
        const int* ids = gathered + rank * kRouteWidth;
        for (int slot = 0; slot < kRouteWidth; ++slot) {
            if (reference[slot] != ids[slot]) {
                atomicCAS(sticky_status, 0, 3);
                return;
            }
        }
    }
}
}

bool allocate_graph_route_workspace(GraphRouteWorkspace& workspace) {
    if (workspace.gathered_ids != nullptr || workspace.sticky_status != nullptr) return false;
    if (cudaMalloc(&workspace.gathered_ids, kWorldSize * kRouteWidth * sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.sticky_status, sizeof(int)) != cudaSuccess) {
        cudaFree(workspace.gathered_ids);
        workspace.gathered_ids = nullptr;
        return false;
    }
    if (cudaMemset(workspace.sticky_status, 0, sizeof(int)) != cudaSuccess) {
        free_graph_route_workspace(workspace);
        return false;
    }
    return true;
}

void free_graph_route_workspace(GraphRouteWorkspace& workspace) noexcept {
    if (workspace.sticky_status != nullptr) cudaFree(workspace.sticky_status);
    if (workspace.gathered_ids != nullptr) cudaFree(workspace.gathered_ids);
    workspace = {};
}

bool clear_graph_route_status(int* sticky_status, void* stream) {
    return sticky_status != nullptr && stream != nullptr &&
           cudaMemsetAsync(sticky_status, 0, sizeof(int), static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

bool enqueue_graph_route_comparison(const int* gathered_ids, int* sticky_status, void* stream) {
    if (gathered_ids == nullptr || sticky_status == nullptr || stream == nullptr) return false;
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    compare_routes<<<1, 1, 0, cuda_stream>>>(gathered_ids, sticky_status);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace strata::distributed
