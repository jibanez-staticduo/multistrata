#pragma once

#include <cstdint>

namespace strata::distributed {

struct GraphRouteWorkspace {
    int* gathered_ids = nullptr;
    int* sticky_status = nullptr;
};

bool allocate_graph_route_workspace(GraphRouteWorkspace& workspace);
void free_graph_route_workspace(GraphRouteWorkspace& workspace) noexcept;
bool clear_graph_route_status(int* sticky_status, void* stream);
bool enqueue_graph_route_comparison(const int* gathered_ids, int* sticky_status, void* stream);

} // namespace strata::distributed
