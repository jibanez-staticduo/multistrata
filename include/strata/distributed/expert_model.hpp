#pragma once

#include "strata/distributed/model.hpp"

#include <array>
#include <memory>
#include <string>

namespace strata::distributed {

enum class ExpertDownPolicy { ExactLaneReference, PartialReduce };
enum class ExpertCollectivePolicy { Sequential, Batched };
struct ExpertFinishGraphStats {
    uint64_t graph_count = 0;
    uint64_t node_count = 0;
    uint64_t output_buffer_bytes = 0;
};
// ExactLaneReference gathers native FMA operands before the original warp sum;
// PartialReduce keeps the measured reassociation path for component benchmarks.

// Original GGUF routed experts, each rank owns only its gate/up output rows
// and down input blocks. The core applies router weights and shared output.
// Route readback has a monotonic deadline (1..3600000 ms). A transfer error or
// timeout makes the adapter unusable; the request owner must abort its session.
class NativeRoutedExperts final : public RoutedExperts {
public:
    NativeRoutedExperts(std::string pack_dir, std::string native_gguf,
                        const core::ModelGeometry&, int rank, LayerCollectives&,
                        std::array<uint64_t, 3> valid = {256, 192, 192},
                        uint32_t readback_timeout_ms = 60000,
                        ExpertDownPolicy down_policy = ExpertDownPolicy::ExactLaneReference,
                        ExpertCollectivePolicy collective_policy = ExpertCollectivePolicy::Sequential,
                        bool graph_finish = false);
    ~NativeRoutedExperts();
    NativeRoutedExperts(const NativeRoutedExperts&) = delete;
    NativeRoutedExperts& operator=(const NativeRoutedExperts&) = delete;
    bool initialize(std::string& err);
    bool forward(int64_t layer, const float* input, const int* ids,
                 int64_t k, float* parts, void* stream, std::string& err) override;
    uint64_t owned_weight_bytes() const;
    // Partial compute graphs only: exact batched finish after lane all-gather.
    ExpertFinishGraphStats finish_graph_stats() const;
    bool requires_process_exit() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::distributed
