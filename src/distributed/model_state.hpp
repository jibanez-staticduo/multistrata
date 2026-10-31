#pragma once

#include "strata/core/session.hpp"
#include "strata/distributed/model_checkpoint.hpp"

namespace strata::distributed::detail {

// Synchronous completion contract; production injects bounded pinned CUDA
// transfers, CPU tests inject ordinary copies over the real SessionState layout.
using ModelStateCopy = std::function<bool(void* host, void* device, std::size_t,
                                        bool to_device, std::string&)>;
struct ModelStateArrays {
    std::vector<StateEvidenceArray> sources;
    std::vector<ModelCheckpointDestination::Array> destinations;
};

// Only plain resident FP16 with a verified identity page table is supported.
// Every control field is included at reset as well as after a real token.
bool describe_checkpoint_state(core::SessionState&, const core::ModelGeometry&,
                               const std::vector<GdnStateSource>&, uint64_t tokens,
                               const ModelStateCopy&, ModelStateArrays&, std::string& err);

} // namespace strata::distributed::detail
