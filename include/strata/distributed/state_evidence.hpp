#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core { struct SessionState; struct ModelGeometry; }

namespace strata::distributed {

// A successful readback must finish copying before returning. Callers own stream
// synchronization and deadlines; this helper has no CUDA runtime dependency.
using StateReadback = std::function<bool(void*, const void*, std::size_t, std::string&)>;
using StateArrayRead = std::function<bool(void*, std::size_t, std::string&)>;

struct StateEvidenceDescriptor {
    std::string kind; // kv, gdn, indexer, ple, residual, or metadata
    std::string name;
    std::string dtype; // explicit little endian: <f2, <f4, <f8, <i4, <i8, |u1
    std::vector<uint64_t> shape; // stored C-order shape
    int64_t layer = -1;
    // Nonempty only for rank-local arrays. Reconstruction is required before
    // numerical comparison; an individual shard never claims full coverage.
    std::vector<uint64_t> logical_shape;
    int shard_axis = -1;
    std::vector<uint64_t> axis_indices;
};

struct StateEvidenceArray {
    StateEvidenceDescriptor descriptor;
    StateArrayRead read;
};

struct StateEvidenceRequest {
    std::string request_id;
    uint64_t position = 0; // zero-based teacher-forced position just completed
    uint64_t token_count = 0;
    uint64_t max_bytes = 0; // mandatory total output bound, never silently truncate
    std::vector<StateEvidenceDescriptor> required;
    bool reset = false; // position=0, token_count=0; KV has zero loaded rows
};

std::vector<StateEvidenceDescriptor> state_evidence_inventory(const std::vector<StateEvidenceArray>&);
bool state_evidence_bytes(const StateEvidenceDescriptor&, uint64_t& bytes, std::string& err);
StateEvidenceArray borrowed_state_array(StateEvidenceDescriptor, const void*, StateReadback);
// FP16 physical [slot,head,page_size,dim] -> loaded [token,head,dim].
// Only loaded logical rows are emitted, including a partially filled last page.
StateArrayRead paged_kv_state_reader(const void* pool, uint64_t slots, uint64_t page_size,
                                    uint64_t heads, uint64_t dim, uint64_t tokens,
                                    std::vector<int32_t> page_table, StateReadback);

// Writes into a new directory. manifest.json is published last and only after
// every required array has been written; a failed capture is never complete.
bool write_state_evidence(const std::string& directory, const StateEvidenceRequest&,
                          const std::vector<StateEvidenceArray>&, std::string& err);

struct GdnStateSource {
    int64_t layer = -1;
    const float* recurrent = nullptr;
    const float* convolution = nullptr;
    // Local value heads and local q|k|v channels -> original global indices.
    // Empty maps denote a complete TP1 array.
    std::vector<uint64_t> value_heads;
    std::vector<uint64_t> channels;
};

struct StateAxisShard {
    const void* data = nullptr;
    std::size_t bytes = 0;
    std::vector<uint64_t> indices;
};

// Reconstructs any C-order axis. Rejects missing, repeated and out-of-range
// ownership; never fabricates zeros for another rank's state.
bool reconstruct_state_axis(const std::vector<uint64_t>& logical_shape, unsigned axis,
                            std::size_t element_bytes, const std::vector<StateAxisShard>&,
                            std::vector<uint8_t>& output, std::string& err);

// Definition enabled only with STRATA_STATE_EVIDENCE_WITH_CORE. Supports plain
// resident FP16 QSA; rejects compressed/streamed/ring state rather than claiming
// incomplete coverage. KV is reconstructed to loaded tokens, excluding unused
// capacity and page padding. No scratch tensors or RoPE tables are evidence.
bool describe_session_state(const core::SessionState&, const core::ModelGeometry&,
                            uint64_t token_count, const StateReadback&,
                            const std::vector<GdnStateSource>& gdn_overrides,
                            std::vector<StateEvidenceArray>& output, std::string& err);

} // namespace strata::distributed
