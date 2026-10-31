#pragma once

#include "strata/distributed/state_evidence.hpp"

#include <chrono>

namespace strata::distributed {

using CheckpointDeadline = std::chrono::steady_clock::time_point;
using StateArrayWrite = std::function<bool(const void*, std::size_t, std::string&)>;

struct ModelCheckpointIdentity {
    std::string model_id;
    std::string request_id;
    uint64_t generation = 0;
    uint32_t slot = 0;
    uint32_t rank = 0;
    uint64_t max_context = 0;
    int32_t position_base = 0;
};

struct ModelCheckpointRequest {
    ModelCheckpointIdentity identity;
    uint64_t next_position = 0;
    uint64_t max_bytes = 0;
    // Use the model's complete inventory at next_position, including zero-row
    // KV after reset. Deriving this from a subset of readers cannot prove coverage.
    std::vector<StateEvidenceDescriptor> required;
};

struct ModelCheckpointDestination {
    ModelCheckpointIdentity identity;
    // Describe the restored position, not the destination's current KV length.
    std::vector<StateEvidenceDescriptor> required;
    struct Array {
        StateEvidenceDescriptor descriptor;
        StateArrayWrite write;
    };
    std::vector<Array> arrays;
    // Publish next_position only after every write has completed. Host PLE
    // token-window state must also be committed by its explicit array writer.
    std::function<bool(uint64_t, std::string&)> commit_position;
    std::function<void()> invalidate;
};

// In-memory request state, not evidence files or writable device pointers.
// Callbacks must finish before returning and honor the operation deadline.
// The caller serializes the model stream for the entire capture/restore.
class ModelCheckpoint final {
public:
    bool capture(const ModelCheckpointRequest&, const std::vector<StateEvidenceArray>&,
                 CheckpointDeadline, std::string& err);
    bool restore(const ModelCheckpointDestination&, CheckpointDeadline, std::string& err) const;
    void clear() noexcept;
    bool complete() const noexcept { return complete_; }
    uint64_t bytes() const noexcept { return bytes_; }
    uint64_t next_position() const noexcept { return next_position_; }
private:
    struct Array {
        StateEvidenceDescriptor descriptor;
        std::vector<uint8_t> bytes;
    };
    ModelCheckpointIdentity identity_;
    uint64_t next_position_ = 0;
    uint64_t bytes_ = 0;
    std::vector<Array> arrays_;
    bool complete_ = false;
};

#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
// Allocate pinned staging and a disable-timing event before entering a timed
// operation. These resources are borrowed. On timeout/error, quarantine staging,
// event AND device allocations until the stream finishes or the worker exits;
// pending DMA must never reference freed storage. This bounds event polling,
// not time spent inside CUDA driver calls. No synchronizing cleanup here.
struct CudaCheckpointTransfer {
    void* pinned_staging = nullptr;
    std::size_t staging_bytes = 0;
    void* completion_event = nullptr;
    bool pending = false;
    bool poisoned = false;
};
bool cuda_checkpoint_copy(CudaCheckpointTransfer&, void* host, void* device,
                          std::size_t bytes, bool to_device, void* stream,
                          CheckpointDeadline, std::string& err);
#endif

} // namespace strata::distributed
