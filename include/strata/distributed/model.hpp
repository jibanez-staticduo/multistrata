#pragma once

#include "strata/core/session.hpp"
#include "strata/core/native_head.hpp"
#include "strata/distributed/session.hpp"
#include "strata/distributed/worker.hpp"
#include "strata/distributed/state_evidence.hpp"
#include "strata/distributed/parity_trace.hpp"
#include "strata/distributed/model_checkpoint.hpp"

#include <array>
#include <memory>

namespace strata::distributed {

// The model only communicates at named, ordered seams. Implementations must
// agree the descriptor, complete the operation and propagate async failures.
class LayerCollectives {
public:
    virtual ~LayerCollectives() = default;
    virtual bool agree_routes(int64_t layer, const int* device_ids, int64_t k,
                              void* stream, std::string& err) = 0;
    virtual bool reduce(int64_t layer, const char* stage, const float* input,
                        float* output, size_t elements, void* stream, std::string& err) = 0;
    virtual bool gather(int64_t layer, const char* stage, const float* input,
                        float* output, std::array<uint64_t, 3> valid,
                        void* stream, std::string& err) = 0;
};

// Host elapsed time, in nanoseconds, accumulated until reset_profile(). Attempts
// include failed phases; completed_collectives counts only successful completion.
struct LayerCollectiveProfile {
    uint64_t collective_calls = 0;
    uint64_t completed_collectives = 0;
    uint64_t route_calls = 0;
    uint64_t route_pending_polls = 0;
    // Per-rank device buffer extents, including gather padding, not network traffic.
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t stage_metadata_ns = 0;
    uint64_t operation_agreement_ns = 0;
    uint64_t enqueue_ns = 0;
    uint64_t wait_ns = 0;
    uint64_t route_readback_ns = 0;
    uint64_t route_agreement_ns = 0;
};

class SessionCollectives final : public LayerCollectives {
public:
    SessionCollectives(Session& session, GuardedTransport& transport, const Clock& clock,
                       uint32_t timeout_ms);
    void step(uint64_t step) { step_ = step; }
    uint64_t sequence() const { return sequence_; }
    void reset_observations() { observed_.clear(); }
    const std::vector<ScheduledCollective>& observed() const { return observed_; }
    void expect_schedule(const std::vector<ScheduledCollective>& expected) { expected_ = expected; }
    void preagreed(bool enabled) { preagreed_ = enabled; }
    // Opt-in diagnostics. Disabled by default; toggling does not clear accumulated counters.
    void profile(bool enabled) { profiling_ = enabled; }
    const std::array<LayerCollectiveProfile, 48>& profile() const { return profile_; }
    void reset_profile() { profile_.fill({}); }
    bool agree_routes(int64_t, const int*, int64_t, void*, std::string&) override;
    bool reduce(int64_t, const char*, const float*, float*, size_t, void*, std::string&) override;
    bool gather(int64_t, const char*, const float*, float*, std::array<uint64_t, 3>,
                void*, std::string&) override;
private:
    bool execute(CollectiveDescriptor, const char*, const float*, float*, void*, std::string&);
    Session& session_;
    GuardedTransport& transport_;
    const Clock& clock_;
    uint32_t timeout_ms_;
    uint64_t sequence_ = 1;
    uint64_t step_ = 0;
    std::vector<ScheduledCollective> observed_;
    std::vector<ScheduledCollective> expected_;
    bool profiling_ = false;
    bool preagreed_ = false;
    std::array<LayerCollectiveProfile, 48> profile_{};
};

// A provider fills this layer's k route outputs, already globally reduced,
// before core::moe_combine_parts adds the replicated shared expert once.
class RoutedExperts {
public:
    virtual ~RoutedExperts() = default;
    virtual bool forward(int64_t layer, const float* input, const int* ids,
                         int64_t k, float* parts, void* stream, std::string& err) = 0;
};

// Borrowed immutable weights and per-request core state; local GDN recurrence
// is owned separately. initialize() allocates device scratch; token() performs
// no device allocation. Control agreement may allocate host metadata.
class ModelForward final : private core::DistributedLayerHooks {
public:
    ModelForward(const core::WeightTable&, const core::ModelGeometry&,
                 core::SessionState&, int rank, LayerCollectives&, RoutedExperts&);
    ~ModelForward();
    ModelForward(const ModelForward&) = delete;
    ModelForward& operator=(const ModelForward&) = delete;
    bool initialize(std::string& err);
    bool configure_prefill_batch_exactnative(bool enabled, std::string& err);
    void native_head(const core::NativeHead* head);
    // Optional uncaptured diagnostics; an empty observer performs no copies.
    void parity_observer(ParityObserver observer);
    bool reset(void* stream, std::string& err);
    bool token(int32_t token, int64_t position, int32_t position_base,
               float* logits, void* stream, std::string& err);
    // Layer-major exact-native prompt chunk. Writes logits for the final row.
    bool tokens(const std::vector<int32_t>& tokens, int64_t position, int32_t position_base,
                float* logits, void* stream, std::string& err);
    const core::ModelGeometry& geometry() const;
    int64_t next_position() const;
    std::vector<GdnStateSource> state_sources() const;
    // Explicit request switching/rollback, outside graph capture. The caller
    // serializes stream access and supplies the frozen model/request identity.
    bool capture_checkpoint(ModelCheckpoint&, ModelCheckpointIdentity, uint64_t max_bytes,
                            void* stream, CheckpointDeadline, std::string& err);
    bool restore_checkpoint(const ModelCheckpoint&, ModelCheckpointIdentity,
                            void* stream, CheckpointDeadline, std::string& err);
    // If true, the worker must exit without unwinding device/session owners:
    // a timed-out DMA can still reference borrowed SessionState allocations.
    bool requires_process_exit() const noexcept;
    void invalidate() noexcept;
private:
    bool mixer(int64_t, int64_t, int32_t, const float*, float*, void*, std::string&) override;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::distributed
