#pragma once

#include "strata/core/session.hpp"
#include "strata/core/native_head.hpp"
#include "strata/distributed/session.hpp"
#include "strata/distributed/worker.hpp"
#include "strata/distributed/state_evidence.hpp"
#include "strata/distributed/parity_trace.hpp"

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

class SessionCollectives final : public LayerCollectives {
public:
    SessionCollectives(Session& session, GuardedTransport& transport, const Clock& clock,
                       uint32_t timeout_ms);
    void step(uint64_t step) { step_ = step; }
    uint64_t sequence() const { return sequence_; }
    void reset_observations() { observed_.clear(); }
    const std::vector<ScheduledCollective>& observed() const { return observed_; }
    void expect_schedule(const std::vector<ScheduledCollective>& expected) { expected_ = expected; }
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
    void native_head(const core::NativeHead* head);
    // Optional uncaptured diagnostics; an empty observer performs no copies.
    void parity_observer(ParityObserver observer);
    bool reset(void* stream, std::string& err);
    bool token(int32_t token, int64_t position, int32_t position_base,
               float* logits, void* stream, std::string& err);
    const core::ModelGeometry& geometry() const;
    int64_t next_position() const;
    std::vector<GdnStateSource> state_sources() const;
    void invalidate() noexcept;
private:
    bool mixer(int64_t, int64_t, int32_t, const float*, float*, void*, std::string&) override;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::distributed
