#pragma once

#include <cstdint>
#include <string>

namespace strata::kernels { struct FusedGrArgs; }
namespace strata::core {
struct SessionState;
struct TensorParallelSegment {
    int32_t slot = 0;
    int64_t first = 0, count = 0, position = 0;
};

// One row is either a prompt position or an independent decode slot. Device
// tables remain alive until the owning stream completes; callbacks never wait
// for the host while a captured graph is being recorded.
struct TensorParallelRows {
    int64_t count = 0;
    int64_t position = 0;
    const int32_t* positions = nullptr;
    const int32_t* slots = nullptr;
    bool independent = false;
    int32_t slot = 0; // contiguous prompt/window when slots is null
    int32_t position_stride = 1;
    // Capture-only host metadata. The caller's graph key fixes these slot IDs;
    // providers may use them to select stable state pointers while recording.
    // Never retained or dereferenced by GPU kernels.
    const int* capture_slots = nullptr;
    bool speculative = false;
    // Host metadata for contiguous prompt segments, valid for this enqueue only.
    const TensorParallelSegment* segments = nullptr;
    int segment_count = 0;
};

// The production TP seam exchanges final projection outputs. The provider owns
// rank-local weights and scratch; the executor owns request state and residuals.
// It is configured before initialization and outlives all captured executables.
class TensorParallelHooks {
public:
    virtual ~TensorParallelHooks() = default;
    virtual bool bind_sessions(SessionState* const*, int, std::string&) { return true; }
    virtual bool reset_slots(const int32_t*, int, void*, std::string&) { return true; }
    virtual bool begin_slots(const int32_t*, const int32_t*, int, bool, void*, std::string&) { return true; }
    virtual bool commit_slots(const int32_t*, const int32_t*, int, void*, std::string&) { return true; }
    virtual bool rollback_slots(const int32_t*, const int64_t*, int, void*, std::string&) { return true; }
    // Sticky device contract failures are checked after bounded completion and
    // before any rank publishes output. No host readback belongs inside capture.
    virtual bool validate_execution_status(void*, std::string&) { return true; }
    virtual bool has_output_head() const { return false; }
    virtual bool output_head(const void*, float*, int, void*, std::string&) { return false; }
    virtual bool compact_greedy_head() const { return false; }
    virtual void configure_compact_head(bool) {}
    virtual bool output_head_greedy(const void*, int32_t*, uint32_t*, int, int, void*, std::string&) { return false; }
    virtual bool has_dense_output_head(int) const { return false; }
    virtual bool dense_output_head(const float*, float*, int32_t*, uint32_t*, int, int, void*, std::string&) { return false; }
    virtual bool has_fast_hc() const { return false; }
    virtual bool use_fast_hc(int) const { return has_fast_hc(); }
    virtual bool hyper_connection(const kernels::FusedGrArgs*, int, void*, std::string&) { return false; }
    virtual bool has_sharded_shared() const { return false; }
    // Add the rank's gated shared-expert partial before the routed reduction.
    virtual bool shared_expert_local_add(int64_t, const TensorParallelRows&, const float*,
                                         float*, void*, std::string&) { return false; }

    // False is a failure, never a request to fall back to replicated full work.
    // mixed/output are [rows, hidden]. The mixer updates exactly the request
    // states identified by rows and returns its globally reduced output.
    virtual bool mixer(int64_t layer, const TensorParallelRows& rows,
                       const float* mixed, float* output, void* stream,
                       std::string& error) = 0;

    // IDs/weights are [rows, top_k] in token-major order. Return the weighted,
    // globally reduced routed sum [rows, hidden]. With has_sharded_shared(),
    // include the shared sum too; otherwise the executor adds its shared result.
    virtual bool experts(int64_t layer, const TensorParallelRows& rows,
                         const float* mixed, const int32_t* ids,
                         const float* weights, int top_k, float* routed_sum,
                         void* stream, std::string& error) = 0;
};

} // namespace strata::core
