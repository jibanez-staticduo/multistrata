#pragma once

#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/distributed/parity_trace.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::distributed {

// Independent C1 oracle: original monolithic block pre/post and grouped CUDA
// experts. Host-mapped GGUF weights are staged by selected expert, so this is
// a numerical reference and does not provide comparable throughput timings.
class MonolithicReference final {
public:
    MonolithicReference(const core::WeightTable&, const core::ModelGeometry&,
                        core::SessionState&, std::string pack, std::string native,
                        uint32_t timeout_ms = 60000);
    ~MonolithicReference();
    MonolithicReference(const MonolithicReference&) = delete;
    MonolithicReference& operator=(const MonolithicReference&) = delete;
    bool initialize(std::string& err);
    void native_head(const core::NativeHead* head);
    void parity_observer(ParityObserver observer);
    bool reset(void* stream, std::string& err);
    // logits is device memory. Success includes bounded stream completion.
    bool token(int32_t token, int64_t position, int32_t position_base,
               float* logits, void* stream, std::string& err);
    int64_t next_position() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::distributed
