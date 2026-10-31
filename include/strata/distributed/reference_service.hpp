#pragma once

#include "strata/distributed/worker.hpp"

#include <functional>
#include <set>
#include <string>
#include <string_view>

namespace strata::distributed {

// A real TP1 execution boundary: forward completes the model and checks every logit before returning.
struct ReferenceExecution {
    std::function<Failure()> reset;
    std::function<Failure(int32_t token, uint64_t position, bool& outputs_finite)> forward;
};

class ReferenceService final {
public:
    ReferenceService(int32_t vocabulary, uint32_t context, ReferenceExecution execution);
    Failure execute(StepPhase phase, std::string_view request, int32_t token, uint64_t position);
    bool output_ready() const { return output_ready_; }
    uint64_t next_position() const { return position_; }
private:
    int32_t vocabulary_;
    uint32_t context_;
    ReferenceExecution execution_;
    bool initialized_ = false, failed_ = false, pending_ = false, decoding_ = false;
    bool pending_decode_ = false, output_ready_ = false;
    uint64_t position_ = 0;
    std::string request_;
    std::set<std::string> retired_;
};

std::string reference_event(std::string_view type, std::string_view request_id = {},
                            int32_t token = -1, uint32_t context = 0,
                            std::string_view state = "ready");

#ifdef STRATA_REFERENCE_CUDA
class MonolithicReference;
// Call before loading weights so model diagnostics cannot enter the JSONL channel.
void prepare_reference_service(uint32_t timeout_ms = 60000);
// Owns fixed sampler/readback buffers and the bounded JSONL channel until shutdown.
Failure run_reference_service(MonolithicReference& model, int32_t vocabulary, uint32_t context,
                              float* device_logits, void* stream, uint32_t timeout_ms = 60000);
#endif

} // namespace strata::distributed
