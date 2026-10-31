#pragma once

#include "strata/distributed/serve_protocol.hpp"
#include "strata/distributed/speculative.hpp"

#include <memory>

namespace strata::core { class WeightTable; class NativeHead; struct ModelGeometry; struct SessionState; }

namespace strata::distributed {

// Root-only, synchronous MTP adapter. Library load/capture/draft calls require an external process
// watchdog; they do not inherit the bounded Worker event waits. No output token is trusted here.
class MtpService final {
public:
    MtpService();
    ~MtpService();
    MtpService(const MtpService&) = delete;
    MtpService& operator=(const MtpService&) = delete;
    bool initialize(const std::string& runtime, uint32_t drafts, const core::ModelGeometry&,
                    core::SessionState&, const core::WeightTable&, const core::NativeHead&,
                    void* model_stream, std::string& err);
    Failure begin_request(const ServeRequest&);
    Failure prompt_row(int32_t next_token, uint64_t position);
    Failure propose(const std::vector<int32_t>& history, int32_t feedback, uint64_t draw,
                    std::vector<int32_t>& drafts);
    Failure record_target_row(size_t row, int32_t input, int32_t next);
    Failure accepted_window(const SpeculativeWindow&, const SpeculativeResult&);
    uint64_t vram_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::distributed
