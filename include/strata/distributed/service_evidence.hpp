#pragma once

#include "strata/distributed/state_evidence.hpp"
#include "strata/distributed/worker.hpp"

#include <optional>
#include <string_view>

namespace strata::distributed {

// Tracks inputs consumed by the target, including speculative replay. Sampled
// output that has not been fed back into the target is deliberately absent.
class ServiceTranscript final {
public:
    bool record_success(StepPhase, std::string_view request, int32_t token,
                        uint64_t position, uint64_t next_position, std::string& error);
    const std::string& request_id() const { return request_; }
    const std::vector<int32_t>& ids() const { return ids_; }
private:
    std::string request_;
    std::vector<int32_t> ids_;
    std::optional<size_t> checkpoint_;
};

// Publishes a fresh rank/request/{complete,cancelled} directory atomically.
// The caller supplies only committed host logits, on rank zero, when available.
bool write_service_evidence(const std::string& directory, uint32_t rank, bool cancelled,
                            const ServiceTranscript&, const StateEvidenceRequest&,
                            const std::vector<StateEvidenceArray>&, const float* logits,
                            size_t logits_count, std::string& error);

} // namespace strata::distributed
