#include "strata/distributed/service_evidence.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>

namespace strata::distributed {
namespace {
bool fail(std::string& error, const char* text) { error = text; return false; }
bool safe_id(std::string_view id) {
    return !id.empty() && id.size() <= 128 && std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}
}

bool ServiceTranscript::record_success(StepPhase phase, std::string_view request, int32_t token,
                                       uint64_t position, uint64_t next, std::string& error) {
    if (!safe_id(request)) return fail(error, "invalid service transcript request identity");
    const bool compute = phase == StepPhase::prefill || phase == StepPhase::decode;
    if (request_ != request && (!request_.empty() || !compute || position || !ids_.empty()))
        return fail(error, "service transcript request identity changed before reset");
    if (compute) {
        if (token < 0 || position != ids_.size() || next != ids_.size() + 1)
            return fail(error, "service transcript compute position mismatch");
        if (request_.empty()) request_ = request;
        ids_.push_back(token);
        return true;
    }
    if (phase == StepPhase::rollback) {
        if (!checkpoint_ || next != *checkpoint_ || *checkpoint_ > ids_.size())
            return fail(error, "service transcript rollback has no matching checkpoint");
        ids_.resize(*checkpoint_);
        return true;
    }
    if (phase == StepPhase::reset || phase == StepPhase::cancel) {
        if (next) return fail(error, "service transcript reset did not clear model position");
        request_.clear(); ids_.clear(); checkpoint_.reset();
        return true;
    }
    if ((phase != StepPhase::commit && phase != StepPhase::checkpoint) || next != ids_.size())
        return fail(error, "service transcript command position mismatch");
    if (phase == StepPhase::checkpoint) checkpoint_ = ids_.size();
    return true;
}

bool write_service_evidence(const std::string& directory, uint32_t rank, bool cancelled,
                            const ServiceTranscript& transcript, const StateEvidenceRequest& request,
                            const std::vector<StateEvidenceArray>& arrays, const float* logits,
                            size_t logits_count, std::string& error) {
    try {
        if (!std::filesystem::path(directory).is_absolute() || rank >= kTpWorld ||
            !safe_id(request.request_id) || request.request_id != transcript.request_id() ||
            request.token_count != transcript.ids().size() || request.reset != (request.token_count == 0) ||
            (logits_count && (!logits || rank != 0)) ||
            logits_count > size_t(std::numeric_limits<std::streamsize>::max()) / sizeof(float))
            return fail(error, "invalid service evidence identity, count or logits");
        const auto parent = std::filesystem::path(directory) / ("rank" + std::to_string(rank));
        std::filesystem::create_directories(parent);
        const auto fresh = parent / request.request_id;
        if (!std::filesystem::create_directory(fresh)) return fail(error, "service request evidence must be fresh");
        std::filesystem::permissions(fresh, std::filesystem::perms::owner_all);
        const auto phase = cancelled ? "cancelled" : "complete";
        const auto staging = fresh / (std::string(phase) + ".partial");
        if (!write_state_evidence(staging.string(), request, arrays, error)) return false;
        std::ofstream ids(staging / "input-ids.json", std::ios::binary);
        ids.exceptions(std::ios::badbit | std::ios::failbit);
        ids << '[';
        for (size_t i = 0; i < transcript.ids().size(); ++i) {
            if (i) ids << ',';
            ids << transcript.ids()[i];
        }
        ids << "]\n"; ids.close();
        if (logits_count) {
            std::ofstream file(staging / "logits.f32", std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(reinterpret_cast<const char*>(logits), std::streamsize(logits_count * sizeof(float)));
            file.close();
        }
        std::filesystem::rename(staging, fresh / phase);
        return true;
    } catch (const std::exception& e) { error = std::string("service evidence: ") + e.what(); return false; }
}
} // namespace strata::distributed
