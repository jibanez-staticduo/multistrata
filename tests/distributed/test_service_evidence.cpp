#include "strata/distributed/service_evidence.hpp"
#include "strata/distributed/speculative.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace {
using namespace strata::distributed;
int failures = 0;
void check(bool ok, const char* message) { if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; } }
std::string read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void speculative_transcript() {
    ServiceTranscript transcript;
    std::string error;
    uint64_t next = 0, checkpoint = 0;
    next = 1;
    check(transcript.record_success(StepPhase::prefill, "mtp-1", 2, 0, next, error), "prompt starts transcript");
    // A real verifier consumes every proposed row, rolls back, then replays the
    // accepted prefix. Its final correction is output but has not been consumed.
    SpeculativeVerifier verifier(0, 100, 32);
    SpeculativeCallbacks callbacks;
    callbacks.broadcast = [](std::string_view, std::string&) { return Failure{}; };
    callbacks.execute = [&](StepPhase phase, int32_t token, uint64_t position) {
        if (phase == StepPhase::checkpoint) checkpoint = next;
        if (phase == StepPhase::rollback) next = checkpoint;
        if (phase == StepPhase::decode) ++next;
        if (!transcript.record_success(phase, "mtp-1", token, position, next, error))
            return Failure::fail(FailureCode::bad_operation, error);
        return Failure{};
    };
    callbacks.sample = [](const std::vector<int32_t>&, uint64_t draw, int32_t& token) {
        token = draw == 0 ? 20 : 91; return Failure{};
    };
    SpeculativeResult result;
    check(!verifier.verify({"mtp-1", 1, 1, 0, {10,20,30,40}, {2}, 8}, callbacks, result), "real speculative rejection and replay succeeds");
    check(result.consumed_inputs == std::vector<int32_t>({10,20}), "verifier retained the canonical prefix");
    check(transcript.ids() == std::vector<int32_t>({2,10,20}) && next == 3,
          "discarded speculative suffix and unconsumed correction absent from transcript");
    check(transcript.record_success(StepPhase::checkpoint, "mtp-1", -1, 3, 3, error), "later checkpoint replaces old checkpoint");
    check(transcript.record_success(StepPhase::decode, "mtp-1", 91, 3, 4, error), "feedback is recorded only when consumed");
    check(transcript.record_success(StepPhase::rollback, "mtp-1", -1, 0, 3, error) && transcript.ids().size() == 3,
          "later rollback retains its own canonical prefix");
    check(!transcript.record_success(StepPhase::decode, "other", 7, 3, 4, error), "live request identity cannot change");
    check(!transcript.record_success(StepPhase::decode, "mtp-1", 7, 4, 5, error), "skipped consumed position rejected");
    check(transcript.record_success(StepPhase::reset, "mtp-1", -1, 0, 0, error), "completed reset clears request");
    check(transcript.record_success(StepPhase::prefill, "new-request", 9, 0, 1, error), "reused slot begins a fresh identity");
    check(!transcript.record_success(StepPhase::rollback, "new-request", -1, 0, 3, error), "old request checkpoint cannot leak into new request");
    check(transcript.ids() == std::vector<int32_t>({9}), "failed stale rollback preserves new request transcript");
    check(transcript.record_success(StepPhase::cancel, "new-request", -1, 0, 0, error), "cancellation clears transcript");
    check(!transcript.record_success(StepPhase::prefill, "../escape", 1, 0, 1, error), "unsafe identity cannot become a path");
}
void publication() {
    const auto root = std::filesystem::temp_directory_path() / ("strata-service-evidence-" + std::to_string(getpid()));
    std::filesystem::remove_all(root);
    ServiceTranscript transcript;
    std::string error;
    check(transcript.record_success(StepPhase::prefill, "request-1", 19, 0, 1, error), "publication transcript ready");
    float state = 7, logits[] = {1,2,3};
    const StateReadback readback = [](void* dst, const void* src, size_t n, std::string&) { std::memcpy(dst, src, n); return true; };
    std::vector<StateEvidenceArray> arrays = {borrowed_state_array({"gdn","recurrent","<f4",{1},0,{},-1,{}}, &state, readback)};
    StateEvidenceRequest request;
    request.request_id = "request-1"; request.token_count = 1; request.max_bytes = 1024;
    request.required = state_evidence_inventory(arrays);
    check(write_service_evidence(root.string(), 0, false, transcript, request, arrays, logits, 3, error), "complete service evidence published");
    const auto complete = root / "rank0/request-1/complete";
    check(read(complete / "input-ids.json") == "[19]\n", "exact canonical IDs written next to manifest");
    check(read(complete / "logits.f32") == std::string(reinterpret_cast<const char*>(logits), sizeof(logits)), "committed host logits written exactly");
    check(std::filesystem::exists(complete / "manifest.json") && !std::filesystem::exists(root / "rank0/request-1/complete.partial"), "publication has no incomplete final phase");
    check(!write_service_evidence(root.string(), 0, true, transcript, request, arrays, logits, 3, error), "request artifact cannot be overwritten with a different phase");
    check(write_service_evidence(root.string(), 1, true, transcript, request, arrays, nullptr, 0, error), "follower cancellation evidence published");
    check(!std::filesystem::exists(root / "rank1/request-1/cancelled/logits.f32"), "follower has no logits");
    check(!write_service_evidence(root.string(), 2, false, transcript, request, arrays, logits, 3, error), "follower cannot publish root logits");
    auto broken = arrays;
    broken[0].read = [](void*, size_t, std::string& e) { e = "injected read failure"; return false; };
    check(!write_service_evidence(root.string(), 2, false, transcript, request, broken, nullptr, 0, error), "injected readback fails capture");
    check(!std::filesystem::exists(root / "rank2/request-1/complete"), "failed capture never publishes final evidence");
    ++request.token_count;
    check(!write_service_evidence(root.string(), 2, false, transcript, request, arrays, nullptr, 0, error), "state position must match canonical transcript");
    std::filesystem::remove_all(root);
}
}
int main() {
    speculative_transcript(); publication();
    if (failures) return 1;
    std::cout << "service evidence transcript and publication checks passed\n";
}
