#include "strata/distributed/reference_service.hpp"
#include "strata/distributed/autoregressive.hpp"
#include "strata/distributed/limits.hpp"

#include <cstdio>
#include <deque>
#include <stdexcept>
#include <vector>

namespace {
using namespace strata::distributed;
int checks = 0, failures = 0;
void check(bool value, const char* label) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", label); }
}
struct Model {
    std::vector<int32_t> state;
    std::vector<std::vector<int32_t>> completed;
    bool finite = true, deadline = false;
    unsigned resets = 0;
    ReferenceExecution hooks() {
        return {
            [&]() -> Failure { completed.push_back(state); state.clear(); ++resets; return {}; },
            [&](int32_t token, uint64_t position, bool& valid) -> Failure {
                if (deadline) return Failure::fail(FailureCode::deadline_expired, "injected compute deadline");
                if (position != state.size()) return Failure::fail(FailureCode::bad_operation, "model position mismatch");
                state.push_back(token); valid = finite; return {};
            }
        };
    }
};
void state_machine() {
    Model model;
    ReferenceService worker(16, 4, model.hooks());
    auto execute = [&](StepPhase phase, const char* id = "a", int32_t token = -1, uint64_t position = 0) {
        return worker.execute(phase, id, token, position);
    };
    check(bool(execute(StepPhase::prefill, "a", 1)), "compute before initialization rejected");
    check(!worker.execute(StepPhase::init, {}, -1, 0), "initial reset completes");
    check(bool(worker.execute(StepPhase::init, {}, -1, 0)), "duplicate initialization rejected");
    check(bool(execute(StepPhase::decode, "a", 1)), "request must start with prefill");
    check(bool(execute(StepPhase::prefill, "a", 16)), "invalid token rejected before model");
    check(!execute(StepPhase::prefill, "a", 1), "first prompt row completes");
    check(!worker.output_ready() && worker.next_position() == 0, "computed finite row stays unpublished before commit");
    check(bool(execute(StepPhase::prefill, "a", 2)), "second compute cannot overwrite pending row");
    check(bool(execute(StepPhase::commit, "wrong")), "wrong request cannot commit");
    check(bool(execute(StepPhase::commit, "a", -1, 1)), "invalid commit position rejected");
    check(bool(execute(StepPhase::reset)), "uncommitted reset rejected");
    check(!execute(StepPhase::commit), "finite row commits");
    check(worker.output_ready() && worker.next_position() == 1, "commit publishes exactly one row");
    check(bool(execute(StepPhase::commit)), "duplicate commit rejected");
    check(bool(execute(StepPhase::prefill, "a", 2, 0)), "stale position rejected");
    check(bool(execute(StepPhase::prefill, "b", 2, 1)), "concurrent request rejected");
    check(!execute(StepPhase::decode, "a", 2, 1), "decode follows committed prompt");
    check(!worker.output_ready(), "new compute clears published row");
    check(!execute(StepPhase::commit), "decode commits");
    check(bool(execute(StepPhase::prefill, "a", 3, 2)), "prefill after decode rejected");
    check(!execute(StepPhase::decode, "a", 3, 2) && !execute(StepPhase::commit), "third row commits");
    check(!execute(StepPhase::decode, "a", 4, 3) && !execute(StepPhase::commit), "last capacity row commits");
    check(bool(execute(StepPhase::decode, "a", 5, 4)), "context budget enforced before model");
    check(!execute(StepPhase::reset), "terminal request reset succeeds");
    check(model.state.empty() && !worker.output_ready() && worker.next_position() == 0, "reset clears state and publication");
    check(bool(execute(StepPhase::prefill, "a", 1)), "retired request identity rejected");
    check(!execute(StepPhase::prefill, "b", 6), "fresh request starts from zero");
    check(!execute(StepPhase::cancel, "b"), "completed pending row can be safely cancelled");
    check(model.state.empty() && !worker.output_ready(), "cancellation resets actual model and clears output");
    check(bool(execute(StepPhase::prefill, "b", 1)), "cancelled identity cannot be reused");
}
void context_capacity() {
    Model model;
    ReferenceService supported(16, kDistributedMaxContext, model.hooks());
    check(!supported.execute(StepPhase::init, {}, -1, 0), "reference accepts expanded context capacity");
    ReferenceService oversized(16, kDistributedMaxContext + 1, model.hooks());
    check(bool(oversized.execute(StepPhase::init, {}, -1, 0)) && model.resets == 1,
          "oversized reference capacity fails before model reset");
    check(reference_event("ready", {}, -1, kDistributedMaxContext) ==
          "{\"type\":\"ready\",\"alive_ranks\":[0],\"protocol\":\"strata-reference-v1\",\"mode\":\"autoregressive\",\"world\":1,\"slots\":1,\"max_context\":65792}",
          "reference publishes expanded capacity with its independent identity");
    bool threw = false;
    try { reference_event("ready", {}, -1, kDistributedMaxContext + 1); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "oversized reference ready capacity rejected");
}
void failed_completion() {
    for (bool deadline : {false, true}) {
        Model model;
        model.finite = false; model.deadline = deadline;
        ReferenceService worker(16, 8, model.hooks());
        check(!worker.execute(StepPhase::init, {}, -1, 0), "failure fixture initializes");
        const auto result = worker.execute(StepPhase::prefill, "bad", 1, 0);
        check(bool(result) && !worker.output_ready(), "nonfinite or expired compute never publishes output");
        check(bool(worker.execute(StepPhase::commit, "bad", -1, 0)), "failed completion cannot commit");
        check(bool(worker.execute(StepPhase::cancel, "bad", -1, 0)), "failed worker cannot be reused through cancellation");
    }
}
std::string generate(const char* id, const char* ids, unsigned count) {
    return std::string("{\"type\":\"generate\",\"request_id\":\"") + id + "\",\"ids\":[" + ids +
        "],\"max_new\":" + std::to_string(count) + ",\"sampling\":{\"temperature\":0,\"seed\":7}}";
}
void coordinator() {
    Model model;
    ReferenceService worker(16, 16, model.hooks());
    check(!worker.execute(StepPhase::init, {}, -1, 0), "coordinator worker initializes");
    std::deque<std::string> input{generate("first", "2,3", 3)};
    std::vector<int32_t> sampled;
    unsigned poll_count = 0;
    std::vector<std::vector<int32_t>> histories;
    std::vector<uint64_t> draws;
    std::vector<std::string> events;
    AutoregressiveCallbacks callbacks;
    callbacks.poll = [&](uint32_t, std::string& line, bool& eof) -> Failure {
        line.clear(); eof = false;
        ++poll_count;
        if (!input.empty()) { line = input.front(); input.pop_front(); }
        else if (poll_count == 5) line = generate("second", "9", 1);
        else if (poll_count > 5) eof = true;
        return {};
    };
    callbacks.broadcast = [](std::string_view, std::string&) -> Failure { return {}; };
    callbacks.health = []() -> Failure { return {}; };
    callbacks.execute = [&](StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
        return worker.execute(phase, request, token, position);
    };
    callbacks.sample = [&](const ServeRequest&, const auto& history, uint64_t draw, int32_t& token) -> Failure {
        check(worker.output_ready(), "coordinator samples only committed logits");
        histories.push_back(history); draws.push_back(draw);
        token = history.back() + 1; sampled.push_back(token); return {};
    };
    callbacks.emit = [&](std::string_view type, std::string_view request, int32_t token, std::string_view state) {
        events.push_back(reference_event(type, request, token, 16, state));
    };
    check(!run_autoregressive(0, 16, 16, callbacks), "world-one coordinator runs to clean EOF");
    check(sampled == std::vector<int32_t>({4,5,6,10}), "first sample follows last prompt logit and feedback follows sampled token");
    check(histories == std::vector<std::vector<int32_t>>({{2,3},{2,3,4},{2,3,4,5},{9}}), "penalty history contains prompt plus consumed feedback only");
    check(draws == std::vector<uint64_t>({0,1,2,0}), "logical draw resets once for fresh request");
    check(model.completed == std::vector<std::vector<int32_t>>({{}, {2,3,4,5}, {9}}), "fresh request resets all actual model state");
    check(!events.empty() && events.front().find("\"world\":1") != std::string::npos &&
          events.front().find("strata-reference-v1") != std::string::npos, "READY reports actual TP1 protocol");
    check(std::all_of(events.begin(), events.end(), [](const auto& event) {
        return event.find("\"alive_ranks\":[0]") != std::string::npos;
    }), "every event reports exactly rank zero");
}
void cancellation() {
    Model model;
    ReferenceService worker(16, 16, model.hooks());
    check(!worker.execute(StepPhase::init, {}, -1, 0), "cancel worker initializes");
    std::deque<std::string> input{generate("cancel", "2,3", 4),
        "{\"type\":\"cancel\",\"request_id\":\"cancel\"}", generate("next", "7", 1)};
    std::vector<std::string> terminal;
    AutoregressiveCallbacks callbacks;
    callbacks.poll = [&](uint32_t, std::string& line, bool& eof) -> Failure {
        eof = input.empty(); line.clear();
        if (!eof) { line = input.front(); input.pop_front(); }
        return {};
    };
    callbacks.broadcast = [](std::string_view, std::string&) -> Failure { return {}; };
    callbacks.health = []() -> Failure { return {}; };
    callbacks.execute = [&](StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
        return worker.execute(phase, request, token, position);
    };
    callbacks.sample = [&](const ServeRequest&, const auto&, uint64_t, int32_t& token) -> Failure {
        token = 8; return {};
    };
    callbacks.emit = [&](std::string_view type, std::string_view, int32_t, std::string_view state) {
        if (type == "cancelled" || type == "done") { check(state == "ready", "terminal event follows reset"); terminal.emplace_back(type); }
    };
    check(!run_autoregressive(0, 16, 16, callbacks), "cancel is drained between completed committed rows");
    check(terminal == std::vector<std::string>({"cancelled", "done"}), "cancelled request cannot emit tokens or done");
    check(model.completed == std::vector<std::vector<int32_t>>({{}, {2}, {7}}), "cancellation resets partial prompt before next request");
}
}
int main() {
    state_machine(); context_capacity(); failed_completion(); coordinator(); cancellation();
    std::printf("reference_service: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
