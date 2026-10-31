#include "strata/distributed/autoregressive.hpp"

#include <cstdio>
#include <deque>
#include <limits>
#include <string>
#include <utility>

namespace {
using namespace strata::distributed;
int failures = 0, checks = 0;
void check(bool value, const char* label) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", label); }
}
struct FakeBackend final : ExecutionBackend {
    std::vector<StepCommand> commands;
    bool fail_commit = false;
    Failure plan(const StepCommand&, std::vector<ScheduledCollective>& out) const override { out.clear(); return {}; }
    Failure agree(const StepCommand&) override { return {}; }
    Failure begin(const StepCommand& c) override { commands.push_back(c); return {}; }
    Failure collective(const ScheduledCollective&) override { return {}; }
    Failure finish(const StepCommand& c, BackendCompletion& out) override {
        if (fail_commit && c.phase == StepPhase::commit)
            return Failure::fail(FailureCode::comm_failed, "injected commit failure");
        out = {true, true, true, "all-model-state"}; return {};
    }
    void abort() noexcept override {}
    void discard_session() noexcept override {}
};
struct Exchange { std::string label, bytes; };
struct Rig {
    FakeBackend backend;
    Worker worker;
    uint64_t step = 1;
    std::deque<std::string> input;
    std::vector<Exchange> wire;
    std::vector<std::string> events;
    std::vector<std::vector<int32_t>> sampled_history;
    std::map<std::string, uint64_t> draws;
    size_t chunks = 0, samples = 0, health = 0;
    bool fail_token_broadcast = false;
    Rig(uint32_t rank = 0) : worker({"session", "manifest", rank}, backend) {
        auto result = worker.execute({"session", {}, 0, step++, StepPhase::init, {}, {}, "manifest", {}});
        check(result.status == StepStatus::completed, "fake Worker init succeeds");
    }
    AutoregressiveCallbacks callbacks(uint32_t rank = 0, const std::vector<Exchange>* replay = nullptr) {
        AutoregressiveCallbacks cb;
        cb.poll = [&](uint32_t wait, std::string& line, bool& eof) {
            check(wait <= 1000, "idle poll is bounded by one second");
            eof = input.empty();
            line = eof ? std::string{} : std::move(input.front());
            if (!eof) input.pop_front();
            return Failure{};
        };
        cb.broadcast = [&, rank, replay, index = size_t(0)](std::string_view label, std::string& bytes) mutable {
            if (rank != 0) {
                check(index < replay->size(), "follower consumes a recorded root broadcast");
                if (index >= replay->size()) return Failure::fail(FailureCode::comm_failed, "missing replay");
                check(bytes.empty() && label == (*replay)[index].label, "follower uses same ordering and supplies no input");
                bytes = (*replay)[index++].bytes;
            } else {
                check(bytes.size() <= kAutoregressiveChunk, "every broadcast chunk is bounded");
                if (fail_token_broadcast && label.starts_with("serve-sampled-id"))
                    return Failure::fail(FailureCode::comm_failed, "injected token broadcast failure");
                wire.push_back({std::string(label), bytes});
                if (label.starts_with("serve-message-v1:")) ++chunks;
            }
            return Failure{};
        };
        cb.health = [&] { ++health; return Failure{}; };
        cb.execute = [&](StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
            StepCommand c{"session", std::string(request), 0, step++, phase, {}, {}, "manifest", {}};
            if (phase == StepPhase::prefill || phase == StepPhase::decode) { c.tokens = {token}; c.positions = {position}; }
            const auto result = worker.execute(c);
            return result.status == StepStatus::completed || result.status == StepStatus::cancelled ? Failure{} : result.failure;
        };
        cb.sample = [&](const ServeRequest& request, const std::vector<int32_t>& history, uint64_t draw, int32_t& token) {
            check(worker.state() == WorkerState::active && draw == draws[request.request_id]++, "sampling follows commit and monotonic per-request draw index");
            ++samples; sampled_history.push_back(history); token = 40 + int32_t(draw); return Failure{};
        };
        cb.emit = [&](std::string_view type, std::string_view id, int32_t, std::string_view state) {
            if (type == "token") check(worker.state() == WorkerState::active, "token event follows committed source logits");
            if (type == "done") check(worker.state() == WorkerState::idle && state == "ready", "done follows group reset to idle");
            events.push_back(std::string(type) + ":" + std::string(id));
        };
        return cb;
    }
    size_t count(StepPhase phase) const {
        size_t result = 0; for (const auto& c : backend.commands) result += c.phase == phase; return result;
    }
};
std::string generate(std::string id, std::string ids = "1,2,3", uint32_t max_new = 2) {
    return "{\"type\":\"generate\",\"request_id\":\"" + id + "\",\"ids\":[" + ids +
           "],\"max_new\":" + std::to_string(max_new) + ",\"sampling\":{\"seed\":123}}";
}
std::string cancel(std::string id) { return "{\"type\":\"cancel\",\"request_id\":\"" + id + "\"}"; }
void normal_and_followers() {
    Rig root;
    root.input = {generate("a"), "", "", "", cancel("a"), cancel("a"), "{\"type\":\"shutdown\"}"};
    auto cb = root.callbacks();
    check(!run_autoregressive(0, 100, 64, cb), "generate, late cancellation and shutdown complete");
    check(root.count(StepPhase::prefill) == 3 && root.count(StepPhase::decode) == 1 &&
          root.count(StepPhase::commit) == 4 && root.count(StepPhase::reset) == 1 && root.count(StepPhase::cancel) == 0,
          "each input commits; final output resets; late cancellation never resets idle Worker");
    check(root.sampled_history == std::vector<std::vector<int32_t>>{{1,2,3},{1,2,3,40}},
          "penalty history includes prompt and fed-back generated token");
    check(root.events == std::vector<std::string>{"ready:","health:","health:","token:a","health:","token:a","done:a","cancelled:a"},
          "only completed output events publish, and late cancellation acknowledges once");
    for (uint32_t rank : {1u, 2u}) {
        Rig follower(rank);
        auto follow = follower.callbacks(rank, &root.wire);
        follow.poll = {}; follow.sample = {}; follow.emit = {};
        check(!run_autoregressive(rank, 100, 64, follow) && follower.worker.state() == WorkerState::idle,
              "followers never poll stdin/sample/emit and execute same common Worker transitions");
        check(follower.backend.commands.size() == root.backend.commands.size(), "followers execute every root step");
    }
}

void short_operation_budget() {
    Rig rig;
    rig.input = {"", "{\"type\":\"shutdown\"}"};
    auto cb = rig.callbacks();
    auto poll = cb.poll;
    cb.poll = [&](uint32_t wait, std::string& line, bool& eof) {
        check(wait <= 25, "idle polling leaves time for peers within a 100 ms operation budget");
        return poll(wait, line, eof);
    };
    check(!run_autoregressive(0, 100, 64, cb, 100), "short operation budget remains healthy while idle");
}
void failures_and_cancel() {
    {
        Rig rig; rig.input = {generate("first", "1", 1), generate("second", "9", 1)};
        check(!run_autoregressive(0, 100, 64, rig.callbacks()), "a fresh request starts after done without stale state");
        check(rig.sampled_history == std::vector<std::vector<int32_t>>{{1},{9}} &&
              rig.count(StepPhase::reset) == 2 && rig.draws["first"] == 1 && rig.draws["second"] == 1,
              "history and sampling counter restart per request while Worker step stays monotonic");
    }
    {
        Rig rig; rig.input = {generate("cancel-me"), cancel("cancel-me")};
        check(!run_autoregressive(0, 100, 64, rig.callbacks()), "active prompt cancellation completes group cancel");
        check(rig.samples == 0 && rig.count(StepPhase::cancel) == 1 && rig.worker.state() == WorkerState::idle,
              "cancel between steps finishes previous commit and clears request state");
    }
    for (int mode = 0; mode < 7; ++mode) {
        Rig rig;
        rig.input = {generate("a", "1", 1)};
        if (mode == 0) rig.backend.fail_commit = true;
        if (mode == 1) rig.fail_token_broadcast = true;
        if (mode == 2) rig.input = {cancel("unknown")};
        if (mode == 3) rig.input = {generate("a"), generate("b")};
        if (mode == 4) rig.input = {generate("a", "1", 1), generate("a")};
        if (mode == 5) rig.input = {std::string(kAutoregressiveMaxLine + 1, 'x')};
        if (mode == 6) rig.input = {generate("a"), cancel("wrong")};
        const auto result = run_autoregressive(0, 100, 64, rig.callbacks());
        check(bool(result), "commit/broadcast/identity/busy/size failures are terminal");
        if (mode < 2) check(rig.events == std::vector<std::string>{"ready:"}, "failed commit or token agreement publishes no token or done");
    }
}
void large_chunked_prompt() {
    std::string ids = "1";
    for (int i = 1; i < 20000; ++i) ids += ",1";
    Rig rig; rig.input = {generate("large", ids, 1), cancel("large")};
    check(!run_autoregressive(0, 100, 65536, rig.callbacks()) && rig.chunks >= 3,
          "large prompt crosses bounded 24KiB chunks with declared length and accepts 64K context");
    check(rig.count(StepPhase::prefill) == 1 && rig.count(StepPhase::cancel) == 1,
          "large prompt can cancel between committed tokens without completing full prefill");
}
}
int main() {
    normal_and_followers(); short_operation_budget(); failures_and_cancel(); large_chunked_prompt();
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
