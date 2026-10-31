#include "strata/distributed/autoregressive.hpp"

#include <cstdio>
#include <deque>
#include <limits>
#include <memory>
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
    std::vector<int32_t> state, checkpoint;
    std::vector<std::vector<int32_t>> completed;
    int32_t pending = -1;
    bool fail_commit = false;
    Failure plan(const StepCommand&, std::vector<ScheduledCollective>& out) const override { out.clear(); return {}; }
    Failure agree(const StepCommand&) override { return {}; }
    Failure begin(const StepCommand& c) override { commands.push_back(c); return {}; }
    Failure collective(const ScheduledCollective&) override { return {}; }
    Failure finish(const StepCommand& c, BackendCompletion& out) override {
        if (fail_commit && c.phase == StepPhase::commit)
            return Failure::fail(FailureCode::comm_failed, "injected commit failure");
        if (c.phase == StepPhase::prefill || c.phase == StepPhase::decode) pending = c.tokens.at(0);
        if (c.phase == StepPhase::commit) { state.push_back(pending); pending = -1; }
        if (c.phase == StepPhase::checkpoint) checkpoint = state;
        if (c.phase == StepPhase::rollback) { state = checkpoint; pending = -1; }
        if (c.phase == StepPhase::reset) completed.push_back(state);
        if (c.phase == StepPhase::reset || c.phase == StepPhase::cancel) {
            state.clear(); checkpoint.clear(); pending = -1;
        }
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
        cb.broadcast = [&, rank, replay, cursor = std::make_shared<size_t>(0)](std::string_view label, std::string& bytes) {
            if (rank != 0) {
                auto& index = *cursor;
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
    for (int i = 20000; i < 65536; ++i) ids += ",1";
    Rig boundary; boundary.input = {generate("boundary", ids, 256), cancel("boundary")};
    check(!run_autoregressive(0, 100, kDistributedMaxContext, boundary.callbacks()) &&
          boundary.count(StepPhase::prefill) == 1 && boundary.count(StepPhase::cancel) == 1,
          "64K prompt plus 256 output admission crosses wire chunks without truncation");
    Rig invalid;
    check(bool(run_autoregressive(0, 100, kDistributedMaxContext + 1, invalid.callbacks())) &&
          invalid.events.empty() && invalid.wire.empty(), "oversized coordinator context fails before readiness");
}

int32_t target_token(const std::vector<int32_t>& history, uint64_t draw) {
    uint64_t value = 17 + draw * 997;
    for (auto token : history) value = value * 131 + uint32_t(token);
    return int32_t(value % 97);
}
struct MtpRig : Rig {
    using Rig::Rig;
    std::vector<int32_t> output;
    std::vector<std::string> requests;
    std::vector<std::pair<int32_t, uint64_t>> prompt_rows;
    std::vector<SpeculativeWindow> windows;
    std::vector<SpeculativeResult> results;
    size_t proposal_calls = 0, records = 0;
    size_t request_output_start = 0;
    int mismatch = -1;
    int fail_hook = -1;
    bool fail_resolved = false;
    AutoregressiveCallbacks mtp_callbacks(bool enabled = true) {
        auto cb = callbacks();
        cb.mtp_enabled = enabled;
        auto emit = cb.emit;
        cb.emit = [&, emit](auto type, auto id, int32_t token, auto state) {
            emit(type, id, token, state);
            if (type == "token") output.push_back(token);
        };
        cb.sample = [&](const ServeRequest&, const auto& history, uint64_t draw, int32_t& token) {
            check(worker.state() == WorkerState::active && backend.state == history && backend.pending == -1,
                  "target sampling sees exactly the committed model state and penalty history");
            ++samples; sampled_history.push_back(history); token = target_token(history, draw); return Failure{};
        };
        const auto injected = [&] { return Failure::fail(FailureCode::comm_failed, "injected MTP hook failure"); };
        cb.begin_request = [&, injected](const ServeRequest& request) {
            requests.push_back(request.request_id);
            request_output_start = output.size();
            return fail_hook == 0 ? injected() : Failure{};
        };
        cb.prompt_row = [&, injected](int32_t next, uint64_t position) {
            check(worker.state() == WorkerState::active && backend.state.size() == position + 1,
                  "prompt adapter runs after committed prompt row");
            prompt_rows.emplace_back(next, position);
            return fail_hook == 1 ? injected() : Failure{};
        };
        cb.propose = [&, injected](const auto& history, int32_t feedback, uint64_t draw, auto& drafts) {
            ++proposal_calls;
            check(backend.state == history && !output.empty() && output.back() == feedback,
                  "drafter receives consumed history and already published feedback");
            auto tail = history;
            for (int row = 0; row < 4; ++row) {
                tail.push_back(feedback);
                feedback = target_token(tail, draw + uint64_t(row));
                if (row == mismatch) feedback = (feedback + 1) % 97;
                drafts.push_back(feedback);
            }
            return fail_hook == 2 ? injected() : Failure{};
        };
        cb.record_target_row = [&, injected](size_t, int32_t input, int32_t) {
            ++records;
            check(!backend.state.empty() && backend.state.back() == input && backend.pending == -1,
                  "target residual is recorded after its commit");
            return fail_hook == 3 ? injected() : Failure{};
        };
        cb.accepted_window = [&, injected](const auto& window, const auto& result) {
            auto expected = window.history;
            expected.insert(expected.end(), result.consumed_inputs.begin(), result.consumed_inputs.end());
            check(backend.state == expected && backend.pending == -1,
                  "accepted adapter observes canonical state after any rollback and replay");
            check(output.size() - request_output_start == window.draw_base,
                  "speculative tokens remain unpublished until adapter catch-up succeeds");
            windows.push_back(window); results.push_back(result);
            return fail_hook == 4 ? injected() : Failure{};
        };
        auto broadcast = cb.broadcast;
        cb.broadcast = [&, broadcast, injected](auto label, auto& bytes) {
            if (fail_resolved && label.starts_with("spec-resolved-v1")) return injected();
            return broadcast(label, bytes);
        };
        return cb;
    }
};
void mtp_parity_and_followers() {
    for (int mismatch : {-1, 0, 2}) {
        MtpRig ordinary, root;
        ordinary.input.push_back(generate("parity", "1,2,3", 10));
        for (int i = 0; i < 20; ++i) ordinary.input.push_back("");
        root.input = ordinary.input; root.mismatch = mismatch;
        check(!run_autoregressive(0, 100, 64, ordinary.mtp_callbacks(false)), "ordinary deterministic oracle completes");
        check(!run_autoregressive(0, 100, 64, root.mtp_callbacks()), "MTP accepted and mismatched windows complete");
        check(root.output == ordinary.output && root.backend.completed == ordinary.backend.completed && root.output.size() == 10,
              "MTP preserves exact ordinary output and complete consumed model history");
        check(root.requests == std::vector<std::string>{"parity"} &&
              root.prompt_rows == std::vector<std::pair<int32_t, uint64_t>>{{2,0},{3,1}},
              "adapter begins once and feeds known next prompt tokens without duplicating final prompt row");
        check(!root.windows.empty() && root.windows.front().draw_base == 1 &&
              root.windows.front().history == std::vector<int32_t>{1,2,3} &&
              root.windows.front().inputs.front() == ordinary.output.front(),
              "first output uses ordinary draw zero before any speculative window");
        size_t expected_rollbacks = 0;
        for (size_t i = 0; i < root.windows.size(); ++i)
            expected_rollbacks += root.results[i].accepted_rows < std::min(root.windows[i].inputs.size(), size_t(root.windows[i].output_budget));
        check(root.count(StepPhase::rollback) == expected_rollbacks && (mismatch < 0 || expected_rollbacks > 0),
              "mismatch restores checkpoint while matching drafts retain committed state");
        for (uint32_t rank : {1u, 2u}) {
            Rig follower(rank);
            auto cb = follower.callbacks(rank, &root.wire);
            cb.mtp_enabled = true; cb.poll = {}; cb.sample = {}; cb.emit = {};
            check(!run_autoregressive(rank, 100, 64, cb) && follower.backend.completed == root.backend.completed &&
                  follower.backend.commands.size() == root.backend.commands.size() && follower.worker.state() == WorkerState::idle,
                  "MTP follower reconstructs identical canonical state without any root adapter hook");
        }
    }
}
void mtp_failures_withhold_output() {
    for (int failure = 0; failure < 7; ++failure) {
        MtpRig rig;
        rig.input = {generate("failure", "1,2,3", 6), "", "", "", ""};
        rig.fail_hook = failure < 5 ? failure : -1;
        rig.fail_resolved = failure == 5;
        auto cb = rig.mtp_callbacks();
        if (failure == 6) {
            auto broadcast = cb.broadcast;
            cb.broadcast = [broadcast](auto label, auto& bytes) {
                if (label == "serve-mtp-accepted-v1") return Failure::fail(FailureCode::comm_failed, "catch-up agreement failed");
                return broadcast(label, bytes);
            };
        }
        check(bool(run_autoregressive(0, 100, 64, cb)), "MTP adapter and agreement failures terminate service");
        check(rig.output.size() == (failure < 2 ? 0u : 1u) && rig.count(StepPhase::reset) == 0,
              "failure withholds entire speculative output and done while retaining prior ordinary token");
    }
}
void mtp_cancel_fresh_and_budget() {
    {
        MtpRig rig;
        rig.input = {generate("cancelled", "1", 10), "", cancel("cancelled"), generate("fresh", "9", 2), ""};
        check(!run_autoregressive(0, 100, 64, rig.mtp_callbacks()), "MTP cancellation permits a fresh request");
        check(rig.requests == std::vector<std::string>{"cancelled","fresh"} && rig.windows.size() == 2 &&
              rig.windows[1].history == std::vector<int32_t>{9} && rig.windows[1].draw_base == 1 &&
              rig.windows[1].window_id > rig.windows[0].window_id && rig.count(StepPhase::cancel) == 1,
              "fresh request clears draft history and draw counter while window identity remains monotonic");
    }
    for (uint32_t budget : {1u, 2u, 4u, 6u}) {
        MtpRig rig;
        rig.input = {generate("budget", "1,2,3", budget), "", "", "", ""};
        check(!run_autoregressive(0, 100, 16, rig.mtp_callbacks()) && rig.output.size() == budget,
              "MTP never exceeds requested output budget even with four proposed drafts");
        check(rig.count(StepPhase::decode) == budget - 1 && rig.count(StepPhase::reset) == 1 &&
              rig.proposal_calls == size_t(budget > 1),
              "final budget trims verification work and one-token request avoids drafter entirely");
    }
}
}
int main() {
    normal_and_followers(); short_operation_budget(); failures_and_cancel(); large_chunked_prompt();
    mtp_parity_and_followers(); mtp_failures_withhold_output(); mtp_cancel_fresh_and_budget();
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
