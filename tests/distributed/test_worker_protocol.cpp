// CPU protocol boundary only: this backend neither executes a model nor moves tensors.
// Build: g++ -std=c++20 -Iinclude -Wall -Wextra -Werror tests/distributed/test_worker_protocol.cpp
//        src/distributed/worker.cpp -o test_worker_protocol (one command).
// CMake (inside STRATA_BUILD_DISTRIBUTED_TESTS):
// add_executable(test_worker_protocol tests/distributed/test_worker_protocol.cpp src/distributed/worker.cpp)
// target_include_directories(test_worker_protocol PRIVATE include)
// target_link_libraries(test_worker_protocol PRIVATE strata_warnings)
// add_test(NAME test_worker_protocol COMMAND test_worker_protocol)
#include "strata/distributed/worker.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace strata::distributed;
int checks = 0;
int failures = 0;
void check(bool value, const char* label) {
    ++checks;
    if (!value) { ++failures; std::printf("FAIL: %s\n", label); }
}

ScheduledCollective operation(uint64_t sequence, uint64_t step, std::string route) {
    ScheduledCollective value;
    value.descriptor.operation = {sequence, 0, step};
    value.descriptor.elements = 4;
    value.route = std::move(route);
    return value;
}

class BoundaryBackend final : public ExecutionBackend {
 public:
    mutable int plans = 0;
    std::vector<std::string> calls;
    Failure injected;
    std::string fail_at;
    bool throw_on_finish = false;
    bool complete = true;
    bool finite = true;
    bool committed = true;
    bool internally_driven = false;
    int aborts = 0;
    int discards = 0;
    uint64_t first_sequence = 1;
    std::function<void()> observe_finish;
    std::vector<int32_t> model_tokens;
    std::vector<int32_t> saved_tokens;
    bool fail_capture = false;
    bool fail_restore = false;

    bool model_driven_collectives() const override { return internally_driven; }

    Failure plan(const StepCommand& command, std::vector<ScheduledCollective>& out) const override {
        ++plans;
        out.clear();
        if (command.phase == StepPhase::prefill || command.phase == StepPhase::decode) {
            out.push_back(operation(first_sequence, command.step, "global-route:experts=2,7"));
            out.push_back(operation(first_sequence + 1, command.step, "global-route:down"));
        }
        return {};
    }
    Failure enter(std::string label) {
        calls.push_back(label);
        return fail_at == label ? injected : Failure{};
    }
    Failure agree(const StepCommand&) override { return enter("agree"); }
    Failure begin(const StepCommand& command) override {
        if (auto f = enter("begin")) return f;
        if (command.phase == StepPhase::prefill || command.phase == StepPhase::decode)
            model_tokens.insert(model_tokens.end(), command.tokens.begin(), command.tokens.end());
        else if (command.phase == StepPhase::checkpoint) {
            if (fail_capture) return Failure::fail(FailureCode::comm_failed, "capture failed");
            saved_tokens = model_tokens;
        } else if (command.phase == StepPhase::rollback) {
            if (fail_restore) return Failure::fail(FailureCode::comm_failed, "restore failed");
            model_tokens = saved_tokens;
        } else if (command.phase == StepPhase::reset || command.phase == StepPhase::cancel) {
            model_tokens.clear();
            saved_tokens.clear();
        }
        if (internally_driven) {
            for (const auto& item : command.expected_collective_schedule)
                if (auto f = collective(item)) return f;
        }
        return {};
    }
    Failure collective(const ScheduledCollective& value) override {
        return enter("collective:" + std::to_string(value.descriptor.operation.sequence));
    }
    Failure finish(const StepCommand&, BackendCompletion& out) override {
        if (observe_finish) observe_finish();
        if (throw_on_finish) throw std::runtime_error("backend exception");
        out = {complete, finite, committed, "fake-boundary-state"};
        return enter("finish");
    }
    void abort() noexcept override { ++aborts; }
    void discard_session() noexcept override { ++discards; model_tokens.clear(); saved_tokens.clear(); }
};

struct Fixture {
    BoundaryBackend backend;
    Worker worker{{"session-A", "manifest-A", 0}, backend};
    uint64_t step = 1;

    StepCommand command(StepPhase phase, std::string request = "request-A",
                        std::vector<int32_t> tokens = {}, std::vector<uint64_t> positions = {}) {
        StepCommand value{"session-A", std::move(request), 0, step, phase,
                          std::move(tokens), std::move(positions), "manifest-A", {}};
        backend.plan(value, value.expected_collective_schedule);
        return value;
    }
    StepResult run(StepPhase phase, std::string request = "request-A",
                   std::vector<int32_t> tokens = {}, std::vector<uint64_t> positions = {}) {
        auto value = command(phase, std::move(request), std::move(tokens), std::move(positions));
        auto result = worker.execute(value);
        if (result.status != StepStatus::rejected && result.status != StepStatus::failed) {
            ++step;
            backend.first_sequence += value.expected_collective_schedule.size();
        }
        return result;
    }
    void init() { check(run(StepPhase::init, "").status == StepStatus::completed, "init completed"); }
    void prefill() { check(run(StepPhase::prefill, "request-A", {4, 9}, {0, 1}).status == StepStatus::completed,
                           "prefill staged"); }
    void commit() { check(run(StepPhase::commit).output_ready, "commit opens coordinator output gate"); }
    void expect_rejected(StepCommand command, const char* label) {
        const auto before = backend.calls.size();
        const auto step_before = worker.next_step();
        const auto result = worker.execute(command);
        check(result.status == StepStatus::rejected && !result.output_ready &&
                  backend.calls.size() == before && worker.next_step() == step_before, label);
    }
};

void happy_path() {
    Fixture f;
    f.expect_rejected(f.command(StepPhase::decode, "request-A", {4}, {0}), "decode before init rejected");
    f.init();
    f.expect_rejected(f.command(StepPhase::commit), "commit before compute rejected");
    f.backend.calls.clear();
    auto result = f.run(StepPhase::prefill, "request-A", {4, 9}, {0, 1});
    check(result.status == StepStatus::completed && !result.output_ready, "no output before commit");
    check(f.backend.calls == std::vector<std::string>{"agree", "begin", "collective:1", "collective:2", "finish"},
          "agreement precedes compute and collective order is exact");
    f.expect_rejected(f.command(StepPhase::decode, "request-A", {3}, {2}), "pending compute blocks another decode");
    f.backend.observe_finish = [&] { check(f.worker.state() == WorkerState::pending,
                                         "worker remains pending until backend commit returns"); };
    result = f.run(StepPhase::commit);
    f.backend.observe_finish = {};
    check(result.output_ready && result.state_digest.starts_with("noncrypto:"), "committed state explicitly noncrypto");
    result = f.run(StepPhase::decode, "request-A", {8}, {2});
    check(!result.output_ready && result.status == StepStatus::completed, "teacher-forced decode staged");
    f.commit();
    f.expect_rejected(f.command(StepPhase::prefill, "request-A", {1}, {3}), "prefill after decode rejected");
    check(f.run(StepPhase::reset).status == StepStatus::completed, "reset request");
    f.expect_rejected(f.command(StepPhase::prefill, "request-A", {1}, {0}), "retired request cannot be replayed");
    check(f.run(StepPhase::prefill, "request-B", {1}, {0}).status == StepStatus::completed, "second request starts empty");
    check(f.run(StepPhase::commit, "request-B").output_ready, "second request commits");
}

void validation() {
    Fixture f;
    f.init();
    auto valid = f.command(StepPhase::prefill, "request-A", {1, 2}, {0, 1});
    auto bad = valid; bad.session = "session-B"; f.expect_rejected(bad, "foreign session rejected before mutation");
    bad = valid; bad.manifest_hash = "wrong"; f.expect_rejected(bad, "manifest mismatch rejected before mutation");
    bad = valid; bad.slot = 1; f.expect_rejected(bad, "C1 rejects other slot");
    bad = valid; --bad.step; f.expect_rejected(bad, "stale step rejected");
    bad = valid; ++bad.step; f.expect_rejected(bad, "skipped step rejected");
    bad = valid; bad.phase = static_cast<StepPhase>(99); f.expect_rejected(bad, "invalid phase rejected");
    bad = valid; bad.tokens[0] = -1; f.expect_rejected(bad, "negative token rejected");
    bad = valid; bad.positions.pop_back(); f.expect_rejected(bad, "position count mismatch rejected");
    bad = valid; bad.positions[1] = 3; f.expect_rejected(bad, "position gap rejected");
    bad = valid; bad.expected_collective_schedule[0].route = "different-experts";
    f.expect_rejected(bad, "route mismatch rejected before backend agreement/mutation");
    bad = valid; bad.expected_collective_schedule.pop_back(); f.expect_rejected(bad, "missing collective rejected");
    bad = valid; ++bad.expected_collective_schedule[0].descriptor.elements;
    f.expect_rejected(bad, "collective extent mismatch rejected");
    bad = valid; bad.expected_collective_schedule[0].descriptor.type = ScalarType::bf16;
    f.expect_rejected(bad, "collective dtype mismatch rejected");
    bad = valid; ++bad.expected_collective_schedule[0].descriptor.operation.sequence;
    f.expect_rejected(bad, "collective sequence mismatch rejected");
    bad = valid; ++bad.expected_collective_schedule[0].descriptor.operation.step;
    f.expect_rejected(bad, "collective step mismatch rejected");
    bad = valid; ++bad.expected_collective_schedule[0].descriptor.operation.layer;
    f.expect_rejected(bad, "collective layer mismatch rejected");
    bad = valid; bad.expected_collective_schedule[0].descriptor.kind = static_cast<CollectiveKind>(99);
    f.expect_rejected(bad, "unknown collective kind rejected");
    bad = valid; bad.expected_collective_schedule[0].descriptor.valid_counts = {4, 4, 4};
    f.expect_rejected(bad, "reduction with gather counts rejected");
    bad = valid; bad.expected_collective_schedule[0].descriptor.kind = CollectiveKind::all_gather_padded;
    bad.expected_collective_schedule[0].descriptor.valid_counts = {4, 5, 4};
    f.expect_rejected(bad, "gather count above slot rejected");
    bad = valid; bad.expected_collective_schedule[0].descriptor.type = static_cast<ScalarType>(99);
    f.expect_rejected(bad, "invalid scalar enum rejected");
    check(f.worker.execute(valid).status == StepStatus::completed, "valid command accepted after local rejections");
    f.expect_rejected(valid, "duplicate command rejected");
    ++f.step; f.backend.first_sequence += 2;
    bad = f.command(StepPhase::commit); bad.request = "other"; f.expect_rejected(bad, "wrong request cannot commit");
    bad = f.command(StepPhase::commit); bad.tokens = {1}; f.expect_rejected(bad, "commit cannot inject tokens");
}

void failure_paths() {
    for (const std::string where : {"agree", "begin", "collective:1", "collective:2", "finish"}) {
        Fixture f;
        f.init();
        f.backend.fail_at = where;
        f.backend.injected = Failure::fail(FailureCode::peer_closed, "dropped peer");
        auto result = f.run(StepPhase::prefill, "request-A", {1}, {0});
        check(result.status == StepStatus::failed && !result.output_ready && result.state_digest.empty(),
              "failure never returns output or reusable digest");
        check(f.worker.state() == WorkerState::failed && f.backend.aborts == 1 && f.backend.discards == 1,
              "backend failure aborts and discards entire session");
        const auto count = f.backend.calls.size();
        result = f.worker.execute(f.command(StepPhase::reset));
        check(result.status == StepStatus::failed && f.backend.calls.size() == count,
              "failed session cannot be reset/reused");
        result = f.worker.execute(f.command(StepPhase::init, ""));
        check(result.status == StepStatus::failed, "failed session cannot be reinitialized");
    }
    for (const StepPhase phase : {StepPhase::commit, StepPhase::reset, StepPhase::cancel}) {
        Fixture f; f.init(); f.prefill();
        f.backend.committed = false;
        auto result = f.run(phase);
        check(result.status == StepStatus::failed && !result.output_ready && f.backend.discards == 1,
              "partial commit/reset/cancel is fail-stop");
    }
    {
        Fixture f; f.init(); f.backend.finite = false;
        const auto result = f.run(StepPhase::prefill, "request-A", {1}, {0});
        check(result.status == StepStatus::failed && !result.output_ready, "nonfinite candidate output rejected");
    }
    {
        Fixture f; f.init(); f.prefill(); f.backend.finite = false;
        check(f.run(StepPhase::commit).status == StepStatus::failed, "nonfinite commit rejected");
    }
    {
        Fixture f; f.init(); f.backend.complete = false;
        check(f.run(StepPhase::prefill, "request-A", {1}, {0}).status == StepStatus::failed,
              "incomplete backend work cannot advance protocol");
    }
    {
        Fixture f; f.init(); f.backend.throw_on_finish = true;
        check(f.run(StepPhase::prefill, "request-A", {1}, {0}).status == StepStatus::failed,
              "backend exception fails session");
    }
}

void cancel_abort_and_follower() {
    Fixture f; f.init(); f.prefill();
    check(f.backend.calls.size() >= 8, "prefill completed its entire scheduled collective list before cancel");
    check(f.run(StepPhase::cancel).status == StepStatus::cancelled && f.worker.state() == WorkerState::idle,
          "cancel at safe command boundary discards request");
    f.expect_rejected(f.command(StepPhase::commit), "cancelled candidate cannot commit");
    check(f.run(StepPhase::prefill, "request-B", {2}, {0}).status == StepStatus::completed,
          "new request after cancel");
    check(f.run(StepPhase::abort, "request-B").status == StepStatus::aborted && f.backend.aborts == 1 &&
              f.backend.discards == 1, "group abort discards pending state");
    check(f.run(StepPhase::abort, "request-B").status == StepStatus::aborted && f.backend.aborts == 1,
          "abort cleanup is idempotent");
    check(f.run(StepPhase::init, "").status == StepStatus::aborted, "aborted session cannot initialize");

    BoundaryBackend backend;
    Worker follower({"session-A", "manifest-A", 1}, backend);
    StepCommand command{"session-A", "", 0, 1, StepPhase::init, {}, {}, "manifest-A", {}};
    check(follower.execute(command).status == StepStatus::completed, "follower init");
    command = {"session-A", "request-A", 0, 2, StepPhase::prefill, {1}, {0}, "manifest-A", {}};
    backend.plan(command, command.expected_collective_schedule);
    check(follower.execute(command).status == StepStatus::completed, "follower stages input");
    command = {"session-A", "request-A", 0, 3, StepPhase::commit, {}, {}, "manifest-A", {}};
    const auto result = follower.execute(command);
    check(result.status == StepStatus::completed && !result.output_ready, "follower never owns sampling/output gate");
}

void prefill_chunks_and_reset() {
    Fixture f; f.init(); f.prefill(); f.commit();
    auto result = f.run(StepPhase::prefill, "request-A", {7, 8}, {2, 3});
    check(result.status == StepStatus::completed && !result.output_ready, "contiguous second prefill chunk staged");
    f.commit();
    f.expect_rejected(f.command(StepPhase::decode, "request-A", {1}, {2}), "old position after chunk commit rejected");
    result = f.run(StepPhase::decode, "request-A", {5}, {4});
    check(result.status == StepStatus::completed, "decode follows all committed prefill chunks");
    f.commit();
    check(f.run(StepPhase::reset).status == StepStatus::completed, "committed request reset");
    check(f.run(StepPhase::prefill, "request-C", {3}, {0}).status == StepStatus::completed,
          "reset clears committed position for next request");
    f.backend.fail_at = "finish";
    f.backend.injected = Failure::fail(FailureCode::deadline_expired, "reset timeout after partial clear");
    check(f.run(StepPhase::reset, "request-C").status == StepStatus::failed && f.backend.discards == 1,
          "reset failure cannot preserve or reuse any prior state");
}

void model_driven_no_replay() {
    Fixture f;
    f.backend.internally_driven = true;
    f.init();
    f.backend.calls.clear();
    const auto result = f.run(StepPhase::prefill, "request-A", {1}, {0});
    check(result.status == StepStatus::completed && !result.output_ready &&
          f.backend.calls == std::vector<std::string>{"agree", "begin", "collective:1", "collective:2", "finish"},
          "model-driven boundary executes each scheduled collective once without worker replay");
    f.commit();
}

void checkpoint_rollback() {
    Fixture f; f.init();
    f.expect_rejected(f.command(StepPhase::checkpoint), "idle checkpoint rejected");
    f.prefill();
    f.expect_rejected(f.command(StepPhase::checkpoint), "pending checkpoint rejected");
    f.expect_rejected(f.command(StepPhase::rollback), "rollback without snapshot rejected");
    f.commit();
    auto result = f.run(StepPhase::checkpoint);
    check(result.status == StepStatus::completed && !result.output_ready &&
          f.backend.saved_tokens == std::vector<int32_t>{4, 9}, "checkpoint saves committed model state without output");
    auto stale_checkpoint = f.command(StepPhase::checkpoint); --stale_checkpoint.step;
    f.expect_rejected(stale_checkpoint, "stale checkpoint step rejected");
    auto bad = f.command(StepPhase::rollback); bad.request = "foreign";
    f.expect_rejected(bad, "snapshot cannot restore foreign request");
    bad = f.command(StepPhase::checkpoint); bad.tokens = {1};
    f.expect_rejected(bad, "checkpoint cannot inject tokens");
    bad = f.command(StepPhase::rollback); bad.expected_collective_schedule = {operation(3, f.step, "illegal")};
    f.expect_rejected(bad, "rollback cannot carry collectives");
    check(f.run(StepPhase::decode, "request-A", {8}, {2}).status == StepStatus::completed,
          "post checkpoint decode staged");
    f.commit();
    const auto step_before = f.worker.next_step();
    const auto sequence_before = f.backend.first_sequence;
    result = f.run(StepPhase::rollback);
    check(result.status == StepStatus::completed && !result.output_ready && f.worker.state() == WorkerState::active &&
          f.backend.model_tokens == std::vector<int32_t>{4, 9}, "rollback restores committed state and closes output gate");
    check(f.worker.next_step() == step_before + 1 && f.backend.first_sequence == sequence_before,
          "rollback advances command step without rewinding collective sequence");
    check(f.run(StepPhase::prefill, "request-A", {7, 8}, {2, 3}).status == StepStatus::completed,
          "rollback restores prefill mode and exact next position");
    result = f.run(StepPhase::rollback);
    check(result.status == StepStatus::completed && !result.output_ready &&
          f.backend.model_tokens == std::vector<int32_t>{4, 9}, "pending rollback discards candidate model state");
    f.expect_rejected(f.command(StepPhase::commit), "rolled back candidate cannot commit");
    check(f.run(StepPhase::decode, "request-A", {8}, {2}).status == StepStatus::completed,
          "same token replays at snapshot position");
    f.commit();
    check(f.backend.model_tokens == std::vector<int32_t>{4, 9, 8}, "replay reconstructs original sequence");
    check(f.run(StepPhase::checkpoint).status == StepStatus::completed, "checkpoint replacement after committed decode");
    check(f.run(StepPhase::decode, "request-A", {1}, {3}).status == StepStatus::completed,
          "decode after replacement checkpoint");
    check(f.run(StepPhase::rollback).status == StepStatus::completed, "replacement snapshot restores pending decode");
    f.expect_rejected(f.command(StepPhase::prefill, "request-A", {1}, {3}), "rollback restores saved decode mode exactly");
    check(f.run(StepPhase::reset).status == StepStatus::completed, "reset invalidates snapshot");
    f.expect_rejected(f.command(StepPhase::rollback), "reset snapshot cannot be restored");
    check(f.run(StepPhase::prefill, "request-B", {2}, {0}).status == StepStatus::completed, "new request after snapshot reset");
    check(f.run(StepPhase::commit, "request-B").status == StepStatus::completed, "new request committed");
    f.expect_rejected(f.command(StepPhase::rollback, "request-B"), "new request cannot restore prior snapshot");
    check(f.run(StepPhase::checkpoint, "request-B").status == StepStatus::completed, "new request snapshot");
    check(f.run(StepPhase::cancel, "request-B").status == StepStatus::cancelled, "cancel invalidates snapshot");
    f.expect_rejected(f.command(StepPhase::rollback, "request-B"), "cancelled snapshot cannot be restored");
}

void checkpoint_failures() {
    for (const auto phase : {StepPhase::checkpoint, StepPhase::rollback}) {
        for (const bool partial_commit : {false, true}) {
            Fixture f; f.init(); f.prefill(); f.commit();
            if (phase == StepPhase::rollback)
                check(f.run(StepPhase::checkpoint).status == StepStatus::completed, "rollback failure fixture snapshot");
            f.backend.fail_capture = !partial_commit && phase == StepPhase::checkpoint;
            f.backend.fail_restore = !partial_commit && phase == StepPhase::rollback;
            f.backend.committed = !partial_commit;
            const auto result = f.run(phase);
            check(result.status == StepStatus::failed && !result.output_ready && result.state_digest.empty() &&
                  f.worker.state() == WorkerState::failed && f.backend.aborts == 1 && f.backend.discards == 1 &&
                  f.backend.saved_tokens.empty(), "capture/restore failure or partial all-rank agreement poisons session");
            const auto calls = f.backend.calls.size();
            check(f.run(StepPhase::rollback).status == StepStatus::failed && f.backend.calls.size() == calls,
                  "failed checkpoint operation cannot retry or restore old state");
        }
    }
}
}  // namespace

int main() {
    happy_path();
    validation();
    failure_paths();
    cancel_abort_and_follower();
    prefill_chunks_and_reset();
    model_driven_no_replay();
    checkpoint_rollback();
    checkpoint_failures();
    std::printf("worker protocol boundary: %d checks, %d failures (no model/GPU execution)\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
