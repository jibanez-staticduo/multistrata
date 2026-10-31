// CPU fake for the opt-in NCCL graph-session contract.
#include "distributed/nccl_api.hpp"

#include <cstdio>
#include <string>

using namespace strata::distributed;
using namespace strata::distributed::nccl;

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", label);
    if (!ok) ++failures;
}

class DefaultNcclApi : public NcclApi {
 public:
    Failure identity(std::string& version, std::string& provenance) const override {
        version = "2.30.7";
        provenance = "sha256:test";
        return ok_failure();
    }
    Failure comm_init(const NcclConfig&) override { return ok_failure(); }
    Failure enqueue_all_reduce_sum(const void*, void*, uint64_t, ScalarType, cudaStream_t) override {
        return ok_failure();
    }
    Failure enqueue_all_gather(const void*, void*, uint64_t, ScalarType, cudaStream_t) override {
        return ok_failure();
    }
    Failure poll_completion(Deadline) override { return ok_failure(); }
    bool async_failure(Failure&) const override { return false; }
    void comm_abort() noexcept override {}
};

class FakeGraphApi final : public DefaultNcclApi {
 public:
    Failure enqueue_all_reduce_sum(const void*, void*, uint64_t, ScalarType, cudaStream_t stream) override {
        return enqueue(stream);
    }
    Failure enqueue_all_gather(const void*, void*, uint64_t, ScalarType, cudaStream_t stream) override {
        return enqueue(stream);
    }
    Failure poll_completion(Deadline) override {
        ++eager_polls;
        return ok_failure();
    }
    Failure prepare_graph_session(cudaStream_t stream, uint64_t expected) override {
        if (state != State::inactive || stream == nullptr || expected == 0) {
            return fail(FailureCode::invalid_config, "bad graph preparation");
        }
        state = State::prepared;
        stream_ = stream;
        expected_ = expected;
        count_ = 0;
        return ok_failure();
    }
    Failure check_graph_capture() override {
        if (state == State::failed) return sticky_;
        if (state != State::prepared && state != State::capturing) return fail(FailureCode::not_ready, "no capture");
        if (capture_query != FailureCode::none) return poison(capture_query, "capture query failed");
        state = State::capturing;
        return ok_failure();
    }
    Failure finish_graph_capture() override {
        if (state != State::capturing) return fail(FailureCode::not_ready, "no capture");
        if (async_error) return poison(FailureCode::async_error, "NCCL async error during capture");
        if (count_ != expected_) return poison(FailureCode::bad_extent, "captured collective count mismatch");
        state = State::captured;
        return ok_failure();
    }
    Failure begin_captured_replay() override {
        if (state != State::captured || replay_pending_) return fail(FailureCode::bad_operation, "not ready");
        replay_pending_ = true;
        return ok_failure();
    }
    Failure wait_captured_replay(Deadline deadline) override {
        if (state != State::captured || !replay_pending_) return fail(FailureCode::not_ready, "no replay");
        if (async_error) return poison(FailureCode::async_error, "NCCL async error during replay");
        if (query_error) return poison(FailureCode::async_error, "unknown CUDA event query");
        if (timeout || std::chrono::steady_clock::now() >= deadline.end) {
            process_exit = true;
            return poison(FailureCode::deadline_expired, "replay did not complete");
        }
        replay_pending_ = false;
        ++completed_replays;
        return ok_failure();
    }
    Failure end_graph_session() override {
        if (state != State::captured || replay_pending_) return fail(FailureCode::in_flight, "work pending");
        state = State::inactive;
        return ok_failure();
    }
    bool requires_process_exit() const override { return process_exit; }
    bool async_failure(Failure& out) const override {
        if (sticky_) {
            out = sticky_;
            return true;
        }
        if (async_error) {
            out = Failure::fail(FailureCode::async_error, "fake NCCL async error");
            return true;
        }
        return false;
    }
    void comm_abort() noexcept override {
        if (replay_pending_) process_exit = true;
        aborted = true;
    }

    Failure enqueue(cudaStream_t stream) {
        ++async_queries;
        if (state != State::capturing) {
            return fail(FailureCode::bad_operation, "capture was not explicitly checked");
        }
        if (stream != stream_) return poison(FailureCode::bad_pointer, "wrong stream");
        if (async_error) return poison(FailureCode::async_error, "NCCL async error");
        if (count_ >= expected_) return poison(FailureCode::bad_extent, "too many collectives");
        ++count_;
        state = State::capturing;
        return ok_failure();
    }

    void set_capture_query(FailureCode code) { capture_query = code; }
    bool aborted = false;
    bool async_error = false;
    bool query_error = false;
    bool timeout = false;
    int eager_polls = 0;
    int completed_replays = 0;
    int async_queries = 0;

 private:
    enum class State { inactive, prepared, capturing, captured, failed };
    Failure fail(FailureCode code, std::string_view detail) {
        return Failure::fail(code, detail);
    }
    Failure poison(FailureCode code, std::string_view detail) {
        if (replay_pending_) process_exit = true;
        state = State::failed;
        sticky_ = Failure::fail(code, detail);
        return sticky_;
    }
    State state = State::inactive;
    cudaStream_t stream_ = nullptr;
    uint64_t expected_ = 0;
    uint64_t count_ = 0;
    bool replay_pending_ = false;
    bool process_exit = false;
    FailureCode capture_query = FailureCode::none;
    Failure sticky_{};
};

void success_replays() {
    FakeGraphApi api;
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x40});
    float input = 0, output = 0;
    const Deadline deadline{std::chrono::steady_clock::now() + std::chrono::seconds(1)};
    bool okay = !api.prepare_graph_session(stream, 3) && !api.check_graph_capture() &&
                !api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
                !api.enqueue_all_gather(&input, &output, 1, ScalarType::fp32, stream) &&
                !api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
                !api.finish_graph_capture() && api.eager_polls == 0 && api.async_queries == 3;
    for (int replay = 0; replay < 100 && okay; ++replay) {
        okay = !api.begin_captured_replay() && !api.wait_captured_replay(deadline);
    }
    check(okay && api.completed_replays == 100, "three captured collectives complete 100 ordered replays");
    check(!api.end_graph_session(), "a fully drained graph session closes");
}

void failures_are_sticky_and_bounded() {
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x40});
    float input = 0, output = 0;
    FakeGraphApi unknown;
    unknown.set_capture_query(FailureCode::async_error);
    check(!unknown.prepare_graph_session(stream, 1) &&
              unknown.check_graph_capture().code == FailureCode::async_error &&
              unknown.check_graph_capture().code == FailureCode::async_error,
          "an unknown capture query poisons the graph and remains sticky");

    FakeGraphApi mismatch;
    check(!mismatch.prepare_graph_session(stream, 2) && !mismatch.check_graph_capture() &&
              !mismatch.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
              mismatch.finish_graph_capture().code == FailureCode::bad_extent,
          "capture aborts when the prepared collective count does not match");

    FakeGraphApi failed;
    check(!failed.prepare_graph_session(stream, 1) && !failed.check_graph_capture() &&
              !failed.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
              !failed.finish_graph_capture() && !failed.begin_captured_replay(),
          "async-error case starts a replay");
    failed.async_error = true;
    Failure health;
    check(failed.wait_captured_replay(Deadline{}).code == FailureCode::async_error &&
              failed.async_failure(health) && health.code == FailureCode::async_error,
          "an NCCL replay error remains visible through health polling");

    FakeGraphApi timeout;
    check(!timeout.prepare_graph_session(stream, 1) && !timeout.check_graph_capture() &&
              !timeout.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
              !timeout.finish_graph_capture() && !timeout.begin_captured_replay(),
          "timeout case starts a replay");
    timeout.timeout = true;
    check(timeout.wait_captured_replay(Deadline{std::chrono::steady_clock::now() +
                                                std::chrono::seconds(1)}).code == FailureCode::deadline_expired &&
              timeout.requires_process_exit(),
          "an uncertain replay timeout requests owner process exit");

    FakeGraphApi aborted;
    check(!aborted.prepare_graph_session(stream, 1) && !aborted.check_graph_capture() &&
              !aborted.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
              !aborted.finish_graph_capture() && !aborted.begin_captured_replay(),
          "abort case starts a replay");
    aborted.comm_abort();
    check(aborted.aborted && aborted.requires_process_exit(), "abort with pending replay signals process exit");
}
}  // namespace

int main() {
    DefaultNcclApi unsupported;
    check(unsupported.prepare_graph_session(nullptr, 1).code == FailureCode::not_supported,
          "a default NcclApi implementation does not enable graph mode");
    FakeGraphApi not_started;
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x40});
    check(!not_started.prepare_graph_session(stream, 1) &&
              not_started.enqueue_all_reduce_sum(nullptr, nullptr, 1, ScalarType::fp32, stream)
                      .code == FailureCode::bad_operation,
          "prepared graph sessions fail closed until capture is explicitly checked");
    success_replays();
    failures_are_sticky_and_bounded();
    return failures == 0 ? 0 : 1;
}
