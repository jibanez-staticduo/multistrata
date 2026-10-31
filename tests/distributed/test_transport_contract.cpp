// tests/distributed/test_transport_contract.cpp - the transport contract, at the boundary a caller actually
// calls, with no GPU and no NCCL.
//
// What is real here and what is not: `GuardedTransport` and `NcclBackend` are the same code the device build
// runs, and every check that can be done before a collective goes on the wire is exercised against them.  The
// library underneath is a fake that says "accepted", "not done yet", "failed" or "here is a late error".  What
// this cannot say: that NCCL sums correctly, that RoCE carries the bytes, that a kernel finishes, or that a
// graph replays.  Those need the CUDA window (transport plan Task 5).
//
// Build and run, CPU-only (no CUDA, no NCCL, no CMake):
//   g++ -std=c++20 -I include -I src -Wall -Wextra tests/distributed/test_transport_contract.cpp -o test_transport_contract
//   ./test_transport_contract
// CMake lines for the integrator (default OFF, no CUDA dependency):
//   add_executable(test_transport_contract tests/distributed/test_transport_contract.cpp)
//   target_include_directories(test_transport_contract PRIVATE include src)
//   add_test(NAME test_transport_contract COMMAND test_transport_contract)
#include "distributed/nccl_api.hpp"
#include "strata/distributed/transport.hpp"

#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace strata::distributed;
using strata::distributed::nccl::NcclApproval;
using strata::distributed::nccl::NcclBackend;
using strata::distributed::nccl::NcclConfig;
using strata::distributed::nccl::NcclApi;

int g_fail = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("  %-86s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// A clock the test moves by hand, so "the deadline passed" is a fact rather than a sleep.
class FakeClock final : public Clock {
 public:
    std::chrono::steady_clock::time_point now() const override { return at_; }
    void advance_ms(int64_t ms) { at_ += std::chrono::milliseconds(ms); }

 private:
    std::chrono::steady_clock::time_point at_{};
};

// The library, as far as the contract is concerned: it records what it was asked for and replays a scripted
// answer.  It does not imitate NCCL's algorithms - only its yes/no/late-error surface.
class FakeBackend final : public Backend {
 public:
    FakeBackend(uint32_t world, uint32_t rank, FakeClock& clock) : world_(world), rank_(rank), clock_(clock) {}

    uint32_t world_size() const override { return world_; }
    uint32_t local_rank() const override { return rank_; }

    Failure raw_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output, uint64_t elements,
                              cudaStream_t) override {
        ++enqueues;
        last_call = "all_reduce_sum";
        last_elements = elements;
        last_input = input;
        last_output = output;
        return reject_enqueue;
    }

    Failure raw_all_gather(const DeviceBuffer& input, const DeviceBuffer& output, uint64_t slot_elements,
                          cudaStream_t) override {
        ++enqueues;
        last_call = "all_gather";
        last_slot = slot_elements;
        last_elements = output.elements;
        last_input = input;
        last_output = output;
        return reject_enqueue;
    }

    Failure raw_poll(Deadline) override {
        ++polls;
        clock_.advance_ms(1);  // time really does pass while a rank waits, even in a test
        if (poll_failure) return poll_failure;
        if (polls >= completes_after_polls) return ok_failure();
        return Failure::fail(FailureCode::deadline_expired, "not finished yet");
    }

    bool raw_async_failure(Failure& out) const override {
        if (!async_pending) return false;
        out = async_failure;
        return true;
    }

    void raw_abort() noexcept override { ++abort_calls; }

    uint32_t world_ = kTpWorld;
    uint32_t rank_ = 0;
    FakeClock& clock_;

    int enqueues = 0;
    int polls = 0;
    int abort_calls = 0;
    std::string last_call;
    uint64_t last_elements = 0;
    uint64_t last_slot = 0;
    DeviceBuffer last_input{};
    DeviceBuffer last_output{};
    Failure reject_enqueue{};
    Failure poll_failure{};
    int completes_after_polls = 1;
    bool async_pending = false;
    Failure async_failure{};
};

// The same answers one level down, at the NCCL-shaped interface.
class FakeNccl final : public NcclApi {
 public:
    Failure identity(std::string& version, std::string& provenance) const override {
        if (identity_failure) return identity_failure;
        version = reported_version;
        provenance = reported_provenance;
        return ok_failure();
    }

    Failure comm_init(const NcclConfig& config) override {
        ++init_calls;
        seen_bootstrap_id = config.bootstrap_id;
        return init_failure;
    }

    Failure enqueue_all_reduce_sum(const void*, void*, uint64_t elements, ScalarType type, cudaStream_t) override {
        ++reduce_calls;
        last_elements = elements;
        last_type = type;
        return enqueue_failure;
    }

    Failure enqueue_all_gather(const void*, void*, uint64_t slot_elements, ScalarType type, cudaStream_t) override {
        ++gather_calls;
        last_slot = slot_elements;
        last_type = type;
        return enqueue_failure;
    }

    Failure poll_completion(Deadline) override {
        ++polls;
        if (poll_failure) return poll_failure;
        return ok_failure();
    }

    bool async_failure(Failure& out) const override {
        if (!async_pending) return false;
        out = async;
        return true;
    }

    void comm_abort() noexcept override { ++abort_calls; }

    std::string reported_version = "2.28.7";
    std::string reported_provenance = "sha256:aaaa triangle-patch-3";
    Failure identity_failure{};
    Failure init_failure{};
    Failure enqueue_failure{};
    Failure poll_failure{};
    bool async_pending = false;
    Failure async{};

    int init_calls = 0;
    int reduce_calls = 0;
    int gather_calls = 0;
    int polls = 0;
    int abort_calls = 0;
    std::string seen_bootstrap_id;
    uint64_t last_elements = 0;
    uint64_t last_slot = 0;
    ScalarType last_type = ScalarType::fp32;
};

alignas(64) unsigned char g_pool_a[8192];
alignas(64) unsigned char g_pool_b[8192];

void* pool_a(uint64_t offset = 0) { return g_pool_a + offset; }
void* pool_b(uint64_t offset = 0) { return g_pool_b + offset; }

DeviceBuffer buf_a(uint64_t elements, ScalarType type, uint64_t offset = 0) {
    return DeviceBuffer{pool_a(offset), elements, type};
}

DeviceBuffer buf_b(uint64_t elements, ScalarType type, uint64_t offset = 0) {
    return DeviceBuffer{pool_b(offset), elements, type};
}

Operation op(uint64_t sequence, uint64_t layer = 0, uint64_t step = 0) {
    return Operation{sequence, layer, step};
}

bool detail_has(const Failure& f, const std::string& needle) {
    return f.detail.find(needle) != std::string::npos;
}

// Let an accepted operation finish, so the next call is judged on its own shape and not on the
// rule that one collective at a time may be in flight.
void drain(GuardedTransport& transport, FakeClock& clock) {
    const Failure f = transport.wait(Deadline{clock.now() + std::chrono::milliseconds(1000)});
    if (f) std::printf("    (drain: %s)\n", f.detail.c_str());
}

// ---------------------------------------------------------------- the checks that must not reach the wire

void rejects_before_enqueue() {
    std::printf("all_reduce_sum: rejected before anything is enqueued\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 1, clock);
    GuardedTransport transport(backend, clock);

    // equal extents, matching dtype, disjoint buffers: the call the model plan makes every layer
    const auto ok_call = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32),
                                                 op(1), nullptr);
    check(!ok_call && backend.enqueues == 1, "a disjoint fp32 sum of equal extents is enqueued");
    drain(transport, clock);

    struct Case {
        DeviceBuffer in;
        DeviceBuffer out;
        FailureCode code;
        const char* what;
    };
    const DeviceBuffer in_ok = buf_a(4, ScalarType::fp32);
    const std::vector<Case> cases = {
        {in_ok, buf_b(4, ScalarType::bf16), FailureCode::bad_dtype, "input and output of different types"},
        {in_ok, buf_b(3, ScalarType::fp32), FailureCode::bad_extent, "input and output of different lengths"},
        {DeviceBuffer{pool_a(), 4, static_cast<ScalarType>(7)}, buf_b(4, ScalarType::fp32),
         FailureCode::bad_dtype, "a scalar type this build does not know"},
        {DeviceBuffer{nullptr, 4, ScalarType::fp32}, buf_b(4, ScalarType::fp32), FailureCode::bad_pointer,
         "a non-empty buffer with no address"},
        {buf_a(4, ScalarType::fp32, 2), buf_b(4, ScalarType::fp32), FailureCode::bad_pointer,
         "an fp32 buffer that does not start on a 4-byte edge"},
        {buf_a(2, ScalarType::bf16, 1), buf_b(2, ScalarType::bf16), FailureCode::bad_pointer,
         "a bf16 buffer on an odd address"},
        {DeviceBuffer{pool_a(), UINT64_MAX / 2 + 1, ScalarType::fp32},
         DeviceBuffer{pool_b(), UINT64_MAX / 2 + 1, ScalarType::fp32}, FailureCode::overflow,
         "an element count whose byte count would wrap"},
        {buf_a(4, ScalarType::fp32), buf_a(4, ScalarType::fp32, 4), FailureCode::overlap,
         "an output that starts inside the input"},
        {buf_a(4, ScalarType::fp32), buf_a(3, ScalarType::fp32), FailureCode::bad_extent,
         "the same address with unequal extents (not exact in-place)"},
        {buf_a(4, ScalarType::fp32), buf_a(4, ScalarType::bf16), FailureCode::bad_dtype,
         "the same address with a different type"},
    };
    for (const auto& c : cases) {
        const int before = backend.enqueues;
        const Failure f = transport.all_reduce_sum(c.in, c.out, op(2), nullptr);
        check(f.code == c.code && backend.enqueues == before,
              std::string("refused before enqueue: ") + c.what);
    }

    // exact in-place is the allowed overlap, and a zero-length collective is still a legal one
    const Failure in_place = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_a(4, ScalarType::fp32),
                                                     op(2), nullptr);
    check(!in_place && backend.enqueues == 2, "exact in-place (same address, same extents) is accepted");
    check(backend.last_elements == 4, "the in-place sum carries the agreed count");
    drain(transport, clock);
}

void gather_contract() {
    std::printf("all_gather_padded: uniform slots, explicit valid counts\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 1, clock);  // the rank with the short shard: 192 of a 256 slot
    GuardedTransport transport(backend, clock);

    const std::array<uint64_t, 3> counts{256, 192, 192};
    const Failure good = transport.all_gather_padded(buf_a(256, ScalarType::fp32), buf_b(768, ScalarType::fp32),
                                                    std::span<const uint64_t>(counts), op(1), nullptr);
    check(!good && backend.enqueues == 1, "256/192/192 travels as three 256-element slots");
    check(backend.last_slot == 256, "the wire slot is the maximum valid count, not this rank's count");
    check(backend.last_elements == 768, "the output is world slots, not the sum of the counts");
    drain(transport, clock);

    struct Case {
        DeviceBuffer in;
        DeviceBuffer out;
        std::array<uint64_t, 3> counts;
        FailureCode code;
        const char* what;
    };
    const std::array<uint64_t, 3> even{256, 192, 192};
    const std::vector<Case> cases = {
        {buf_a(256, ScalarType::fp32), buf_b(640, ScalarType::fp32), even, FailureCode::bad_counts,
         "an output sized by summing the counts (640) instead of by slots (768)"},
        {buf_a(192, ScalarType::fp32), buf_b(768, ScalarType::fp32), even, FailureCode::bad_extent,
         "a local buffer sized to the valid count instead of to the slot"},
        {buf_a(256, ScalarType::fp32), buf_b(768, ScalarType::fp32), std::array<uint64_t, 3>{256, 300, 192},
         FailureCode::bad_counts, "a valid count larger than the buffer behind it"},
        {buf_a(256, ScalarType::fp32), buf_b(768, ScalarType::fp32), std::array<uint64_t, 3>{0, 0, 0},
         FailureCode::bad_extent, "a gather of nothing at all"},
        {buf_a(256, ScalarType::fp32), buf_b(768, ScalarType::bf16), even, FailureCode::bad_dtype,
         "input and output of different types"},
        {buf_a(256, ScalarType::fp32, 1024), buf_a(768, ScalarType::fp32), even, FailureCode::overlap,
         "an input that sits inside the output"},
        {buf_a(256, ScalarType::fp32), buf_a(768, ScalarType::fp32), even, FailureCode::overlap,
         "an all-gather whose output starts on top of its input"},
    };
    for (const auto& c : cases) {
        const int before = backend.enqueues;
        const Failure f = transport.all_gather_padded(c.in, c.out, std::span<const uint64_t>(c.counts), op(2),
                                                     nullptr);
        check(f.code == c.code && backend.enqueues == before,
              std::string("refused before enqueue: ") + c.what);
    }

    // a short local shard is allowed; a count list that is not one per rank is not
    const std::array<uint64_t, 3> short_shard{256, 100, 192};
    check(!transport.all_gather_padded(buf_a(256, ScalarType::fp32), buf_b(768, ScalarType::fp32),
                                      std::span<const uint64_t>(short_shard), op(2), nullptr),
          "a local valid count below the slot is accepted (the pad is transported, not read)");
    drain(transport, clock);

    FakeClock other_clock;
    FakeBackend four_ranks(4, 1, other_clock);
    GuardedTransport four_transport(four_ranks, other_clock);
    const int before = four_ranks.enqueues;
    const Failure wrong_world = four_transport.all_gather_padded(
        buf_a(256, ScalarType::fp32), buf_b(1024, ScalarType::fp32), std::span<const uint64_t>(even), op(1),
        nullptr);
    check(wrong_world.code == FailureCode::bad_counts && four_ranks.enqueues == before,
          "a count list that does not match the communicator's world is refused");
}

// ---------------------------------------------------------------- the collective order

void sequence_and_one_in_flight() {
    std::printf("one ordered stream: sequence order and one operation in flight\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    GuardedTransport transport(backend, clock);

    const Failure first = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32),
                                                  op(5, 3, 1), nullptr);
    check(first.code == FailureCode::bad_operation, "a collective that does not start at sequence 1 is refused");
    check(detail_has(first, "expected 1"), "and the refusal says what was expected");
    check(backend.enqueues == 0, "nothing reached the wire");

    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1, 3, 1), nullptr),
          "sequence 1 is accepted");
    const Failure second = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32),
                                                   op(2, 3, 1), nullptr);
    check(second.code == FailureCode::in_flight && backend.enqueues == 1,
          "a second collective while the first is in flight is refused");

    check(!transport.wait(Deadline{clock.now() + std::chrono::milliseconds(1000)}), "the first one completes");
    check(transport.check_health().code == FailureCode::none, "and the transport is usable again");

    const Failure skipped = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32),
                                                    op(4, 3, 1), nullptr);
    check(skipped.code == FailureCode::bad_operation, "a skipped sequence number is refused");
    const Failure again = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32),
                                                  op(1, 3, 1), nullptr);
    check(again.code == FailureCode::bad_operation, "a repeated sequence number is refused");
    check(backend.enqueues == 1, "and none of them reached the wire");
}

void completed_external_batch_advances_transport_cursor() {
    std::printf("completed NCCL graph batches synchronize the eager transport cursor\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    GuardedTransport transport(backend, clock);
    const std::array<Operation, 3> completed{{op(1, 0, 1), op(2, 1, 1), op(3, 2, 1)}};
    check(!transport.adopt_completed_batch(completed),
          "a contiguous externally completed batch advances local sequence bookkeeping");
    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32),
                                    op(4, 3, 2), nullptr),
          "the next eager operation continues after a graph batch");

    FakeClock bad_clock;
    FakeBackend bad_backend(kTpWorld, 0, bad_clock);
    GuardedTransport bad_transport(bad_backend, bad_clock);
    const std::array<Operation, 2> skipped{{op(1), op(3)}};
    const auto bad = bad_transport.adopt_completed_batch(skipped);
    check(bad.code == FailureCode::bad_operation && bad_backend.enqueues == 0,
          "a non-contiguous batch cannot advance the transport cursor");
}

// ---------------------------------------------------------------- failures are terminal

void enqueue_failure_is_final() {
    std::printf("an enqueue failure ends the session\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    backend.reject_enqueue = Failure::fail(FailureCode::comm_failed, "the library said no");
    GuardedTransport transport(backend, clock);

    const Failure f = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1),
                                              nullptr);
    check(f.code == FailureCode::comm_failed && detail_has(f, "the library said no"),
          "the library's reason is the reason the caller gets");
    check(backend.abort_calls == 1, "the communicator was abandoned once");

    const Failure again = transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(2),
                                                  nullptr);
    check(again.code == FailureCode::comm_failed && backend.enqueues == 1,
          "a failed transport enqueues nothing more: no retry, no output after failure");
    check(transport.wait(Deadline{clock.now()}).code == FailureCode::comm_failed,
          "waiting on a dead transport says why instead of hanging");
}

void pending_async_error_is_reported() {
    std::printf("an asynchronous library error is reported and stays reported\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    backend.async_failure = Failure::fail(FailureCode::async_error, "a peer died mid-collective");
    GuardedTransport transport(backend, clock);

    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1), nullptr),
          "the enqueue itself succeeded");
    backend.async_pending = true;  // what NCCL reports late: after the work is already on the device
    const Failure health = transport.check_health();
    check(health.code == FailureCode::async_error && detail_has(health, "peer died"),
          "health reports the pending error before anyone waits");
    const Failure waited = transport.wait(Deadline{clock.now() + std::chrono::milliseconds(1000)});
    check(waited.code == FailureCode::async_error, "and wait turns it into the session's reason");
    check(transport.check_health().code == FailureCode::async_error,
          "asking twice does not clear it: a rank must not look healthy one step after a peer died");
    check(backend.abort_calls == 1, "abandoned once");

    // The same error seen before an enqueue is not something to walk into: the session is already broken, and
    // a collective issued into it would be one more the other two ranks are waiting for.
    FakeClock before_clock;
    FakeBackend before(kTpWorld, 0, before_clock);
    before.async_pending = true;
    before.async_failure = Failure::fail(FailureCode::async_error, "a peer died in the last step");
    GuardedTransport before_transport(before, before_clock);
    check(before_transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1), nullptr)
                  .code == FailureCode::async_error,
          "no new collective is issued into a session the library already knows is broken");
    check(before.enqueues == 0, "so that call never reached the wire");
}

void deadline_expiry_names_the_operation() {
    std::printf("an expired deadline ends the session, naming the operation\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    backend.completes_after_polls = 1000000;  // never, within this deadline
    GuardedTransport transport(backend, clock);

    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1, 31, 77),
                                    nullptr),
          "the down-projection sum was enqueued");
    const Failure f = transport.wait(Deadline{clock.now() + std::chrono::milliseconds(10)});
    check(f.code == FailureCode::deadline_expired, "the wait returns at its deadline rather than forever");
    check(detail_has(f, "31") && detail_has(f, "77"),
          "the reason names the layer and step that never completed");
    check(backend.polls > 1 && backend.polls < 20, "the wait polled a bounded number of times");
    check(backend.abort_calls == 1, "and the communicator was abandoned");

    // An already-expired deadline is still a bounded wait.
    FakeClock other_clock;
    FakeBackend other(kTpWorld, 0, other_clock);
    other.completes_after_polls = 1000000;
    GuardedTransport other_transport(other, other_clock);
    check(!other_transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1), nullptr),
          "the second case enqueued");
    const Failure gone = other_transport.wait(Deadline{other_clock.now()});
    check(gone.code == FailureCode::deadline_expired,
          "a deadline that already passed is not a permission to block");

    // Nothing in flight: waiting is not an error, and it does not touch the library.
    FakeClock idle_clock;
    FakeBackend idle(kTpWorld, 0, idle_clock);
    GuardedTransport idle_transport(idle, idle_clock);
    check(!idle_transport.wait(Deadline{idle_clock.now()}),
          "waiting with nothing in flight is a no-op, not a failure");
    check(idle.polls == 0, "and it asked the library for nothing");
}

void zero_poll_interval_preserves_completion_and_failure_checks() {
    std::printf("zero poll interval yields while keeping completion and failure checks\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    backend.completes_after_polls = 4;
    GuardedTransport transport(backend, clock);
    transport.set_poll_interval_ms(0);
    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1), nullptr),
          "zero interval accepts the agreed collective");
    check(!transport.wait(Deadline{clock.now() + std::chrono::milliseconds(10)}) && backend.polls == 4,
          "zero interval keeps polling until the backend completes");
    check(!transport.check_health() && backend.abort_calls == 0,
          "successful zero interval completion leaves the transport healthy");

    backend.polls = 0;
    backend.completes_after_polls = 1000000;
    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(2), nullptr),
          "another collective may follow zero interval completion");
    const Failure expired = transport.wait(Deadline{clock.now() + std::chrono::milliseconds(3)});
    check(expired.code == FailureCode::deadline_expired && backend.polls == 3 && backend.abort_calls == 1,
          "zero interval terminates at the same deadline with bounded polls");
    check(transport.wait(Deadline{clock.now()}).code == FailureCode::deadline_expired && backend.polls == 3,
          "the zero interval deadline failure stays terminal");

    FakeClock error_clock;
    FakeBackend error_backend(kTpWorld, 0, error_clock);
    GuardedTransport error_transport(error_backend, error_clock);
    error_transport.set_poll_interval_ms(0);
    check(!error_transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1), nullptr),
          "the zero interval async-error case enqueues first");
    error_backend.async_pending = true;
    error_backend.async_failure = Failure::fail(FailureCode::async_error, "zero interval peer failure");
    check(error_transport.wait(Deadline{error_clock.now() + std::chrono::milliseconds(10)}).code ==
              FailureCode::async_error && error_backend.polls == 0 && error_backend.abort_calls == 1,
          "zero interval checks asynchronous failure before polling");
}

void abort_is_idempotent() {
    std::printf("abort: the first reason wins, and it is safe to say twice\n");
    FakeClock clock;
    FakeBackend backend(kTpWorld, 0, clock);
    GuardedTransport transport(backend, clock);

    check(!transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(1), nullptr),
          "one collective is in flight");
    transport.abort("the peer's control link closed");
    check(backend.abort_calls == 1, "the communicator was abandoned");
    const Failure health = transport.check_health();
    check(health.code == FailureCode::aborted && detail_has(health, "control link closed"),
          "the reason the caller gave is the reason recorded");

    transport.abort("a second reason");
    check(backend.abort_calls == 1, "a second abort changes nothing");
    check(transport.check_health().detail.find("second reason") == std::string::npos,
          "the first reason is the one that stands");
    check(transport.all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), op(2), nullptr)
                  .code == FailureCode::aborted,
          "and nothing more is enqueued on a session that was aborted");
}

// ---------------------------------------------------------------- the NCCL-shaped boundary

void library_approval_is_checked_before_a_communicator() {
    std::printf("the library must be the approved build before a communicator exists\n");
    const NcclConfig config{0, kTpWorld, "bootstrap-id-from-the-roster", "/opt/strata/lib/libnccl.so.2"};
    const NcclApproval approval{"2.28.7", "sha256:aaaa triangle-patch-3"};

    auto good = std::make_unique<FakeNccl>();
    FakeNccl* good_raw = good.get();
    NcclBackend good_backend(std::move(good), config, approval);
    check(!good_backend.init() && good_raw->init_calls == 1, "the approved build is accepted");
    check(good_raw->seen_bootstrap_id == "bootstrap-id-from-the-roster",
          "and the communicator was given the id the roster verified");

    const std::vector<std::pair<std::string, std::pair<std::string, FailureCode>>> wrong = {
        {"2.27.7", {"sha256:aaaa triangle-patch-3", FailureCode::library_mismatch}},
        {"2.28.7", {"sha256:bbbb other-triangle", FailureCode::library_mismatch}},
        {"", {"", FailureCode::library_mismatch}},
    };
    for (const auto& [version, expected] : wrong) {
        auto api = std::make_unique<FakeNccl>();
        FakeNccl* raw = api.get();
        raw->reported_version = version;
        raw->reported_provenance = expected.first;
        NcclBackend backend(std::move(api), config, approval);
        const Failure f = backend.init();
        check(f.code == expected.second && raw->init_calls == 0,
              "refused: version '" + version + "' provenance '" + expected.first + "'");
    }

    auto silent = std::make_unique<FakeNccl>();
    FakeNccl* silent_raw = silent.get();
    silent_raw->identity_failure = Failure::fail(FailureCode::comm_failed, "dlopen said no");
    NcclBackend silent_backend(std::move(silent), config, approval);
    check(silent_backend.init().code == FailureCode::library_mismatch && silent_raw->init_calls == 0,
          "a library that will not say what it is is refused by name, not by silence");

    NcclBackend no_approval(std::make_unique<FakeNccl>(), config, NcclApproval{"", ""});
    check(no_approval.init().code == FailureCode::library_mismatch,
          "a build with nothing approved approves nothing");

    NcclBackend no_bootstrap(std::make_unique<FakeNccl>(), NcclConfig{0, kTpWorld, "", "/lib"}, approval);
    check(no_bootstrap.init().code == FailureCode::invalid_config,
          "and a communicator needs the bootstrap id the session verified");

    NcclBackend wrong_rank(std::make_unique<FakeNccl>(), NcclConfig{7, kTpWorld, "id", "/lib"}, approval);
    check(wrong_rank.init().code == FailureCode::invalid_config, "a rank outside the world never gets a comm");

    auto failing_init = std::make_unique<FakeNccl>();
    FakeNccl* failing_raw = failing_init.get();
    failing_raw->init_failure = Failure::fail(FailureCode::comm_failed, "a peer never came");
    NcclBackend failing(std::move(failing_init), config, approval);
    const Failure f = failing.init();
    check(f.code == FailureCode::comm_failed && detail_has(f, "a peer never came"),
          "a communicator that did not form is reported as such");
    check(!failing.initialized(), "and the backend says it is not usable");
}

void backend_gates_every_call() {
    std::printf("the backend answers only when it has a live communicator\n");
    const NcclConfig config{1, kTpWorld, "bootstrap", "/lib"};
    const NcclApproval approval{"2.28.7", "sha256:aaaa triangle-patch-3"};  // what the fake library reports

    auto api = std::make_unique<FakeNccl>();
    FakeNccl* raw = api.get();
    NcclBackend backend(std::move(api), config, approval);

    check(backend.raw_all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), 4, nullptr)
              .code == FailureCode::not_ready,
          "an enqueue before init is refused");
    check(raw->reduce_calls == 0, "and never reached the library");

    check(!backend.init(), "init passes");
    check(!backend.raw_all_gather(buf_a(256, ScalarType::bf16), buf_b(768, ScalarType::bf16), 256, nullptr),
          "a padded gather is handed to the library");
    check(raw->gather_calls == 1 && raw->last_slot == 256 && raw->last_type == ScalarType::bf16,
          "with the slot count and the dtype the contract agreed");

    backend.raw_abort();
    check(raw->abort_calls == 1, "the communicator was abandoned");
    check(backend.raw_all_reduce_sum(buf_a(4, ScalarType::fp32), buf_b(4, ScalarType::fp32), 4, nullptr)
              .code == FailureCode::aborted,
          "and after that every call is refused");
    backend.raw_abort();
    check(raw->abort_calls == 1, "abandoning twice is one abandonment");
}

void contract_sits_above_the_library() {
    std::printf("the contract runs before the library is asked\n");
    FakeClock clock;
    auto api = std::make_unique<FakeNccl>();
    FakeNccl* raw = api.get();
    NcclBackend backend(std::move(api), NcclConfig{1, kTpWorld, "bootstrap", "/lib"},
                        NcclApproval{"2.28.7", "sha256:aaaa triangle-patch-3"});
    check(!backend.init(), "the backend is up");
    GuardedTransport transport(backend, clock);

    const std::array<uint64_t, 3> counts{256, 192, 192};
    const Failure bad = transport.all_gather_padded(buf_a(192, ScalarType::fp32), buf_b(640, ScalarType::fp32),
                                                   std::span<const uint64_t>(counts), op(1), nullptr);
    check(bad && raw->gather_calls == 0, "a bad padded gather is refused before NCCL is named");

    check(!transport.all_gather_padded(buf_a(256, ScalarType::fp32), buf_b(768, ScalarType::fp32),
                                      std::span<const uint64_t>(counts), op(1), nullptr),
          "the same call with the agreed shapes goes through");
    check(raw->gather_calls == 1 && raw->last_slot == 256, "and reaches it once, with the slot it agreed to");
    check(!transport.wait(Deadline{clock.now() + std::chrono::milliseconds(100)}), "the fake says it finished");

    raw->async_pending = true;
    raw->async = Failure::fail(FailureCode::async_error, "ncclAsyncTest failed");
    check(transport.check_health().code == FailureCode::async_error,
          "a late library error reaches the caller through health");
}

// ---------------------------------------------------------------- the small helpers the contract rests on

void helper_rules() {
    std::printf("the helpers the validators are built from\n");
    check(!ranges_overlap(pool_a(), 0, pool_a(), 8) && !ranges_overlap(pool_a(), 8, pool_a(), 0),
          "a zero-length range never overlaps anything, not even itself");
    check(!ranges_overlap(nullptr, 0, pool_a(), 8), "a null pointer with nothing behind it is not a collision");
    check(ranges_overlap(pool_a(), 8, pool_a(4), 8), "two ranges that share one byte overlap");

    uint64_t bytes = 0;
    check(!element_bytes(4, static_cast<ScalarType>(9), bytes), "an unknown type has no byte width");
    check(element_bytes(3, ScalarType::bf16, bytes) && bytes == 6, "bf16 is two bytes an element");
    check(!element_bytes(UINT64_MAX / 2 + 1, ScalarType::fp32, bytes), "a wrapped byte count is refused");

    bool all_named = true;
    for (uint32_t raw = 0; raw <= static_cast<uint32_t>(FailureCode::closed); ++raw) {
        const std::string name = failure_code_name(static_cast<FailureCode>(raw));
        if (name.empty() || name == "unrecognised_code") all_named = false;
    }
    check(all_named, "every failure code in the range has a name a log can show");
}

}  // namespace

int main() {
    // Nothing here should ever wait for a human.  If a case does, the alarm says which run died rather than
    // leaving the suite hanging in CI.
    alarm(120);
    std::printf("transport contract (CPU, no NCCL, no GPU)\n");
    rejects_before_enqueue();
    gather_contract();
    sequence_and_one_in_flight();
    completed_external_batch_advances_transport_cursor();
    enqueue_failure_is_final();
    pending_async_error_is_reported();
    deadline_expiry_names_the_operation();
    zero_poll_interval_preserves_completion_and_failure_checks();
    abort_is_idempotent();
    library_approval_is_checked_before_a_communicator();
    backend_gates_every_call();
    contract_sits_above_the_library();
    helper_rules();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
