// include/strata/distributed/transport.hpp - what a TP3 worker may ask the network to do: sum one buffer across
// three ranks, or gather three buffers of unequal valid length into uniform wire slots.  Nothing else.
//
// The shape comes from the transport plan (docs/superpowers/plans/2026-10-30-strata-tp3-transport.md Task 1) and
// from the padded all-gather the partition contract needs (docs/superpowers/specs/2026-10-30-strata-tp3-design.md,
// "Quantization boundaries"): unequal shards travel as slots sized to the maximum with explicit per-rank valid
// counts, never as unequal counts handed to a uniform collective.
//
// The rules this header enforces, and why:
//   * One ordered stream, one operation in flight.  A step issues collectives in one order on every rank; a
//     second enqueue before the first completes means the caller lost track of which collective it is on, and
//     a mismatched order deadlocks the other two ranks until their deadline fires.
//   * Everything checkable is checked before the enqueue: dtype, extent, pointer validity, byte-count overflow,
//     aliasing and the gather counts.  A rejected call touches no wire and consumes no sequence number, so it
//     is a caller bug found at the caller, not a three-way deadlock found by timeout.
//   * Aliasing: exact in-place (same pointer, same extent, same dtype) or fully disjoint.  Partial overlap is
//     rejected: with an in-flight read and a write into the same bytes the result depends on NCCL's internal
//     chunking, which is not a contract anyone wants to depend on.
//   * A collective that fails is fatal to the session.  There is no retry path here, because a rank that
//     retried alone would be one collective out of step with its peers.  After any enqueue failure, any async
//     error or any expired deadline the transport is dead until it is destroyed; only `abort()` and
//     destruction change that, and both are idempotent.
//   * A failure carries a code and a sentence.  `Failure` is returned, not written to an out-parameter, so a
//     caller that ignores the result still has to look at it to get the reason out.
//
// Deviations from the sketch in the plan, all deliberate:
//   * `Operation` passes by const reference (it is three numbers but a struct; by value would copy it for no
//     reason and by reference says "we only read it").
//   * The void returns became `Failure`.  The plan's `void wait(Deadline)` and `void check_health()` cannot
//     report why they are unhappy, and "why" is the whole point when three ranks disagree.
//   * `check_health()` is const: asking whether a transport is usable must not change it.
//
// CPU-only build surface: this header never includes a CUDA or NCCL header.  `cudaStream_t` is the plan's type
// name, and the device backend (Task 3) passes a real stream through it, so the name is kept and given a
// stand-in when the CUDA headers are not part of the build (see STRATA_DISTRIBUTED_CUDA below).  Nothing here
// dereferences a stream, so the stand-in cannot be mistaken for something usable: it only keeps the signature
// identical in both builds, which is what makes a CPU contract test meaningful.
#pragma once

#if defined(STRATA_DISTRIBUTED_CUDA)
// The device build has the CUDA runtime; use its real type so the backend's overrides match.
#include <cuda_runtime.h>
#else
// No CUDA runtime on this build surface (the NAS builds CPU-only and has no nccl.h either).  An opaque
// pointer type stands in for cudaStream_t: the CPU contract never uses the stream's value, and the type is
// still distinct from every integer and pointer the tests pass by accident.  A CUDA stream cannot convert to
// it, so a CPU TU can never be handed a real stream by mistake.
using cudaStream_t = struct StrataCpuOnlyStream*;
#endif

#include <cstdint>
#include <chrono>
#include <span>
#include <string>
#include <string_view>
#include <thread>

namespace strata::distributed {

// TP3 is three ranks.  The session refuses any other world size; the transport validators stay generic so a
// later two-rank experiment does not need a second copy of the aliasing rules.
inline constexpr uint32_t kTpWorld = 3;

// The only element types the first transport accepts.  The plan's numerical gate is FP32-exact and BF16 within
// declared tolerances, and NCCL's Sum is defined for both.
enum class ScalarType : uint32_t { fp32 = 0, bf16 = 1 };

// Bytes per element, or 0 for a value that is not a supported type.  Callers must not assume 4 or 2: an
// out-of-range enum (a torn config, a bad cast) reads as 0 and is refused by the validators.
constexpr uint64_t scalar_type_bytes(ScalarType type) {
    switch (type) {
        case ScalarType::fp32: return 4;
        case ScalarType::bf16: return 2;
    }
    return 0;
}

// One device buffer.  The caller owns `data` and the stream until the operation completes; the transport owns
// the communicator, the control sockets and any completion event.  `data` is read-only for an input and
// written once for an output; nothing here keeps a reference past a completed `wait()`.
struct DeviceBuffer {
    void* data = nullptr;
    uint64_t elements = 0;
    ScalarType type = ScalarType::fp32;
};

// Which collective this is, in the model's own terms.  The transport does not interpret it: it requires the
// sequence to advance by one and records the rest so a rank that deadlocks can say which layer and step it was
// waiting on.  That is the difference between "layer 31 down-projection sum timed out" and "a collective timed
// out" in the log.
struct Operation {
    uint64_t sequence = 0;
    uint64_t layer = 0;
    uint64_t step = 0;
};

// When to stop waiting.  A steady_clock point: monotonic on every host, and never compared across hosts (the
// control protocol exchanges durations, not machine timestamps).
struct Deadline {
    std::chrono::steady_clock::time_point end{};
};

// Read-only monotonic clock.  Injected so a contract test can expire a deadline on the spot instead of
// sleeping, and so one test binary can prove "no case waits indefinitely" rather than hoping.
class Clock {
 public:
    virtual ~Clock() = default;
    virtual std::chrono::steady_clock::time_point now() const = 0;
};

// The clock a production transport uses.
class SystemClock final : public Clock {
 public:
    std::chrono::steady_clock::time_point now() const override {
        return std::chrono::steady_clock::now();
    }
};

inline bool deadline_passed(const Clock& clock, Deadline at) {
    return !(clock.now() < at.end);
}

// Every way this transport can be unhappy.  `none` means "no failure", so a Failure with code `none` is the
// ok value and a function can return one instead of a bool plus an out-string.
enum class FailureCode : uint32_t {
    none = 0,
    // rejected before the enqueue: the call is wrong, the wire and the sequence are untouched
    bad_dtype,
    bad_extent,
    bad_pointer,
    overflow,
    overlap,
    bad_counts,
    bad_operation,
    in_flight,
    invalid_config,
    not_ready,
    // data plane after a successful enqueue: any of these kills the session
    enqueue_failed,
    async_error,
    deadline_expired,
    comm_failed,
    library_mismatch,
    aborted,
    // control plane
    bad_frame,
    oversized_frame,
    truncated_frame,
    protocol_version,
    unknown_frame,
    credential,
    rank_range,
    duplicate_rank,
    missing_rank,
    uuid_mismatch,
    stale_join,
    model_mismatch,
    config_mismatch,
    backend_mismatch,
    nccl_version_mismatch,
    provenance_mismatch,
    roster_mismatch,
    peer_closed,
    connect_failed,
    listen_failed,
    not_supported,
    closed,
};

// A failure with a code and one sentence about it, or nothing at all.
struct Failure {
    FailureCode code = FailureCode::none;
    std::string detail;

    explicit operator bool() const { return code != FailureCode::none; }

    static Failure fail(FailureCode code, std::string_view detail) {
        Failure f;
        f.code = code;
        f.detail.assign(detail);
        return f;
    }
};

inline Failure ok_failure() { return Failure{}; }

// The name of a failure code, for a log line or a test report.  A new code has to be named here, which is the
// point: a failure nobody can name is a failure nobody can act on.
inline const char* failure_code_name(FailureCode code) {
    switch (code) {
        case FailureCode::none: return "none";
        case FailureCode::bad_dtype: return "bad_dtype";
        case FailureCode::bad_extent: return "bad_extent";
        case FailureCode::bad_pointer: return "bad_pointer";
        case FailureCode::overflow: return "overflow";
        case FailureCode::overlap: return "overlap";
        case FailureCode::bad_counts: return "bad_counts";
        case FailureCode::bad_operation: return "bad_operation";
        case FailureCode::in_flight: return "in_flight";
        case FailureCode::invalid_config: return "invalid_config";
        case FailureCode::not_ready: return "not_ready";
        case FailureCode::enqueue_failed: return "enqueue_failed";
        case FailureCode::async_error: return "async_error";
        case FailureCode::deadline_expired: return "deadline_expired";
        case FailureCode::comm_failed: return "comm_failed";
        case FailureCode::library_mismatch: return "library_mismatch";
        case FailureCode::aborted: return "aborted";
        case FailureCode::bad_frame: return "bad_frame";
        case FailureCode::oversized_frame: return "oversized_frame";
        case FailureCode::truncated_frame: return "truncated_frame";
        case FailureCode::protocol_version: return "protocol_version";
        case FailureCode::unknown_frame: return "unknown_frame";
        case FailureCode::credential: return "credential";
        case FailureCode::rank_range: return "rank_range";
        case FailureCode::duplicate_rank: return "duplicate_rank";
        case FailureCode::missing_rank: return "missing_rank";
        case FailureCode::uuid_mismatch: return "uuid_mismatch";
        case FailureCode::stale_join: return "stale_join";
        case FailureCode::model_mismatch: return "model_mismatch";
        case FailureCode::config_mismatch: return "config_mismatch";
        case FailureCode::backend_mismatch: return "backend_mismatch";
        case FailureCode::nccl_version_mismatch: return "nccl_version_mismatch";
        case FailureCode::provenance_mismatch: return "provenance_mismatch";
        case FailureCode::roster_mismatch: return "roster_mismatch";
        case FailureCode::peer_closed: return "peer_closed";
        case FailureCode::connect_failed: return "connect_failed";
        case FailureCode::listen_failed: return "listen_failed";
        case FailureCode::not_supported: return "not_supported";
        case FailureCode::closed: return "closed";
    }
    return "unrecognised_code";
}

// Overflow-safe size in bytes.  Returns false instead of wrapping: a collective whose byte count wrapped
// would ask the network for a few bytes and write into a buffer the caller sized for billions.
inline bool element_bytes(uint64_t elements, ScalarType type, uint64_t& out) {
    const uint64_t per = scalar_type_bytes(type);
    if (per == 0) return false;
    if (elements > UINT64_MAX / per) return false;
    out = elements * per;
    return true;
}

// Two half-open device ranges intersect.  Zero-length ranges never overlap (a null pointer with no elements
// is a legitimate "nothing to send", not a collision).
inline bool ranges_overlap(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0 || a == nullptr || b == nullptr) return false;
    const auto pa = reinterpret_cast<uintptr_t>(a);
    const auto pb = reinterpret_cast<uintptr_t>(b);
    return pa < pb + b_bytes && pb < pa + a_bytes;
}

// A buffer a collective may use: supported dtype, and a usable pointer at a sane offset for that dtype.
inline Failure validate_buffer(const DeviceBuffer& buffer, std::string_view what) {
    const uint64_t per = scalar_type_bytes(buffer.type);
    if (per == 0) return Failure::fail(FailureCode::bad_dtype, std::string(what) + ": unsupported scalar type");
    uint64_t bytes = 0;
    if (!element_bytes(buffer.elements, buffer.type, bytes)) {
        return Failure::fail(FailureCode::overflow, std::string(what) + ": element count overflows a byte count");
    }
    if (buffer.elements > 0 && buffer.data == nullptr) {
        return Failure::fail(FailureCode::bad_pointer, std::string(what) + ": non-empty buffer with no address");
    }
    if (buffer.data != nullptr && (reinterpret_cast<uintptr_t>(buffer.data) % per) != 0) {
        // A pointer that is not on an element boundary is not the buffer the plan describes: it starts half an
        // element into it, so every element would straddle two reads.
        return Failure::fail(FailureCode::bad_pointer, std::string(what) + ": address is not on an element edge");
    }
    return ok_failure();
}

// What an all-reduce sum of `count` elements needs checked.
struct AllReducePlan {
    uint64_t elements = 0;
    uint64_t bytes = 0;
    bool in_place = false;
};

// Check an all-reduce sum before it goes anywhere.  Equal extents are required in both the in-place and the
// disjoint case (a sum of two buffers of different length is not one operation), the dtypes must match, and
// the ranges must either be the same buffer or not touch at all.
inline Failure validate_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output,
                                       AllReducePlan& plan) {
    if (input.type != output.type) {
        return Failure::fail(FailureCode::bad_dtype, "all_reduce_sum: input and output are different types");
    }
    if (input.elements != output.elements) {
        return Failure::fail(FailureCode::bad_extent, "all_reduce_sum: input and output hold different counts");
    }
    if (auto f = validate_buffer(input, "all_reduce_sum: input")) return f;
    if (auto f = validate_buffer(output, "all_reduce_sum: output")) return f;

    uint64_t bytes = 0;
    if (!element_bytes(input.elements, input.type, bytes)) {
        return Failure::fail(FailureCode::overflow, "all_reduce_sum: byte count overflows");
    }
    const bool same_address = input.data == output.data;
    if (!same_address && ranges_overlap(input.data, bytes, output.data, bytes)) {
        return Failure::fail(FailureCode::overlap,
                            "all_reduce_sum: input and output partially overlap; use exact in-place or disjoint buffers");
    }
    plan.elements = input.elements;
    plan.bytes = bytes;
    plan.in_place = same_address;
    return ok_failure();
}

// What a padded all-gather agreed to put on the wire.  `slot_elements` is the uniform slot (the maximum valid
// count); `wire_elements` is what the output buffer must hold exactly.
struct PaddedGatherPlan {
    uint64_t slot_elements = 0;
    uint64_t wire_elements = 0;
};

// Check an ordered all-gather with equal padded slots and explicit valid counts.
//
// The counts are the contract: rank i's contribution is `valid_counts[i]` elements of a slot sized to
// `max(valid_counts)`, so every rank pays for the same wire shape and the padding stays transport-only.  Three
// rejections here are the ones that matter:
//   * A count list that does not describe the world, or a local count larger than the buffer passed.
//   * A local buffer sized to the valid count instead of to the slot.  A uniform collective reads one whole slot
//     from every rank, so a 192-element buffer behind a 256-element slot would have the network read past its
//     end.  The pad is transported and never read by anything that matters, so it must be addressable, not
//     initialised.
//   * An output sized by summing the counts (640 for 256/192/192) instead of by slots (768).  That is unequal
//     counts smuggled into a uniform collective, and it silently loses or invents elements.
// All-gather is never in place: the output contains the input, so any overlap is a self-overwrite.
inline Failure validate_all_gather_padded(const DeviceBuffer& input, const DeviceBuffer& output,
                                          std::span<const uint64_t> valid_counts, uint32_t world,
                                          uint32_t local_rank, PaddedGatherPlan& plan) {
    if (world == 0 || local_rank >= world) {
        return Failure::fail(FailureCode::invalid_config, "all_gather_padded: rank is not inside the world");
    }
    if (valid_counts.size() != world) {
        return Failure::fail(FailureCode::bad_counts,
                            "all_gather_padded: one valid count per rank or none of the contract holds");
    }
    if (input.type != output.type) {
        return Failure::fail(FailureCode::bad_dtype, "all_gather_padded: input and output are different types");
    }
    if (output.elements == 0) {
        return Failure::fail(FailureCode::bad_extent, "all_gather_padded: output is empty");
    }
    if (auto f = validate_buffer(input, "all_gather_padded: input")) return f;
    if (auto f = validate_buffer(output, "all_gather_padded: output")) return f;

    uint64_t slot = 0;
    for (size_t i = 0; i < valid_counts.size(); ++i) {
        const uint64_t c = valid_counts[i];
        if (c > UINT64_MAX / world) {
            return Failure::fail(FailureCode::overflow, "all_gather_padded: a valid count overflows the wire");
        }
        if (c > slot) slot = c;
    }
    const uint64_t wire = slot * world;
    if (valid_counts[local_rank] > input.elements) {
        return Failure::fail(FailureCode::bad_counts,
                            "all_gather_padded: this rank's valid count does not fit the buffer passed");
    }
    if (input.elements != slot) {
        return Failure::fail(FailureCode::bad_extent,
                            "all_gather_padded: the local buffer is not one whole wire slot; a uniform "
                            "collective reads max(valid_counts) elements from every rank");
    }
    if (output.elements != wire) {
        return Failure::fail(FailureCode::bad_counts,
                            "all_gather_padded: output is not world slots of the maximum valid count");
    }
    uint64_t wire_bytes = 0;
    if (!element_bytes(wire, output.type, wire_bytes)) {
        return Failure::fail(FailureCode::overflow, "all_gather_padded: wire byte count overflows");
    }
    uint64_t in_bytes = 0;
    if (!element_bytes(slot, input.type, in_bytes)) {
        return Failure::fail(FailureCode::overflow, "all_gather_padded: input byte count overflows");
    }
    if (ranges_overlap(input.data, in_bytes, output.data, wire_bytes)) {
        return Failure::fail(FailureCode::overlap,
                            "all_gather_padded: input overlaps the output; an all-gather is never in place");
    }
    plan.slot_elements = slot;
    plan.wire_elements = wire;
    return ok_failure();
}

// The ordered-collective bookkeeping: sequence must advance by exactly one, and only one operation may be in
// flight.  Used by the transport for the data plane and by the session for its rank-0 ordering check, so a
// step that issues collectives in a different order than it did on the previous step is caught by the rank
// that can see all of them.
class OperationTracker {
 public:
    explicit OperationTracker(uint64_t first_sequence = 1) : next_(first_sequence) {}

    // Accept one operation, or say why it is out of order.  A rejected operation consumes nothing.
    Failure accept(const Operation& op) {
        if (in_flight_) {
            return Failure::fail(FailureCode::in_flight,
                                 "operation " + std::to_string(current_.sequence) +
                                     " is still in flight; one collective at a time");
        }
        if (op.sequence != next_) {
            return Failure::fail(FailureCode::bad_operation,
                                 "operation sequence " + std::to_string(op.sequence) + ", expected " +
                                     std::to_string(next_));
        }
        current_ = op;
        in_flight_ = true;
        return ok_failure();
    }

    // The in-flight operation is over (it completed, or its failure killed the session).  The sequence still
    // advances: a collective that failed was still issued, and the session is dead either way.
    void complete() {
        if (!in_flight_) return;
        in_flight_ = false;
        ++next_;
        last_ = current_;
    }

    bool in_flight() const { return in_flight_; }
    uint64_t next_sequence() const { return next_; }
    Operation current() const { return current_; }
    Operation last() const { return last_; }

 private:
    uint64_t next_ = 1;
    bool in_flight_ = false;
    Operation current_{};
    Operation last_{};
};

// The minimal raw surface a device backend offers.  It is deliberately smaller than `Transport`: no
// validation, no sequencing, no health.  `GuardedTransport` owns those rules so the CPU contract tests and the
// device backend exercise the same code path, and a backend only has to talk to the library.
class Backend {
 public:
    virtual ~Backend() = default;

    virtual uint32_t world_size() const = 0;
    virtual uint32_t local_rank() const = 0;

    // Enqueue only: return as soon as the library accepted the work.  Success here is not completion.
    virtual Failure raw_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output,
                                       uint64_t elements, cudaStream_t stream) = 0;
    virtual Failure raw_all_gather(const DeviceBuffer& input, const DeviceBuffer& output,
                                  uint64_t slot_elements, cudaStream_t stream) = 0;

    // One look at the in-flight operation.  `none`: the work is done.  `deadline_expired`: not yet, look
    // again.  Anything else: it failed, and the session is over.
    virtual Failure raw_poll(Deadline deadline) = 0;

    // A failure the library reports asynchronously (a peer that died mid-collective).  Sticky on success:
    // polling health must not clear it.
    virtual bool raw_async_failure(Failure& out) const = 0;

    // Abandon the communicator.  Idempotent; never throws.
    virtual void raw_abort() noexcept = 0;
};

// The transport the worker calls.  It is the only place the contract rules live: `GuardedTransport` checks
// them, tracks the sequence, keeps one operation in flight, and turns every failure into a sticky, reported
// session death.
class Transport {
 public:
    virtual ~Transport() = default;

    // Sum `input` across the world into `output`.  Exact in-place or disjoint buffers.
    virtual Failure all_reduce_sum(DeviceBuffer input, DeviceBuffer output, const Operation& operation,
                                   cudaStream_t stream) = 0;

    // Gather each rank's input into uniform slots of max(valid_counts), in rank order, with the padding
    // excluded from anything the model reads.
    virtual Failure all_gather_padded(DeviceBuffer input, DeviceBuffer output,
                                      std::span<const uint64_t> valid_counts, const Operation& operation,
                                      cudaStream_t stream) = 0;

    // Block until the in-flight operation completes or `at` passes.  Completion and failure are reported by
    // the returned Failure and stay in `check_health()`.
    virtual Failure wait(Deadline at) = 0;

    // Whether the next collective may be issued: no sticky failure and nothing in flight.
    virtual Failure check_health() const = 0;

    // End the session.  Idempotent; the first reason wins.  Never throws, because the caller is usually
    // already unwinding a failure.
    virtual void abort(std::string_view reason) noexcept = 0;
};

// The transport used by both build surfaces: the CPU contract tests drive it over a fake backend, and the
// device build drives it over the NCCL backend.  Single-threaded by design (one worker thread, one ordered
// stream, one operation in flight), as the design document's runtime section requires.
class GuardedTransport final : public Transport {
 public:
    GuardedTransport(Backend& backend, const Clock& clock) : backend_(backend), clock_(clock) {}

    Failure all_reduce_sum(DeviceBuffer input, DeviceBuffer output, const Operation& operation,
                           cudaStream_t stream) override;
    Failure all_gather_padded(DeviceBuffer input, DeviceBuffer output,
                              std::span<const uint64_t> valid_counts, const Operation& operation,
                              cudaStream_t stream) override;
    Failure wait(Deadline at) override;
    Failure check_health() const override;
    void abort(std::string_view reason) noexcept override;

    // How long a poll waits before looking again, and how long `wait` may poll at all.  The poll interval is
    // a real sleep between looks at the library, so it is only used when the backend reports "not yet".
    void set_poll_interval_ms(uint32_t ms) { poll_interval_ms_ = ms == 0 ? 1u : ms; }

 private:
    Failure health_unlocked() const;
    void kill(Failure f);

    Backend& backend_;
    const Clock& clock_;
    OperationTracker tracker_{};
    Failure failure_{};
    bool aborted_ = false;
    uint32_t poll_interval_ms_ = 1u;
};

inline Failure GuardedTransport::all_reduce_sum(DeviceBuffer input, DeviceBuffer output,
                                                const Operation& operation, cudaStream_t stream) {
    if (auto f = check_health()) return f;

    AllReducePlan plan;
    if (auto f = validate_all_reduce_sum(input, output, plan)) return f;
    if (auto f = tracker_.accept(operation)) return f;

    const DeviceBuffer fixed_in{input.data, plan.elements, input.type};
    const DeviceBuffer fixed_out{output.data, plan.elements, output.type};
    if (auto f = backend_.raw_all_reduce_sum(fixed_in, fixed_out, plan.elements, stream)) {
        tracker_.complete();
        kill(f.code == FailureCode::none ? Failure::fail(FailureCode::enqueue_failed,
                                                         "all_reduce_sum: the library refused the enqueue")
                                         : f);
        return failure_;
    }
    return ok_failure();
}

inline Failure GuardedTransport::all_gather_padded(DeviceBuffer input, DeviceBuffer output,
                                                  std::span<const uint64_t> valid_counts,
                                                  const Operation& operation, cudaStream_t stream) {
    if (auto f = check_health()) return f;

    PaddedGatherPlan plan;
    if (auto f = validate_all_gather_padded(input, output, valid_counts, backend_.world_size(),
                                            backend_.local_rank(), plan)) {
        return f;
    }
    if (auto f = tracker_.accept(operation)) return f;

    const DeviceBuffer fixed_out{output.data, plan.wire_elements, output.type};
    if (auto f = backend_.raw_all_gather(input, fixed_out, plan.slot_elements, stream)) {
        tracker_.complete();
        kill(f.code == FailureCode::none ? Failure::fail(FailureCode::enqueue_failed,
                                                         "all_gather_padded: the library refused the enqueue")
                                         : f);
        return failure_;
    }
    return ok_failure();
}

inline Failure GuardedTransport::wait(Deadline at) {
    if (failure_) return failure_;
    if (!tracker_.in_flight()) return ok_failure();

    for (;;) {
        Failure pending;
        if (backend_.raw_async_failure(pending)) {
            tracker_.complete();
            kill(pending.code == FailureCode::none
                     ? Failure::fail(FailureCode::async_error, "an asynchronous library error is pending")
                     : pending);
            return failure_;
        }
        const Failure polled = backend_.raw_poll(at);
        if (polled.code == FailureCode::none) {
            tracker_.complete();
            return ok_failure();
        }
        if (polled.code != FailureCode::deadline_expired) {
            tracker_.complete();
            kill(polled);
            return failure_;
        }
        // The backend says "not yet".  Trust the clock over the library: a deadline that has passed is
        // expired no matter what a poll claims next, and an expired deadline ends the session.
        if (deadline_passed(clock_, at)) {
            tracker_.complete();
            kill(Failure::fail(FailureCode::deadline_expired,
                               "operation " + std::to_string(tracker_.last().sequence) +
                                   " (layer " + std::to_string(tracker_.last().layer) + ", step " +
                                   std::to_string(tracker_.last().step) + ") did not complete before its deadline"));
            return failure_;
        }
        if (poll_interval_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms_));
        }
    }
}

inline Failure GuardedTransport::check_health() const { return health_unlocked(); }

inline Failure GuardedTransport::health_unlocked() const {
    if (failure_) return failure_;
    Failure pending;
    if (backend_.raw_async_failure(pending)) {
        return pending.code == FailureCode::none
                   ? Failure::fail(FailureCode::async_error, "an asynchronous library error is pending")
                   : pending;
    }
    if (tracker_.in_flight()) {
        return Failure::fail(FailureCode::in_flight,
                             "operation " + std::to_string(tracker_.current().sequence) + " is still in flight");
    }
    return ok_failure();
}

inline void GuardedTransport::abort(std::string_view reason) noexcept {
    if (aborted_) return;
    aborted_ = true;
    tracker_.complete();
    failure_ = Failure::fail(FailureCode::aborted, std::string(reason.empty() ? "aborted" : reason));
    backend_.raw_abort();
}

inline void GuardedTransport::kill(Failure f) {
    if (aborted_) return;
    aborted_ = true;
    failure_ = f;
    backend_.raw_abort();
}

}  // namespace strata::distributed
