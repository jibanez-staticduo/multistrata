// src/distributed/nccl_api.hpp - the one place this tree names NCCL, and the shape the device backend has to fit.
//
// Why a boundary at all: `nccl.h` is not present on the CPU build surface (this NAS has no NCCL headers), and the
// contract that decides whether a collective is legal - dtype, extent, aliasing, padded slots, one operation in
// flight, sequence order - must be testable without a GPU.  So the rules live above this line, in
// `GuardedTransport`, and everything below it is "ask the library".  A CPU test drives the same
// `NcclBackend` + `GuardedTransport` pair the device build uses, with the library replaced; what stays unproven
// here is only what NCCL itself does.
//
// What a production implementation (Task 3, `src/distributed/nccl_transport.cpp`) owes this interface:
//   * `identity()` reports the runtime library's own version and a provenance fingerprint of the resolved file
//     (its SHA-256 plus the patch-set identity).  A matching version number does not say the triangle patches
//     are present, so both are required and `NcclBackend::init()` compares them with the approved values before
//     any communicator exists.
//   * `comm_init()` takes the bootstrap id the session roster verified, and must not block forever: NCCL's
//     communicator init blocks on all ranks, so the caller's connect deadline needs an abort path that works
//     while init is still in progress.  A blocked init that cannot be aborted is a hang, not a slow start.
//   * Enqueue is not completion. `enqueue_*` returning success says the library accepted the work; only
//     `poll_completion()` says the device finished it.  `async_failure()` surfaces what NCCL reports late (a
//     peer that died mid-collective) and stays set once set - clearing it on read would let a rank report a
//     healthy session one step after its peer died.
//   * `comm_abort()` is what `GuardedTransport` calls the moment anything fails, including while a collective is
//     in flight.  It must be safe to call from a different thread than the enqueue, and safe to call twice.
//   * No blocking synchronization primitive without this deadline-carrying poll path behind it.
//   * The stream a collective is enqueued on is the caller's; the backend owns any event it needs to poll it.
//
// Nothing in this header includes `nccl.h`, so it compiles on the CPU surface.
#pragma once

#include "strata/distributed/transport.hpp"

#include <memory>
#include <atomic>
#include <string>
#include <string_view>
#include <utility>

namespace strata::distributed::nccl {

// What the communicator needs, all of it already verified by the session before a device byte is touched.
struct NcclConfig {
    uint32_t rank = 0;
    uint32_t world = kTpWorld;
    std::string bootstrap_id;  // rank 0's id, from the verified roster; never written to a log line
    std::string library_path;  // the libnccl.so.2 this rank resolved, for provenance
    uint32_t initialize_ms = 120000;  // bounded nonblocking initialization, relative to comm_init
};

// The library, as an interface.  One implementation loads the real NCCL; a test supplies another one.
class NcclApi {
 public:
    virtual ~NcclApi() = default;

    // What this library says it is.  Both strings are required: an unnamed library is an unknown library.
    virtual Failure identity(std::string& version, std::string& provenance) const = 0;

    // Build the communicator.  A failure here is fatal to the session, like any other.
    virtual Failure comm_init(const NcclConfig& config) = 0;

    // Enqueue only.  `elements` and `slot_elements` are the counts `GuardedTransport` already checked, and the
    // type is one of the two the transport accepts (fp32, bf16) - NCCL's Sum is defined for both.
    virtual Failure enqueue_all_reduce_sum(const void* send, void* recv, uint64_t elements, ScalarType type,
                                          cudaStream_t stream) = 0;
    virtual Failure enqueue_all_gather(const void* send, void* recv, uint64_t slot_elements, ScalarType type,
                                      cudaStream_t stream) = 0;

    // One look at the in-flight work: `none` when the device finished, `deadline_expired` when it has not,
    // anything else when it failed.
    virtual Failure poll_completion(Deadline deadline) = 0;

    // Opt-in production executor: enqueue a complete compute step on one stream
    // without a device wait between its reductions. The caller agrees the step
    // before entry and waits once after all work (including graph replay).
    // These calls do not replace the strict single-operation or graph APIs.
    virtual Failure begin_stream_step(cudaStream_t) {
        return Failure::fail(FailureCode::not_supported, "NCCL stream steps are not supported");
    }
    virtual Failure finish_stream_step(Deadline) {
        return Failure::fail(FailureCode::not_supported, "NCCL stream steps are not supported");
    }

    // Opt-in graph session. Prepare before beginning CUDA stream capture. Captured enqueues must all use
    // `stream` and exactly `expected_collectives`; they do not record or poll a CUDA event. Check and finish
    // perform NCCL host-side health queries only. After each graph launch, wait_captured_replay() records and
    // polls the completion event on the prepared stream. The owner must destroy every captured graph before
    // end_graph_session(), since a live graph may still reference this communicator. After begin_captured_replay,
    // any graph-launch failure must abort the communicator; abort reports process-exit when completion is unknown.
    // Implementations without graph support refuse these calls, leaving the eager interface unchanged.
    virtual Failure prepare_graph_session(cudaStream_t, uint64_t) {
        return Failure::fail(FailureCode::not_supported, "NCCL graph sessions are not supported");
    }
    virtual Failure check_graph_capture() {
        return Failure::fail(FailureCode::not_supported, "NCCL graph sessions are not supported");
    }
    virtual Failure finish_graph_capture() {
        return Failure::fail(FailureCode::not_supported, "NCCL graph sessions are not supported");
    }
    virtual Failure begin_captured_replay() {
        return Failure::fail(FailureCode::not_supported, "NCCL graph sessions are not supported");
    }
    virtual Failure wait_captured_replay(Deadline) {
        return Failure::fail(FailureCode::not_supported, "NCCL graph sessions are not supported");
    }
    virtual Failure end_graph_session() {
        return Failure::fail(FailureCode::not_supported, "NCCL graph sessions are not supported");
    }
    virtual bool requires_process_exit() const { return false; }

    // A late error the library knows about.  Sticky: reporting it must not clear it.
    virtual bool async_failure(Failure& out) const = 0;

    // Abandon the communicator, including from inside a blocked init.  Idempotent, never throws.
    virtual void comm_abort() noexcept = 0;
};

// The approved library, as the session recorded it.  Both must be non-empty: "any version, any build" is what a
// session without an approval step would accept, and the whole point of the check is not to accept that.
struct NcclApproval {
    std::string version;      // the NCCL version this build was validated against
    std::string provenance;   // that build's file hash plus its patch-set identity
};

// `Backend` over an `NcclApi`: the approval gate and the count/type handoff, with no rules of its own.
// Header-only because it needs no NCCL symbols, so a CPU test runs the same adapter the device build runs.
class NcclBackend final : public Backend {
 public:
    NcclBackend(std::unique_ptr<NcclApi> api, NcclConfig config, NcclApproval approval)
        : api_(std::move(api)), config_(std::move(config)), approval_(std::move(approval)) {}

    // The gate, then the communicator.  A library that is not the approved build is refused before a
    // communicator exists, so the refusal is a message rather than a deadlock between two ranks that did get one.
    Failure init() {
        if (api_ == nullptr) {
            return Failure::fail(FailureCode::invalid_config, "no NCCL API was given to the backend");
        }
        if (approval_.version.empty() || approval_.provenance.empty()) {
            return Failure::fail(FailureCode::library_mismatch,
                                 "this build names no approved NCCL version or library provenance");
        }
        if (config_.world != kTpWorld || config_.rank >= config_.world) {
            return Failure::fail(FailureCode::invalid_config,
                                 "the backend is configured for " + std::to_string(config_.world) +
                                     " ranks and rank " + std::to_string(config_.rank));
        }
        if (config_.bootstrap_id.empty()) {
            return Failure::fail(FailureCode::invalid_config,
                                 "the communicator has no bootstrap id from the verified roster");
        }
        std::string version;
        std::string provenance;
        if (auto f = api_->identity(version, provenance)) {
            return Failure::fail(FailureCode::library_mismatch, "the library would not say what it is: " + f.detail);
        }
        if (version.empty() || provenance.empty()) {
            return Failure::fail(FailureCode::library_mismatch,
                                 "the library reported an empty version or provenance");
        }
        if (version != approval_.version) {
            return Failure::fail(FailureCode::library_mismatch,
                                 "the runtime library is NCCL " + version + ", this build was approved against " +
                                     approval_.version);
        }
        if (provenance != approval_.provenance) {
            return Failure::fail(FailureCode::library_mismatch,
                                 "the library's provenance is '" + provenance + "'; the approved build is '" +
                                     approval_.provenance + "'");
        }
        if (auto f = api_->comm_init(config_)) {
            api_->comm_abort();
            return Failure::fail(f.code == FailureCode::none ? FailureCode::comm_failed : f.code,
                                 "the communicator did not form: " + f.detail);
        }
        if (aborted_) return Failure::fail(FailureCode::aborted, "the backend was abandoned during init");
        ready_ = true;
        return ok_failure();
    }

    uint32_t world_size() const override { return config_.world; }
    uint32_t local_rank() const override { return config_.rank; }

    Failure raw_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output, uint64_t elements,
                               cudaStream_t stream) override {
        if (auto f = usable()) return f;
        return api_->enqueue_all_reduce_sum(input.data, output.data, elements, input.type, stream);
    }

    Failure raw_all_gather(const DeviceBuffer& input, const DeviceBuffer& output, uint64_t slot_elements,
                           cudaStream_t stream) override {
        if (auto f = usable()) return f;
        return api_->enqueue_all_gather(input.data, output.data, slot_elements, input.type, stream);
    }

    Failure raw_poll(Deadline deadline) override {
        if (auto f = usable()) return f;
        return api_->poll_completion(deadline);
    }

    Failure prepare_graph_session(cudaStream_t stream, uint64_t expected_collectives) {
        if (auto f = usable()) return f;
        return api_->prepare_graph_session(stream, expected_collectives);
    }

    Failure check_graph_capture() {
        if (auto f = usable()) return f;
        return api_->check_graph_capture();
    }

    Failure finish_graph_capture() {
        if (auto f = usable()) return f;
        return api_->finish_graph_capture();
    }

    Failure begin_captured_replay() {
        if (auto f = usable()) return f;
        return api_->begin_captured_replay();
    }

    Failure wait_captured_replay(Deadline deadline) {
        if (auto f = usable()) return f;
        return api_->wait_captured_replay(deadline);
    }

    Failure end_graph_session() {
        if (api_ == nullptr) return Failure::fail(FailureCode::invalid_config, "no NCCL API");
        if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        if (!ready_) return Failure::fail(FailureCode::not_ready, "the communicator was never formed");
        return api_->end_graph_session();
    }

    bool requires_process_exit() const {
        return api_ != nullptr && api_->requires_process_exit();
    }
    Failure begin_stream_step(cudaStream_t stream) {
        if (auto f = usable()) return f;
        return api_->begin_stream_step(stream);
    }
    Failure finish_stream_step(Deadline deadline) {
        if (auto f = usable()) return f;
        return api_->finish_stream_step(deadline);
    }

    bool raw_async_failure(Failure& out) const override {
        if (api_ == nullptr) return false;
        return api_->async_failure(out);
    }

    void raw_abort() noexcept override {
        if (api_ == nullptr || aborted_.exchange(true)) return;
        ready_ = false;
        api_->comm_abort();
    }

    bool initialized() const { return ready_ && !aborted_; }

 private:
    Failure usable() const {
        if (api_ == nullptr) return Failure::fail(FailureCode::invalid_config, "no NCCL API");
        if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        if (!ready_) return Failure::fail(FailureCode::not_ready, "the communicator was never formed");
        return ok_failure();
    }

    std::unique_ptr<NcclApi> api_;
    NcclConfig config_;
    NcclApproval approval_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> aborted_{false};
};

// The real library, loaded the way the approved build says to load it.  Defined only in a build that has NCCL
// (Task 3, CUDA/ARM64); naming it in a CPU link is a build-configuration error, and it will read as one.
// The implementation resolves the file, opens it, and reports what it found through `NcclApi::identity()`.
bool make_nccl_bootstrap_id(const std::string& library_path, std::string& out, std::string& err);

std::unique_ptr<NcclApi> make_production_nccl_api(const NcclConfig& config, std::string& err);

}  // namespace strata::distributed::nccl
