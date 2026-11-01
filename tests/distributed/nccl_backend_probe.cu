// tests/distributed/nccl_backend_probe.cu - compile the NCCL backend the way the device build compiles it.
//
// Why this exists.  Until now every test in `tests/distributed/` was built by g++ on the CPU surface, where
// `cudaStream_t` is a stand-in pointer type and no NCCL symbol exists.  That proves the contract; it proves
// nothing about the path that passes a real stream to a real library, because nothing has ever compiled that
// path.  This file is that path: it includes the production translation unit and is built by nvcc with
// `-DSTRATA_DISTRIBUTED_CUDA` and the pinned NCCL headers, so the collective calls, the CUDA event calls and
// the `ncclConfig_t` use are checked by a CUDA compiler against the pinned header.
//
// What it checks at compile time (a wrong signature here is a build failure, not a test failure):
//   * `cudaStream_t` is the real CUDA stream type in this build surface, so the backend's overrides match the
//     contract only when CUDA is really present;
//   * the boundary's enqueue signatures take exactly that type;
//   * the two element types the contract allows are the two NCCL names for them (checked at run time, below,
//     because `ncclDataType_t` values are not constexpr-visible to the contract's enum).
// What it checks at run time, on the host that runs it: the bootstrap id encoding round-trips, the version
// decoder says what the pinned header means, and the SHA-256 used for library provenance matches the published
// FIPS 180-4 vectors.  A provenance gate whose hash is wrong refuses every library, so the hash gets vectors.
//
// Build (compile only, no GPU work, no run - this is what the NAS can prove):
//   /usr/local/cuda-12.8/bin/nvcc -std=c++20 -I include -I src -I <nccl-headers> -DSTRATA_DISTRIBUTED_CUDA -gencode arch=compute_86,code=sm_86 -c tests/distributed/nccl_backend_probe.cu -o <evidence>/probe.o
// On a Spark, the same translation unit is built into `strata-tp3-nccl-probe` and run before the smoke test.
//
// CMake lines for the parent integrator (not applied here; the root CMakeLists.txt belongs to the integrator):
//   add_executable(strata-tp3-nccl-probe tests/distributed/nccl_backend_probe.cu)
//   target_compile_definitions(strata-tp3-nccl-probe PRIVATE STRATA_DISTRIBUTED_CUDA)
//   target_include_directories(strata-tp3-nccl-probe PRIVATE include src ${STRATA_NCCL_INCLUDE_DIR})
//   target_link_libraries(strata-tp3-nccl-probe PRIVATE ${CMAKE_DL_LIBS})
//   add_test(NAME distributed_nccl_probe COMMAND strata-tp3-nccl-probe)
//   set_tests_properties(distributed_nccl_probe PROPERTIES LABELS "distributed;cuda")
// The NCCL library itself is not linked: `make_production_nccl_api` resolves it with `dlopen`, which is what
// lets a CPU build carry this contract without the library.  The probe fails loudly, and reports why, when no
// library is there to load.
#include "distributed/nccl_transport.cpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>

namespace {

using namespace strata::distributed;
using namespace strata::distributed::nccl;

// The stream type must be CUDA's own here.  If this fires, the file was compiled without
// `-DSTRATA_DISTRIBUTED_CUDA` and the probe is proving the wrong thing.
static_assert(std::is_same_v<cudaStream_t, CUstream_st*>,
              "the probe must be built with the real CUDA stream type");

// The enqueue signatures of the boundary, checked against the same type.
static_assert(std::is_same_v<decltype(&NcclApi::enqueue_all_reduce_sum),
                             Failure (NcclApi::*)(const void*, void*, uint64_t, ScalarType, cudaStream_t)>,
              "all_reduce_sum does not take a real cudaStream_t");
static_assert(std::is_same_v<decltype(&NcclApi::enqueue_all_gather),
                             Failure (NcclApi::*)(const void*, void*, uint64_t, ScalarType, cudaStream_t)>,
              "all_gather does not take a real cudaStream_t");
static_assert(std::is_same_v<decltype(&NcclApi::begin_captured_replay),
                             Failure (NcclApi::*)()>,
              "captured replay must be explicitly declared before launch");
static_assert(std::is_same_v<decltype(&NcclApi::wait_captured_replay),
                             Failure (NcclApi::*)(Deadline)>,
              "captured replay completion must carry a deadline");

// The contract's one-in-flight rule reaches the library through `Backend`, so the adapter must be a `Backend`
// whose raw calls carry a real stream.
static_assert(std::is_same_v<decltype(&Backend::raw_all_reduce_sum),
                             Failure (Backend::*)(const DeviceBuffer&, const DeviceBuffer&, uint64_t,
                                                  cudaStream_t)>,
              "Backend::raw_all_reduce_sum lost its stream");

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failed;
}

namespace fake {
std::atomic<int> init_calls{0}, polls{0}, aborts{0}, records{0}, reduces{0}, gathers{0}, stream_checks{0};
std::atomic<ncclResult_t> state{ncclSuccess};
std::atomic<int> enqueue_pending_polls{0};
int pending_polls = 0;
cudaError_t event_state = cudaErrorNotReady;
cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusActive;
cudaError_t capture_query = cudaSuccess;
void reset() {
    init_calls = 0; polls = 0; aborts = 0; records = 0; reduces = 0; gathers = 0; stream_checks = 0;
    state = ncclSuccess; pending_polls = 0; enqueue_pending_polls = 0; event_state = cudaErrorNotReady;
    capture_status = cudaStreamCaptureStatusActive; capture_query = cudaSuccess;
}
ncclResult_t init(ncclComm_t* comm, int, ncclUniqueId, int, ncclConfig_t* cfg) {
    ++init_calls;
    *comm = reinterpret_cast<ncclComm_t>(uintptr_t{0x10});
    return cfg->blocking == 0 ? ncclInProgress : ncclInvalidUsage;
}
ncclResult_t async(ncclComm_t, ncclResult_t* out) {
    const int n = ++polls;
    int remaining = enqueue_pending_polls.load();
    while (remaining > 0 && !enqueue_pending_polls.compare_exchange_weak(remaining, remaining - 1)) {}
    *out = remaining > 0 || n <= pending_polls ? ncclInProgress : state.load();
    return ncclSuccess;
}
ncclResult_t count(ncclComm_t, int* out) { *out = 3; return ncclSuccess; }
ncclResult_t abort(ncclComm_t) { ++aborts; return ncclSuccess; }
ncclResult_t reduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) {
    ++reduces;
    enqueue_pending_polls = 2;
    return ncclInProgress;
}
ncclResult_t gather(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
    ++gathers;
    enqueue_pending_polls = 2;
    return ncclInProgress;
}
cudaError_t create(cudaEvent_t* event, unsigned) {
    *event = reinterpret_cast<cudaEvent_t>(uintptr_t{0x20}); return cudaSuccess;
}
cudaError_t record(cudaEvent_t, cudaStream_t) {
    ++records;
    return cudaSuccess;
}
cudaError_t query(cudaEvent_t) { return event_state; }
cudaError_t destroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t is_capturing(cudaStream_t, cudaStreamCaptureStatus* out) {
    ++stream_checks;
    *out = capture_status;
    return capture_query;
}
detail::NcclSymbols symbols() {
    detail::NcclSymbols out;
    out.comm_init_rank_config = &init; out.comm_get_async_error = &async;
    out.comm_count = &count; out.comm_abort = &abort;
    out.all_reduce = &reduce; out.all_gather = &gather;
    return out;
}
detail::CudaCalls cuda() { return {&create, &record, &query, &destroy, &is_capturing}; }
NcclConfig config() {
    NcclConfig out;
    out.bootstrap_id.assign(NCCL_UNIQUE_ID_BYTES * 2, 'a');
    out.initialize_ms = 100;
    return out;
}
}

void check_nonblocking_lifecycle() {
    fake::reset();
    fake::pending_polls = 3;
    auto config = fake::config();
    {
        ProductionNcclApi api(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
        check(!api.comm_init(config) && fake::init_calls == 1 && fake::polls == 4,
              "nonblocking init is called once, then only async state is polled");
        const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x30});
        float input = 1, output = 0;
        fake::state = ncclInProgress;
        check(!api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) && fake::records == 0,
              "NCCL background enqueue does not record an event prematurely");
        check(api.enqueue_all_gather(&input, &output, 1, ScalarType::fp32, stream).code == FailureCode::in_flight,
              "the production backend refuses a second collective in flight");
        const Deadline at{std::chrono::steady_clock::now() + std::chrono::seconds(1)};
        check(api.poll_completion(at).code == FailureCode::deadline_expired && fake::records == 0,
              "poll cannot complete or record while NCCL is enqueueing");
        fake::state = ncclSuccess;
        fake::event_state = cudaSuccess;
        // The fake enqueue itself stays pending for two async queries.
        while (fake::enqueue_pending_polls > 0) (void)api.poll_completion(at);
        check(!api.poll_completion(at) && fake::records == 1,
              "the completion event is recorded only after background enqueue succeeds");
        fake::state = ncclInProgress;
        check(!api.enqueue_all_gather(&input, &output, 1, ScalarType::fp32, stream) && fake::records == 1,
              "all-gather uses the same deferred completion rule");
        check(api.poll_completion(at).code == FailureCode::deadline_expired,
              "a completed previous event cannot complete a new background all-gather");
        fake::state = ncclRemoteError;
        fake::enqueue_pending_polls = 0;
        Failure failure;
        check(api.async_failure(failure) && failure.code == FailureCode::async_error,
              "a communicator error reaches health polling");
        fake::state = ncclSuccess;
        check(api.async_failure(failure), "an observed asynchronous failure stays sticky");
        api.comm_abort(); api.comm_abort();
        check(fake::aborts == 1, "abort is idempotent after an operation");
    }
    fake::reset();
    fake::state = ncclInProgress;
    {
        ProductionNcclApi api(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
        config.initialize_ms = 10;
        check(api.comm_init(config).code == FailureCode::deadline_expired && fake::init_calls == 1,
              "an init stuck in background work returns at its deadline");
        api.comm_abort();
        check(fake::aborts == 1, "an init that timed out retains an abortable handle");
    }
    fake::reset();
    fake::state = ncclInProgress;
    {
        ProductionNcclApi api(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
        config.initialize_ms = 1000;
        Failure outcome;
        std::thread worker([&] { outcome = api.comm_init(config); });
        while (fake::polls == 0) std::this_thread::yield();
        api.comm_abort();
        worker.join();
        check(outcome.code == FailureCode::aborted && fake::init_calls == 1 && fake::aborts == 1,
              "watchdog abort reaches the published init handle without double abort or reinit");
    }
}

void check_stream_step_lifecycle() {
    fake::reset();
    fake::capture_status = cudaStreamCaptureStatusNone;
    auto config = fake::config();
    ProductionNcclApi api(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
    auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x30});
    float input = 1, output = 0;
    check(!api.comm_init(config) && !api.begin_stream_step(stream), "a stream step starts on an idle communicator");
    check(api.begin_stream_step(stream).code == FailureCode::in_flight, "nested stream steps are refused");
    check(!api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
          !api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) && fake::records == 0,
          "two reductions enqueue without a GPU completion event between them");
    check(api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32,
          reinterpret_cast<cudaStream_t>(uintptr_t{0x40})).code == FailureCode::bad_pointer,
          "a stream step refuses another stream");
    fake::capture_status = cudaStreamCaptureStatusActive;
    check(api.finish_stream_step({std::chrono::steady_clock::now() + std::chrono::seconds(1)}).code == FailureCode::bad_operation,
          "stream completion cannot record an event into an unfinished capture");
    fake::capture_status = cudaStreamCaptureStatusNone;
    fake::event_state = cudaSuccess;
    check(!api.finish_stream_step({std::chrono::steady_clock::now() + std::chrono::seconds(1)}) && fake::records == 1,
          "one completion event finishes the whole compute step");
    check(!api.begin_stream_step(stream), "a completed step allows the next one");
    fake::event_state = cudaErrorNotReady;
    check(api.finish_stream_step({std::chrono::steady_clock::now()}).code == FailureCode::deadline_expired &&
          api.requires_process_exit(), "unknown DMA completion quarantines the step allocations");
}

void check_graph_lifecycle() {
    fake::reset();
    auto config = fake::config();
    ProductionNcclApi api(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
    check(!api.comm_init(config), "graph test communicator initializes");
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x30});
    float input = 1, output = 0;

    check(!api.prepare_graph_session(stream, 3), "graph session prepares with an exact collective count");
    const auto early_replay = api.begin_captured_replay();
    check(early_replay.code == FailureCode::not_ready || early_replay.code == FailureCode::bad_operation,
          "replay cannot begin until graph capture is finalized");
    check(!api.check_graph_capture(), "an explicitly prepared stream can be validated during capture");
    check(!api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
              !api.enqueue_all_gather(&input, &output, 1, ScalarType::fp32, stream) &&
              !api.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream),
          "three collectives enqueue in order without host completion polling");
    check(fake::records == 0 && fake::polls > fake::reduces + fake::gathers,
          "capture enqueues record no CUDA event and only query NCCL async health");
    check(api.enqueue_all_gather(&input, &output, 1, ScalarType::fp32, stream).code == FailureCode::bad_extent,
          "a capture cannot enqueue beyond its prepared collective count");
    check(fake::gathers == 1, "the over-count collective never reaches NCCL");

    fake::reset();
    // A fresh instance below covers successful capture because over-counting is terminal by contract.
    ProductionNcclApi good(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
    check(!good.comm_init(config) && !good.prepare_graph_session(stream, 3) && !good.check_graph_capture(),
          "fresh graph session begins capture");
    check(!good.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream) &&
              !good.enqueue_all_gather(&input, &output, 1, ScalarType::fp32, stream) &&
              !good.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream),
          "three graph operations enqueue successfully");
    fake::capture_status = cudaStreamCaptureStatusNone;
    check(!good.finish_graph_capture(), "capture finalizes only after the exact ordered sequence");
    check(!good.begin_captured_replay(), "a finalized graph begins an explicit replay");
    fake::event_state = cudaSuccess;
    const Deadline at{std::chrono::steady_clock::now() + std::chrono::seconds(1)};
    bool replay_ok = true;
    for (int i = 0; i < 100; ++i) {
        if (i != 0 && good.begin_captured_replay()) replay_ok = false;
        if (good.wait_captured_replay(at)) replay_ok = false;
        if (fake::records != i + 1) replay_ok = false;
    }
    check(replay_ok, "100 graph replays each record and observe one completion event");
    check(fake::polls == 215, "every replay checks NCCL async state before event query");
    check(!good.end_graph_session(), "a drained graph session closes cleanly");

    fake::reset();
    ProductionNcclApi unknown(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
    check(!unknown.comm_init(config) && !unknown.prepare_graph_session(stream, 1) && !unknown.check_graph_capture() &&
              !unknown.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream),
          "unknown-query case has a captured operation");
    fake::capture_status = cudaStreamCaptureStatusNone;
    check(!unknown.finish_graph_capture() && !unknown.begin_captured_replay(), "unknown-query graph is finalized");
    fake::event_state = cudaErrorUnknown;
    check(unknown.wait_captured_replay(at).code == FailureCode::async_error &&
              unknown.requires_process_exit(),
          "an unknown CUDA event result poisons the graph and requests process exit for uncertain DMA");

    fake::reset();
    ProductionNcclApi timeout(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
    check(!timeout.comm_init(config) && !timeout.prepare_graph_session(stream, 1) && !timeout.check_graph_capture() &&
              !timeout.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream),
          "timeout case has a captured operation");
    fake::capture_status = cudaStreamCaptureStatusNone;
    check(!timeout.finish_graph_capture() && !timeout.begin_captured_replay(), "timeout graph is finalized");
    fake::event_state = cudaErrorNotReady;
    const Deadline expired{std::chrono::steady_clock::now()};
    check(timeout.wait_captured_replay(expired).code == FailureCode::deadline_expired &&
              timeout.requires_process_exit(),
          "an unfinished replay deadline requests owner process exit");

    fake::reset();
    ProductionNcclApi launch_failed(config, nullptr, fake::symbols(), "fake", "sha256:fake", fake::cuda());
    check(!launch_failed.comm_init(config) && !launch_failed.prepare_graph_session(stream, 1) &&
              !launch_failed.check_graph_capture() &&
              !launch_failed.enqueue_all_reduce_sum(&input, &output, 1, ScalarType::fp32, stream),
          "launch-failure case has a captured operation");
    fake::capture_status = cudaStreamCaptureStatusNone;
    check(!launch_failed.finish_graph_capture() && !launch_failed.begin_captured_replay(),
          "launch-failure graph declares a pending replay");
    launch_failed.comm_abort();
    check(launch_failed.requires_process_exit() && fake::aborts == 1,
          "abort after a failed graph launch requests process exit and preserves pending DMA resources");
}

void check_mapped_provenance() {
    char name[] = "/tmp/strata-nccl-provenance-XXXXXX";
    const int fd = ::mkstemp(name);
    check(fd >= 0, "a file for mapped provenance regression");
    if (fd < 0) return;
    const char original[] = "original mapped library bytes";
    ::write(fd, original, sizeof original);
    void* mapped = ::mmap(nullptr, sizeof original, PROT_READ, MAP_PRIVATE, fd, 0);
    std::string hash, err, reference;
    detail::sha256_of_file(name, reference, err);
    check(mapped != MAP_FAILED && detail::mapped_library_hash(name, mapped, hash, err) && hash == reference,
          "provenance hashes the descriptor matching the mapped inode");
    ::unlink(name);
    const int replacement = ::open(name, O_CREAT | O_WRONLY | O_EXCL, 0600);
    ::write(replacement, "replacement", 11);
    ::close(replacement);
    check(!detail::mapped_library_hash(name, mapped, hash, err),
          "a replaced dladdr pathname cannot authenticate different mapped bytes");
    if (mapped != MAP_FAILED) ::munmap(mapped, sizeof original);
    ::close(fd); ::unlink(name);
}

// The FIPS 180-4 vectors, as published.  The 56-character one is the one that matters most: its length is
// exactly where the padding rule changes shape, so a hash that gets all three right is not accidentally
// right about the tail of a 300 MB library.
void check_sha256_vectors() {
    struct Case {
        const char* text;
        const char* hex;
    };
    const Case cases[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    for (const Case& c : cases) {
        detail::sha256_ns::Sha256 hash;
        hash.update(reinterpret_cast<const uint8_t*>(c.text), std::strlen(c.text));
        const auto digest = hash.finish();
        const std::string got = detail::sha256_ns::hex(digest.data(), digest.size());
        check(got == c.hex, (std::string("sha256 of \"") + c.text + "\"").c_str());
        if (got != c.hex) std::printf("     got %s\n     want %s\n", got.c_str(), c.hex);
    }
}

void check_bootstrap_encoding() {
    // A 128-byte id where every byte is distinct, so a swapped nibble or a short read cannot pass.
    ncclUniqueId id;
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        id.internal[i] = static_cast<char>(i * 7 + 3);
    }
    const std::string hex = detail::hex_of_id(id);
    check(hex.size() == NCCL_UNIQUE_ID_BYTES * 2, "a 128-byte id encodes as 256 hex characters");
    ncclUniqueId back;
    std::string err;
    check(detail::hex_to_id(hex, back, err) &&
              std::memcmp(id.internal, back.internal, NCCL_UNIQUE_ID_BYTES) == 0,
          "the bootstrap id survives encode and decode unchanged");

    // The two refusals a torn roster produces: the wrong length, and something that is not hex.
    check(!detail::hex_to_id("00", back, err), "a short id is refused");
    check(!detail::hex_to_id(std::string(NCCL_UNIQUE_ID_BYTES * 2, 'g'), back, err),
          "a non-hex id is refused");
    check(!detail::hex_to_id(std::string(NCCL_UNIQUE_ID_BYTES * 2, 'A'), back, err),
          "an uppercase hex id is refused: the encoding is lowercase");
}

void check_version_decoder() {
    check(detail::version_string(NCCL_VERSION_CODE) == "2.30.7",
          "the pinned version code decodes to 2.30.7");
    check(detail::version_string(22707) == "2.27.7", "a 2.27 build decodes with the same shape");
    // The pre-2.9 shape from the header's own `NCCL_VERSION(X,Y,Z)` macro: 2*1000 + 8*100 + 7.
    check(detail::version_string(2807) == "2.8.7", "a pre-2.9 code decodes with the older shape");
    check(detail::version_string(20900) == "2.9.0", "the new shape starts where the header says it does");
}

// The two dtype handoffs.  A wrong value here sends the wrong element width over the wire, which is a wrong
// answer rather than an error, so it is checked against the header's own names.
void check_dtype_mapping(cudaStream_t stream) {
    ncclDataType_t dtype = ncclNumTypes;
    check(sizeof(float) == 4 && sizeof(__nv_bfloat16) == 2, "fp32 is 4 bytes and bf16 is 2 bytes here");
    // `ncclFloat32` and `ncclBfloat16` are the names the pinned header gives these two widths, and both are
    // legal for `ncclSum`.  Nothing in the contract allows a third.
    dtype = ncclFloat32;
    check(dtype == ncclFloat32, "fp32 has a name in this header");
    dtype = ncclBfloat16;
    check(dtype == ncclBfloat16, "bf16 has a name in this header");
    (void)stream;
}

}  // namespace

// Run on a host with a CUDA driver.  Here it is only compiled.
//
// `strata-tp3-nccl-probe [file]`: with an argument, print that file's SHA-256 as this code computes it, so a
// rank's approved provenance string can be read off the library it will actually load, and checked against
// `sha256sum` rather than trusted.
int main(int argc, char** argv) {
    const bool host_only = argc > 1 && std::strcmp(argv[1], "--host-only") == 0;
    if (argc > 1 && !host_only) {
        std::string hex;
        std::string err;
        if (!detail::sha256_of_file(argv[1], hex, err)) {
            std::printf("FAIL %s\n", err.c_str());
            return 1;
        }
        std::printf("%s  %s\n", hex.c_str(), argv[1]);
        return 0;
    }
    check_sha256_vectors();
    check_bootstrap_encoding();
    check_version_decoder();
    check_nonblocking_lifecycle();
    check_stream_step_lifecycle();
    check_graph_lifecycle();
    check_mapped_provenance();
    if (host_only) return g_failed == 0 ? 0 : 1;

    // Real stream, real library: the part a CPU test cannot reach.  If no NCCL is installed, the factory says
    // so and the rest of the probe is skipped rather than pretending.
    NcclConfig config;
    config.rank = 0;
    config.world = kTpWorld;
    config.bootstrap_id = std::string(NCCL_UNIQUE_ID_BYTES * 2, 'a');
    config.library_path.clear();  // the loader's own search order

    std::string err;
    std::unique_ptr<NcclApi> api = make_production_nccl_api(config, err);
    if (api == nullptr) {
        std::printf("skip  no NCCL library to load: %s\n", err.c_str());
        return g_failed == 0 ? 0 : 1;
    }

    std::string version;
    std::string provenance;
    const Failure identity = api->identity(version, provenance);
    check(!identity, "the library will say what it is");
    std::printf("info  library version '%s' provenance '%s'\n", version.c_str(), provenance.c_str());

    check_dtype_mapping(nullptr);

    // The gate is the interesting half: with an approval that cannot match, no communicator may exist.
    NcclBackend refused(std::move(api), config, NcclApproval{"0.0.0", "sha256:not-this-build"});
    const Failure gate = refused.init();
    check(gate.code == FailureCode::library_mismatch,
          "a library that is not the approved build is refused before a communicator exists");
    check(refused.initialized() == false, "the refused backend never reports itself ready");
    // Teardown through the transport the worker actually uses, so the CUDA build is compiled against
    // the same pairing the runtime has: `GuardedTransport` over `NcclBackend`.
    SystemClock clock;
    GuardedTransport transport(refused, clock);
    transport.abort("probe finished");

    std::printf("%s\n", g_failed == 0 ? "probe: all checks passed" : "probe: failures above");
    return g_failed == 0 ? 0 : 1;
}
