#include "strata/distributed/roce_backend.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::distributed;
using namespace strata::distributed::roce;

struct roce_ctx_t;
extern "C" uint64_t roce_blob_bytes(void);

namespace {

struct Proxy {
    bool failed = false;
    bool connect_fails = false;
    bool start_fails = false;
    uint32_t completed = 0;
};

struct FakeState {
    int abi = 10;
    bool create_fails = false;
    cudaError_t capture_query = cudaSuccess;
    cudaError_t event_query = cudaSuccess;
    cudaError_t event_record = cudaSuccess;
    bool capturing = false;
    uint32_t captured_launches = 0;
    uint32_t replay_launches = 0;
    uint32_t state_sequence = 0;
    strata::distributed::roce::cuda::Kind last_kind =
        strata::distributed::roce::cuda::Kind::all_reduce_sum;
    uint64_t last_global_first = 0;
    uint64_t last_global_total = 0;
    int proxy_gid_index = INT32_MIN;
    uint8_t* region = nullptr;
    uint64_t ctrl_off = 0;
    void* device_state = nullptr;
    Proxy* proxy = nullptr;
    int live_allocations = 0;
    int frees = 0;
    int proxy_destroys = 0;
    int event_records = 0;
} fake;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void put_u32(void* base, uint32_t offset, uint32_t value) {
    std::memcpy(static_cast<uint8_t*>(base) + offset, &value, sizeof(value));
}

Config valid_config() {
    Config config;
    config.rank = 0;
    config.hca_names = {"fake0"};
    config.route_hca = {-1, 0, 0};
    config.slot_bytes = 4096;
    config.blocks = 1;
    config.threads = 2;
    config.approved_proxy_sha256 = "test-proxy-hash";
    return config;
}

void setup_fake_sysfs(const std::filesystem::path& root) {
    auto add_hca = [&](const char* name, int gid_index, const char* gid, const char* type) {
        const auto hca = root / name;
        std::filesystem::create_directories(hca / "ports/1/gids");
        std::filesystem::create_directories(hca / "ports/1/gid_attrs/types");
        std::ofstream(hca / "ports/1/state") << "4: ACTIVE\n";
        std::ofstream(hca / "ports/1/gids" / std::to_string(gid_index)) << gid << '\n';
        std::ofstream(hca / "ports/1/gid_attrs/types" / std::to_string(gid_index)) << type << '\n';
    };
    add_hca("fake0", 3, "0000:0000:0000:0000:0000:ffff:c0a8:0101", "RoCE v2");
    add_hca("fake1", 4, "0000:0000:0000:0000:0000:ffff:c0a8:0102", "RoCE v2");
    add_hca("fake2", 3, "fe80:0000:0000:0000:0000:0000:0000:0001", "RoCE v2");
    add_hca("fake3", 5, "0000:0000:0000:0000:0000:ffff:c0a8:01zz", "RoCE v2");
    add_hca("fake4", 6, "0000:0000:0000:0000:0000:ffff:c0a8:0104", "RoCE v1");
    setenv("STRATA_ROCE_TEST_SYSFS_ROOT", root.c_str(), 1);
}

void connect_backend(RoceBackend& backend) {
    check(!backend.create(), "valid RoCE config did not create");
    std::vector<uint8_t> blobs(roce_blob_bytes() * kTpWorld);
    check(!backend.connect(blobs), "fake proxy did not connect");
}

struct BackendFixture {
    std::filesystem::path sysfs;
    explicit BackendFixture(const char* name) {
        sysfs = std::filesystem::temp_directory_path() / name;
        std::filesystem::remove_all(sysfs);
        setup_fake_sysfs(sysfs);
        fake = {};
    }
    ~BackendFixture() {
        unsetenv("STRATA_ROCE_TEST_SYSFS_ROOT");
        std::filesystem::remove_all(sysfs);
    }
};

void test_config_and_setup_failures(const std::filesystem::path& sysfs) {
    check(discover_hcas(-1) == std::vector<std::string>({"fake0", "fake1"}),
          "automatic discovery did not select per-HCA IPv4 RoCEv2 entries");
    check(discover_hcas(3) == std::vector<std::string>({"fake0", "fake2"}),
          "explicit GID index no longer selects populated entries");
    {
        Config config = valid_config();
        config.blocks = 9;
        RoceBackend backend(config);
        check(backend.create().code == FailureCode::invalid_config, "invalid geometry was accepted");
    }
    {
        Config config = valid_config();
        config.blocks = 8;
        config.threads = 512;
        RoceBackend backend(config);
        check(!backend.create(), "explicit cooperative geometry was rejected");
    }
    {
        Config config = valid_config();
        config.approved_proxy_sha256 = "wrong";
        RoceBackend backend(config);
        check(backend.create().code == FailureCode::provenance_mismatch, "wrong proxy hash was accepted");
    }
    {
        fake.abi = 9;
        RoceBackend backend(valid_config());
        check(backend.create().code == FailureCode::provenance_mismatch, "wrong ABI was accepted");
        fake.abi = 10;
    }
    {
        fake.create_fails = true;
        RoceBackend backend(valid_config());
        check(backend.create().code == FailureCode::comm_failed, "proxy create failure was not reported");
        check(fake.live_allocations == 0, "create failure leaked CUDA test resources");
        fake.create_fails = false;
    }
    {
        RoceBackend backend(valid_config());
        check(!backend.create(), "backend setup failed before connect test");
        std::vector<uint8_t> blobs(roce_blob_bytes() * kTpWorld);
        fake.proxy->connect_fails = true;
        check(backend.connect(blobs).code == FailureCode::comm_failed, "proxy connect failure was not reported");
        fake.proxy->connect_fails = false;
        fake.proxy->start_fails = true;
        check(backend.connect(blobs).code == FailureCode::comm_failed, "proxy start failure was not reported");
    }
    {
        Config config = valid_config();
        config.hca_names = {"fake0", "fake1"};
        config.route_hca = {-1, 0, 1};
        config.gid_index = -1;
        RoceBackend backend(config);
        check(!backend.create(), "automatic per-HCA GID selection failed");
        check(fake.proxy_gid_index == -1, "automatic selection sentinel was not passed to the ABI-10 proxy");
    }
    {
        Config config = valid_config();
        config.gid_index = 3;
        RoceBackend backend(config);
        check(!backend.create(), "explicit GID selection failed");
        check(fake.proxy_gid_index == 3, "explicit GID index changed before proxy creation");
    }
    (void)sysfs;
}

void test_single_enqueue_and_abort() {
    BackendFixture fixture("strata-roce-state-single");
    {
        RoceBackend backend(valid_config());
        connect_backend(backend);
        alignas(16) std::array<uint8_t, 64> input{}, output{};
        auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(1));
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "valid enqueue failed");
        fake.event_query = cudaErrorNotReady;
        const auto pending = backend.raw_poll({std::chrono::steady_clock::now() + std::chrono::seconds(1)});
        check(pending.code == FailureCode::deadline_expired, "not-ready event was not pending");
        check(static_cast<bool>(backend.raw_poll({std::chrono::steady_clock::now()})),
              "expired in-flight operation was not reported");
        const int frees_before_abort = fake.frees;
        backend.raw_abort();
        check(backend.requires_process_exit(), "timeout abort did not require process exit");
        check(fake.frees == frees_before_abort, "timeout abort freed borrowed transport memory");
        check(fake.live_allocations == 2, "timeout path unexpectedly released resources");
        fake.event_query = cudaSuccess;
        (void)input;
        (void)output;
        (void)stream;
    }
    check(fake.frees == 0, "unsafe backend destructor released resources");
    std::free(fake.region);
    std::free(fake.device_state);
    delete fake.proxy;
    fake.region = nullptr;
    fake.device_state = nullptr;
    fake.proxy = nullptr;
    fake.live_allocations = 0;
}

void test_ring_sum_validation_and_forwarding() {
    BackendFixture fixture("strata-roce-state-ring-sum");
    alignas(16) std::array<uint8_t, 64> input{}, output{};
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(7));
    RoceBackend backend(valid_config());

    check(backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::bf16},
                                          {output.data(), 16, ScalarType::bf16},
                                          16, 0, 32, stream).code == FailureCode::bad_extent,
          "ring sum accepted BF16 before backend setup");
    check(backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32},
                                          16, 0, 30, stream).code == FailureCode::bad_extent,
          "ring sum accepted a total outside SIMPLE8 geometry");
    check(backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32},
                                          16, 2, 32, stream).code == FailureCode::bad_extent,
          "ring sum accepted an unaligned global offset");
    check(backend.raw_all_reduce_ring_sum({input.data(), 12, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32},
                                          16, 0, 32, stream).code == FailureCode::bad_extent,
          "ring sum accepted input metadata that differs from its element count");
    check(backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 12, ScalarType::fp32},
                                          16, 0, 32, stream).code == FailureCode::bad_extent,
          "ring sum accepted output metadata that differs from its element count");
    check(backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32},
                                          16, 20, 32, stream).code == FailureCode::bad_extent,
          "ring sum accepted a chunk extending past the global reduction");
    check(backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32},
                                          16, 0, 32, stream).code == FailureCode::not_ready,
          "ring sum did not defer valid spans to backend readiness");

    connect_backend(backend);
    check(!backend.raw_all_reduce_ring_sum({input.data(), 16, ScalarType::fp32},
                                           {output.data(), 16, ScalarType::fp32},
                                           16, 4, 32, stream),
          "valid ring sum enqueue failed");
    check(fake.last_kind == strata::distributed::roce::cuda::Kind::all_reduce_ring_sum &&
              fake.last_global_first == 4 && fake.last_global_total == 32,
          "ring sum kind or global range was not forwarded to the CUDA launcher");
    check(!backend.raw_poll({std::chrono::steady_clock::now() + std::chrono::seconds(1)}),
          "valid ring sum completion failed");
}

void test_capture_and_replay() {
    BackendFixture fixture("strata-roce-state-graphs");
    {
        RoceBackend backend(valid_config());
        connect_backend(backend);
        alignas(16) std::array<uint8_t, 64> input{}, output{};
        auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(2));
        fake.capturing = true;
        for (uint32_t i = 0; i < 3; ++i) {
            check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                              {output.data(), 16, ScalarType::fp32}, 16, stream),
                  "multi-operation capture enqueue failed");
            check(!backend.raw_poll({std::chrono::steady_clock::now()}),
                  "capture acknowledgement reported a pending operation");
        }
        check(fake.captured_launches == 3, "capture did not retain all launches");
        fake.capturing = false;
        for (uint32_t replay = 0; replay < 100; ++replay) {
            check(!backend.wait_captured_replay(
                      {std::chrono::steady_clock::now() + std::chrono::seconds(1)}),
                  "captured graph replay did not complete");
            check(fake.proxy->completed == (replay + 1) * 3,
                  "replay sequence did not advance dynamically");
        }
        backend.raw_abort();
        check(!backend.requires_process_exit(), "completed replay unnecessarily requires process exit");
    }
}

void test_stream_steps() {
    BackendFixture fixture("strata-roce-state-stream-step");
    {
        RoceBackend backend(valid_config());
        connect_backend(backend);
        alignas(16) std::array<uint8_t, 64> input{}, output{};
        auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(6));
        const auto deadline = [] {
            return Deadline{std::chrono::steady_clock::now() + std::chrono::seconds(1)};
        };

        check(!backend.begin_stream_step(stream), "eager stream step did not begin");
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "first eager stream-step enqueue failed");
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "second eager stream-step enqueue failed");
        check(fake.event_records == 0, "stream step recorded an event per collective");
        check(!backend.finish_stream_step(deadline()), "multiple eager stream operations did not finish");
        check(fake.event_records == 1 && fake.proxy->completed == 2,
              "eager stream step did not record one event and observe both operations");

        fake.capturing = true;
        check(!backend.begin_stream_step(stream), "captured stream step did not begin");
        for (int i = 0; i < 3; ++i) {
            check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                              {output.data(), 16, ScalarType::fp32}, 16, stream),
                  "first captured stream-step enqueue failed");
        }
        fake.capturing = false;
        check(!backend.finish_stream_step(deadline()), "first captured stream step did not finish");
        check(fake.proxy->completed == 5, "first captured stream step lost operations");

        fake.capturing = true;
        check(!backend.begin_stream_step(stream), "second captured stream step did not begin");
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "second captured stream-step enqueue failed");
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "second captured stream-step enqueue failed");
        fake.capturing = false;
        check(!backend.finish_stream_step(deadline()), "different-size captured stream step did not finish");
        check(fake.proxy->completed == 7, "different-size captured stream step used a static count");

        fake.replay_launches = 0;
        check(!backend.begin_stream_step(stream), "empty stream step did not begin");
        check(!backend.finish_stream_step(deadline()), "empty stream step was rejected");
        check(fake.proxy->completed == 7, "empty stream step changed the completion sequence");

        fake.replay_launches = 3;
        check(!backend.begin_stream_step(stream), "cached graph replay step did not begin");
        check(!backend.finish_stream_step(deadline()), "cached graph without host enqueues did not finish");
        check(fake.proxy->completed == 10, "cached graph replay lost its dynamic completion sequence");
        fake.replay_launches = 0;
        check(!backend.begin_stream_step(stream), "mixed eager/capture step did not begin");
        check(!backend.raw_all_reduce_sum({input.data(),16,ScalarType::fp32},
            {output.data(),16,ScalarType::fp32},16,stream), "mixed step eager enqueue failed");
        fake.capturing=true;
        check(!backend.raw_all_reduce_sum({input.data(),16,ScalarType::fp32},
            {output.data(),16,ScalarType::fp32},16,stream), "mixed step capture enqueue failed");
        fake.capturing=false;
        check(!backend.finish_stream_step(deadline()), "mixed eager/capture step did not finish");
    }
}

void test_uncertain_capture_lifetime() {
    BackendFixture fixture("strata-roce-state-capture-lifetime");
    {
        RoceBackend backend(valid_config());
        connect_backend(backend);
        alignas(16) std::array<uint8_t, 64> input{}, output{};
        auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(5));
        fake.capturing = true;
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "capture-lifetime enqueue failed");
        check(backend.requires_process_exit(), "captured graph did not require lifetime protection");
        const int frees_before = fake.frees;
        check(fake.capture_query == cudaSuccess, "fake capture query setup failed");
        fake.capturing = false;
        fake.capture_query = cudaErrorUnknown;
        check(static_cast<bool>(backend.wait_captured_replay(
                  {std::chrono::steady_clock::now() + std::chrono::seconds(1)})),
              "capture query error was ignored");
        backend.raw_abort();
        check(fake.frees == frees_before, "capture error released graph-owned resources");
    }
    check(fake.frees == 0, "captured backend destructor released graph-owned resources");
    std::free(fake.region);
    std::free(fake.device_state);
    delete fake.proxy;
    fake.region = nullptr;
    fake.device_state = nullptr;
    fake.proxy = nullptr;
    fake.live_allocations = 0;
}

void test_event_error_requires_exit() {
    BackendFixture fixture("strata-roce-state-event-error");
    void* leaked_region = nullptr;
    void* leaked_state = nullptr;
    Proxy* leaked_proxy = nullptr;
    {
        RoceBackend backend(valid_config());
        connect_backend(backend);
        alignas(16) std::array<uint8_t, 64> input{}, output{};
        auto stream = reinterpret_cast<cudaStream_t>(uintptr_t(3));
        check(!backend.raw_all_reduce_sum({input.data(), 16, ScalarType::fp32},
                                          {output.data(), 16, ScalarType::fp32}, 16, stream),
              "event-error enqueue failed");
        fake.event_query = cudaErrorUnknown;
        check(static_cast<bool>(backend.raw_poll(
                  {std::chrono::steady_clock::now() + std::chrono::seconds(1)})),
              "event error was ignored");
        leaked_region = fake.region;
        leaked_state = fake.device_state;
        leaked_proxy = fake.proxy;
        const int frees_before_abort = fake.frees;
        backend.raw_abort();
        check(backend.requires_process_exit(), "event error did not protect in-flight memory");
        check(fake.frees == frees_before_abort, "event error path freed unsafe resources");
    }
    check(fake.frees == 0, "event error backend destructor released resources");
    std::free(leaked_region);
    std::free(leaked_state);
    delete leaked_proxy;
    fake.region = nullptr;
    fake.device_state = nullptr;
    fake.proxy = nullptr;
    fake.live_allocations = 0;
}

}  // namespace

extern "C" {
int roce_abi_version(void) { return fake.abi; }
int roce_layout(int world, uint64_t slot_bytes, uint64_t* out) {
    strata::distributed::roce::cuda::Layout layout;
    if (!strata::distributed::roce::cuda::layout(world, slot_bytes, layout)) return -1;
    out[0] = layout.recv_off;
    out[1] = layout.flag_off;
    out[2] = layout.send_off;
    out[3] = layout.ctrl_off;
    out[4] = layout.total_bytes;
    out[5] = layout.flag_stride;
    out[6] = layout.slots;
    return 0;
}
uint64_t roce_blob_bytes(void) { return 16; }
roce_ctx_t* roce_create(int, int, const char* const*, int, const int*, int, int gid_index, int, int,
                        void* region, uint64_t, uint64_t, char* error, uint64_t error_size) {
    fake.proxy_gid_index = gid_index;
    if (fake.create_fails) {
        std::snprintf(error, static_cast<size_t>(error_size), "injected create failure");
        return nullptr;
    }
    fake.region = static_cast<uint8_t*>(region);
    fake.proxy = new Proxy;
    return reinterpret_cast<roce_ctx_t*>(fake.proxy);
}
int roce_local_blob(roce_ctx_t*, void* out, uint64_t bytes) {
    if (bytes < 16) return -1;
    std::memset(out, 0x5a, 16);
    return 0;
}
int roce_connect(roce_ctx_t* ctx, const void*, uint64_t) {
    return reinterpret_cast<Proxy*>(ctx)->connect_fails ? -1 : 0;
}
int roce_start(roce_ctx_t* ctx) { return reinterpret_cast<Proxy*>(ctx)->start_fails ? -1 : 0; }
void roce_stop(roce_ctx_t*) {}
int roce_failed(roce_ctx_t* ctx) { return reinterpret_cast<Proxy*>(ctx)->failed ? 1 : 0; }
const char* roce_error(roce_ctx_t*) { return "injected proxy failure"; }
uint64_t roce_stat(roce_ctx_t* ctx, int which) {
    return which == 2 ? reinterpret_cast<Proxy*>(ctx)->completed : 0;
}
void roce_destroy(roce_ctx_t* ctx) {
    ++fake.proxy_destroys;
    delete reinterpret_cast<Proxy*>(ctx);
    fake.proxy = nullptr;
}
}

cudaError_t cudaHostAlloc(void** ptr, size_t bytes, unsigned int) {
    *ptr = std::malloc(bytes);
    if (!*ptr) return cudaErrorMemoryAllocation;
    ++fake.live_allocations;
    return cudaSuccess;
}
cudaError_t cudaHostGetDevicePointer(void** device, void* host, unsigned int) {
    *device = host;
    fake.region = static_cast<uint8_t*>(host);
    return cudaSuccess;
}
cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    if (!*ptr) return cudaErrorMemoryAllocation;
    ++fake.live_allocations;
    fake.device_state = *ptr;
    return cudaSuccess;
}
cudaError_t cudaMemcpy(void* destination, const void* source, size_t bytes, enum cudaMemcpyKind) {
    std::memcpy(destination, source, bytes);
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
    *event = reinterpret_cast<cudaEvent_t>(uintptr_t(4));
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) {
    if (fake.capture_query != cudaSuccess) return fake.capture_query;
    *status = fake.capturing ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) {
    if (fake.event_record != cudaSuccess) return fake.event_record;
    ++fake.event_records;
    if (!fake.capturing && fake.replay_launches != 0) {
        fake.state_sequence += fake.replay_launches;
        put_u32(fake.device_state, 0, fake.state_sequence);
        const uint32_t ctrl = static_cast<uint32_t>(fake.ctrl_off);
        put_u32(fake.region, ctrl + strata::distributed::roce::cuda::kCtrlCompletedSeq,
                fake.state_sequence);
        if (fake.proxy) fake.proxy->completed = fake.state_sequence;
        fake.captured_launches = 0;
    }
    return cudaSuccess;
}
cudaError_t cudaEventQuery(cudaEvent_t) { return fake.event_query; }
cudaError_t cudaFree(void* ptr) {
    if (ptr) {
        std::free(ptr);
        --fake.live_allocations;
        ++fake.frees;
        if (ptr == fake.device_state) fake.device_state = nullptr;
    }
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void* ptr) {
    if (ptr) {
        std::free(ptr);
        --fake.live_allocations;
        ++fake.frees;
        if (ptr == fake.region) fake.region = nullptr;
    }
    return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "fake CUDA error"; }

namespace strata::distributed::roce::cuda {
bool layout(uint32_t world, uint64_t slot_bytes, Layout& out) {
    if (world != 3 || slot_bytes == 0 || slot_bytes % 4096 != 0) return false;
    const uint64_t recv_bytes = uint64_t(world) * 2 * slot_bytes;
    const uint64_t flag_bytes = uint64_t(world) * 2 * 4 * 128;
    out.recv_off = 0;
    out.flag_off = recv_bytes;
    out.send_off = recv_bytes + flag_bytes;
    out.ctrl_off = out.send_off + 2 * slot_bytes;
    out.total_bytes = out.ctrl_off + 128;
    out.flag_stride = 128;
    out.slots = 2;
    fake.ctrl_off = out.ctrl_off;
    return true;
}
cudaError_t launch(const DeviceBuffer&, const DeviceBuffer&, Kind kind, uint32_t, const Layout&,
                   void* region, uint64_t, void* device_state, uint32_t, cudaStream_t,
                   uint32_t, uint32_t, uint64_t global_first, uint64_t global_total) {
    fake.last_kind = kind;
    fake.last_global_first = global_first;
    fake.last_global_total = global_total;
    fake.region = static_cast<uint8_t*>(region);
    fake.device_state = device_state;
    if (fake.capturing) {
        ++fake.captured_launches;
        fake.replay_launches = fake.captured_launches;
        return cudaSuccess;
    }
    ++fake.state_sequence;
    put_u32(device_state, 0, fake.state_sequence);
    put_u32(region, static_cast<uint32_t>(fake.ctrl_off) + kCtrlCompletedSeq, fake.state_sequence);
    if (fake.proxy) fake.proxy->completed = fake.state_sequence;
    return cudaSuccess;
}
}  // namespace strata::distributed::roce::cuda

int main() {
    try {
        BackendFixture setup("strata-roce-state-setup");
        test_config_and_setup_failures(setup.sysfs);
        test_single_enqueue_and_abort();
        test_ring_sum_validation_and_forwarding();
        test_capture_and_replay();
        test_stream_steps();
        test_uncertain_capture_lifetime();
        test_event_error_requires_exit();
        check(fake.live_allocations >= 0, "invalid fake allocation count");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
