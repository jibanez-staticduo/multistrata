#include "strata/distributed/roce_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>

extern "C" {
typedef struct roce_ctx_t roce_ctx_t;
int roce_abi_version(void);
int roce_layout(int world, uint64_t slot_bytes, uint64_t* out);
uint64_t roce_blob_bytes(void);
roce_ctx_t* roce_create(int world, int rank, const char* const* hca_names, int n_hca,
                        const int* route_hca, int n_stripes, int gid_index,
                        int traffic_class, int topology, void* region,
                        uint64_t region_bytes, uint64_t slot_bytes,
                        char* err, uint64_t err_len);
int roce_local_blob(roce_ctx_t* ctx, void* out, uint64_t out_len);
int roce_connect(roce_ctx_t* ctx, const void* blobs, uint64_t blobs_len);
int roce_start(roce_ctx_t* ctx);
void roce_stop(roce_ctx_t* ctx);
int roce_failed(roce_ctx_t* ctx);
const char* roce_error(roce_ctx_t* ctx);
uint64_t roce_stat(roce_ctx_t* ctx, int which);
void roce_destroy(roce_ctx_t* ctx);
}

namespace strata::distributed::roce {
namespace {

#ifndef STRATA_ROCE_PROXY_SHA256
#error "The isolated RoCE build must pin STRATA_ROCE_PROXY_SHA256"
#endif
constexpr char kProxySourceSha256[] = STRATA_ROCE_PROXY_SHA256;
constexpr uint32_t kControlFailure = cuda::kCtrlFailed;
constexpr uint32_t kControlCompleted = cuda::kCtrlCompletedSeq;
constexpr uint32_t kControlPeer = cuda::kCtrlErrorPeer;
constexpr uint32_t kControlWaitedUs = cuda::kCtrlWaitedUs;

struct DeviceState {
    uint32_t epoch = 0;
    uint32_t error = 0;
    uint32_t sequence = 0;
};
static_assert(sizeof(DeviceState) == cuda::kDeviceStateBytes);

Failure cuda_failure(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return ok_failure();
    return Failure::fail(FailureCode::comm_failed,
                         std::string(operation) + ": " + cudaGetErrorString(error));
}

bool read_u32(const void* base, uint32_t offset, uint32_t& value) {
    if (base == nullptr) return false;
    const auto* word = reinterpret_cast<const uint32_t*>(
        static_cast<const uint8_t*>(base) + offset);
    value = __atomic_load_n(word, __ATOMIC_ACQUIRE);
    return true;
}

bool has_populated_gid(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string gid;
    if (!(in >> gid)) return false;
    return gid.find_first_not_of("0: \t\r\n") != std::string::npos;
}

std::filesystem::path infiniband_sysfs_root() {
#ifdef STRATA_ROCE_TEST_SYSFS_ROOT_ENV
    if (const char* root = std::getenv("STRATA_ROCE_TEST_SYSFS_ROOT"); root && root[0] != '\0') {
        return root;
    }
#endif
    return "/sys/class/infiniband";
}

}  // namespace

std::vector<std::string> discover_hcas(int gid_index, size_t maximum) {
    std::vector<std::string> found;
    if (gid_index < 0 || maximum == 0) return found;
    const std::filesystem::path root = infiniband_sysfs_root();
    std::error_code error;
    for (std::filesystem::directory_iterator it(root, error), end; !error && it != end; it.increment(error)) {
        std::ifstream state(it->path() / "ports/1/state");
        std::string state_value;
        if (!std::getline(state, state_value) || state_value.find("ACTIVE") == std::string::npos) continue;
        if (!has_populated_gid(it->path() / "ports/1/gids" / std::to_string(gid_index))) continue;
        found.push_back(it->path().filename().string());
        if (found.size() == maximum) break;
    }
    std::sort(found.begin(), found.end());
    return found;
}

RoceBackend::RoceBackend(Config config) : config_(std::move(config)) {}

RoceBackend::~RoceBackend() {
    if (unsafe_to_destroy_ || in_flight_ || captured_pending_ || capture_replay_pending_) return;
    release_safe();
}

Failure RoceBackend::create() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (created_) return Failure::fail(FailureCode::invalid_config, "RoCE backend was already created");
    if (config_.world != kTpWorld || config_.rank >= kTpWorld || config_.stripes != 1 ||
        config_.hca_names.empty() || config_.hca_names.size() > 4 ||
        config_.route_hca.size() != size_t(kTpWorld) * config_.stripes ||
        config_.gid_index < 0 || config_.traffic_class < 0 || config_.traffic_class > 255 ||
        config_.timeout_ms == 0 || config_.timeout_ms > (INT32_MAX / 1000u) ||
        config_.blocks != 1 || config_.threads < 2 || config_.threads > 1024 ||
        config_.slot_bytes == 0 || config_.slot_bytes % 4096 != 0) {
        return Failure::fail(FailureCode::invalid_config, "invalid TP3 direct RoCE geometry");
    }
    if (config_.approved_proxy_sha256.empty() ||
        config_.approved_proxy_sha256 != kProxySourceSha256) {
        return Failure::fail(FailureCode::provenance_mismatch,
                             "the RoCE proxy source hash is not the approved build");
    }
    if (roce_abi_version() != static_cast<int>(kProxyAbi)) {
        return Failure::fail(FailureCode::provenance_mismatch, "the RoCE proxy ABI is not version 10");
    }
    for (uint32_t peer = 0; peer < kTpWorld; ++peer) {
        const int route = config_.route_hca[peer];
        if ((peer == config_.rank && route != -1) ||
            (peer != config_.rank && (route < 0 || size_t(route) >= config_.hca_names.size()))) {
            return Failure::fail(FailureCode::invalid_config, "TP3 route table is incomplete or invalid");
        }
    }
    const auto discovered = discover_hcas(config_.gid_index);
    for (const auto& hca : config_.hca_names) {
        if (std::find(discovered.begin(), discovered.end(), hca) == discovered.end()) {
            return Failure::fail(FailureCode::invalid_config,
                                 "configured HCA is inactive or has an empty selected GID: " + hca);
        }
    }
    if (!cuda::layout(config_.world, config_.slot_bytes, layout_)) {
        return Failure::fail(FailureCode::invalid_config, "the requested RoCE region layout overflows");
    }
    std::array<uint64_t, 7> c_layout{};
    if (roce_layout(static_cast<int>(config_.world), config_.slot_bytes, c_layout.data()) != 0 ||
        c_layout[0] != layout_.recv_off || c_layout[1] != layout_.flag_off ||
        c_layout[2] != layout_.send_off || c_layout[3] != layout_.ctrl_off ||
        c_layout[4] != layout_.total_bytes || c_layout[5] != layout_.flag_stride ||
        c_layout[6] != layout_.slots) {
        return Failure::fail(FailureCode::provenance_mismatch, "proxy and CUDA region layouts disagree");
    }
    if (auto f = cuda_failure(cudaHostAlloc(&region_, layout_.total_bytes,
                                           cudaHostAllocMapped | cudaHostAllocPortable),
                              "cudaHostAlloc(mapped RoCE region)")) return f;
    std::memset(region_, 0, static_cast<size_t>(layout_.total_bytes));
    if (auto f = cuda_failure(cudaHostGetDevicePointer(&region_device_, region_, 0),
                              "cudaHostGetDevicePointer(RoCE region)")) {
        release_safe();
        return f;
    }
    if (auto f = cuda_failure(cudaMalloc(&device_state_, sizeof(DeviceState)),
                              "cudaMalloc(RoCE epoch state)")) {
        release_safe();
        return f;
    }
    DeviceState zero{};
    if (auto f = cuda_failure(cudaMemcpy(device_state_, &zero, sizeof(zero), cudaMemcpyHostToDevice),
                              "initialize RoCE epoch state")) {
        release_safe();
        return f;
    }
    cudaEvent_t event = nullptr;
    if (auto f = cuda_failure(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                              "cudaEventCreateWithFlags(RoCE completion)")) {
        release_safe();
        return f;
    }
    event_ = event;

    std::vector<const char*> names;
    names.reserve(config_.hca_names.size());
    for (const auto& name : config_.hca_names) names.push_back(name.c_str());
    std::array<char, 512> error{};
    proxy_ = roce_create(static_cast<int>(config_.world), static_cast<int>(config_.rank),
                         names.data(), static_cast<int>(names.size()), config_.route_hca.data(),
                         static_cast<int>(config_.stripes), config_.gid_index, config_.traffic_class,
                         0, region_, layout_.total_bytes, config_.slot_bytes, error.data(), error.size());
    if (proxy_ == nullptr) {
        release_safe();
        return Failure::fail(FailureCode::comm_failed,
                             std::string("roce_create: ") + (error[0] ? error.data() : "unknown error"));
    }
    const uint32_t timeout_us = config_.timeout_ms * 1000u;
    std::memcpy(static_cast<uint8_t*>(region_) + layout_.ctrl_off + cuda::kCtrlTimeoutUs,
                &timeout_us, sizeof(timeout_us));
    provenance_ = std::string("abi10:sha256:") + kProxySourceSha256;
    created_ = true;
    return ok_failure();
}

Failure RoceBackend::local_blob(std::vector<uint8_t>& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!created_ || proxy_ == nullptr) {
        return Failure::fail(FailureCode::not_ready, "the RoCE local blob requires a created proxy");
    }
    const uint64_t bytes = roce_blob_bytes();
    if (bytes == 0 || bytes > 4096) {
        return Failure::fail(FailureCode::provenance_mismatch, "the RoCE proxy blob size is invalid");
    }
    out.assign(static_cast<size_t>(bytes), 0);
    if (roce_local_blob(static_cast<roce_ctx_t*>(proxy_), out.data(), bytes) != 0) {
        out.clear();
        return Failure::fail(FailureCode::comm_failed, "roce_local_blob failed");
    }
    return ok_failure();
}

Failure RoceBackend::connect(std::span<const uint8_t> blobs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!created_ || proxy_ == nullptr) return Failure::fail(FailureCode::not_ready, "RoCE proxy is not created");
    if (connected_) return Failure::fail(FailureCode::invalid_config, "RoCE proxy is already connected");
    const uint64_t blob_bytes = roce_blob_bytes();
    if (blob_bytes == 0 || blobs.size() != blob_bytes * config_.world) {
        return Failure::fail(FailureCode::bad_extent, "RoCE roster must contain exactly one blob per rank");
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(blobs.data());
    if (roce_connect(static_cast<roce_ctx_t*>(proxy_), bytes, blobs.size()) != 0) {
        const char* detail = roce_error(static_cast<roce_ctx_t*>(proxy_));
        return Failure::fail(FailureCode::comm_failed,
                             std::string("roce_connect: ") + (detail ? detail : "unknown error"));
    }
    if (roce_start(static_cast<roce_ctx_t*>(proxy_)) != 0) {
        const char* detail = roce_error(static_cast<roce_ctx_t*>(proxy_));
        return Failure::fail(FailureCode::comm_failed,
                             std::string("roce_start: ") + (detail ? detail : "unknown error"));
    }
    connected_ = true;
    return ok_failure();
}

Failure RoceBackend::raw_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output,
                                        uint64_t elements, cudaStream_t stream) {
    return enqueue(input, output, cuda::Kind::all_reduce_sum, elements, stream);
}

Failure RoceBackend::raw_all_gather(const DeviceBuffer& input, const DeviceBuffer& output,
                                    uint64_t slot_elements, cudaStream_t stream) {
    if (slot_elements > UINT64_MAX / config_.world) {
        return Failure::fail(FailureCode::overflow, "RoCE all-gather output extent overflowed");
    }
    DeviceBuffer normalized_output = output;
    normalized_output.elements = slot_elements * config_.world;
    return enqueue(input, normalized_output, cuda::Kind::all_gather, slot_elements, stream);
}

Failure RoceBackend::enqueue(const DeviceBuffer& input, const DeviceBuffer& output,
                             cuda::Kind kind, uint64_t slot_elements, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_) return Failure::fail(FailureCode::aborted, "RoCE backend was aborted");
    if (!created_ || !connected_ || proxy_ == nullptr) {
        return Failure::fail(FailureCode::not_ready, "RoCE proxy is not connected");
    }
    if (in_flight_) return Failure::fail(FailureCode::in_flight, "a RoCE operation is already in flight");
    if (stream == nullptr || input.data == nullptr || output.data == nullptr || slot_elements == 0) {
        return Failure::fail(FailureCode::bad_pointer, "RoCE enqueue needs non-empty CUDA buffers and a stream");
    }
    if (input.type != ScalarType::fp32 && input.type != ScalarType::bf16) {
        return Failure::fail(FailureCode::bad_dtype, "RoCE supports FP32 and BF16 only");
    }
    uint64_t input_bytes = 0;
    if (!element_bytes(slot_elements, input.type, input_bytes) || input_bytes > config_.slot_bytes ||
        input_bytes % cuda::kPackBytes != 0 || input_bytes > UINT32_MAX ||
        reinterpret_cast<uintptr_t>(input.data) % cuda::kPackBytes != 0 ||
        reinterpret_cast<uintptr_t>(output.data) % cuda::kPackBytes != 0) {
        return Failure::fail(FailureCode::bad_extent, "RoCE payload must fit a slot and be aligned to 16 bytes");
    }
    if (kind == cuda::Kind::all_reduce_sum && output.elements != slot_elements) {
        return Failure::fail(FailureCode::bad_extent, "RoCE all-reduce input and output sizes differ");
    }
    if (kind == cuda::Kind::all_gather && output.elements != slot_elements * config_.world) {
        return Failure::fail(FailureCode::bad_extent, "RoCE all-gather output is not three equal slots");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t capture_query = cudaStreamIsCapturing(stream, &capture_status);
    if (capture_query != cudaSuccess) return cuda_failure(capture_query, "cudaStreamIsCapturing(RoCE)");
    capture_active_ = capture_status != cudaStreamCaptureStatusNone;
    graph_capture_seen_ = graph_capture_seen_ || capture_active_;
    if (!capture_active_ && capture_replay_pending_) {
        return Failure::fail(FailureCode::in_flight,
                             "a captured RoCE operation must be completed by wait_captured_replay");
    }
    const cudaError_t launched = cuda::launch(input, output, kind, config_.rank, layout_,
                                               region_device_, config_.slot_bytes, device_state_,
                                               config_.timeout_ms * 1000u, stream,
                                               config_.blocks, config_.threads);
    if (launched != cudaSuccess) return cuda_failure(launched, "launch RoCE collective");
    pending_stream_ = stream;
    in_flight_ = true;
    if (capture_active_) {
        if (capture_stream_ == nullptr) {
            capture_stream_ = stream;
            capture_baseline_ = last_completed_;
        } else if (capture_stream_ != stream) {
            sticky_ = Failure::fail(FailureCode::invalid_config,
                                    "RoCE graph collectives must use one capture stream");
            unsafe_to_destroy_ = true;
            return sticky_;
        }
        if (captured_collectives_ == UINT32_MAX ||
            captured_collectives_ >= UINT32_MAX - 1u - capture_baseline_) {
            sticky_ = Failure::fail(FailureCode::overflow, "RoCE graph sequence range is exhausted");
            unsafe_to_destroy_ = true;
            return sticky_;
        }
        ++captured_collectives_;
        captured_pending_ = true;
        in_flight_ = false;
        return ok_failure();
    }
    const cudaError_t recorded = cudaEventRecord(static_cast<cudaEvent_t>(event_), stream);
    if (recorded != cudaSuccess) {
        sticky_ = cuda_failure(recorded, "cudaEventRecord(RoCE collective)");
        unsafe_to_destroy_ = true;
        return sticky_;
    }
    return ok_failure();
}

Failure RoceBackend::raw_poll(Deadline deadline) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sticky_) return sticky_;
    if (aborted_) return Failure::fail(FailureCode::aborted, "RoCE backend was aborted");
    if (!in_flight_) return ok_failure();
    if (captured_pending_ && capture_active_) return ok_failure();
    if (proxy_ != nullptr && roce_failed(static_cast<roce_ctx_t*>(proxy_))) {
        const char* detail = roce_error(static_cast<roce_ctx_t*>(proxy_));
        sticky_ = Failure::fail(FailureCode::async_error,
                                std::string("RoCE proxy failed: ") + (detail ? detail : "unknown error"));
        unsafe_to_destroy_ = true;
        return sticky_;
    }
    uint32_t failed = 0, peer = UINT32_MAX, waited = 0;
    read_u32(region_, layout_.ctrl_off + kControlFailure, failed);
    read_u32(region_, layout_.ctrl_off + kControlPeer, peer);
    read_u32(region_, layout_.ctrl_off + kControlWaitedUs, waited);
    if (failed != 0) {
        sticky_ = Failure::fail(FailureCode::async_error,
                                "RoCE device wait failed at peer " + std::to_string(peer) +
                                    " after " + std::to_string(waited) + " us");
        unsafe_to_destroy_ = true;
        return sticky_;
    }
    const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(event_));
    if (status == cudaErrorNotReady) {
        return Failure::fail(FailureCode::deadline_expired,
                             std::chrono::steady_clock::now() >= deadline.end
                                 ? "the RoCE GPU operation missed its deadline"
                                 : "the RoCE GPU operation is still running");
    }
    if (status != cudaSuccess) {
        sticky_ = cuda_failure(status, "cudaEventQuery(RoCE collective)");
        unsafe_to_destroy_ = true;
        return sticky_;
    }
    uint32_t completed = 0;
    read_u32(region_, layout_.ctrl_off + kControlCompleted, completed);
    const uint32_t expected = capture_replay_pending_
                                  ? capture_baseline_ + captured_collectives_
                                  : last_completed_ + 1u;
    if (completed == 0 || completed != expected) {
        return Failure::fail(FailureCode::deadline_expired, "RoCE host completion sequence is not visible yet");
    }
    if (roce_stat(static_cast<roce_ctx_t*>(proxy_), 2) != completed) {
        return Failure::fail(FailureCode::deadline_expired,
                             "RoCE GPU completed before the proxy drained its RDMA completions");
    }
    last_completed_ = completed;
    in_flight_ = false;
    if (capture_replay_pending_) {
        capture_replay_pending_ = false;
        captured_pending_ = false;
        capture_baseline_ = completed;
    }
    return ok_failure();
}

bool RoceBackend::raw_async_failure(Failure& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sticky_) {
        out = sticky_;
        return true;
    }
    if (proxy_ != nullptr && roce_failed(static_cast<roce_ctx_t*>(proxy_))) {
        const char* detail = roce_error(static_cast<roce_ctx_t*>(proxy_));
        out = Failure::fail(FailureCode::async_error,
                            std::string("RoCE proxy failed: ") + (detail ? detail : "unknown error"));
        return true;
    }
    uint32_t failed = 0;
    if (region_ != nullptr) read_u32(region_, layout_.ctrl_off + kControlFailure, failed);
    if (failed != 0) {
        uint32_t peer = UINT32_MAX, waited = 0;
        read_u32(region_, layout_.ctrl_off + kControlPeer, peer);
        read_u32(region_, layout_.ctrl_off + kControlWaitedUs, waited);
        out = Failure::fail(FailureCode::async_error,
                            "RoCE device wait failed at peer " + std::to_string(peer) +
                                " after " + std::to_string(waited) + " us");
        return true;
    }
    return false;
}

Failure RoceBackend::wait_captured_replay(Deadline deadline) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (captured_collectives_ == 0 || pending_stream_ == nullptr || capture_replay_pending_) {
            return Failure::fail(FailureCode::not_ready, "there is no captured RoCE replay to wait for");
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t query = cudaStreamIsCapturing(pending_stream_, &capture_status);
        if (query != cudaSuccess) {
            sticky_ = cuda_failure(query, "cudaStreamIsCapturing(RoCE replay)");
            unsafe_to_destroy_ = true;
            return sticky_;
        }
        if (capture_status != cudaStreamCaptureStatusNone) {
            return Failure::fail(FailureCode::invalid_config,
                                 "wait_captured_replay must run after CUDA graph launch, outside capture");
        }
        const cudaError_t record = cudaEventRecord(static_cast<cudaEvent_t>(event_), pending_stream_);
        if (record != cudaSuccess) {
            sticky_ = cuda_failure(record, "cudaEventRecord(RoCE graph replay)");
            unsafe_to_destroy_ = true;
            return sticky_;
        }
        capture_active_ = false;
        capture_replay_pending_ = true;
        captured_pending_ = true;
        in_flight_ = true;
    }
    for (;;) {
        Failure pending;
        if (raw_async_failure(pending)) {
            std::lock_guard<std::mutex> lock(mutex_);
            sticky_ = pending;
            unsafe_to_destroy_ = true;
            return sticky_;
        }
        const Failure result = raw_poll(deadline);
        if (!result) {
            std::lock_guard<std::mutex> lock(mutex_);
            return ok_failure();
        }
        if (result.code != FailureCode::deadline_expired) return result;
        if (std::chrono::steady_clock::now() >= deadline.end) return result;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool RoceBackend::requires_process_exit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return unsafe_to_destroy_ || in_flight_ || captured_pending_ || capture_replay_pending_;
}

void RoceBackend::raw_abort() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_.exchange(true)) return;
    sticky_ = Failure::fail(FailureCode::aborted, "RoCE backend was aborted");
    if (in_flight_ || captured_pending_ || capture_replay_pending_) {
        unsafe_to_destroy_ = true;
        return;
    }
    release_safe();
}

void RoceBackend::release_safe() noexcept {
    if (unsafe_to_destroy_) return;
    if (proxy_ != nullptr) {
        roce_stop(static_cast<roce_ctx_t*>(proxy_));
        roce_destroy(static_cast<roce_ctx_t*>(proxy_));
        proxy_ = nullptr;
    }
    if (event_ != nullptr) {
        (void)cudaEventDestroy(static_cast<cudaEvent_t>(event_));
        event_ = nullptr;
    }
    if (device_state_ != nullptr) {
        (void)cudaFree(device_state_);
        device_state_ = nullptr;
    }
    if (region_ != nullptr) {
        (void)cudaFreeHost(region_);
        region_ = nullptr;
        region_device_ = nullptr;
    }
    created_ = false;
    connected_ = false;
}

}  // namespace strata::distributed::roce
