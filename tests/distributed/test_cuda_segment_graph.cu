#include "strata/distributed/cuda_segment_graph.hpp"
#include "strata/distributed/process_exit_guard.hpp"
#include "strata/kernels/tp_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = strata::kernels::tp::cuda;
namespace d = strata::distributed;
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Stream {
    cudaStream_t stream = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Stream() { if (stream) (void)cudaStreamDestroy(stream); }
};
struct Device {
    float* data = nullptr;
    explicit Device(std::size_t count) { check(cudaMalloc(&data, count * sizeof(float))); }
    ~Device() { if (data) (void)cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
void qualify(int type, bool compact) {
    constexpr int width = 2560, ff = 640, k = 10;
    constexpr std::array<uint64_t, 3> valid = {256, 192, 192};
    constexpr std::size_t output_count = std::size_t(k) * width;
    constexpr int full_slot_width = tp::down_lane_payload_width;
    const int slot_width = compact ? tp::compact_down_payload_width(type, ff, valid) : full_slot_width;
    const std::size_t payload_count = 3 * output_count * slot_width;
    const int qk = type == 20 ? 32 : 64;
    require(slot_width == (compact ? (type == 20 ? 32 : 16) : full_slot_width),
            "finish graph payload width regression");
    constexpr std::size_t full_payload_count = 3 * output_count * full_slot_width;
    Stream stream;
    Device wire(payload_count), fixed(output_count), ordinary(output_count), copied(output_count);
    std::vector<float> payload(payload_count, 0), expected(output_count), actual(output_count);
    std::unique_ptr<Device> full_wire, full_reference;
    std::vector<float> full_payload, full_actual;
    if (compact) {
        full_wire = std::make_unique<Device>(full_payload_count);
        full_reference = std::make_unique<Device>(output_count);
        full_payload.resize(full_payload_count);
        full_actual.resize(output_count);
    }
    check(cudaMemsetAsync(wire.data, 0, payload_count * sizeof(float), stream.stream));
    check(cudaStreamSynchronize(stream.stream));
    d::CudaSegmentGraph graph;
    d::PendingDmaExitGuard graph_guard(graph);
    const auto enqueue = [&](void* s) {
        if (compact)
            tp::finish_down_lanes_compact(wire.data, type, ff, width, k, valid, slot_width, fixed.data, s);
        else
            tp::finish_down_lanes(wire.data, type, ff, width, k, valid, fixed.data, s);
    };
    std::string err;
    require(graph.initialize(enqueue, 60000, err), err.c_str());
    require(graph.ready() && graph.node_count() == 1, "finish graph must contain exactly one compute node");
    for (int iteration = 0; iteration < 100; ++iteration) {
        // Keep equivalent global calls in both layouts; iteration 50 simulates reset.
        std::fill(payload.begin(), payload.end(), 0.0f);
        if (compact) std::fill(full_payload.begin(), full_payload.end(), 0.0f);
        int begin_call = 0;
        for (int rank = 0; rank < 3; ++rank) {
            const int rank_calls = int(valid[rank] / qk * 2);
            for (int row = 0; row < k; ++row) for (int out = 0; out < width; ++out) {
                const auto base = ((std::size_t(rank) * k + row) * width + out) * slot_width;
                const auto full_base = ((std::size_t(rank) * k + row) * width + out) * full_slot_width;
                for (int local_call = 0; local_call < rank_calls; ++local_call) {
                    const int call = begin_call + local_call;
                    const auto pair = compact ? std::size_t(local_call) * 2 :
                        std::size_t(call / 32) * 64 + std::size_t(call % 32) * 2;
                    const float scale = iteration == 50 ? 0.0f :
                        float(int((std::size_t(row * width + out) * 67 + call * 13 +
                                   std::size_t(iteration) * 17) % 63) - 31) * 0.00390625f;
                    const float dot = iteration == 50 ? 0.0f :
                        float(int((std::size_t(row * width + out) * 97 + call * 29 +
                                   std::size_t(iteration) * 31) % 2047) - 1023);
                    payload[base + pair] = scale;
                    payload[base + pair + 1] = dot;
                    if (compact) {
                        full_payload[full_base + std::size_t(call) * 2] = scale;
                        full_payload[full_base + std::size_t(call) * 2 + 1] = dot;
                    }
                }
            }
            begin_call += rank_calls;
        }
        check(cudaMemcpyAsync(wire.data, payload.data(), payload.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream.stream));
        if (compact)
            tp::finish_down_lanes_compact(wire.data, type, ff, width, k, valid, slot_width,
                                          ordinary.data, stream.stream);
        else
            tp::finish_down_lanes(wire.data, type, ff, width, k, valid, ordinary.data, stream.stream);
        if (compact) {
            check(cudaMemcpyAsync(full_wire->data, full_payload.data(), full_payload.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream.stream));
            tp::finish_down_lanes(full_wire->data, type, ff, width, k, valid,
                                  full_reference->data, stream.stream);
        }
        require(graph.launch(stream.stream, err), err.c_str());
        check(cudaMemcpyAsync(copied.data, fixed.data, output_count * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream.stream));
        check(cudaMemcpyAsync(expected.data(), ordinary.data, output_count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.stream));
        check(cudaMemcpyAsync(actual.data(), copied.data, output_count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.stream));
        if (compact)
            check(cudaMemcpyAsync(full_actual.data(), full_reference->data, output_count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream.stream));
        check(cudaStreamSynchronize(stream.stream));
        require(std::memcmp(expected.data(), actual.data(), output_count * sizeof(float)) == 0,
                "finish graph replay differs from ordinary finalizer bits");
        if (compact)
            require(std::memcmp(full_actual.data(), actual.data(), output_count * sizeof(float)) == 0,
                    "compact finish graph differs from full-layout finalizer bits");
        for (auto value : actual) require(std::isfinite(value), "non-finite graph output");
        if (iteration == 50)
            for (auto value : actual) require(value == 0.0f, "reset operands left stale output");
    }
    // A rejected replay invalidates this graph, including subsequent valid calls.
    require(!graph.launch(nullptr, err), "invalid replay stream accepted");
    require(!graph.ready() && !graph.launch(stream.stream, err), "failed graph was reused");
    require(!graph.initialize(enqueue, 60000, err), "failed graph was recaptured");

    // A throw during active capture must end capture and leave the object terminal.
    d::CudaSegmentGraph broken;
    d::PendingDmaExitGuard broken_guard(broken);
    int calls = 0;
    const auto throws_in_capture = [&](void* s) {
        enqueue(s);
        if (++calls == 2) throw std::runtime_error("injected capture failure");
    };
    require(!broken.initialize(throws_in_capture, 60000, err), "capture failure was accepted");
    require(err == "injected capture failure", "capture failure detail lost");
    require(!broken.launch(stream.stream, err) && !broken.initialize(enqueue, 60000, err),
            "capture failure was reused");
    d::CudaSegmentGraph after_failure;
    d::PendingDmaExitGuard after_failure_guard(after_failure);
    require(after_failure.initialize(enqueue, 60000, err), "failed capture contaminated a fresh graph");
    after_failure.poison();
    require(!after_failure.launch(stream.stream, err), "poisoned graph was replayed");

    d::CudaSegmentGraph no_compute;
    d::PendingDmaExitGuard no_compute_guard(no_compute);
    require(!no_compute.initialize([](void*) {}, 60000, err), "empty graph was accepted");
    std::printf("finish graph codec=%d operands=%s slot_width=%d K=10 width=2560 ff=640 "
                "replays=100 exact_bits=PASS reset=PASS terminal=PASS\n",
                type, compact ? "compact" : "full", slot_width);
}
}  // namespace

int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "SKIP: CUDA device unavailable\n");
        return 77;
    }
    try {
        qualify(20, false);
        qualify(20, true);
        qualify(42, false);
        qualify(42, true);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
