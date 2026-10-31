#include "strata/distributed/cuda_segment_graph.hpp"
#include "strata/distributed/process_exit_guard.hpp"
#include "strata/kernels/tp_cuda.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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
void qualify(int type) {
    constexpr int width = 2560, ff = 640, k = 10;
    constexpr std::array<uint64_t, 3> valid = {256, 192, 192};
    constexpr std::size_t output_count = std::size_t(k) * width;
    constexpr std::size_t payload_count = 3 * output_count * tp::down_lane_payload_width;
    Stream stream;
    Device wire(payload_count), fixed(output_count), ordinary(output_count), copied(output_count);
    std::vector<float> payload(payload_count, 0), expected(output_count), actual(output_count);
    check(cudaMemsetAsync(wire.data, 0, payload_count * sizeof(float), stream.stream));
    check(cudaStreamSynchronize(stream.stream));
    d::CudaSegmentGraph graph;
    d::PendingDmaExitGuard graph_guard(graph);
    const auto enqueue = [&](void* s) {
        tp::finish_down_lanes(wire.data, type, ff, width, k, valid, fixed.data, s);
    };
    std::string err;
    require(graph.initialize(enqueue, 60000, err), err.c_str());
    require(graph.ready() && graph.node_count() == 1, "finish graph must contain exactly one compute node");
    for (int iteration = 0; iteration < 100; ++iteration) {
        // Change every scale/dot operand while keeping captured addresses fixed.
        // Iteration 50 simulates a request reset; the same graph must see zeros.
        for (std::size_t i = 0; i < payload.size(); i += 2) {
            payload[i] = iteration == 50 ? 0.0f :
                         float(int((i / 2 + std::size_t(iteration) * 13) % 63) - 31) * 0.00390625f;
            payload[i + 1] = iteration == 50 ? 0.0f :
                             float(int((i / 2 * 7 + std::size_t(iteration) * 29) % 2047) - 1023);
        }
        check(cudaMemcpyAsync(wire.data, payload.data(), payload.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream.stream));
        tp::finish_down_lanes(wire.data, type, ff, width, k, valid, ordinary.data, stream.stream);
        require(graph.launch(stream.stream, err), err.c_str());
        check(cudaMemcpyAsync(copied.data, fixed.data, output_count * sizeof(float),
                              cudaMemcpyDeviceToDevice, stream.stream));
        check(cudaMemcpyAsync(expected.data(), ordinary.data, output_count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.stream));
        check(cudaMemcpyAsync(actual.data(), copied.data, output_count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.stream));
        check(cudaStreamSynchronize(stream.stream));
        require(std::memcmp(expected.data(), actual.data(), output_count * sizeof(float)) == 0,
                "finish graph replay differs from ordinary finalizer bits");
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
    std::printf("finish graph codec=%d K=10 width=2560 ff=640 replays=100 exact_bits=PASS reset=PASS terminal=PASS\n", type);
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
        qualify(20);
        qualify(42);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
