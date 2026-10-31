#include "strata/distributed/parity_trace_writer.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace strata::distributed {
namespace {
bool identifier(const std::string& value) {
    if (value.empty() || value.size() > 128) return false;
    for (const unsigned char c : value)
        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') &&
            !(c >= '0' && c <= '9') && c != '_' && c != '-') return false;
    return true;
}
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

struct ParityTraceWriter::Impl {
    std::filesystem::path directory, current;
    uint32_t timeout_ms;
    std::size_t capacity, sequence = 0;
    float* pinned = nullptr;
    cudaEvent_t event = nullptr;
    bool pending = false, failed = false;
    std::ofstream manifest;

    Impl(std::string path, uint32_t timeout, std::size_t maximum)
        : directory(std::move(path)), timeout_ms(timeout), capacity(maximum) {
        if (directory.empty() || timeout < 100 || timeout > 600000 || !maximum ||
            maximum > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::invalid_argument("invalid parity trace directory, timeout or capacity");
        checked(cudaMallocHost(reinterpret_cast<void**>(&pinned), maximum * sizeof(float)));
        const auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (status != cudaSuccess) { cudaFreeHost(pinned); pinned = nullptr; checked(status); }
    }
    ~Impl() {
        // A timed-out D2H may still own this host memory. Keep it alive until
        // process teardown; freeing it could block or race the pending DMA.
        if (!pending) {
            if (event) cudaEventDestroy(event);
            if (pinned) cudaFreeHost(pinned);
        }
    }
};

ParityTraceWriter::ParityTraceWriter(std::string directory, uint32_t timeout_ms, std::size_t max_elements)
    : impl_(std::make_unique<Impl>(std::move(directory), timeout_ms, max_elements)) {}
ParityTraceWriter::~ParityTraceWriter() = default;

bool ParityTraceWriter::request(const std::string& name, std::string& err) {
    auto& p = *impl_;
    try {
        if (p.failed || p.pending || !identifier(name))
            throw std::runtime_error("parity trace: failed writer or invalid request name");
        std::filesystem::create_directories(p.directory);
        const auto next = p.directory / name;
        if (!std::filesystem::create_directory(next))
            throw std::runtime_error("parity trace: request directory already exists");
        p.manifest.close(); p.manifest.clear();
        p.manifest.open(next / "manifest.jsonl", std::ios::out | std::ios::binary);
        if (!p.manifest) throw std::runtime_error("parity trace: cannot open manifest");
        p.current = next; p.sequence = 0;
        return true;
    } catch (const std::exception& e) { p.failed = true; err = e.what(); return false; }
}

bool ParityTraceWriter::capture(const ParityTraceView& v, void* stream, std::string& err) {
    auto& p = *impl_;
    try {
        if (p.failed || p.pending || p.current.empty() || !v.stage || !identifier(v.stage) ||
            !v.device_data || !stream || !v.elements || v.elements > p.capacity || v.layer < 0 || v.position < 0)
            throw std::runtime_error("parity trace: invalid capture or failed writer");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(p.timeout_ms);
        // An API error can surface earlier asynchronous work. Conservatively
        // retain the destination unless the completion event proves it idle.
        p.pending = true;
        checked(cudaMemcpyAsync(p.pinned, v.device_data, v.elements * sizeof(float),
                                cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)));
        checked(cudaEventRecord(p.event, static_cast<cudaStream_t>(stream)));
        for (;;) {
            const auto status = cudaEventQuery(p.event);
            if (status == cudaSuccess) { p.pending = false; break; }
            if (status != cudaErrorNotReady) checked(status);
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("parity trace: D2H completion deadline expired");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto filename = "trace-" + std::to_string(p.sequence++) + ".bin";
        std::ofstream output(p.current / filename, std::ios::out | std::ios::binary);
        output.write(reinterpret_cast<const char*>(p.pinned), std::streamsize(v.elements * sizeof(float)));
        output.close();
        if (!output) throw std::runtime_error("parity trace: cannot write trace array");
        p.manifest << "{\"stage\":\"" << v.stage << "\",\"layer\":" << v.layer
                   << ",\"position\":" << v.position << ",\"dtype\":\"<f4\",\"elements\":" << v.elements
                   << ",\"path\":\"" << filename << "\"}\n";
        p.manifest.flush();
        if (!p.manifest) throw std::runtime_error("parity trace: cannot append manifest");
        return true;
    } catch (const std::exception& e) { p.failed = true; err = e.what(); return false; }
}
} // namespace strata::distributed
