#include "strata/distributed/parity_trace_writer.hpp"
#include <cuda_runtime.h>

#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

// Host-only CUDA stand-ins: exercise trace ordering and failure lifetime
// without a CUDA context, device, runtime library, or inference process.
namespace {
bool stalled = false;
int copies = 0, frees = 0;
void check(bool ok) { if (!ok) throw std::runtime_error("parity trace writer test failed"); }
}
extern "C" {
cudaError_t cudaMallocHost(void** p, size_t n) { *p = std::malloc(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t cudaFreeHost(void* p) { ++frees; std::free(p); return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
    *event = reinterpret_cast<cudaEvent_t>(uintptr_t(1)); return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaEventQuery(cudaEvent_t) { return stalled ? cudaErrorNotReady : cudaSuccess; }
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind, cudaStream_t) {
    ++copies; std::memcpy(dst, src, bytes); return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "fake CUDA error"; }
}

int main(int argc, char** argv) {
    check(argc == 2);
    namespace fs = std::filesystem;
    using namespace strata::distributed;
    const fs::path directory(argv[1]);
    check(!fs::exists(directory));
    float values[] = {1.0f, -2.5f, 3.25f};
    auto stream = reinterpret_cast<void*>(uintptr_t(1));
    std::string err;
    {
        ParityTraceWriter writer(directory.string(), 100, 3);
        check(writer.request("request-0", err));
        check(writer.capture({"mixer_input", 35, 0, values, 3}, stream, err));
        check(copies == 1);
        float result[3]{};
        std::ifstream input(directory / "request-0/trace-0.bin", std::ios::binary);
        input.read(reinterpret_cast<char*>(result), sizeof(result));
        check(input && std::memcmp(values, result, sizeof(values)) == 0);
        std::ifstream manifest(directory / "request-0/manifest.jsonl");
        std::string line; std::getline(manifest, line);
        check(line.find("\"layer\":35") != std::string::npos);
        check(line.find("\"stage\":\"mixer_input\"") != std::string::npos);
        check(!writer.capture({"expert_output", 35, 0, values, 4}, stream, err));
        check(copies == 1);
        check(!writer.request("request-1", err));
    }
    check(frees == 1);
    {
        ParityTraceWriter writer(directory.string(), 100, 3);
        check(!writer.request("request-0", err));
        check(fs::file_size(directory / "request-0/trace-0.bin") == sizeof(values));
    }
    {
        ParityTraceWriter writer(directory.string(), 100, 3);
        check(!writer.request("../escape", err));
    }
    const int freed_before_timeout = frees;
    {
        ParityTraceWriter writer(directory.string(), 100, 3);
        check(writer.request("timed-out", err));
        stalled = true;
        check(!writer.capture({"residual_post", 0, 0, values, 3}, stream, err));
        check(err.find("deadline") != std::string::npos);
        check(!fs::exists(directory / "timed-out/trace-0.bin"));
        check(fs::file_size(directory / "timed-out/manifest.jsonl") == 0);
        stalled = false;
        const int before_retry = copies;
        check(!writer.capture({"residual_post", 0, 0, values, 3}, stream, err));
        check(copies == before_retry);
    }
    check(frees == freed_before_timeout);
    fs::remove_all(directory);
}
