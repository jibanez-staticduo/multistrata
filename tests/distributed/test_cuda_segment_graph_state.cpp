// CPU execution of the production lifecycle with fake CUDA entry points.
// Link cuda_segment_graph.cpp, without cudart. Numerical qualification is in
// test_cuda_segment_graph.cu and must run separately on CUDA hardware.
#include "strata/distributed/cuda_segment_graph.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
enum class Failure { None, Begin, End, Instantiate, Upload, Launch, Query, Timeout };
Failure failure = Failure::None;
bool capturing = false;
int device = 0, launches = 0, ends = 0, execs = 0, graphs = 0, streams = 0;
cudaGraphNodeType node_type = cudaGraphNodeTypeKernel;
std::size_t node_count = 1;
template<class T> T handle() { return reinterpret_cast<T>(0x1000); }
cudaError_t fail(Failure stage) { return failure == stage ? cudaErrorUnknown : cudaSuccess; }
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void reset() {
    require(!capturing && execs == 0 && graphs == 0 && streams == 0, "CUDA resources leaked");
    failure = Failure::None; device = 0; launches = 0; ends = 0;
    node_type = cudaGraphNodeTypeKernel; node_count = 1;
}
}  // namespace

extern "C" {
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA failure"; }
cudaError_t cudaGetDevice(int* result) { *result = device; return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* result, unsigned int flags) {
    require(flags == cudaStreamNonBlocking, "capture stream must be nonblocking");
    *result = handle<cudaStream_t>(); ++streams; return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t) {
    require(!capturing, "destroyed a stream still in capture"); --streams; return cudaSuccess;
}
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode mode) {
    require(mode == cudaStreamCaptureModeThreadLocal, "wrong capture isolation");
    if (failure == Failure::Begin) return cudaErrorUnknown;
    capturing = true; return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* result) {
    require(capturing, "capture was ended twice");
    capturing = false; ++ends;
    if (failure == Failure::End) { *result = nullptr; return cudaErrorUnknown; }
    *result = handle<cudaGraph_t>(); ++graphs; return cudaSuccess;
}
cudaError_t cudaStreamQuery(cudaStream_t) {
    if (failure == Failure::Timeout) return cudaErrorNotReady;
    return fail(Failure::Query);
}
cudaError_t cudaGraphDestroy(cudaGraph_t) { --graphs; return cudaSuccess; }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { --execs; return cudaSuccess; }
cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t* result, std::size_t* count) {
    *count = node_count;
    if (result && node_count) result[0] = handle<cudaGraphNode_t>();
    return cudaSuccess;
}
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType* result) {
    *result = node_type; return cudaSuccess;
}
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* result, cudaGraph_t, unsigned long long flags) {
    require(flags == 0, "unexpected graph instantiation flags");
    if (failure == Failure::Instantiate) return cudaErrorUnknown;
    *result = handle<cudaGraphExec_t>(); ++execs; return cudaSuccess;
}
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t) { return fail(Failure::Upload); }
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { ++launches; return fail(Failure::Launch); }
}

int main() {
    using strata::distributed::CudaSegmentGraph;
    const auto enqueue = [](void*) {};
    std::string err;
    try {
        for (const auto stage : {Failure::Begin, Failure::End, Failure::Instantiate,
                                 Failure::Upload, Failure::Launch, Failure::Query, Failure::Timeout}) {
            reset();
            const bool pending_failure = stage == Failure::Upload || stage == Failure::Launch ||
                                         stage == Failure::Query || stage == Failure::Timeout;
            // Pending-work failures intentionally retain CUDA handles and the
            // owner must _Exit. Exercise that contract in a disposable child.
            const auto child = pending_failure ? fork() : -1;
            if (pending_failure) require(child >= 0, "fork failed");
            if (child > 0) {
                int status = 0;
                require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                        "pending failure released CUDA resources or lost terminal state");
                continue;
            }
            {
                CudaSegmentGraph graph;
                failure = stage;
                require(!graph.initialize(enqueue, 2, err), "injected initialization failure accepted");
                require(graph.requires_process_exit() == pending_failure, "wrong pending failure latch");
                if (stage == Failure::Timeout)
                    require(err.find("deadline expired") != std::string::npos, "timeout reason lost");
                const int previous_launches = launches;
                failure = Failure::None;
                require(!graph.ready() && !graph.launch(handle<cudaStream_t>(), err), "failed graph replayed");
                require(!graph.initialize(enqueue, 2, err), "failed graph recaptured");
                require(launches == previous_launches, "failed graph reached CUDA launch");
            }
            if (pending_failure) {
                require(streams == 1, "pending stream destroyed");
                if (stage == Failure::Upload || stage == Failure::Launch)
                    require(graphs == 1 && execs == 1, "pending graph/executable destroyed");
                _Exit(0);
            }
        }
        reset();
        {
            CudaSegmentGraph graph;
            int enqueues = 0;
            const auto throws = [&](void*) {
                if (++enqueues == 2) throw std::runtime_error("injected capture enqueue failure");
            };
            require(!graph.initialize(throws, 100, err), "capture exception accepted");
            require(ends == 1 && !capturing, "exception left active capture");
            require(!graph.launch(handle<cudaStream_t>(), err), "exception allowed replay");
        }
        reset();
        for (auto rejected : {cudaGraphNodeTypeHost, cudaGraphNodeTypeMemAlloc, cudaGraphNodeTypeMemcpy}) {
            CudaSegmentGraph graph;
            node_type = rejected;
            require(!graph.initialize(enqueue, 100, err), "non-compute graph accepted");
        }
        reset();
        {
            CudaSegmentGraph graph;
            node_count = 0;
            require(!graph.initialize(enqueue, 100, err), "empty graph accepted");
        }
        reset();
        {
            CudaSegmentGraph graph;
            require(graph.initialize(enqueue, 100, err), "valid graph rejected");
            require(graph.ready() && graph.node_count() == 1, "graph not ready after initialization");
            for (int i = 0; i < 100; ++i)
                require(graph.launch(handle<cudaStream_t>(), err), "valid replay rejected");
            require(!graph.requires_process_exit(), "completed replay incorrectly requires exit");
            failure = Failure::Launch;
            require(!graph.launch(handle<cudaStream_t>(), err), "launch failure accepted");
            const int previous_launches = launches;
            failure = Failure::None;
            require(!graph.launch(handle<cudaStream_t>(), err), "failed launch retried");
            require(launches == previous_launches, "failed graph reached CUDA again");
        }
        reset();
        {
            CudaSegmentGraph graph;
            require(graph.initialize(enqueue, 100, err), "valid graph rejected");
            device = 1;
            const int previous_launches = launches;
            require(!graph.launch(handle<cudaStream_t>(), err), "device mismatch accepted");
            device = 0;
            require(!graph.launch(handle<cudaStream_t>(), err), "device mismatch allowed reuse");
            require(launches == previous_launches, "device mismatch reached launch");
        }
        reset();
        {
            CudaSegmentGraph graph;
            require(graph.initialize(enqueue, 100, err), "valid graph rejected");
            graph.poison();
            require(!graph.ready() && !graph.launch(handle<cudaStream_t>(), err), "poison allowed replay");
        }
        reset();
        {
            const auto child = fork();
            require(child >= 0, "fork failed");
            if (child == 0) {
                {
                    CudaSegmentGraph graph;
                    require(graph.initialize(enqueue, 100, err), "valid graph rejected");
                    require(graph.launch(handle<cudaStream_t>(), err), "valid replay rejected");
                    failure = Failure::Timeout;
                    graph.poison();
                    require(graph.requires_process_exit(), "pending replay poison lost exit latch");
                    require(!graph.launch(handle<cudaStream_t>(), err), "pending poisoned graph replayed");
                    failure = Failure::None;
                    require(graph.requires_process_exit(), "unsafe exit latch was cleared");
                }
                require(execs == 1, "pending graph executable destroyed");
                _Exit(0);
            }
            int status = 0;
            require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                    "pending replay destructor violated fail-stop contract");
        }
        reset();
        std::puts("CUDA segment graph state: PASS (CPU fake driver; no numerical/GPU claim)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
