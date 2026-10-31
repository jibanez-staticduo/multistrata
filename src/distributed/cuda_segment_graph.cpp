#include "strata/distributed/cuda_segment_graph.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::distributed {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct CaptureStream {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    bool capturing = false;
    bool* pending = nullptr;
    ~CaptureStream() {
        if (pending && *pending) return;
        // Invalid capture must still be ended before destroying its stream.
        if (capturing) (void)cudaStreamEndCapture(stream, &graph);
        if (graph) (void)cudaGraphDestroy(graph);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
void wait(cudaStream_t stream, std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        require(std::chrono::steady_clock::now() < deadline,
                "TP CUDA segment graph: initialization deadline expired");
        const auto status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) check(status);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
}  // namespace

struct CudaSegmentGraph::Impl {
    enum class State { Empty, Ready, Failed };
    cudaGraphExec_t exec = nullptr;
    State state = State::Empty;
    std::size_t nodes = 0;
    int device = -1;
    cudaStream_t pending_stream = nullptr;
    bool pending = false;
    bool unsafe = false;
    ~Impl() { if (exec) (void)cudaGraphExecDestroy(exec); }
};

CudaSegmentGraph::CudaSegmentGraph() : impl_(std::make_unique<Impl>()) {}
CudaSegmentGraph::~CudaSegmentGraph() {
    if (requires_process_exit()) (void)impl_.release();
}

bool CudaSegmentGraph::initialize(const Enqueue& enqueue, uint32_t timeout_ms, std::string& err) {
    try {
        require(impl_->state == Impl::State::Empty, "TP CUDA segment graph: initialize called in invalid state");
        impl_->state = Impl::State::Failed;
        require(enqueue && timeout_ms > 0 && timeout_ms <= 3600000,
                "TP CUDA segment graph: invalid enqueue/deadline");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        check(cudaGetDevice(&impl_->device));
        CaptureStream capture;
        capture.pending = &impl_->pending;
        check(cudaStreamCreateWithFlags(&capture.stream, cudaStreamNonBlocking));
        impl_->pending_stream = capture.stream;
        // Resolve lazy kernel loading and validate the fixed buffers before capture.
        impl_->pending = true;
        enqueue(capture.stream);
        wait(capture.stream, deadline);
        impl_->pending = false;
        check(cudaStreamBeginCapture(capture.stream, cudaStreamCaptureModeThreadLocal));
        capture.capturing = true;
        enqueue(capture.stream);
        const auto end_status = cudaStreamEndCapture(capture.stream, &capture.graph);
        capture.capturing = false;
        check(end_status);
        check(cudaGraphGetNodes(capture.graph, nullptr, &impl_->nodes));
        require(impl_->nodes > 0, "TP CUDA segment graph: empty compute segment");
        std::vector<cudaGraphNode_t> nodes(impl_->nodes);
        check(cudaGraphGetNodes(capture.graph, nodes.data(), &impl_->nodes));
        for (auto node : nodes) {
            cudaGraphNodeType type;
            check(cudaGraphNodeGetType(node, &type));
            require(type == cudaGraphNodeTypeKernel, "TP CUDA segment graph: only compute kernel nodes are allowed");
        }
        check(cudaGraphInstantiateWithFlags(&impl_->exec, capture.graph, 0));
        impl_->pending = true;
        check(cudaGraphUpload(impl_->exec, capture.stream));
        check(cudaGraphLaunch(impl_->exec, capture.stream));
        wait(capture.stream, deadline);
        impl_->pending = false;
        impl_->pending_stream = nullptr;
        impl_->state = Impl::State::Ready;
        return true;
    } catch (const std::exception& e) {
        impl_->unsafe = impl_->unsafe || impl_->pending;
        impl_->state = Impl::State::Failed;
        err = e.what();
        return false;
    }
}

bool CudaSegmentGraph::launch(void* stream, std::string& err) {
    try {
        require(ready() && stream, "TP CUDA segment graph: invalid replay state/stream");
        int device = -1;
        check(cudaGetDevice(&device));
        require(device == impl_->device, "TP CUDA segment graph: replay device changed");
        require(!impl_->pending || impl_->pending_stream == static_cast<cudaStream_t>(stream),
                "TP CUDA segment graph: request stream changed with pending replay");
        impl_->pending_stream = static_cast<cudaStream_t>(stream);
        impl_->pending = true;
        check(cudaGraphLaunch(impl_->exec, static_cast<cudaStream_t>(stream)));
        return true;
    } catch (const std::exception& e) {
        if (impl_->pending && cudaStreamQuery(impl_->pending_stream) == cudaSuccess)
            impl_->pending = false;
        impl_->unsafe = impl_->unsafe || impl_->pending;
        impl_->state = Impl::State::Failed;
        err = e.what();
        return false;
    }
}

void CudaSegmentGraph::poison() noexcept {
    (void)requires_process_exit();
    impl_->state = Impl::State::Failed;
}
bool CudaSegmentGraph::ready() const noexcept { return impl_->state == Impl::State::Ready; }
bool CudaSegmentGraph::requires_process_exit() const noexcept {
    if (impl_->unsafe) return true;
    if (impl_->pending) {
        if (cudaStreamQuery(impl_->pending_stream) != cudaSuccess) {
            impl_->unsafe = true;
            return true;
        }
        impl_->pending = false;
    }
    return false;
}
std::size_t CudaSegmentGraph::node_count() const noexcept { return impl_->nodes; }
}  // namespace strata::distributed
