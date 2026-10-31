#include "strata/distributed/limits.hpp"
#include "strata/distributed/reference_service.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace strata::distributed {
namespace {
Failure bad(std::string_view text) { return Failure::fail(FailureCode::bad_operation, text); }
bool safe_id(std::string_view value) {
    return !value.empty() && value.size() <= 128 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}
}

ReferenceService::ReferenceService(int32_t vocabulary, uint32_t context, ReferenceExecution execution)
    : vocabulary_(vocabulary), context_(context), execution_(std::move(execution)) {}

Failure ReferenceService::execute(StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
    if (failed_) return bad("reference worker is terminally failed");
    auto reset = [&]() -> Failure {
        output_ready_ = false;
        if (auto f = execution_.reset()) { failed_ = true; return f; }
        position_ = 0; pending_ = false; decoding_ = false; pending_decode_ = false;
        return {};
    };
    if (phase == StepPhase::init) {
        if (initialized_ || vocabulary_ <= 0 || context_ < 4 || context_ > kDistributedMaxContext ||
            !execution_.reset || !execution_.forward || !request.empty() || token != -1 || position)
            return bad("invalid reference initialization");
        if (auto f = reset()) return f;
        initialized_ = true;
        return {};
    }
    if (!initialized_ || !safe_id(request)) return bad("reference requires initialization and a valid request");
    if (phase == StepPhase::prefill || phase == StepPhase::decode) {
        if (pending_ || token < 0 || token >= vocabulary_ || position != position_ || position >= context_ ||
            (!request_.empty() && request != request_) ||
            (request_.empty() && (phase != StepPhase::prefill || position || retired_.contains(std::string(request)))) ||
            (phase == StepPhase::prefill && decoding_)) return bad("invalid reference compute order, identity or position");
        output_ready_ = false;
        bool finite = false;
        if (auto f = execution_.forward(token, position, finite)) { failed_ = true; return f; }
        if (!finite) { failed_ = true; return bad("reference produced non-finite logits"); }
        request_ = request;
        pending_ = true;
        pending_decode_ = phase == StepPhase::decode;
        return {};
    }
    if (request_ != request || token != -1 || position) return bad("invalid reference control identity or position");
    if (phase == StepPhase::commit) {
        if (!pending_) return bad("reference commit requires a completed row");
        ++position_; pending_ = false; decoding_ = decoding_ || pending_decode_; output_ready_ = true;
        return {};
    }
    if (phase == StepPhase::reset || phase == StepPhase::cancel) {
        if (pending_ && phase == StepPhase::reset) return bad("reference reset requires committed state");
        if (auto f = reset()) return f;
        retired_.insert(request_); request_.clear();
        return {};
    }
    return bad("unsupported reference worker phase");
}

std::string reference_event(std::string_view type, std::string_view request_id, int32_t token,
                            uint32_t context, std::string_view state, const RequestTiming* timing) {
    if (type != "ready" && type != "health" && type != "token" && type != "done" && type != "cancelled")
        throw std::invalid_argument("unknown reference event");
    std::string out = "{\"type\":\"" + std::string(type) + "\",\"alive_ranks\":[0]";
    if (type == "ready") {
        if (context < 4 || context > kDistributedMaxContext) throw std::invalid_argument("invalid reference context");
        out += ",\"protocol\":\"strata-reference-v1\",\"mode\":\"autoregressive\",\"world\":1,\"slots\":1,\"max_context\":" + std::to_string(context);
    } else if (type == "health") {
        if (state != "ready" && state != "busy") throw std::invalid_argument("invalid reference health state");
        out += ",\"state\":\"" + std::string(state) + '"';
    } else {
        if (!safe_id(request_id)) throw std::invalid_argument("invalid reference request identity");
        out += ",\"request_id\":\"" + std::string(request_id) + '"';
        if (type == "token") {
            if (token < 0) throw std::invalid_argument("invalid reference token");
            out += ",\"committed\":true,\"id\":" + std::to_string(token);
        } else {
            out += ",\"state\":\"ready\"";
            if (type == "done" && timing) out += ",\"timing\":" + request_timing_json(*timing);
        }
    }
    return out + '}';
}
} // namespace strata::distributed

#ifdef STRATA_REFERENCE_CUDA
#include "strata/distributed/autoregressive.hpp"
#include "strata/distributed/reference.hpp"
#include "strata/distributed/process_exit_guard.hpp"

#include <cuda_runtime.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <poll.h>
#include <random>
#include <thread>
#include <unistd.h>

namespace strata::distributed {
namespace {
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* detail) { if (!condition) throw std::runtime_error(detail); }
class JsonInput {
 public:
    JsonInput() {
        previous_ = ::fcntl(STDIN_FILENO, F_GETFL);
        require(previous_ >= 0 && ::fcntl(STDIN_FILENO, F_SETFL, previous_ | O_NONBLOCK) == 0,
                "cannot set nonblocking serve input");
    }
    ~JsonInput() { if (previous_ >= 0) ::fcntl(STDIN_FILENO, F_SETFL, previous_); }
    Failure poll(uint32_t wait_ms, std::string& line, bool& eof) {
        line.clear(); eof = false;
        auto extract = [&]() {
            const auto end = pending_.find('\n');
            if (end == std::string::npos) return false;
            line = pending_.substr(0, end); pending_.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) throw std::runtime_error("empty serve input line");
            return true;
        };
        if (extract()) return {};
        if (eof_) { eof = true; return {}; }
        pollfd fd{STDIN_FILENO, POLLIN, 0};
        const int ready = ::poll(&fd, 1, int(wait_ms));
        if (ready < 0) {
            if (errno == EINTR) return {};
            return Failure::fail(FailureCode::comm_failed, "serve stdin poll failed");
        }
        if (!ready) return {};
        char buffer[4096];
        // One bounded read per turn also keeps an endlessly writing pipe from delaying group health.
        const auto n = ::read(STDIN_FILENO, buffer, sizeof buffer);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return {};
            return Failure::fail(FailureCode::comm_failed, "serve stdin read failed");
        }
        if (!n) {
            eof_ = true;
            if (!pending_.empty()) return Failure::fail(FailureCode::truncated_frame, "serve input ended before newline");
            eof = true; return {};
        }
        pending_.append(buffer, size_t(n));
        const auto first = pending_.find('\n');
        if ((first == std::string::npos ? pending_.size() : first) > kAutoregressiveMaxLine)
            return Failure::fail(FailureCode::oversized_frame, "serve input exceeds one MiB");
        extract(); return {};
    }
 private:
    int previous_ = -1;
    bool eof_ = false;
    std::string pending_;
};

class JsonOutput {
 public:
    JsonOutput(uint32_t timeout) : timeout_(timeout) {
        {
            fd_ = ::dup(STDOUT_FILENO);
            require(fd_ >= 0, "cannot preserve JSON output channel");
            const int flags = ::fcntl(fd_, F_GETFL);
            require(flags >= 0 && ::fcntl(fd_, F_SETFL, flags | O_NONBLOCK) == 0, "cannot bound JSON output");
        }
        // Core and library diagnostics (including C stdio) go to stderr in this opt-in mode.
        require(::dup2(STDERR_FILENO, STDOUT_FILENO) >= 0, "cannot route service diagnostics to stderr");
    }
    ~JsonOutput() { if (fd_ >= 0) ::close(fd_); }
    void emit(std::string_view json) {
        const std::string line = std::string(json) + '\n';
        size_t offset = 0;
        const auto at = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_);
        while (offset < line.size()) {
            const auto n = ::write(fd_, line.data() + offset, line.size() - offset);
            if (n > 0) { offset += size_t(n); continue; }
            if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                throw std::runtime_error("JSON output channel failed");
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(at - std::chrono::steady_clock::now()).count();
            require(left > 0, "JSON output deadline expired");
            pollfd fd{fd_, POLLOUT, 0};
            const int result = ::poll(&fd, 1, int(std::min<int64_t>(left, 100)));
            require(result >= 0 || errno == EINTR, "JSON output poll failed");
        }
    }
 private:
    int fd_ = -1;
    uint32_t timeout_;
};


std::unique_ptr<JsonOutput> output;
bool request_timing_enabled = false;
struct DeviceBuffer {
    void* pointer = nullptr;
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    void allocate(size_t bytes) { cuda_check(cudaMalloc(&pointer, bytes)); }
};
struct HostBuffer {
    void* pointer = nullptr;
    ~HostBuffer() { if (pointer) cudaFreeHost(pointer); }
    void allocate(size_t bytes) { cuda_check(cudaMallocHost(&pointer, bytes)); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda_check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
struct PendingWork {
    bool pending = false;
    bool requires_process_exit() const noexcept { return pending; }
};

// All allocations and sampler scratch warmup finish before READY. DMA failure exits before owners unwind.
class ReferenceSampler final {
public:
    ReferenceSampler(int32_t vocabulary, uint32_t context, cudaStream_t stream, uint32_t timeout)
        : vocabulary_(vocabulary), context_(context), stream_(stream), timeout_(timeout) {
        warm_logits_.allocate(size_t(vocabulary) * sizeof(float));
        bias_.allocate(size_t(vocabulary) * sizeof(float));
        history_.allocate(size_t(context) * sizeof(int32_t)); token_.allocate(sizeof(int32_t));
        host_logits_.allocate(size_t(vocabulary) * sizeof(float));
        host_bias_.allocate(size_t(vocabulary) * sizeof(float));
        host_history_.allocate(size_t(context) * sizeof(int32_t)); host_token_.allocate(sizeof(int32_t));
        PendingWork work;
        PendingDmaExitGuard guard(work);
        work.pending = true;
        cuda_check(cudaMemsetAsync(warm_logits_.pointer, 0, size_t(vocabulary) * sizeof(float), stream_));
        kernels::SamplerParams params;
        params.top_k = 64; params.top_p = 1.0f; params.seed = 1;
        kernels::sample_tokens(static_cast<const float*>(warm_logits_.pointer), 1, vocabulary_, nullptr, 0,
                               params, static_cast<int*>(token_.pointer), stream_);
        cuda_check(cudaGetLastError());
        wait();
        work.pending = false;
    }
    bool finite(const float* logits) {
        PendingWork work;
        PendingDmaExitGuard guard(work);
        work.pending = true;
        cuda_check(cudaMemcpyAsync(host_logits_.pointer, logits, size_t(vocabulary_) * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        wait();
        work.pending = false;
        const auto* begin = static_cast<const float*>(host_logits_.pointer);
        return std::all_of(begin, begin + vocabulary_, [](float value) { return std::isfinite(value); });
    }
    Failure sample(const float* logits, const ServeRequest& request, const std::vector<int32_t>& history,
                   uint64_t draw, int32_t& out) {
        if (history.empty() || history.size() > context_)
            return bad("reference sampler history exceeds fixed context capacity");
        PendingWork work;
        PendingDmaExitGuard guard(work);
        if (draw == 0) {
            params_ = request.sampling;
            params_.penalty_last_n = std::min(params_.penalty_last_n, int(context_));
            if (!params_.seed) {
                std::random_device entropy;
                params_.seed = (uint64_t(entropy()) << 32) ^ uint64_t(entropy());
                if (!params_.seed) params_.seed = 1;
            }
            auto* bias = static_cast<float*>(host_bias_.pointer);
            std::fill(bias, bias + vocabulary_, 0.0f);
            for (const auto& [id, value] : request.logit_bias) bias[id] = value;
            work.pending = true;
            cuda_check(cudaMemcpyAsync(bias_.pointer, bias, size_t(vocabulary_) * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_));
            params_.logit_bias = request.logit_bias.empty() ? nullptr : static_cast<const float*>(bias_.pointer);
        }
        params_.counter = request.sampling.counter + draw;
        std::memcpy(host_history_.pointer, history.data(), history.size() * sizeof(int32_t));
        work.pending = true;
        cuda_check(cudaMemcpyAsync(history_.pointer, host_history_.pointer, history.size() * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream_));
        kernels::sample_tokens(logits, 1, vocabulary_, static_cast<const int*>(history_.pointer), int(history.size()),
                               params_, static_cast<int*>(token_.pointer), stream_);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(host_token_.pointer, token_.pointer, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        wait();
        work.pending = false;
        out = *static_cast<const int32_t*>(host_token_.pointer);
        return {};
    }
private:
    void wait() {
        cuda_check(cudaEventRecord(event_.value, stream_));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_);
        for (;;) {
            const auto result = cudaEventQuery(event_.value);
            if (result == cudaSuccess) return;
            if (result != cudaErrorNotReady) cuda_check(result);
            cuda_check(cudaGetLastError());
            require(std::chrono::steady_clock::now() < deadline, "reference sampler completion deadline expired");
            std::this_thread::yield();
        }
    }
    int32_t vocabulary_;
    uint32_t context_;
    cudaStream_t stream_;
    uint32_t timeout_;
    DeviceBuffer warm_logits_, bias_, history_, token_;
    HostBuffer host_logits_, host_bias_, host_history_, host_token_;
    Event event_;
    kernels::SamplerParams params_;
};
} // namespace

void prepare_reference_service(uint32_t timeout_ms, bool request_timing) {
    require(!output && timeout_ms >= 100 && timeout_ms <= 600000, "invalid reference output initialization");
    std::signal(SIGPIPE, SIG_IGN);
    request_timing_enabled = request_timing;
    output = std::make_unique<JsonOutput>(timeout_ms);
}

Failure run_reference_service(MonolithicReference& model, int32_t vocabulary, uint32_t context,
                              float* device_logits, void* stream, uint32_t timeout_ms) {
    if (!output || !stream || !device_logits || vocabulary <= 0 || context < 4 || context > kDistributedMaxContext ||
        timeout_ms < 100 || timeout_ms > 600000)
        return Failure::fail(FailureCode::invalid_config, "invalid native reference service configuration");
    JsonInput input;
    ReferenceSampler sampler(vocabulary, context, static_cast<cudaStream_t>(stream), timeout_ms);
    ReferenceExecution execution;
    execution.reset = [&]() -> Failure {
        std::string error;
        // The legacy model cannot attest DMA drainage after failure. Exit before CUDA owners unwind.
        if (!model.reset(stream, error)) {
            std::fprintf(stderr, "reference reset failed: %s\n", error.c_str());
            std::fflush(stderr); std::_Exit(2);
        }
        return {};
    };
    execution.forward = [&](int32_t token, uint64_t position, bool& finite) -> Failure {
        std::string error;
        if (!model.token(token, int64_t(position), 0, device_logits, stream, error)) {
            std::fprintf(stderr, "reference forward failed: %s\n", error.c_str());
            std::fflush(stderr); std::_Exit(2);
        }
        finite = sampler.finite(device_logits);
        return {};
    };
    ReferenceService worker(vocabulary, context, std::move(execution));
    if (auto f = worker.execute(StepPhase::init, {}, -1, 0)) return f;
    AutoregressiveCallbacks callbacks;
    callbacks.poll = [&](uint32_t wait, std::string& line, bool& eof) { return input.poll(wait, line, eof); };
    // There are no peers in world=1. The coordinator's byte exchange is the identity operation.
    callbacks.broadcast = [](std::string_view, std::string&) -> Failure { return {}; };
    callbacks.health = []() -> Failure {
        const auto result = cudaGetLastError();
        return result == cudaSuccess ? Failure{} : Failure::fail(FailureCode::comm_failed, cudaGetErrorString(result));
    };
    callbacks.execute = [&](StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
        return worker.execute(phase, request, token, position);
    };
    callbacks.sample = [&](const ServeRequest& request, const auto& history, uint64_t draw, int32_t& out) {
        if (!worker.output_ready()) return bad("reference sampler requires committed finite logits");
        return sampler.sample(device_logits, request, history, draw, out);
    };
    callbacks.emit = [&](std::string_view type, std::string_view request, int32_t token, std::string_view state) {
        output->emit(reference_event(type, request, token, context, state));
    };
    if (request_timing_enabled)
        callbacks.emit_timing = [&](std::string_view type, std::string_view request, int32_t token,
                                    std::string_view state, const RequestTiming& timing) {
            output->emit(reference_event(type, request, token, context, state,
                                         type == "done" ? &timing : nullptr));
        };
    return run_autoregressive(0, vocabulary, context, callbacks, timeout_ms);
}
} // namespace strata::distributed
#endif
