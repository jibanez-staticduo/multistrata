#include "strata/distributed/limits.hpp"
// Opt-in, uncaptured C1 driver. Teacher forcing keeps the original per-rank input;
// autoregressive mode lets rank zero coordinate JSONL requests and sampled IDs.
#include "distributed/nccl_transport.cpp"
#include "strata/distributed/hybrid_backend.hpp"
#include "strata/kernels/native_mmvq.hpp"
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
#include "strata/distributed/roce_backend.hpp"
#endif
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/distributed/expert_model.hpp"
#include "strata/distributed/model_worker.hpp"
#include "strata/distributed/autoregressive.hpp"
#include "strata/distributed/capture_positions.hpp"
#include "strata/distributed/mtp_service.hpp"
#include "strata/distributed/multislot_service.hpp"
#include "strata/distributed/slot_pool.hpp"
#include "strata/distributed/service_evidence.hpp"
#include "strata/distributed/token_timing.hpp"
#include "strata/distributed/parity_trace_writer.hpp"
#include "strata/distributed/process_exit_guard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <poll.h>
#include <random>
#include <thread>
#include <unistd.h>

namespace {
using namespace strata;
using namespace strata::distributed;
void cuda_check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void require(bool ok, const std::string& err) { if (!ok) throw std::runtime_error(err); }
void require(Failure f) { if (f) throw std::runtime_error(f.detail); }
std::string json_quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(char(c)); }
        else if (c < 32) { out += "\\u00"; out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]); }
        else out.push_back(char(c));
    }
    return out + '"';
}
struct Allocation {
    void* pointer = nullptr;
    ~Allocation() { if (pointer) cudaFree(pointer); }
    void allocate(uint64_t bytes) { cuda_check(cudaMalloc(&pointer, bytes)); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct SessionLifetime {
    core::SessionState& state;
    ~SessionLifetime() { core::session_release(state); }
};

struct PinnedAllocation {
    void* pointer = nullptr;
    ~PinnedAllocation() { if (pointer) cudaFreeHost(pointer); }
    void allocate(size_t bytes) { cuda_check(cudaMallocHost(&pointer, bytes)); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda_check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming)); }
    ~Event() { if (value) cudaEventDestroy(value); }
};

#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
struct RoceLifetime {
    strata::distributed::roce::RoceBackend& backend;
    bool requires_process_exit() const noexcept { return backend.requires_process_exit(); }
};

std::vector<std::string> split_csv(const std::string& value, const char* what) {
    std::vector<std::string> result;
    std::istringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) throw std::runtime_error(std::string("invalid ") + what);
        result.push_back(std::move(item));
    }
    if (result.empty() || value.back() == ',') throw std::runtime_error(std::string("invalid ") + what);
    return result;
}

int signed_number(const std::string& value, const char* what) {
    size_t used = 0;
    const long parsed = std::stol(value, &used);
    if (used != value.size() || parsed < INT32_MIN || parsed > INT32_MAX)
        throw std::runtime_error(std::string("invalid ") + what);
    return static_cast<int>(parsed);
}
#endif

class ServiceStateReadback final {
public:
    ServiceStateReadback(cudaStream_t stream, Session& session, GuardedTransport& transport)
        : stream_(stream), session_(session), transport_(transport) { staging_.allocate(capacity_); }
    bool requires_process_exit() const noexcept { return pending_; }
    bool read(void* destination, const void* source, size_t bytes, std::string& error) {
        PendingDmaExitGuard lifetime(*this);
        Deadline deadline;
        if (auto f = session_.operation_deadline(deadline)) { error = f.detail; return false; }
        for (size_t offset = 0; offset < bytes;) {
            const auto n = std::min(capacity_, bytes - offset);
            pending_ = true;
            cuda_check(cudaMemcpyAsync(staging_.pointer, static_cast<const uint8_t*>(source) + offset,
                                       n, cudaMemcpyDeviceToHost, stream_));
            cuda_check(cudaEventRecord(event_.value, stream_));
            for (;;) {
                const auto status = cudaEventQuery(event_.value);
                if (status == cudaSuccess) break;
                if (status != cudaErrorNotReady) { error = cudaGetErrorString(status); return false; }
                if (auto f = transport_.check_health()) { error = f.detail; return false; }
                if (std::chrono::steady_clock::now() >= deadline.end) {
                    error = "service state readback deadline expired"; return false;
                }
                std::this_thread::yield();
            }
            pending_ = false;
            std::memcpy(static_cast<uint8_t*>(destination) + offset, staging_.pointer, n);
            offset += n;
        }
        return true;
    }
private:
    static constexpr size_t capacity_ = size_t(1) << 20;
    cudaStream_t stream_;
    Session& session_;
    GuardedTransport& transport_;
    PinnedAllocation staging_;
    Event event_;
    bool pending_ = false;
};

void write_profile(const SessionCollectives& collectives, const std::string& directory, uint32_t rank) {
    std::filesystem::create_directories(directory);
    const auto path = std::filesystem::path(directory) / ("rank" + std::to_string(rank) + ".csv");
    require(!std::filesystem::exists(path), "profile output must be fresh");
    std::ofstream file(path);
    file << "layer,collective_calls,completed_collectives,route_calls,route_pending_polls,input_bytes,output_bytes,"
            "stage_metadata_ns,operation_agreement_ns,enqueue_ns,wait_ns,route_readback_ns,route_agreement_ns\n";
    for (size_t i = 0; i < collectives.profile().size(); ++i) {
        const auto& row = collectives.profile()[i];
        file << i << ',' << row.collective_calls << ',' << row.completed_collectives << ',' << row.route_calls << ','
             << row.route_pending_polls << ',' << row.input_bytes << ',' << row.output_bytes << ','
             << row.stage_metadata_ns << ',' << row.operation_agreement_ns << ',' << row.enqueue_ns << ','
             << row.wait_ns << ',' << row.route_readback_ns << ',' << row.route_agreement_ns << '\n';
    }
    require(bool(file), "cannot publish profile output");
}

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
    JsonOutput(bool coordinator, uint32_t timeout) : timeout_(timeout) {
        if (coordinator) {
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

// CUDA sampler buffers are fixed before READY. Host staging is pinned, and completion uses the same
// bounded event/health polling as ModelWorker; no stream/device synchronize enters the service loop.
class ServiceSampler {
    struct PendingWork {
        bool pending = false;
        bool requires_process_exit() const noexcept { return pending; }
    };
 public:
    ServiceSampler(size_t vocabulary, size_t context, cudaStream_t stream, Session& session,
                   GuardedTransport& transport, ModelWorkerBackend& execution)
        : vocabulary_(vocabulary), context_(context), stream_(stream), session_(session), transport_(transport), execution_(execution) {
        logits_.allocate(vocabulary * sizeof(float)); bias_.allocate(vocabulary * sizeof(float));
        history_.allocate(context * sizeof(int32_t)); token_.allocate(sizeof(int32_t));
        host_logits_.allocate(vocabulary * sizeof(float)); host_bias_.allocate(vocabulary * sizeof(float));
        host_history_.allocate(context * sizeof(int32_t)); host_token_.allocate(sizeof(int32_t));
        PendingWork work;
        PendingDmaExitGuard lifetime(work);
        // sample_tokens caches split top-k scratch per stream on its first sampled call. Populate it before
        // READY so no request needs to allocate it; these dummy logits never enter model/request state.
        work.pending = true;
        cuda_check(cudaMemsetAsync(logits_.pointer, 0, vocabulary * sizeof(float), stream_));
        kernels::SamplerParams warm;
        warm.top_k = 64; warm.top_p = 1.0f; warm.seed = 1;
        kernels::sample_tokens(static_cast<const float*>(logits_.pointer), 1, int(vocabulary), nullptr, 0,
                               warm, static_cast<int*>(token_.pointer), stream_);
        cuda_check(cudaGetLastError());
        require(wait());
        work.pending = false;
    }
    Failure sample(const ServeRequest& request, const std::vector<int32_t>& history, uint64_t draw, int32_t& out) {
        const auto* logits = execution_.committed_logits();
        if (!logits || execution_.logits_size() != vocabulary_)
            return Failure::fail(FailureCode::bad_operation, "sampler requires committed rank-zero logits");
        if (history.empty() || history.size() > context_)
            return Failure::fail(FailureCode::bad_operation, "sampler history exceeds fixed context capacity");
        // This local guard also covers constructor failure, before an outer owner can observe the sampler.
        PendingWork work;
        PendingDmaExitGuard lifetime(work);
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
            cuda_check(cudaMemcpyAsync(bias_.pointer, bias, vocabulary_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
            params_.logit_bias = request.logit_bias.empty() ? nullptr : static_cast<const float*>(bias_.pointer);
        }
        params_.counter = request.sampling.counter + draw;
        std::memcpy(host_logits_.pointer, logits, vocabulary_ * sizeof(float));
        std::memcpy(host_history_.pointer, history.data(), history.size() * sizeof(int32_t));
        work.pending = true;
        cuda_check(cudaMemcpyAsync(logits_.pointer, host_logits_.pointer, vocabulary_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
        cuda_check(cudaMemcpyAsync(history_.pointer, host_history_.pointer, history.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        kernels::sample_tokens(static_cast<const float*>(logits_.pointer), 1, int(vocabulary_),
                               static_cast<const int*>(history_.pointer), int(history.size()), params_,
                               static_cast<int*>(token_.pointer), stream_);
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(host_token_.pointer, token_.pointer, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
        if (auto f = wait()) return f;
        work.pending = false;
        out = *static_cast<const int32_t*>(host_token_.pointer);
        return {};
    }
 private:
    Failure wait() {
        cuda_check(cudaEventRecord(event_.value, stream_));
        Deadline deadline;
        if (auto f = session_.operation_deadline(deadline)) return f;
        for (;;) {
            const auto status = cudaEventQuery(event_.value);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) return Failure::fail(FailureCode::comm_failed, cudaGetErrorString(status));
            if (auto f = transport_.check_health()) return f;
            if (std::chrono::steady_clock::now() >= deadline.end)
                return Failure::fail(FailureCode::deadline_expired, "sampler completion deadline expired");
            std::this_thread::yield();
        }
        return transport_.check_health();
    }
    size_t vocabulary_;
    size_t context_;
    cudaStream_t stream_;
    Session& session_;
    GuardedTransport& transport_;
    ModelWorkerBackend& execution_;
    Allocation logits_, bias_, history_, token_;
    PinnedAllocation host_logits_, host_bias_, host_history_, host_token_;
    Event event_;
    kernels::SamplerParams params_;
};

void wire_ple(const core::WeightTable& weights, core::SessionState& state,
              kernels::PleTable& table, Allocation& embedding, Allocation& scratch,
              std::vector<float>& host) {
    auto get = [&](const char* name) -> const core::WeightRef& {
        auto* w = weights.find(std::string("blk.1.") + name);
        if (!w || (!w->data && !w->native_data)) throw std::runtime_error("PLE tensor is missing");
        return *w;
    };
    const auto& key = get("ple_key.weight");
    const auto& conv = get("ple_conv1d.weight");
    if (conv.bytes != uint64_t(conv.elements) * 2 ||
        (conv.kind != core::WeightKind::F16InF32 &&
         !(conv.kind == core::WeightKind::Verbatim && !conv.quantized())))
        throw std::runtime_error("PLE convolution must retain F16 engine storage");
    if (!key.quantized()) state.ple.w.key_bf16 = static_cast<const uint16_t*>(key.data);
    else if (key.native_data) {
        state.ple.w.key_native_data = key.native_data;
        state.ple.w.key_native_type = key.native_type;
        state.ple.w.key_native_q8_1 = key.native_q8_1;
    } else {
        state.ple.w.key_codes = static_cast<const uint8_t*>(key.data);
        state.ple.w.key_scales = reinterpret_cast<const float*>(state.ple.w.key_codes + key.codes_bytes);
    }
    state.ple.w.value_bf16 = static_cast<const uint16_t*>(get("ple_value.weight").data);
    state.ple.w.norm_key = static_cast<const float*>(get("ple_norm_key.weight").data);
    state.ple.w.norm_query = static_cast<const float*>(get("ple_norm_query.weight").data);
    state.ple.w.norm_conv = static_cast<const float*>(get("ple_norm_conv.weight").data);
    state.ple.w.conv1d_f16 = static_cast<const uint16_t*>(conv.data);
    embedding.allocate(kernels::NG_N_EMBD * sizeof(float));
    scratch.allocate(core::ple_run_scratch_bytes());
    host.resize(kernels::NG_N_EMBD);
    state.ple.table = &table; state.ple.hist = state.ple_hist;
    state.ple.token = &state.ple_token; state.ple.prev = state.ple_prev;
    state.ple.emb_host = host.data(); state.ple.emb_dev = static_cast<float*>(embedding.pointer);
    state.ple.scratch = static_cast<float*>(scratch.pointer);
    state.ple.consts = kernels::ple_artifact_consts();
}
}

int main(int argc, char** argv) {
    using namespace strata;
    using namespace strata::distributed;
    using namespace strata::distributed::nccl;
    std::cout.setf(std::ios::unitbuf);
    try {
        std::map<std::string, std::string> args;
        std::set<std::string> allowed = {"rank", "rank0-address", "port", "session-uuid",
            "library-path", "nccl-version", "library-provenance", "model-hash", "config-fingerprint",
            "pack", "native", "ple", "context", "logits-dir", "logits-positions", "timeout-ms",
            "state-dir", "state-positions", "state-max-bytes", "trace-dir", "mode", "poll-interval-ms",
            "expert-collectives", "graph-finish", "mtp-path", "mtp-drafts", "mtp-hash", "parallel", "profile-dir",
            "collective-agreement", "service-state-dir", "timing-file", "expert-operands", "request-timing",
            "prefill-batch-size"};
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        allowed.insert("gather-transport");
        allowed.insert("roce-hca-list");
        allowed.insert("roce-route-hca");
        allowed.insert("roce-gid-index");
#endif
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i + 1 == argc || !key.starts_with("--") || !allowed.contains(key.substr(2)) || args.contains(key.substr(2)))
                throw std::runtime_error("unknown, duplicate or incomplete argument");
            args[key.substr(2)] = argv[i + 1];
        }
        auto get = [&](const char* key) -> std::string {
            if (!args.contains(key) || args.at(key).empty()) throw std::runtime_error(std::string("missing --") + key);
            return args.at(key);
        };
        auto number = [&](const char* key, uint64_t fallback) {
            if (!args.contains(key)) return fallback;
            size_t used = 0; const auto n = std::stoull(args.at(key), &used);
            if (used != args.at(key).size()) throw std::runtime_error("invalid numeric argument");
            return uint64_t(n);
        };
        std::set<uint64_t> state_positions;
        bool state_last = false;
        if (args.contains("state-dir")) {
            std::istringstream values(args.contains("state-positions") ? args.at("state-positions") : "0,3,last");
            std::string value;
            while (std::getline(values, value, ',')) {
                if (value == "last") { state_last = true; continue; }
                size_t used = 0;
                const auto position = std::stoull(value, &used);
                require(used == value.size() && state_positions.insert(position).second,
                        "invalid or duplicate state position");
            }
        } else require(!args.contains("state-positions") && !args.contains("state-max-bytes"),
                       "state options require state-dir");
        const uint64_t rank = number("rank", 3), port = number("port", 0);
        const auto timeout = number("timeout-ms", 60000), context = number("context", 256);
        const auto poll_interval = number("poll-interval-ms", 1);
        require(poll_interval <= 1000, "poll interval must be in [0, 1000] ms");
        const auto expert_collectives = args.contains("expert-collectives") ? args.at("expert-collectives") : "sequential";
        require(expert_collectives == "sequential" || expert_collectives == "batched", "invalid expert collective policy");
        const bool batched_experts = expert_collectives == "batched";
        const auto expert_operands_value = args.contains("expert-operands") ? args.at("expert-operands") : "full";
        require(expert_operands_value == "full" || expert_operands_value == "compact",
                "invalid expert operand policy");
        const bool compact_expert_operands = expert_operands_value == "compact";
        const auto gather_transport = args.contains("gather-transport") ? args.at("gather-transport") : "nccl";
        require(gather_transport == "nccl" || gather_transport == "roce", "invalid gather transport");
        const bool hybrid_roce = gather_transport == "roce";
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        require(!hybrid_roce || (args.contains("roce-hca-list") && args.contains("roce-route-hca")),
                "--gather-transport roce requires --roce-hca-list and --roce-route-hca");
        require(hybrid_roce || (!args.contains("roce-hca-list") && !args.contains("roce-route-hca") &&
                                !args.contains("roce-gid-index")),
                "RoCE options require --gather-transport roce");
#else
        require(!hybrid_roce && !args.contains("roce-hca-list") && !args.contains("roce-route-hca") &&
                    !args.contains("roce-gid-index"),
                "RoCE gather transport is unavailable in this build");
#endif
        const auto agreement = args.contains("collective-agreement") ? args.at("collective-agreement") : "per-op";
        require(agreement == "per-op" || agreement == "step", "invalid collective agreement policy");
        const bool preagreed_collectives = agreement == "step";
        const auto graph_finish_value = number("graph-finish", 0);
        require(graph_finish_value <= 1 && (!graph_finish_value || batched_experts),
                "--graph-finish must be 0 or 1 and requires batched expert collectives");
        const bool graph_finish = graph_finish_value != 0;
        const auto mode = args.contains("mode") ? args.at("mode") : "teacher-forced";
        require(mode == "teacher-forced" || mode == "autoregressive" || mode == "multislot", "invalid distributed mode");
        const bool autoregressive = mode == "autoregressive";
        const bool multislot = mode == "multislot";
        const auto prefill_batch_size = number("prefill-batch-size", 1);
        require(prefill_batch_size >= 1 && prefill_batch_size <= 8 &&
                    (prefill_batch_size == 1 || !multislot),
                "--prefill-batch-size 2..8 currently requires C1 mode");
        require(prefill_batch_size == 1 || autoregressive ||
                    (!args.contains("timing-file") && args.contains("logits-positions")),
                "batched prefill qualification requires explicit --logits-positions and no per-token timing");
        if (prefill_batch_size > 1) kernels::native_mmvq_set_multi_exact(true);
        const auto request_timing_value = args.contains("request-timing") ? args.at("request-timing") : "0";
        require((request_timing_value == "0" || request_timing_value == "1") &&
                    (request_timing_value == "0" || autoregressive || multislot),
                "--request-timing 1 requires autoregressive or multislot mode");
        const bool request_timing = request_timing_value == "1";
        require(!args.contains("timing-file") || (!autoregressive && !multislot),
                "--timing-file requires teacher-forced mode");
        require(!args.contains("timing-file") || !args.contains("trace-dir"),
                "--timing-file cannot include trace artifact work inside model forward");
        const bool service_evidence = args.contains("service-state-dir");
        require(!service_evidence || ((autoregressive || multislot) &&
                std::filesystem::path(get("service-state-dir")).is_absolute()),
                "--service-state-dir requires an absolute path and autoregressive or multislot mode");
        const auto parallel = number("parallel", 1);
        require((parallel == 1 || parallel == 4 || parallel == 8) && (multislot || !args.contains("parallel")),
                "--parallel 1,4,8 requires --mode multislot");
        const bool mtp_enabled = args.contains("mtp-path");
        require(prefill_batch_size == 1 || !mtp_enabled,
                "batched prompt processing does not yet supply every target residual row required by MTP");
        const auto mtp_drafts = number("mtp-drafts", 4);
        require((mtp_enabled || (!args.contains("mtp-drafts") && !args.contains("mtp-hash"))) &&
                (!mtp_enabled || (autoregressive && mtp_drafts >= 1 && mtp_drafts <= 4 &&
                                  std::filesystem::path(get("mtp-path")).is_absolute())),
                "MTP requires autoregressive mode, an absolute --mtp-path and --mtp-drafts in [1,4]");
        if (mtp_enabled) {
            const auto expected = get("mtp-hash");
            require(expected.size() == 64 && expected.find_first_not_of("0123456789abcdef") == std::string::npos,
                    "--mtp-hash must be the approved lowercase SHA256 of the runtime file-hash manifest");
            if (rank == 0) {
                require(!std::filesystem::exists(std::filesystem::path(get("mtp-path")) / "draft_vocab.bin"),
                        "distributed MTP requires the full vocabulary; remove the unapproved draft subset");
                std::string manifest = "strata-mtp-runtime-v1\n", error;
                for (const char* file : {"dense.txt", "dense.bin", "experts.bin"}) {
                    std::string hash;
                    require(detail::sha256_of_file((std::filesystem::path(get("mtp-path")) / file).string(), hash, error), error);
                    manifest += std::string(file) + ":" + hash + "\n";
                }
                detail::sha256_ns::Sha256 hash;
                hash.update(reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size());
                const auto digest = hash.finish();
                require(detail::sha256_ns::hex(digest.data(), digest.size()) == expected, "MTP runtime content hash mismatch");
            }
        }
        if (rank >= 3 || port < 1024 || port > 65535 || timeout < 100 || timeout > 600000 || context < 4 ||
            context > kDistributedMaxContext)
            throw std::runtime_error("invalid rank, port, deadline or context");
        const bool selected_logits = args.contains("logits-positions");
        require(!selected_logits || (args.contains("logits-dir") && !get("logits-dir").empty() &&
                                    !autoregressive && !multislot),
                "--logits-positions requires --logits-dir and teacher-forced mode");
        const auto logits_positions = selected_logits ?
            CapturePositions::parse(args.at("logits-positions"), context) : CapturePositions{};
        std::unique_ptr<JsonOutput> json;
        if (autoregressive || multislot) {
            require(!args.contains("trace-dir") && !args.contains("state-dir") && !args.contains("logits-dir"),
                    "autoregressive mode does not accept teacher-forced evidence options");
            json = std::make_unique<JsonOutput>(rank == 0, uint32_t(timeout));
            std::signal(SIGPIPE, SIG_IGN);
        }
        const auto pack = get("pack"), native = get("native"), ple = get("ple");
        const auto shards = gguf_split_paths(native);
        const GgufFile metadata(shards.front());
        core::ModelGeometry geometry;
        require(read_geometry(metadata, geometry).empty(), "invalid model geometry");
        require(geometry_unsupported(geometry).empty(), "unsupported model geometry");
        cuda_check(cudaSetDevice(0));
        Stream stream;

        // The native arithmetic policy is explicit for this executable only.
        kernels::gr_set_fp32_activations(true); kernels::gr_set_native_mmvf(true);
        core::layer_set_native_bf16(true); core::layer_set_fused_gr(false);
        core::layer_set_fused_gdn(false); core::layer_set_fast_attn(false);
        core::layer_set_fast_select(true);
        core::layer_set_native_flash_attn_short(false); core::layer_set_shared_early(true);
        kernels::native_grouped_set_v1(true); kernels::iq_set_old_kernels(true);
        kernels::native_expert_set_mode(0, 0); kernels::native_expert_set_expdec(0);
        kernels::native_gdn_set_enabled(true); kernels::native_router_set_enabled(true);
        kernels::native_qsa_set_enabled(true); kernels::native_qsa_indexer_set_enabled(true);
        kernels::native_rope_set_enabled(true); kernels::native_moe_combine_set_enabled(true);
        kernels::ple_set_native_bf16(true); kernels::ple_set_native_postops(true);
        kernels::shared_expert_set_native_bf16(true);

        core::WeightTable weights;
        std::set<std::string> skip;
        std::string err;
        require(core::NativeDense::served_names(shards, true, skip, err), err);
        require(core::NativeDense::keep_unquantized_ple_key(pack, skip, err), err);
        skip.insert("output.weight"); skip.insert("token_embd.weight");
        uint64_t weight_bytes = 0;
        require(core::WeightTable::pool_bytes(pack, weight_bytes, err, &skip), err);
        Allocation weight_memory;
        weight_memory.allocate(weight_bytes);
        std::cerr << "LOAD rank=" << rank << " stage=dense-pack bytes=" << weight_bytes << std::endl;
        require(weights.load(pack, weight_memory.pointer, weight_bytes, err, &skip), err);
        std::cerr << "LOAD rank=" << rank << " stage=dense-native" << std::endl;
        require(core::check_all(weights, geometry, err), err);
        core::NativeDense dense;
        require(dense.load(shards, weights, err, true), err);
        core::NativeHead head;
        core::NativeEmbed embedding;
        std::cerr << "LOAD rank=" << rank << " stage=head" << std::endl;
        require(head.load(shards, geometry.n_embd, geometry.n_vocab, err), err);
        std::cerr << "LOAD rank=" << rank << " stage=embedding" << std::endl;
        require(embedding.load(shards, geometry.n_embd, geometry.n_vocab, err), err);
        core::set_native_embed(&embedding);
        core::SessionState state;
        Allocation state_memory, ple_embedding, ple_scratch;
        if (!multislot) {
            state_memory.allocate(core::session_bytes(geometry, context, geometry.n_expert_used));
            require(core::session_init(geometry, context, geometry.n_expert_used, state_memory.pointer, state) != 0,
                    "cannot initialize the full model session");
        }
        SessionLifetime state_lifetime{state};
        kernels::PleTable table;
        kernels::PleIoOptions ple_io; ple_io.mode = kernels::PleIo::Mmap; ple_io.cache_rows = 0;
        require(table.open(ple, err, ple_io), err);
        std::vector<float> ple_host;
        if (!multislot) wire_ple(weights, state, table, ple_embedding, ple_scratch, ple_host);

        Credential credential;
        require(Credential::from_env("STRATA_TP3_CREDENTIAL", credential));
        SystemClock clock;
        NcclConfig nccl_config{uint32_t(rank), 3, {}, get("library-path")};
        auto api = make_production_nccl_api(nccl_config, err);
        require(bool(api), err);
        std::string version, provenance;
        require(api->identity(version, provenance));
        require(version == get("nccl-version") && provenance == get("library-provenance"), "NCCL approval mismatch");
        JoinInfo join;
        join.session_uuid = get("session-uuid"); join.rank = rank; join.world = 3;
        join.model_hash = get("model-hash"); join.config_fingerprint = get("config-fingerprint");
        if (preagreed_collectives) join.config_fingerprint += ":collective-agreement=step-v1";
        if (batched_experts || graph_finish || poll_interval != 1)
            join.config_fingerprint += ":tp-policy-v1:expert=" + expert_collectives +
                                       ":poll=" + std::to_string(poll_interval) +
                                       ":graphfinish=" + std::to_string(graph_finish_value);
        if (mtp_enabled) {
            const auto path = get("mtp-path");
            join.config_fingerprint += ":mtp-v1:path=" + std::to_string(path.size()) + ":" + path +
                                       ":drafts=" + std::to_string(mtp_drafts) + ":context=" + std::to_string(context) +
                                       ":sha256=" + get("mtp-hash");
        }
        if (multislot) join.config_fingerprint += ":multislot-v1:slots=" + std::to_string(parallel) +
                                                  ":context=" + std::to_string(context);
        if (compact_expert_operands) join.config_fingerprint += ":expert-operands-v1:compact";
        if (prefill_batch_size > 1)
            join.config_fingerprint += ":prefill-batch-exactnative-v1:rows=" + std::to_string(prefill_batch_size);
        join.backend_identity = hybrid_roce ? std::string(kHybridBackendIdentity) : "nccl";
        if (hybrid_roce) join.config_fingerprint += ":gather-transport=hybrid-roce-v1";
        join.nccl_version_expected = version; join.library_provenance = provenance;
        join.nccl_bootstrap_id = std::string(256, '0');
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        std::unique_ptr<roce::RoceBackend> roce_backend;
        std::unique_ptr<RoceLifetime> roce_owner;
        std::unique_ptr<PendingDmaExitGuard<RoceLifetime>> roce_lifetime;
        std::vector<uint8_t> local_roce_blob;
        if (hybrid_roce) {
            const auto hca_names = split_csv(get("roce-hca-list"), "--roce-hca-list");
            require(hca_names.size() <= 4, "--roce-hca-list supports at most four HCAs");
            const auto routes = split_csv(get("roce-route-hca"), "--roce-route-hca");
            require(routes.size() == kTpWorld, "--roce-route-hca must contain exactly three integers");
            roce::Config roce_config;
            roce_config.rank = uint32_t(rank);
            roce_config.world = kTpWorld;
            roce_config.hca_names = hca_names;
            for (const auto& route : routes) roce_config.route_hca.push_back(signed_number(route, "--roce-route-hca"));
            roce_config.gid_index = args.contains("roce-gid-index")
                                        ? signed_number(args.at("roce-gid-index"), "--roce-gid-index")
                                        : -1;
            roce_config.timeout_ms = uint32_t(timeout);
            roce_config.approved_proxy_sha256 = STRATA_ROCE_PROXY_SHA256;
            roce_backend = std::make_unique<roce::RoceBackend>(std::move(roce_config));
            roce_owner = std::make_unique<RoceLifetime>(RoceLifetime{*roce_backend});
            // This guard is declared before any transport borrows the RoCE-owned resources.
            roce_lifetime = std::make_unique<PendingDmaExitGuard<RoceLifetime>>(*roce_owner);
            require(roce_backend->create());
            require(roce_backend->local_blob(local_roce_blob));
            join.library_provenance += ";roce=" + roce_backend->provenance();
        }
#endif
        if (rank == 0) require(make_nccl_bootstrap_id(nccl_config.library_path, join.nccl_bootstrap_id, err), err);
        if (hybrid_roce) {
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
            HybridBootstrap bundle;
            if (rank == 0) bundle.nccl_id_hex = join.nccl_bootstrap_id;
            bundle.roce_blob = std::move(local_roce_blob);
            require(encode_hybrid_bootstrap(bundle, join.nccl_bootstrap_id));
#else
            throw std::runtime_error("RoCE gather transport is unavailable in this build");
#endif
        }
        TcpChannelConfig tcp;
        tcp.rank = rank; tcp.world = 3; tcp.self = {get("rank0-address"), uint16_t(port)};
        tcp.rank0 = tcp.self; tcp.credential = credential.secret();
        auto channel = make_tcp_control_channel(tcp, clock, err);
        require(bool(channel), err);
        SessionConfig config;
        config.rank = rank; config.session_uuid = join.session_uuid; config.credential = credential;
        config.approved_nccl_version = version; config.approved_library_provenance = provenance;
        if (hybrid_roce) config.approved_library_provenance = join.library_provenance;
        config.budgets = {uint32_t(timeout), uint32_t(timeout), uint32_t(timeout), uint32_t(timeout)};
        Session session(config, join, *channel, clock);
        std::cerr << "LOAD rank=" << rank << " stage=join" << std::endl;
        require(session.start());
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        if (hybrid_roce) {
            std::array<std::string, kTpWorld> fields;
            for (uint32_t peer = 0; peer < kTpWorld; ++peer)
                fields[peer] = session.roster().ranks[peer].nccl_bootstrap_id;
            std::string bootstrap_hex;
            std::vector<uint8_t> rank_ordered_blobs;
            require(decode_hybrid_bootstrap_roster(fields, bootstrap_hex, rank_ordered_blobs));
            nccl_config.bootstrap_id = std::move(bootstrap_hex);
            require(roce_backend->connect(rank_ordered_blobs));
        } else
#endif
        require(session.bootstrap_id(0, nccl_config.bootstrap_id));
        nccl_config.initialize_ms = std::min(config.budgets.initialize_ms, session.roster().budgets.initialize_ms);
        NcclBackend nccl_backend(std::move(api), nccl_config, {version, provenance});
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        std::unique_ptr<HybridBackend> hybrid_backend;
        Backend* active_backend = &nccl_backend;
        if (hybrid_roce) {
            hybrid_backend = std::make_unique<HybridBackend>(nccl_backend, *roce_backend);
            active_backend = hybrid_backend.get();
        }
        GuardedTransport transport(*active_backend, clock);
#else
        GuardedTransport transport(nccl_backend, clock);
#endif
        transport.set_poll_interval_ms(uint32_t(poll_interval));
        require(nccl_backend.init());
        try {
            SessionCollectives collectives(session, transport, clock, uint32_t(timeout));
            collectives.profile(args.contains("profile-dir"));
            NativeRoutedExperts experts(pack, native, geometry, int(rank), collectives,
                                         {256, 192, 192}, uint32_t(timeout), ExpertDownPolicy::ExactLaneReference,
                                         batched_experts ? ExpertCollectivePolicy::Batched : ExpertCollectivePolicy::Sequential,
                                         graph_finish, compact_expert_operands);
            PendingDmaExitGuard expert_lifetime(experts);
            std::cerr << "LOAD rank=" << rank << " stage=expert-shards" << std::endl;
            require(experts.initialize(err), err);
            if (graph_finish) {
                const auto stats = experts.finish_graph_stats();
                std::cerr << "LOAD rank=" << rank << " stage=expert-finish-graphs count=" << stats.graph_count << '\n';
            }
            std::cerr << "LOAD rank=" << rank << " stage=expert-shards-complete bytes=" << experts.owned_weight_bytes() << std::endl;
            std::unique_ptr<ServiceStateReadback> service_readback;
            if (service_evidence) {
                std::cerr << "LOAD rank=" << rank << " stage=service-state-readback external-watchdog-required=1\n";
                service_readback = std::make_unique<ServiceStateReadback>(stream.value, session, transport);
            }
            auto capture_service = [&](StepPhase phase, std::string_view request_id,
                                       const ServiceTranscript& transcript, const ModelForward& model,
                                       const core::SessionState& model_state, const float* logits, size_t logits_count) {
                const auto count = uint64_t(model.next_position());
                require(request_id == transcript.request_id() && transcript.ids().size() == count,
                        "service state and canonical input transcript disagree");
                const StateReadback readback = [&](void* dst, const void* src, size_t n, std::string& error) {
                    return service_readback->read(dst, src, n, error);
                };
                std::vector<StateEvidenceArray> arrays;
                require(describe_session_state(model_state, geometry, count, readback, model.state_sources(), arrays, err), err);
                StateEvidenceRequest evidence;
                evidence.request_id = std::string(request_id);
                evidence.position = count ? count - 1 : 0;
                evidence.token_count = count;
                evidence.max_bytes = uint64_t(4) << 30;
                evidence.reset = count == 0;
                evidence.required = state_evidence_inventory(arrays);
                require(write_service_evidence(args.at("service-state-dir"), uint32_t(rank), phase == StepPhase::cancel,
                        transcript, evidence, arrays, rank == 0 ? logits : nullptr,
                        rank == 0 && logits ? logits_count : 0, err), err);
            };
            if (multislot) {
                SlotPoolConfig pool_config{uint32_t(parallel), uint32_t(context), uint32_t(timeout),
                                            join.session_uuid, join.model_hash, uint32_t(rank), batched_experts, 0,
                                            preagreed_collectives};
                pool_config.expert_operands = {compact_expert_operands, experts.valid_counts(), experts.down_codecs()};
                SlotPool pool(pool_config, {weights, geometry, head, table, collectives, experts, session, transport,
                                            stream.value, [&] { return experts.requires_process_exit(); }});
                PendingDmaExitGuard pool_lifetime(pool);
                require(pool.initialize());
                std::cerr << "LOAD rank=" << rank << " stage=slot-pool-complete slots=" << parallel
                          << " session-arena-bytes=" << pool.allocated_session_bytes() << '\n';
                std::unique_ptr<JsonInput> input;
                std::vector<std::unique_ptr<ServiceSampler>> samplers;
                if (rank == 0) {
                    input = std::make_unique<JsonInput>();
                    for (uint32_t slot = 0; slot < parallel; ++slot)
                        samplers.push_back(std::make_unique<ServiceSampler>(size_t(geometry.n_vocab), size_t(context),
                                             stream.value, session, transport, pool.execution(slot)));
                }
                PendingDmaExitGuard sampler_lifetime(pool);
                std::vector<ServiceTranscript> transcripts(service_evidence ? size_t(parallel) : 0);
                MultiSlotServiceCallbacks callbacks;
                callbacks.poll = [&](uint32_t wait, std::string& line, bool& eof) { return input->poll(wait, line, eof); };
                callbacks.health = [&] { return transport.check_health(); };
                callbacks.scheduler.broadcast = [&](std::string_view label, std::string& bytes) {
                    Deadline at;
                    if (auto f = session.operation_deadline(at)) return f;
                    return session.broadcast_control_payload(label, bytes, at);
                };
                callbacks.scheduler.execute = [&](uint32_t slot, StepPhase phase, std::string_view request,
                                                     int32_t token, uint64_t position) {
                    if (service_evidence && (phase == StepPhase::reset || phase == StepPhase::cancel))
                        capture_service(phase, request, transcripts.at(slot), pool.model(slot), pool.session_state(slot),
                                        pool.committed_logits(slot), pool.logits_size(slot));
                    const auto result = pool.execute(slot, phase, request, token, position);
                    if (pool.requires_process_exit()) std::_Exit(2);
                    if (service_evidence && (result.status == StepStatus::completed || result.status == StepStatus::cancelled))
                        require(transcripts.at(slot).record_success(phase, request, token, position,
                                uint64_t(pool.model(slot).next_position()), err), err);
                    return result.status == StepStatus::completed || result.status == StepStatus::cancelled ?
                        Failure{} : result.failure ? result.failure : Failure::fail(FailureCode::comm_failed, "slot command failed");
                };
                callbacks.scheduler.sample = [&](uint32_t slot, const ServeRequest& request,
                                                    const std::vector<int32_t>& history, uint64_t draw, int32_t& token) {
                    return samplers.at(slot)->sample(request, history, draw, token);
                };
                callbacks.scheduler.abort_all = [&] { pool.abort(); };
                callbacks.emit = [&](std::string_view type, std::string_view request, int32_t token,
                                       std::string_view status, std::string_view reason) {
                    std::string event;
                    if (type == "rejected") {
                        event = "{\"type\":\"rejected\",\"alive_ranks\":[0,1,2],\"request_id\":" +
                                json_quote(request) + ",\"state\":\"ready\",\"reason\":" + json_quote(reason) + '}';
                    } else event = serve_event(type, request, token, uint32_t(context), status);
                    if (type == "ready") event.insert(event.size() - 1, ",\"slots\":" + std::to_string(parallel));
                    json->emit(event);
                };
                if (request_timing) callbacks.emit_timing = [&](std::string_view type, std::string_view request,
                                                               const RequestTiming& timing) {
                    auto event = serve_event(type, request, -1, uint32_t(context), "ready");
                    event.insert(event.size() - 1, ",\"timing\":" + request_timing_json(timing));
                    json->emit(event);
                };
                const auto service_result = run_multislot_service({uint32_t(rank), uint32_t(parallel),
                     int32_t(geometry.n_vocab), uint32_t(context)}, callbacks,
                     std::min(config.budgets.operation_ms, session.roster().budgets.operation_ms));
                if (pool.requires_process_exit()) std::_Exit(2);
                require(service_result);
            } else {
            ModelForward model(weights, geometry, state, int(rank), collectives, experts);
            model.native_head(&head);
            std::unique_ptr<ParityTraceWriter> trace;
            if (args.contains("trace-dir")) {
                trace = std::make_unique<ParityTraceWriter>(args.at("trace-dir"), uint32_t(timeout));
                require(trace->request("request-0", err), err);
                model.parity_observer([&](const ParityTraceView& view, void* s, std::string& error) {
                    return view.position != 0 || trace->capture(view, s, error);
                });
            }
            WorkerConfig worker_config{join.session_uuid, join.model_hash, uint32_t(rank)};
            ModelWorkerConfig model_worker_config{worker_config, stream.value, 0, batched_experts};
            model_worker_config.expert_operands = {compact_expert_operands, experts.valid_counts(),
                                                   experts.down_codecs()};
            model_worker_config.checkpoint_max_context = uint64_t(state.max_cells);
            model_worker_config.preagreed_collectives = preagreed_collectives;
            model_worker_config.prefill_batch_exactnative = prefill_batch_size > 1;
            ModelWorkerBackend execution(model_worker_config, model, collectives, session, transport);
            Worker worker(worker_config, execution);
            PendingDmaExitGuard checkpoint_lifetime(model);
            uint64_t step = 1;
            uint64_t request_sequence = 0;
            auto execute = [&](StepPhase phase, std::vector<int32_t> tokens = {}, std::vector<uint64_t> positions = {}) {
                const std::string request = phase == StepPhase::init ? std::string{} :
                    "teacher-forcing-" + std::to_string(request_sequence);
                StepCommand command{join.session_uuid, request, 0, step++, phase,
                                    std::move(tokens), std::move(positions), join.model_hash, {}};
                require(execution.plan(command, command.expected_collective_schedule));
                auto result = worker.execute(command);
                if (model.requires_process_exit() || experts.requires_process_exit()) std::_Exit(2);
                if (result.status != StepStatus::completed) throw std::runtime_error(result.failure.detail);
                return result;
            };
            execute(StepPhase::init);
            if (autoregressive) {
                std::unique_ptr<JsonInput> input;
                std::unique_ptr<ServiceSampler> sampler;
                std::unique_ptr<MtpService> mtp;
                if (rank == 0) {
                    input = std::make_unique<JsonInput>();
                    sampler = std::make_unique<ServiceSampler>(size_t(geometry.n_vocab), size_t(context),
                                                               stream.value, session, transport, execution);
                    if (mtp_enabled) {
                        mtp = std::make_unique<MtpService>();
                        std::cerr << "LOAD rank=0 stage=mtp-sync-library-capture external-watchdog-required=1\n";
                        require(mtp->initialize(get("mtp-path"), uint32_t(mtp_drafts), geometry, state, weights, head,
                                                stream.value, err), err);
                        std::cerr << "LOAD rank=0 stage=mtp-complete allocated-vram-bytes=" << mtp->vram_bytes()
                                  << " drafts=" << mtp_drafts << " resident-window=" << context << '\n';
                    }
                }
                AutoregressiveCallbacks callbacks;
                ServiceTranscript transcript;
                callbacks.poll = [&](uint32_t wait, std::string& line, bool& eof) {
                    return input->poll(wait, line, eof);
                };
                callbacks.broadcast = [&](std::string_view label, std::string& bytes) {
                    Deadline at;
                    if (auto f = session.operation_deadline(at)) return f;
                    return session.broadcast_control_payload(label, bytes, at);
                };
                callbacks.health = [&] { return transport.check_health(); };
                callbacks.execute = [&](StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
                    if (service_evidence && (phase == StepPhase::reset || phase == StepPhase::cancel))
                        capture_service(phase, request, transcript, model, state,
                                        execution.committed_logits(), execution.logits_size());
                    StepCommand command{join.session_uuid, std::string(request), 0, step++, phase,
                                        {}, {}, join.model_hash, {}};
                    if (phase == StepPhase::prefill || phase == StepPhase::decode) {
                        command.tokens = {token}; command.positions = {position};
                    }
                    if (auto f = execution.plan(command, command.expected_collective_schedule)) return f;
                    const auto result = worker.execute(command);
                    // Checkpoint DMA can outlive a failed call. Exit before sampler/drafter CUDA destructors.
                    if (model.requires_process_exit() || experts.requires_process_exit()) std::_Exit(2);
                    if (result.status != StepStatus::completed && result.status != StepStatus::cancelled)
                        return result.failure ? result.failure : Failure::fail(FailureCode::bad_operation, "worker did not complete serve step");
                    if (service_evidence) require(transcript.record_success(phase, request, token, position,
                                                 uint64_t(model.next_position()), err), err);
                    return Failure{};
                };
                callbacks.prefill_batch_size = uint32_t(prefill_batch_size);
                if (prefill_batch_size > 1) callbacks.execute_prefill_batch =
                    [&](std::string_view request, std::span<const int32_t> tokens, uint64_t position) {
                        StepCommand command{join.session_uuid, std::string(request), 0, step++, StepPhase::prefill,
                                            {tokens.begin(), tokens.end()}, {}, join.model_hash, {}};
                        for (size_t row = 0; row < tokens.size(); ++row) command.positions.push_back(position + row);
                        if (auto f = execution.plan(command, command.expected_collective_schedule)) return f;
                        const auto result = worker.execute(command);
                        if (model.requires_process_exit() || experts.requires_process_exit()) std::_Exit(2);
                        if (result.status != StepStatus::completed)
                            return result.failure ? result.failure : Failure::fail(FailureCode::comm_failed, "prefill chunk failed");
                        if (service_evidence)
                            for (size_t row = 0; row < tokens.size(); ++row)
                                require(transcript.record_success(StepPhase::prefill, request, tokens[row], position + row,
                                                                 position + row + 1, err), err);
                        return Failure{};
                    };
                callbacks.sample = [&](const ServeRequest& request, const std::vector<int32_t>& history,
                                       uint64_t draw, int32_t& token) { return sampler->sample(request, history, draw, token); };
                callbacks.emit = [&](std::string_view type, std::string_view request, int32_t token, std::string_view status) {
                    json->emit(serve_event(type, request, token, uint32_t(context), status));
                };
                if (request_timing) callbacks.emit_timing = [&](std::string_view type, std::string_view request,
                                                               int32_t token, std::string_view status,
                                                               const RequestTiming& timing) {
                    auto event = serve_event(type, request, token, uint32_t(context), status);
                    event.insert(event.size() - 1, ",\"timing\":" + request_timing_json(timing));
                    json->emit(event);
                };
                callbacks.mtp_enabled = mtp_enabled;
                if (mtp_enabled && rank == 0) {
                    callbacks.begin_request = [&](const ServeRequest& request) { return mtp->begin_request(request); };
                    callbacks.prompt_row = [&](int32_t next, uint64_t position) { return mtp->prompt_row(next, position); };
                    callbacks.propose = [&](const std::vector<int32_t>& history, int32_t feedback, uint64_t draw,
                                             std::vector<int32_t>& drafts) { return mtp->propose(history, feedback, draw, drafts); };
                    callbacks.record_target_row = [&](size_t row, int32_t input_id, int32_t next) {
                        return mtp->record_target_row(row, input_id, next);
                    };
                    callbacks.accepted_window = [&](const SpeculativeWindow& window, const SpeculativeResult& result) {
                        return mtp->accepted_window(window, result);
                    };
                }
                const auto service_result = run_autoregressive(uint32_t(rank), int32_t(geometry.n_vocab), uint32_t(context), callbacks,
                                          std::min(config.budgets.operation_ms, session.roster().budgets.operation_ms));
                if (model.requires_process_exit() || experts.requires_process_exit()) std::_Exit(2);
                require(service_result);
            } else {
            std::unique_ptr<TokenTimingWriter> timing;
            if (args.contains("timing-file")) timing = std::make_unique<TokenTimingWriter>(args.at("timing-file"));
            std::set<uint64_t> dumped_positions;
            auto dump_state = [&](bool reset) {
                if (!args.contains("state-dir")) return;
                const auto count = uint64_t(model.next_position());
                if (!reset && (!count || !dumped_positions.insert(count - 1).second)) return;
                std::vector<StateEvidenceArray> arrays;
                const StateReadback readback = [](void* dst, const void* src, size_t n, std::string& error) {
                    const auto status = cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);
                    if (status != cudaSuccess) { error = cudaGetErrorString(status); return false; }
                    return true;
                };
                require(describe_session_state(state, geometry, count, readback, model.state_sources(), arrays, err), err);
                StateEvidenceRequest request;
                request.request_id = "request-" + std::to_string(request_sequence);
                request.position = count ? count - 1 : 0;
                request.token_count = count;
                request.max_bytes = number("state-max-bytes", uint64_t(4) << 30);
                request.reset = reset;
                request.required = state_evidence_inventory(arrays);
                const auto path = std::filesystem::path(args.at("state-dir")) / request.request_id /
                                  (reset ? "reset" : "position-" + std::to_string(count - 1));
                require(write_state_evidence(path.string(), request, arrays, err), err);
            };
            dump_state(true);
            std::cout << "READY rank=" << rank << " tp=3 pp=1 layers=48 c=1"
                      << " policy=native-original-v1-gather-reference"
                      << " grouped_v1=1 old_iq=1 expert_mode=0 expert_expdec=0"
                      << " fused_gr=0 fused_gdn=0 fast_attn=0 flash_short=0 shared_early=1";
            if (selected_logits) std::cout << " logits_positions=" << args.at("logits-positions")
                                           << " logits_last=gen-segment";
            std::cout << '\n';
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "QUIT") { if (state_last) dump_state(false); break; }
                if (line == "CHECKPOINT") {
                    execute(StepPhase::checkpoint);
                    std::cout << "CHECKPOINT_OK position=" << model.next_position() << '\n';
                    continue;
                }
                if (line == "ROLLBACK") {
                    execute(StepPhase::rollback);
                    require(execution.committed_logits() == nullptr, "rollback exposed stale logits");
                    std::cout << "ROLLBACK_OK position=" << model.next_position() << '\n';
                    continue;
                }
                if (line == "RESET") {
                    if (state_last) dump_state(false);
                    execute(StepPhase::reset); ++request_sequence;
                    if (trace) require(trace->request("request-" + std::to_string(request_sequence), err), err);
                    dumped_positions.clear(); dump_state(true);
                    std::cout << "RESET_OK\n"; continue;
                }
                std::istringstream input(line);
                std::string verb; input >> verb;
                if (verb != "GEN") throw std::runtime_error("expected GEN <teacher-forced token IDs>, RESET or QUIT");
                int64_t id;
                std::vector<int32_t> tokens;
                size_t token_index = 0;
                if (selected_logits) {
                    // Parse only the opt-in path ahead of execution, so "last" is
                    // known without retaining logits or changing commit/readback.
                    const auto position = uint64_t(model.next_position());
                    require(position <= context, "position exceeds context");
                    std::string token_word;
                    while (input >> token_word) {
                        const auto [end, status] = std::from_chars(token_word.data(),
                            token_word.data() + token_word.size(), id);
                        require(status == std::errc{} && end == token_word.data() + token_word.size() &&
                                id >= 0 && id < geometry.n_vocab, "invalid token ID");
                        require(tokens.size() < context - position, "GEN exceeds remaining context");
                        tokens.push_back(int32_t(id));
                    }
                    require(!tokens.empty() && input.eof(), "GEN needs valid integer token IDs");
                }
                auto next_token = [&] {
                    if (!selected_logits) return bool(input >> id);
                    if (token_index == tokens.size()) return false;
                    id = tokens[token_index++];
                    return true;
                };
                bool any = false;
                if (prefill_batch_size > 1) {
                    for (size_t begin = 0; begin < tokens.size();) {
                        const size_t count = std::min(size_t(prefill_batch_size), tokens.size() - begin);
                        const auto first = uint64_t(model.next_position());
                        const auto last = first + count - 1;
                        for (uint64_t inner = first; inner < last; ++inner)
                            require(!logits_positions.contains(inner, false) && !state_positions.contains(inner),
                                    "batched qualification captures must be at chunk boundaries");
                        std::vector<int32_t> chunk(tokens.begin() + begin, tokens.begin() + begin + count);
                        std::vector<uint64_t> positions;
                        for (size_t row = 0; row < count; ++row) positions.push_back(first + row);
                        execute(StepPhase::prefill, chunk, positions);
                        const auto result = execute(StepPhase::commit);
                        if (state_positions.contains(last)) dump_state(false);
                        if (rank == 0 && result.output_ready) {
                            const auto* logits = execution.committed_logits();
                            require(logits != nullptr, "batch commit did not publish logits");
                            if (args.contains("logits-dir") && logits_positions.contains(last, begin + count == tokens.size())) {
                                const auto directory = std::filesystem::path(args.at("logits-dir")) /
                                                       ("request-" + std::to_string(request_sequence));
                                std::filesystem::create_directories(directory);
                                std::ofstream file(directory / ("logits-" + std::to_string(last) + ".f32"), std::ios::binary);
                                file.write(reinterpret_cast<const char*>(logits), geometry.n_vocab * sizeof(float));
                                require(bool(file), "cannot write batched logits artifact");
                            }
                            const auto top = std::max_element(logits, logits + geometry.n_vocab) - logits;
                            std::cout << "CHUNK first=" << first << " count=" << count << " last=" << last
                                      << " argmax=" << top << '\n';
                        }
                        begin += count;
                    }
                    continue;
                }
                while (next_token()) {
                    if (id < 0 || id >= geometry.n_vocab) throw std::runtime_error("invalid token ID");
                    any = true;
                    const auto pos = model.next_position();
                    TokenRowTimer row_timer{bool(timing)};
                    require(uint64_t(pos) < context, "GEN exceeds remaining context");
                    execute(StepPhase::prefill, {int32_t(id)}, {uint64_t(pos)});
                    const auto result = execute(StepPhase::commit);
                    row_timer.forward_done();
                    // Forward includes finite readback and the all-rank commit.
                    row_timer.readback_done();
                    if (state_positions.contains(uint64_t(pos))) dump_state(false);
                    if (rank == 0 && result.output_ready) {
                        const auto* logits = execution.committed_logits();
                        require(logits != nullptr, "commit did not publish logits");
                        const auto top = std::max_element(logits, logits + geometry.n_vocab) - logits;
                        if (args.contains("logits-dir") && logits_positions.contains(uint64_t(pos),
                                selected_logits && token_index == tokens.size())) {
                            const auto directory = std::filesystem::path(args.at("logits-dir")) /
                                                   ("request-" + std::to_string(request_sequence));
                            std::filesystem::create_directories(directory);
                            const auto path = directory / ("logits-" + std::to_string(pos) + ".f32");
                            std::ofstream file(path, std::ios::binary);
                            file.write(reinterpret_cast<const char*>(logits), geometry.n_vocab * sizeof(float));
                            require(bool(file), "cannot write logits artifact");
                        }
                        if (timing) timing->write("teacher-forcing-" + std::to_string(request_sequence), uint64_t(pos),
                                                  int32_t(id), "teacher-forced", row_timer.finish());
                        std::cout << "TOKEN position=" << pos << " input=" << id << " argmax=" << top << '\n';
                    }
                    if (rank != 0 && timing)
                        timing->write("teacher-forcing-" + std::to_string(request_sequence), uint64_t(pos), int32_t(id),
                                      "teacher-forced", row_timer.finish());
                }
                require(any && (selected_logits || input.eof()), "GEN needs valid integer token IDs");
            }
            if (timing) timing->close();
            }
            }
            if (args.contains("profile-dir")) write_profile(collectives, args.at("profile-dir"), uint32_t(rank));
            transport.abort("driver finished");
            require(session.close());
        } catch (...) { transport.abort("model worker failed"); session.abort("model worker failed"); throw; }
        core::set_native_embed(nullptr);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "distributed model: " << e.what() << '\n';
        return 1;
    }
}
