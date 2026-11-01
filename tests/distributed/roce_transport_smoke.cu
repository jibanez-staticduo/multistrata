// Private, opt-in three-host RoCEnante smoke. One process and logical CUDA device 0 per host.
#include "strata/distributed/process_exit_guard.hpp"
#include "strata/distributed/roce_backend.hpp"
#include "strata/distributed/session.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace smoke {
using namespace strata::distributed;
using namespace strata::distributed::roce;
const char* last_failure_code = "runtime_error";

struct Options {
    uint32_t rank = 0;
    uint32_t timeout_ms = 30000;
    uint32_t stripes = 1;
    uint32_t blocks = 1;
    int gid_index = -1;
    uint16_t port = 0;
    std::string rank0_address;
    std::string session_uuid;
    std::string credential_env = "STRATA_TP3_CREDENTIAL";
    std::string hca_list;
    std::string route_hca;
    std::string fault = "none";
    uint32_t fault_rank = 2;
    uint64_t slot_bytes = 4 * 1024 * 1024;
    std::string model_hash;
    std::string config_fingerprint;
    std::string build_commit;
    std::string proxy_sha256 = STRATA_ROCE_PROXY_SHA256;
};

uint64_t number(std::string_view value, uint64_t maximum) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::runtime_error("invalid numeric option");
    size_t used = 0;
    const auto parsed = std::stoull(std::string(value), &used);
    if (used != value.size() || parsed > maximum) throw std::runtime_error("invalid numeric option");
    return parsed;
}

std::vector<std::string> split(std::string_view value, char separator) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(separator, start);
        const auto part = value.substr(start, end == std::string_view::npos ? value.size() - start : end - start);
        if (part.empty()) throw std::runtime_error("empty list element");
        result.emplace_back(part);
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return result;
}

Options parse(int argc, char** argv) {
    const std::vector<std::string> allowed = {
        "rank", "rank0-address", "port", "session-uuid", "credential-env", "hca-list", "route-hca",
        "stripes", "blocks", "gid-index", "timeout-ms", "fault", "fault-rank", "slot-bytes",
        "model-hash", "config-fingerprint", "build-commit"};
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        std::string key = argv[i];
        if (i + 1 >= argc || key.rfind("--", 0) != 0) throw std::runtime_error("option needs a value");
        key.erase(0, 2);
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end() || args.contains(key))
            throw std::runtime_error("unknown or duplicate option");
        args[key] = argv[i + 1];
    }
    auto get = [&](const char* key) -> std::string {
        if (!args.contains(key) || args.at(key).empty()) throw std::runtime_error("missing option");
        return args.at(key);
    };
    Options o;
    o.rank = static_cast<uint32_t>(number(get("rank"), UINT32_MAX));
    o.rank0_address = get("rank0-address");
    const auto port = number(get("port"), UINT16_MAX);
    if (port < 1024) throw std::runtime_error("invalid port");
    o.port = static_cast<uint16_t>(port);
    o.session_uuid = get("session-uuid");
    o.hca_list = args.contains("hca-list") ? args.at("hca-list") : "";
    o.route_hca = get("route-hca");
    o.model_hash = get("model-hash");
    o.config_fingerprint = get("config-fingerprint");
    o.build_commit = get("build-commit");
    if (args.contains("credential-env")) o.credential_env = get("credential-env");
    if (args.contains("timeout-ms")) o.timeout_ms = static_cast<uint32_t>(number(args.at("timeout-ms"), UINT32_MAX));
    if (args.contains("stripes")) o.stripes = static_cast<uint32_t>(number(args.at("stripes"), UINT32_MAX));
    if (args.contains("blocks")) o.blocks = static_cast<uint32_t>(number(args.at("blocks"), 8));
    if (args.contains("gid-index")) {
        if (args.at("gid-index") == "-1") o.gid_index = -1;
        else o.gid_index = static_cast<int>(number(args.at("gid-index"), UINT8_MAX));
    }
    if (args.contains("fault")) o.fault = args.at("fault");
    if (args.contains("fault-rank")) o.fault_rank = static_cast<uint32_t>(number(args.at("fault-rank"), UINT32_MAX));
    if (args.contains("slot-bytes")) o.slot_bytes = number(args.at("slot-bytes"), UINT64_MAX);
    if (o.rank >= kTpWorld || o.fault_rank >= kTpWorld || o.stripes != 1 || (o.blocks != 1 && o.blocks != 8) ||
        o.timeout_ms < 100 || o.timeout_ms > 60000 ||
        (o.fault != "none" && o.fault != "missing-rank" && o.fault != "killed-peer" &&
         o.fault != "graph-killed-peer") ||
        (o.fault == "graph-killed-peer" && o.fault_rank != 2)) {
        throw std::runtime_error("invalid test configuration");
    }
    return o;
}

void require(Failure failure) {
    if (failure) {
        last_failure_code = failure_code_name(failure.code);
        throw std::runtime_error(last_failure_code);
    }
}

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorName(error));
}

class Allocation {
public:
    explicit Allocation(size_t bytes) { cuda_check(cudaMalloc(&data_, bytes)); }
    ~Allocation() { if (data_) (void)cudaFree(data_); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    void* get() const { return data_; }
private:
    void* data_ = nullptr;
};

class Stream {
public:
    Stream() { cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { if (stream_) (void)cudaStreamDestroy(stream_); }
    operator cudaStream_t() const { return stream_; }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_{};
};

// Construct after every CUDA owner so unsafe backend failures exit before borrowed buffers unwind.
Deadline deadline(Session& session) {
    Deadline at;
    require(session.operation_deadline(at));
    return at;
}

template <typename T> float to_float(T value) { return static_cast<float>(value); }
template <> float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> T from_float(float value) { return value; }
template <> __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }

float reduction_input(uint32_t rank, size_t i) {
    return static_cast<float>(rank + 1) * 0.5f + static_cast<float>(i % 29) * 0.03125f;
}

float gather_value(uint32_t rank, size_t i) {
    return static_cast<float>(rank * 10000 + static_cast<uint32_t>(i)) * 0.125f;
}

void wait_stream(cudaStream_t stream, Deadline at) {
    for (;;) {
        const cudaError_t status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) std::_Exit(2);
        if (std::chrono::steady_clock::now() >= at.end) std::_Exit(2);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

class Smoke {
public:
    Smoke(const Options& options, Session& session, RoceBackend& backend, GuardedTransport& transport)
        : options_(options), session_(session), backend_(backend), transport_(transport) {}

    void run() {
        for (const ScalarType type : {ScalarType::fp32, ScalarType::bf16}) {
            if (type == ScalarType::fp32) {
                for (const size_t count : aligned_counts(sizeof(float))) reduce<float>(count, type);
                gather<float>(type);
            } else {
                for (const size_t count : aligned_counts(sizeof(__nv_bfloat16))) reduce<__nv_bfloat16>(count, type);
                gather<__nv_bfloat16>(type);
            }
        }
        graph_replays();
        require(transport_.check_health());
        std::cout << "{\"event\":\"result\",\"status\":\"passed\",\"rank\":" << options_.rank
                  << ",\"reduce_checks\":" << reductions_ << ",\"gather_checks\":" << gathers_
                  << ",\"graph_replays\":100,"
                  << "\"tails\":\"unsupported: payload byte count must be divisible by 16\"}\n";
    }

private:
    static std::vector<size_t> aligned_counts(size_t width) {
        // Include non-power-of-two and boundary cases whose payloads meet the 16-byte wire granularity.
        std::vector<size_t> counts{16 / width, 48 / width, 80 / width, 256 / width, 272 / width, 640 / width,
                                   4096 / width, 65536 / width, (1u << 20) / width};
        return counts;
    }

    Operation next_operation() {
        ++sequence_;
        return {sequence_, 0, sequence_};
    }

    void agree(const Operation& op, CollectiveKind kind, ScalarType type, uint64_t elements,
               std::vector<uint64_t> valid_counts = {}) {
        require(session_.agree_operation({op, kind, type, elements, std::move(valid_counts)}, deadline(session_)));
    }

    void complete(const Operation& op) {
        require(session_.complete_operation(op));
    }

    Deadline graph_replay_deadline() {
        Deadline at = deadline(session_);
        if (options_.fault == "graph-killed-peer") {
            at.end = std::min(at.end, std::chrono::steady_clock::now() + std::chrono::seconds(8));
        }
        return at;
    }

    template <typename T>
    void reduce(size_t count, ScalarType type) {
        std::vector<T> input(count), output(count);
        for (size_t i = 0; i < count; ++i) input[i] = from_float<T>(reduction_input(options_.rank, i));
        Allocation send(count * sizeof(T)), receive(count * sizeof(T));
        PendingDmaExitGuard<RoceBackend> dma_guard(backend_);
        cuda_check(cudaMemcpyAsync(send.get(), input.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream_.get()));
        wait_stream(stream_.get(), deadline(session_));
        const auto op = next_operation();
        agree(op, CollectiveKind::all_reduce_sum, type, count);
        if (op.sequence == 1 && options_.fault == "killed-peer" && options_.rank == options_.fault_rank) {
            std::cout << "{\"event\":\"injected_peer_exit\",\"rank\":" << options_.rank << "}\n";
            std::cout.flush();
            std::_Exit(86);
        }
        require(transport_.all_reduce_sum({send.get(), count, type}, {receive.get(), count, type}, op, stream_));
        require(transport_.wait(deadline(session_)));
        complete(op);
        cuda_check(cudaMemcpyAsync(output.data(), receive.get(), count * sizeof(T), cudaMemcpyDeviceToHost, stream_.get()));
        wait_stream(stream_.get(), deadline(session_));
        for (size_t i = 0; i < count; ++i) {
            float expected = 0.0f;
            for (uint32_t rank = 0; rank < kTpWorld; ++rank)
                expected += to_float(from_float<T>(reduction_input(rank, i)));
            expected = to_float(from_float<T>(expected));
            const float got = to_float(output[i]);
            if (!std::isfinite(got) || got != expected)
                throw std::runtime_error("all_reduce numeric mismatch");
        }
        ++reductions_;
    }

    template <typename T>
    void gather(ScalarType type) {
        constexpr std::array<uint64_t, kTpWorld> valid{256, 192, 192};
        constexpr size_t slot = 256;
        std::vector<T> input(slot, from_float<T>(-999.0f)), output(slot * kTpWorld);
        for (size_t i = 0; i < valid[options_.rank]; ++i)
            input[i] = from_float<T>(gather_value(options_.rank, i));
        Allocation send(slot * sizeof(T)), receive(slot * kTpWorld * sizeof(T));
        PendingDmaExitGuard<RoceBackend> dma_guard(backend_);
        cuda_check(cudaMemcpyAsync(send.get(), input.data(), slot * sizeof(T), cudaMemcpyHostToDevice, stream_.get()));
        wait_stream(stream_.get(), deadline(session_));
        const auto op = next_operation();
        agree(op, CollectiveKind::all_gather_padded, type, slot, {valid.begin(), valid.end()});
        require(transport_.all_gather_padded({send.get(), slot, type},
                                             {receive.get(), slot * kTpWorld, type}, valid, op, stream_));
        require(transport_.wait(deadline(session_)));
        complete(op);
        cuda_check(cudaMemcpyAsync(output.data(), receive.get(), output.size() * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream_.get()));
        wait_stream(stream_.get(), deadline(session_));
        for (uint32_t rank = 0; rank < kTpWorld; ++rank) {
            for (size_t i = 0; i < valid[rank]; ++i) {
                const float expected = to_float(from_float<T>(gather_value(rank, i)));
                const float got = to_float(output[rank * slot + i]);
                if (!std::isfinite(got) || got != expected) throw std::runtime_error("all_gather value mismatch");
            }
        }
        ++gathers_;
    }

    void graph_replays() {
        constexpr std::array<size_t, 3> counts{256, 512, 1024};
        std::array<std::vector<float>, counts.size()> input;
        std::array<std::unique_ptr<Allocation>, counts.size()> sends;
        std::array<std::unique_ptr<Allocation>, counts.size()> receives;
        for (size_t op_index = 0; op_index < counts.size(); ++op_index) {
            input[op_index].resize(counts[op_index]);
            for (size_t i = 0; i < counts[op_index]; ++i)
                input[op_index][i] = reduction_input(options_.rank, i);
            sends[op_index] = std::make_unique<Allocation>(counts[op_index] * sizeof(float));
            receives[op_index] = std::make_unique<Allocation>(counts[op_index] * sizeof(float));
            cuda_check(cudaMemcpy(sends[op_index]->get(), input[op_index].data(),
                                  counts[op_index] * sizeof(float), cudaMemcpyHostToDevice));
        }
        wait_stream(stream_.get(), deadline(session_));
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        struct GraphOwners {
            cudaGraph_t& graph;
            cudaGraphExec_t& executable;
            ~GraphOwners() {
                if (executable) (void)cudaGraphExecDestroy(executable);
                if (graph) (void)cudaGraphDestroy(graph);
            }
        } graph_owners{graph, executable};
        PendingDmaExitGuard<RoceBackend> dma_guard(backend_);
        cuda_check(cudaStreamBeginCapture(stream_.get(), cudaStreamCaptureModeThreadLocal));
        for (size_t op_index = 0; op_index < counts.size(); ++op_index) {
            const auto captured = next_operation();
            agree(captured, CollectiveKind::all_reduce_sum, ScalarType::fp32, counts[op_index]);
            require(transport_.all_reduce_sum({sends[op_index]->get(), counts[op_index], ScalarType::fp32},
                                              {receives[op_index]->get(), counts[op_index], ScalarType::fp32},
                                              captured, stream_));
            require(transport_.wait(deadline(session_)));  // capture-order acknowledgement only
            require(session_.complete_operation(captured));  // capture-order acknowledgement only
        }
        cuda_check(cudaStreamEndCapture(stream_.get(), &graph));
        if (!backend_.graph_capture_seen()) throw std::runtime_error("backend did not observe CUDA graph capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        if (options_.fault == "graph-killed-peer") {
            // All ranks finish capture and instantiation before any graph can touch the RoCE buffers.
            require(session_.agree_control_payload("roce-graph-capture-ready-v1",
                                                   "all_reduce_sum:fp32:256,512,1024", deadline(session_)));
            if (options_.rank == options_.fault_rank) {
                std::cout << "{\"event\":\"injected_graph_peer_exit\",\"rank\":" << options_.rank << "}\n";
                std::cout.flush();
                std::_Exit(86);
            }
        }
        for (uint32_t replay = 0; replay < 100; ++replay) {
            require(cuda_graph_launch(executable, stream_.get()));
            require(backend_.wait_captured_replay(graph_replay_deadline()));
            for (size_t op_index = 0; op_index < counts.size(); ++op_index) {
                std::vector<float> output(counts[op_index]);
                cuda_check(cudaMemcpyAsync(output.data(), receives[op_index]->get(),
                                           output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_.get()));
                wait_stream(stream_.get(), deadline(session_));
                for (size_t i = 0; i < counts[op_index]; ++i) {
                    const float expected = reduction_input(0, i) + reduction_input(1, i) + reduction_input(2, i);
                    if (!std::isfinite(output[i]) || std::fabs(output[i] - expected) > 1e-5f)
                        throw std::runtime_error("graph replay numeric mismatch");
                }
            }
        }
        cuda_check(cudaGraphExecDestroy(executable));
        executable = nullptr;
        cuda_check(cudaGraphDestroy(graph));
        graph = nullptr;
    }

    static Failure cuda_graph_launch(cudaGraphExec_t executable, cudaStream_t stream) {
        const cudaError_t error = cudaGraphLaunch(executable, stream);
        return error == cudaSuccess ? ok_failure()
                                    : Failure::fail(FailureCode::comm_failed, cudaGetErrorName(error));
    }

    const Options& options_;
    Session& session_;
    RoceBackend& backend_;
    GuardedTransport& transport_;
    Stream stream_;
    uint64_t sequence_ = 0;
    uint64_t reductions_ = 0;
    uint64_t gathers_ = 0;
};

std::vector<int> routes(const Options& options) {
    std::vector<int> result;
    for (const auto& entry : split(options.route_hca, ',')) {
        if (entry == "self" || entry == "-1") result.push_back(-1);
        else result.push_back(static_cast<int>(number(entry, INT32_MAX)));
    }
    if (result.size() != kTpWorld || result[options.rank] != -1)
        throw std::runtime_error("route-hca must have three entries and self must be 'self'");
    return result;
}

}  // namespace smoke

int main(int argc, char** argv) {
    using namespace smoke;
    std::cout.setf(std::ios::unitbuf);
    uint32_t rank = UINT32_MAX;
    try {
        const Options options = parse(argc, argv);
        rank = options.rank;
        Credential credential;
        require(Credential::from_env(options.credential_env.c_str(), credential));
        cuda_check(cudaSetDevice(0));

        Config backend_config;
        backend_config.rank = options.rank;
        backend_config.world = kTpWorld;
        backend_config.hca_names = options.hca_list.empty()
                                       ? discover_hcas(options.gid_index, 4)
                                       : split(options.hca_list, ',');
        backend_config.route_hca = routes(options);
        backend_config.stripes = options.stripes;
        backend_config.gid_index = options.gid_index;
        backend_config.timeout_ms = options.timeout_ms;
        backend_config.slot_bytes = options.slot_bytes;
        backend_config.blocks = options.blocks;
        backend_config.threads = options.blocks == 8 ? 512 : 256;
        backend_config.approved_proxy_sha256 = options.proxy_sha256;

        RoceBackend backend(std::move(backend_config));
        PendingDmaExitGuard backend_exit_guard(backend);
        require(backend.create());
        std::vector<uint8_t> local_blob;
        require(backend.local_blob(local_blob));

        JoinInfo join;
        join.session_uuid = options.session_uuid;
        join.rank = options.rank;
        join.world = kTpWorld;
        join.model_hash = options.model_hash;
        join.config_fingerprint = options.config_fingerprint;
        join.backend_identity = "rocenante";
        join.nccl_version_expected = "proxy-abi-10";  // historical session field; this is not an NCCL version
        join.library_provenance = backend.provenance();
        // The existing session roster stores binary bootstrap payloads as lowercase hex; never log these blobs.
        constexpr char hex[] = "0123456789abcdef";
        join.nccl_bootstrap_id.reserve(local_blob.size() * 2);
        for (const uint8_t byte : local_blob) {
            join.nccl_bootstrap_id.push_back(hex[byte >> 4]);
            join.nccl_bootstrap_id.push_back(hex[byte & 15]);
        }

        SystemClock clock;
        if (options.fault == "missing-rank" && options.rank == options.fault_rank) {
            std::cout << "{\"event\":\"injected_missing_rank\",\"rank\":" << options.rank << "}\n";
            std::cout.flush();
            std::_Exit(86);
        }
        TcpChannelConfig tcp;
        tcp.rank = options.rank;
        tcp.world = kTpWorld;
        tcp.self = {options.rank0_address, options.port};
        tcp.rank0 = tcp.self;
        tcp.credential = credential.secret();
        std::string channel_error;
        auto channel = make_tcp_control_channel(tcp, clock, channel_error);
        if (!channel) throw std::runtime_error("cannot create private control channel");
        SessionConfig session_config;
        session_config.rank = options.rank;
        session_config.world = kTpWorld;
        session_config.session_uuid = options.session_uuid;
        session_config.credential = credential;
        session_config.approved_nccl_version = "proxy-abi-10";
        session_config.approved_library_provenance = backend.provenance();
        session_config.budgets = {options.timeout_ms, options.timeout_ms, options.timeout_ms, options.timeout_ms};
        Session session(session_config, std::move(join), *channel, clock);
        require(session.start());

        std::vector<uint8_t> roster_blobs;
        for (uint32_t peer = 0; peer < kTpWorld; ++peer) {
            std::string encoded;
            require(session.bootstrap_id(peer, encoded));
            if (encoded.size() % 2 != 0) throw std::runtime_error("invalid opaque blob encoding");
            const size_t peer_blob_start = roster_blobs.size();
            for (size_t i = 0; i < encoded.size(); i += 2) {
                const auto nibble = [](char c) -> uint8_t {
                    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
                    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
                    throw std::runtime_error("invalid opaque blob encoding");
                };
                roster_blobs.push_back(static_cast<uint8_t>((nibble(encoded[i]) << 4) | nibble(encoded[i + 1])));
            }
            if (roster_blobs.size() - peer_blob_start != local_blob.size())
                throw std::runtime_error("RoCE bootstrap blob size differs between ranks");
        }
        require(backend.connect(roster_blobs));
        GuardedTransport transport(backend, clock);
        try {
            std::cout << "{\"event\":\"ready\",\"rank\":" << rank << ",\"world\":3,\"backend\":\"rocenante\","
                      << "\"proxy_sha256\":\"" << options.proxy_sha256 << "\",\"session_uuid\":\""
                      << options.session_uuid << "\",\"model_hash\":\"" << options.model_hash
                      << "\",\"config_fingerprint\":\"" << options.config_fingerprint
                      << "\",\"build_commit\":\"" << options.build_commit << "\"}\n";
            Smoke(options, session, backend, transport).run();
            transport.abort("smoke completed");
            require(session.close());
        } catch (...) {
            transport.abort("smoke failed");
            session.abort("smoke failed");
            throw;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "smoke failed: " << error.what() << '\n';
        std::cout << "{\"event\":\"failure\",\"status\":\"failed\",\"rank\":" << rank
                  << ",\"failure_code\":\"" << last_failure_code << "\"}\n";
        return 1;
    }
}
