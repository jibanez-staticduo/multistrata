// Opt-in three-host NCCL graph capture/replay smoke. One process and logical device 0 per host.
#include "distributed/nccl_transport.cpp"
#include "strata/distributed/session.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace graph_smoke {
using namespace strata::distributed;
using namespace strata::distributed::nccl;

struct Options {
    uint32_t rank = 0;
    uint32_t world = 3;
    uint32_t timeout_ms = 30000;
    uint32_t fault_rank = 2;
    uint16_t port = 0;
    std::string rank0_address, session_uuid, library_path, nccl_version;
    std::string library_provenance, model_hash, config_fingerprint, build_commit;
    bool kill_peer = false;
};

uint32_t number(const std::string& value) {
    size_t used = 0;
    const auto parsed = std::stoull(value, &used);
    if (used != value.size() || parsed > UINT32_MAX) throw std::runtime_error("invalid numeric option");
    return static_cast<uint32_t>(parsed);
}

Options parse(int argc, char** argv) {
    const std::map<std::string, std::string> allowed = {
        {"rank", ""}, {"world", ""}, {"rank0-address", ""}, {"port", ""}, {"session-uuid", ""},
        {"library-path", ""}, {"nccl-version", ""}, {"library-provenance", ""}, {"model-hash", ""},
        {"config-fingerprint", ""}, {"build-commit", ""}, {"timeout-ms", ""}, {"fault", ""},
        {"fault-rank", ""}};
    auto args = allowed;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || std::strncmp(argv[i], "--", 2) != 0)
            throw std::runtime_error("option needs a value");
        const std::string key = argv[i] + 2;
        if (!args.contains(key) || !args[key].empty()) throw std::runtime_error("unknown or duplicate option");
        args[key] = argv[i + 1];
    }
    auto get = [&](const char* key) -> const std::string& {
        if (args.at(key).empty()) throw std::runtime_error("missing option");
        return args.at(key);
    };
    Options out;
    out.rank = number(get("rank"));
    out.world = number(get("world"));
    const auto port = number(get("port"));
    if (port < 1024 || port > 65535) throw std::runtime_error("invalid port");
    out.port = static_cast<uint16_t>(port);
    out.rank0_address = get("rank0-address");
    out.session_uuid = get("session-uuid");
    out.library_path = get("library-path");
    out.nccl_version = get("nccl-version");
    out.library_provenance = get("library-provenance");
    out.model_hash = get("model-hash");
    out.config_fingerprint = get("config-fingerprint");
    out.build_commit = get("build-commit");
    if (!args["timeout-ms"].empty()) out.timeout_ms = number(args["timeout-ms"]);
    if (!args["fault-rank"].empty()) out.fault_rank = number(args["fault-rank"]);
    if (!args["fault"].empty()) {
        if (args["fault"] != "none" && args["fault"] != "killed-peer")
            throw std::runtime_error("invalid fault mode");
        out.kill_peer = args["fault"] == "killed-peer";
    }
    if (out.world != 3 || out.rank >= out.world || out.fault_rank >= out.world ||
        out.timeout_ms < 100 || out.timeout_ms > 60000)
        throw std::runtime_error("invalid test configuration");
    return out;
}

void require(Failure failure) {
    if (failure) throw std::runtime_error(failure_code_name(failure.code));
}
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorName(error));
}

class DeviceBuffer {
 public:
    explicit DeviceBuffer(size_t bytes) { cuda_check(cudaMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() { if (ptr_ != nullptr) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const { return ptr_; }
 private:
    void* ptr_ = nullptr;
};

class PinnedBuffer {
 public:
    explicit PinnedBuffer(size_t bytes) { cuda_check(cudaHostAlloc(&ptr_, bytes, cudaHostAllocPortable)); }
    ~PinnedBuffer() { if (ptr_ != nullptr) cudaFreeHost(ptr_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    void* get() const { return ptr_; }
 private:
    void* ptr_ = nullptr;
};

class Stream {
 public:
    Stream() { cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { if (stream_ != nullptr) cudaStreamDestroy(stream_); }
    operator cudaStream_t() const { return stream_; }
 private:
    cudaStream_t stream_ = nullptr;
};

class Graph {
 public:
    Graph() = default;
    ~Graph() {
        if (exec_ != nullptr) cudaGraphExecDestroy(exec_);
        if (graph_ != nullptr) cudaGraphDestroy(graph_);
    }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    void instantiate(cudaGraph_t graph) {
        graph_ = graph;
        cuda_check(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
    }
    cudaGraphExec_t exec() const { return exec_; }
    void destroy() {
        if (exec_ != nullptr) {
            cuda_check(cudaGraphExecDestroy(exec_));
            exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            cuda_check(cudaGraphDestroy(graph_));
            graph_ = nullptr;
        }
    }
 private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

void run(const Options& options, Session& session, NcclBackend& backend, cudaStream_t stream) {
    const std::array<uint64_t, 3> sizes{4, 8, 4};
    std::array<DeviceBuffer, 3> sends{DeviceBuffer(4 * sizeof(float)),
                                      DeviceBuffer(8 * sizeof(float)),
                                      DeviceBuffer(4 * sizeof(float))};
    std::array<DeviceBuffer, 3> receives{DeviceBuffer(4 * sizeof(float)),
                                         DeviceBuffer(8 * sizeof(float)),
                                         DeviceBuffer(4 * sizeof(float))};
    std::array<PinnedBuffer, 3> host_inputs{PinnedBuffer(8 * sizeof(float)),
                                            PinnedBuffer(8 * sizeof(float)),
                                            PinnedBuffer(8 * sizeof(float))};
    std::array<float*, 3> input_ptrs{static_cast<float*>(host_inputs[0].get()),
                                     static_cast<float*>(host_inputs[1].get()),
                                     static_cast<float*>(host_inputs[2].get())};
    std::array<std::array<float, 8>, 3> host_output{};
    constexpr std::array<const char*, 3> labels{"graph_reduce_a", "graph_reduce_b", "graph_reduce_c"};
    std::string schedule;
    for (size_t op = 0; op < sizes.size(); ++op) {
        schedule += std::string(labels[op]) + ":all_reduce_sum:fp32:" + std::to_string(sizes[op]) + ";";
    }
    Deadline deadline;
    require(session.operation_deadline(deadline));
    require(session.agree_control_payload("nccl-graph-smoke-v1", schedule, deadline));
    for (size_t op = 0; op < sizes.size(); ++op) {
        for (size_t i = 0; i < sizes[op]; ++i)
            input_ptrs[op][i] = static_cast<float>((options.rank + 1) + i + op);
        cuda_check(cudaMemcpyAsync(sends[op].get(), input_ptrs[op],
                                   sizes[op] * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    require(backend.prepare_graph_session(stream, sizes.size()));
    cudaEvent_t warmup_done = nullptr;
    cuda_check(cudaEventCreateWithFlags(&warmup_done, cudaEventDisableTiming));
    cuda_check(cudaEventRecord(warmup_done, stream));
    const Deadline warmup_deadline{std::chrono::steady_clock::now() +
                                   std::chrono::milliseconds(options.timeout_ms)};
    for (;;) {
        const cudaError_t status = cudaEventQuery(warmup_done);
        if (status == cudaSuccess) break;
        if (status != cudaErrorNotReady) {
            cudaEventDestroy(warmup_done);
            cuda_check(status);
        }
        if (std::chrono::steady_clock::now() >= warmup_deadline.end) {
            cudaEventDestroy(warmup_done);
            throw std::runtime_error("warmup deadline expired");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cuda_check(cudaEventDestroy(warmup_done));
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    require(backend.check_graph_capture());
    for (size_t op = 0; op < sizes.size(); ++op) {
        require(backend.raw_all_reduce_sum({sends[op].get(), sizes[op], ScalarType::fp32},
                                           {receives[op].get(), sizes[op], ScalarType::fp32},
                                           sizes[op], stream));
    }
    cudaGraph_t graph_handle = nullptr;
    cuda_check(cudaStreamEndCapture(stream, &graph_handle));
    require(backend.finish_graph_capture());
    Graph captured;
    captured.instantiate(graph_handle);
    require(session.operation_deadline(deadline));
    require(session.agree_control_payload("nccl-graph-capture-ready", "three-ops", deadline));

    for (size_t replay = 0; replay < 100; ++replay) {
        for (size_t op = 0; op < sizes.size(); ++op) {
            for (size_t i = 0; i < sizes[op]; ++i)
                input_ptrs[op][i] = static_cast<float>((options.rank + 1) * (replay + 1) + i + op);
            cuda_check(cudaMemcpyAsync(sends[op].get(), input_ptrs[op],
                                       sizes[op] * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        if (replay == 2 && options.kill_peer && options.rank == options.fault_rank) {
            std::cout << "{\"event\":\"injected_peer_exit\",\"rank\":" << options.rank << "}\n";
            std::_Exit(86);
        }
        require(backend.begin_captured_replay());
        const auto launch = cudaGraphLaunch(captured.exec(), stream);
        if (launch != cudaSuccess) {
            backend.raw_abort();
            if (backend.requires_process_exit()) {
                std::cerr << "{\"event\":\"fatal_launch_failure\",\"rank\":" << options.rank << "}\n";
                std::_Exit(86);
            }
            cuda_check(launch);
        }
        const Deadline replay_deadline{std::chrono::steady_clock::now() +
                                       std::chrono::milliseconds(options.timeout_ms)};
        const Failure replay_status = backend.wait_captured_replay(replay_deadline);
        if (replay_status && backend.requires_process_exit()) {
            std::cerr << "{\"event\":\"fatal_pending_dma\",\"rank\":" << options.rank
                      << ",\"failure_code\":\"" << failure_code_name(replay_status.code) << "\"}\n";
            std::_Exit(86);
        }
        require(replay_status);
        for (size_t op = 0; op < sizes.size(); ++op) {
            cuda_check(cudaMemcpyAsync(host_output[op].data(), receives[op].get(),
                                       sizes[op] * sizeof(float), cudaMemcpyDeviceToHost, stream));
        }
        cudaEvent_t done = nullptr;
        cuda_check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        cuda_check(cudaEventRecord(done, stream));
        for (;;) {
            const auto status = cudaEventQuery(done);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) cuda_check(status);
            if (std::chrono::steady_clock::now() >= replay_deadline.end) {
                cudaEventDestroy(done);
                throw std::runtime_error("replay result deadline expired");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cuda_check(cudaEventDestroy(done));
        for (size_t op = 0; op < sizes.size(); ++op) {
            for (size_t i = 0; i < sizes[op]; ++i) {
                const float expected = static_cast<float>(6 * (replay + 1) + 3 * (i + op));
                if (host_output[op][i] != expected) throw std::runtime_error("graph replay output mismatch");
            }
        }
        if (replay == 0) {
            require(session.operation_deadline(deadline));
            require(session.agree_control_payload("nccl-graph-replay-1-complete", "replay-1", deadline));
        }
        if (replay == 1 && options.kill_peer && options.rank == options.fault_rank) {
            std::cout << "{\"event\":\"injected_peer_exit\",\"rank\":" << options.rank << "}\n";
            std::_Exit(86);
        }
    }
    captured.destroy();
    require(backend.end_graph_session());
    require(session.operation_deadline(deadline));
    require(session.agree_control_payload("nccl-graph-smoke-complete", "100-replays", deadline));
    std::cout << "{\"event\":\"result\",\"status\":\"passed\",\"rank\":" << options.rank
              << ",\"captured_collectives\":3,\"replays\":100}\n";
}
}  // namespace graph_smoke

int main(int argc, char** argv) {
    using namespace graph_smoke;
    std::cout.setf(std::ios::unitbuf);
    uint32_t rank = UINT32_MAX;
    try {
        const Options options = parse(argc, argv);
        rank = options.rank;
        Credential credential;
        require(Credential::from_env("STRATA_TP3_CREDENTIAL", credential));
        cuda_check(cudaSetDevice(0));
        SystemClock clock;
        NcclConfig nccl_config{options.rank, options.world, {}, options.library_path};
        std::string error, version, provenance;
        auto api = make_production_nccl_api(nccl_config, error);
        if (!api) throw std::runtime_error("cannot load NCCL");
        require(api->identity(version, provenance));
        if (version != options.nccl_version || provenance != options.library_provenance)
            throw std::runtime_error("NCCL identity differs from approval");
        JoinInfo join;
        join.session_uuid = options.session_uuid; join.rank = options.rank; join.world = options.world;
        join.model_hash = options.model_hash; join.config_fingerprint = options.config_fingerprint;
        join.backend_identity = "nccl"; join.nccl_version_expected = version; join.library_provenance = provenance;
        join.nccl_bootstrap_id = std::string(256, '0');
        if (options.rank == 0 && !make_nccl_bootstrap_id(options.library_path, join.nccl_bootstrap_id, error))
            throw std::runtime_error("cannot create NCCL bootstrap id");
        TcpChannelConfig tcp;
        tcp.rank = options.rank; tcp.world = options.world; tcp.self = {options.rank0_address, options.port};
        tcp.rank0 = tcp.self; tcp.credential = credential.secret();
        auto channel = make_tcp_control_channel(tcp, clock, error);
        if (!channel) throw std::runtime_error("cannot create private control channel");
        SessionConfig config;
        config.rank = options.rank; config.world = options.world; config.session_uuid = options.session_uuid;
        config.credential = credential; config.approved_nccl_version = version;
        config.approved_library_provenance = provenance;
        config.budgets = {options.timeout_ms, options.timeout_ms, options.timeout_ms, options.timeout_ms};
        Session session(config, join, *channel, clock);
        require(session.start());
        require(session.bootstrap_id(0, nccl_config.bootstrap_id));
        nccl_config.initialize_ms = session.roster().budgets.initialize_ms;
        NcclBackend backend(std::move(api), nccl_config, {version, provenance});
        Stream stream;
        require(backend.init());
        run(options, session, backend, stream);
        require(session.close());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph smoke failed: " << error.what() << '\n';
        std::cout << "{\"event\":\"failure\",\"status\":\"failed\",\"rank\":" << rank << "}\n";
        return 1;
    }
}
