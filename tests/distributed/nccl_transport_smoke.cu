// Private, opt-in three-host smoke: one process and logical device 0 per host.
// Include the real backend exactly once, as nccl_backend_probe.cu does. Link this
// TU with session.cpp and control_tcp.cpp, CUDA runtime, dl and pthread only.
#include "distributed/nccl_transport.cpp"
#include "strata/distributed/session.hpp"
#include "strata/kernels/quantize_act_dev.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace smoke_test {
using namespace strata::distributed;
using namespace strata::distributed::nccl;
const char* last_failure_code = "runtime_error";

struct Options {
    uint32_t rank = 0, world = 3, timeout_ms = 30000, fault_rank = 2;
    uint16_t port = 0;
    std::string rank0_address, session_uuid, library_path, nccl_version;
    std::string library_provenance, model_hash, config_fingerprint, build_commit;
    std::string fault = "none";
};

uint32_t number(const std::string& value) {
    size_t used = 0;
    const auto n = std::stoull(value, &used);
    if (used != value.size() || n > UINT32_MAX) throw std::runtime_error("invalid numeric option");
    return static_cast<uint32_t>(n);
}

Options parse(int argc, char** argv) {
    std::map<std::string, std::string> args;
    const std::vector<std::string> allowed = {
        "rank", "world", "rank0-address", "port", "session-uuid", "library-path", "nccl-version",
        "library-provenance", "model-hash", "config-fingerprint", "build-commit", "timeout-ms",
        "fault", "fault-rank"};
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
    o.rank = number(get("rank")); o.world = number(get("world"));
    const auto port = number(get("port"));
    if (port < 1024 || port > 65535) throw std::runtime_error("invalid port");
    o.port = static_cast<uint16_t>(port);
    o.rank0_address = get("rank0-address"); o.session_uuid = get("session-uuid");
    o.library_path = get("library-path"); o.nccl_version = get("nccl-version");
    o.library_provenance = get("library-provenance"); o.model_hash = get("model-hash");
    o.config_fingerprint = get("config-fingerprint"); o.build_commit = get("build-commit");
    if (args.contains("timeout-ms")) o.timeout_ms = number(args.at("timeout-ms"));
    if (args.contains("fault-rank")) o.fault_rank = number(args.at("fault-rank"));
    if (args.contains("fault")) o.fault = args.at("fault");
    if (o.world != 3 || o.rank >= 3 || o.fault_rank >= 3 || o.timeout_ms < 100 || o.timeout_ms > 60000 ||
        (o.fault != "none" && o.fault != "metadata-mismatch" && o.fault != "killed-peer" && o.fault != "missing-rank"))
        throw std::runtime_error("invalid test configuration");
    return o;
}

void require(Failure failure) {
    if (failure) {
        last_failure_code = failure_code_name(failure.code);
        throw std::runtime_error(last_failure_code);
    }
}
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorName(status));
}

class Allocation {
 public:
    explicit Allocation(size_t bytes) { cuda_check(cudaMalloc(&data_, bytes)); }
    ~Allocation() { if (data_) cudaFree(data_); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    void* get() const { return data_; }
 private:
    void* data_ = nullptr;
};

class Stream {
 public:
    Stream() { cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    operator cudaStream_t() const { return stream_; }
 private:
    cudaStream_t stream_{};
};

// Construct after the buffers: communicator abort runs before their destructors
// if any CUDA call or numeric check throws while work may still reference them.
struct AbortOnUnwind {
    Session& session;
    GuardedTransport& transport;
    int exceptions = std::uncaught_exceptions();
    ~AbortOnUnwind() {
        if (std::uncaught_exceptions() > exceptions) {
            transport.abort("smoke operation failed");
            session.abort("smoke operation failed");
        }
    }
};

// Poll CUDA work added after the collective (copies/compaction/quantization), too.
// Never use cudaStreamSynchronize on a possibly broken distributed stream.
void wait_stream(cudaStream_t stream, Deadline deadline) {
    for (;;) {
        const auto status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) cuda_check(status);
        if (std::chrono::steady_clock::now() >= deadline.end) throw std::runtime_error("cuda deadline expired");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

template <typename T> float to_float(T value) { return static_cast<float>(value); }
template <> float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> T from_float(float value) { return value; }
template <> __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }

float reduction_input(uint32_t rank, size_t i, bool varied) {
    if (!varied) return static_cast<float>(rank + 1) + static_cast<float>(i % 7) * 0.125f;
    return 0.013f * static_cast<float>(rank + 1) + std::sin(static_cast<float>(i % 97)) * 0.2f;
}

float activation(size_t global) {
    if (global == 300) return 31.75f;
    if (global == 470) return -127.0f;
    return static_cast<float>(static_cast<int>(global % 31) - 15) * 0.25f;
}

template <typename T>
__global__ void compact_gather(const T* wire, float* logical) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 640) {
        const size_t source = i < 448 ? i : i + 64;
        logical[i] = static_cast<float>(wire[source]);
    } else if (i < 768) {
        logical[i] = 0.0f;
    }
}

__global__ void quantize_global(const float* logical, uint8_t* output) {
    if (threadIdx.x == 0) strata::kernels::quantize_q8_K_block(logical + blockIdx.x * 256,
                                                           output + blockIdx.x * 292);
}

// Independent monolithic oracle: signed first maximum, ties-to-even codes,
// and 16-lane sums. It never calls the production CPU or device quantizer.
std::array<uint8_t, 292> reference_block(const float* values) {
    std::array<uint8_t, 292> bytes{};
    float signed_max = 0;
    for (size_t i = 0; i < 256; ++i)
        if (std::fabs(values[i]) > std::fabs(signed_max)) signed_max = values[i];
    if (signed_max == 0) return bytes;
    const float inverse = -127.0f / signed_max;
    const float scale = 1.0f / inverse;
    std::memcpy(bytes.data(), &scale, sizeof(scale));
    std::array<int8_t, 256> codes{};
    for (size_t i = 0; i < codes.size(); ++i)
        codes[i] = static_cast<int8_t>(std::min(127.0f, std::nearbyint(values[i] * inverse)));
    std::memcpy(bytes.data() + 4, codes.data(), codes.size());
    for (size_t group = 0; group < 16; ++group) {
        int16_t sum = 0;
        for (size_t lane = 0; lane < 16; ++lane) sum += codes[group * 16 + lane];
        std::memcpy(bytes.data() + 260 + group * 2, &sum, sizeof(sum));
    }
    return bytes;
}

class Smoke {
 public:
    Smoke(const Options& options, Session& session, GuardedTransport& transport, const Clock& clock)
        : o_(options), session_(session), transport_(transport), clock_(clock) {}

    void run() {
        size_t free = 0, total = 0;
        cuda_check(cudaMemGetInfo(&free, &total));
        std::cout << "{\"event\":\"allocation_budget\",\"free_bytes\":" << free
                  << ",\"total_bytes\":" << total << "}\n";
        // Largest pair is 32 MiB. Reserve headroom instead of probing to OOM.
        if (free < 128ull * 1024 * 1024) throw std::runtime_error("insufficient allocation headroom");
        // Fixture geometry: residual 2560; shared FFN 640, padded 256/192/192.
        // Include scalar/block boundaries, decode rows, and larger prefill rows.
        for (ScalarType type : {ScalarType::fp32, ScalarType::bf16}) {
            const size_t width = scalar_type_bytes(type);
            std::vector<size_t> counts{1, 31, 32, 33, 255, 256, 257, 640, 2560};
            for (size_t bytes : {10u * 1024u, 64u * 1024u, 256u * 1024u, 1024u * 1024u,
                                 2u * 1024u * 1024u, 4u * 1024u * 1024u, 8u * 1024u * 1024u,
                                 16u * 1024u * 1024u}) counts.push_back(bytes / width);
            std::sort(counts.begin(), counts.end());
            counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
            for (size_t count : counts) for (bool varied : {false, true}) {
                if (type == ScalarType::fp32) reduce<float>(count, type, varied);
                else reduce<__nv_bfloat16>(count, type, varied);
            }
            if (type == ScalarType::fp32) gather<float>(type);
            else gather<__nv_bfloat16>(type);
        }
        require(transport_.check_health());
        std::cout << "{\"event\":\"result\",\"status\":\"passed\",\"rank\":" << o_.rank
                  << ",\"reduce_checks\":" << reductions_ << ",\"gather_checks\":2,\"q8_checks\":2}\n";
    }

 private:
    Deadline deadline() {
        Deadline at;
        require(session_.operation_deadline(at));
        return at;
    }
    Operation operation(CollectiveKind kind, ScalarType type, uint64_t elements,
                        std::vector<uint64_t> valid_counts = {}) {
        Operation op{++sequence_, 0, sequence_};
        if (sequence_ == 2 && o_.fault == "metadata-mismatch" && o_.rank == o_.fault_rank) ++elements;
        require(session_.agree_operation({op, kind, type, elements, std::move(valid_counts)}, deadline()));
        return op;
    }
    void complete() {
        require(transport_.wait(deadline()));
        require(session_.complete_operation({sequence_, 0, sequence_}));
        if (sequence_ == 1 && o_.fault == "killed-peer" && o_.rank == o_.fault_rank) {
            std::cout << "{\"event\":\"injected_peer_exit\",\"rank\":" << o_.rank << "}" << std::endl;
            std::_Exit(86);
        }
    }

    template <typename T> void reduce(size_t count, ScalarType type, bool varied) {
        std::vector<T> input(count), output(count);
        for (size_t i = 0; i < count; ++i) input[i] = from_float<T>(reduction_input(o_.rank, i, varied));
        Allocation send(count * sizeof(T)), receive(count * sizeof(T));
        AbortOnUnwind abort_on_error{session_, transport_};
        cuda_check(cudaMemcpyAsync(send.get(), input.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream_));
        const auto op = operation(CollectiveKind::all_reduce_sum, type, count);
        require(transport_.all_reduce_sum({send.get(), count, type}, {receive.get(), count, type}, op, stream_));
        complete();
        cuda_check(cudaMemcpyAsync(output.data(), receive.get(), count * sizeof(T), cudaMemcpyDeviceToHost, stream_));
        wait_stream(stream_, deadline());
        double sum_error = 0, max_error = 0;
        const double absolute = !varied ? 0 : (type == ScalarType::fp32 ? 2e-6 : 0.004);
        const double relative = !varied ? 0 : (type == ScalarType::fp32 ? 2e-6 : 0.016);
        for (size_t i = 0; i < count; ++i) {
            double expected = 0;
            for (uint32_t rank = 0; rank < 3; ++rank)
                expected += to_float(from_float<T>(reduction_input(rank, i, varied)));
            const double got = to_float(output[i]);
            const double error = std::fabs(got - expected);
            if (!std::isfinite(got) || error > absolute + relative * std::fabs(expected))
                throw std::runtime_error("all_reduce numeric mismatch");
            max_error = std::max(max_error, error); sum_error += error;
        }
        ++reductions_;
        std::cout << "{\"event\":\"all_reduce\",\"sequence\":" << sequence_
                  << ",\"dtype\":\"" << (type == ScalarType::fp32 ? "fp32" : "bf16")
                  << "\",\"elements\":" << count << ",\"bytes\":" << count * sizeof(T)
                  << ",\"varied\":" << (varied ? "true" : "false")
                  << ",\"max_error\":" << max_error << ",\"mean_error\":" << sum_error / count
                  << ",\"absolute_tolerance\":" << absolute << ",\"relative_tolerance\":" << relative
                  << ",\"nonfinite\":0}\n";
    }

    template <typename T> void gather(ScalarType type) {
        constexpr std::array<uint64_t, 3> counts{256, 192, 192};
        constexpr std::array<size_t, 3> offsets{0, 256, 448};
        std::vector<T> input(256, from_float<T>(-999));
        for (size_t i = 0; i < counts[o_.rank]; ++i) input[i] = from_float<T>(activation(offsets[o_.rank] + i));
        std::vector<float> gathered(768);
        std::array<uint8_t, 292> actual{};
        Allocation send(256 * sizeof(T)), wire(768 * sizeof(T)), logical(768 * sizeof(float)), q8(3 * 292);
        AbortOnUnwind abort_on_error{session_, transport_};
        cuda_check(cudaMemcpyAsync(send.get(), input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice, stream_));
        const auto op = operation(CollectiveKind::all_gather_padded, type, 256, {256, 192, 192});
        require(transport_.all_gather_padded({send.get(), 256, type}, {wire.get(), 768, type}, counts, op, stream_));
        complete();
        compact_gather<<<3, 256, 0, stream_>>>(static_cast<const T*>(wire.get()), static_cast<float*>(logical.get()));
        cuda_check(cudaGetLastError());
        quantize_global<<<3, 1, 0, stream_>>>(static_cast<const float*>(logical.get()), static_cast<uint8_t*>(q8.get()));
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpyAsync(gathered.data(), logical.get(), 768 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        cuda_check(cudaMemcpyAsync(actual.data(), static_cast<uint8_t*>(q8.get()) + 292, 292, cudaMemcpyDeviceToHost, stream_));
        wait_stream(stream_, deadline());
        std::array<float, 768> monolithic{};
        for (size_t i = 0; i < 640; ++i) monolithic[i] = to_float(from_float<T>(activation(i)));
        if (!std::equal(gathered.begin(), gathered.end(), monolithic.begin()))
            throw std::runtime_error("gather compaction mismatch");
        const auto reference = reference_block(monolithic.data() + 256);
        if (actual != reference) throw std::runtime_error("global Q8_K block crossing 448 mismatch");
        // Prove this fixture would fail if the second block were reset at shard 448.
        std::array<float, 256> wrong_shard{};
        std::copy_n(monolithic.data() + 256, 192, wrong_shard.data());
        const auto wrong = reference_block(wrong_shard.data());
        if (std::equal(reference.begin(), reference.begin() + 4, wrong.begin()))
            throw std::runtime_error("Q8_K fixture has no shard-scale sensitivity");
        std::cout << "{\"event\":\"gather_q8\",\"sequence\":" << sequence_
                  << ",\"dtype\":\"" << (type == ScalarType::fp32 ? "fp32" : "bf16")
                  << "\",\"valid_counts\":[256,192,192],\"wire_elements\":768,\"logical_elements\":640,"
                  << "\"q8_block\":[256,512],\"crosses\":448,\"independent_oracle\":true,\"shard_scale_sensitive\":true}\n";
    }

    const Options& o_;
    Session& session_;
    GuardedTransport& transport_;
    const Clock& clock_;
    Stream stream_;
    uint64_t sequence_ = 0, reductions_ = 0;
};
}  // namespace smoke_test

int main(int argc, char** argv) {
    using namespace smoke_test;
    std::cout.setf(std::ios::unitbuf);
    uint32_t rank = UINT32_MAX;
    try {
        const Options o = parse(argc, argv); rank = o.rank;
        Credential credential;
        require(Credential::from_env("STRATA_TP3_CREDENTIAL", credential));
        cuda_check(cudaSetDevice(0));
        SystemClock clock;
        NcclConfig nccl_config{o.rank, o.world, {}, o.library_path};
        std::string error, version, provenance;
        auto api = make_production_nccl_api(nccl_config, error);
        if (!api) throw std::runtime_error("cannot load NCCL");
        require(api->identity(version, provenance));
        if (version != o.nccl_version || provenance != o.library_provenance)
            throw std::runtime_error("NCCL identity differs from approval");
        JoinInfo join;
        join.session_uuid = o.session_uuid; join.rank = o.rank; join.world = o.world;
        join.model_hash = o.model_hash; join.config_fingerprint = o.config_fingerprint;
        join.backend_identity = "nccl"; join.nccl_version_expected = version;
        join.library_provenance = provenance;
        // The session requires a nonempty opaque field in every join. Followers'
        // all-zero placeholders are never used to initialize a communicator.
        join.nccl_bootstrap_id = std::string(256, '0');
        if (o.rank == 0 && !make_nccl_bootstrap_id(o.library_path, join.nccl_bootstrap_id, error))
            throw std::runtime_error("cannot create NCCL bootstrap id");
        TcpChannelConfig tcp;
        tcp.rank = o.rank; tcp.world = o.world; tcp.self = {o.rank0_address, o.port};
        tcp.rank0 = tcp.self; tcp.credential = credential.secret();
        auto channel = make_tcp_control_channel(tcp, clock, error);
        if (!channel) throw std::runtime_error("cannot create private control channel");
        SessionConfig config;
        config.rank = o.rank; config.world = o.world; config.session_uuid = o.session_uuid;
        config.credential = credential; config.approved_nccl_version = version;
        config.approved_library_provenance = provenance;
        config.budgets = {o.timeout_ms, o.timeout_ms, o.timeout_ms, o.timeout_ms};
        Session session(config, join, *channel, clock);
        require(session.start());
        require(session.bootstrap_id(0, nccl_config.bootstrap_id));
        nccl_config.initialize_ms = session.roster().budgets.initialize_ms;
        NcclBackend backend(std::move(api), nccl_config, {version, provenance});
        GuardedTransport transport(backend, clock);
        try {
            require(backend.init());
            std::cout << "{\"event\":\"ready\",\"rank\":" << rank << ",\"device\":0,\"world\":3,"
                      << "\"headers_version\":" << NCCL_VERSION_CODE << ",\"runtime_version\":\"" << version
                      << "\",\"library_provenance\":\"" << provenance
                      << "\",\"session_uuid\":\"" << o.session_uuid << "\",\"model_hash\":\"" << o.model_hash
                      << "\",\"config_fingerprint\":\"" << o.config_fingerprint
                      << "\",\"build_commit\":\"" << o.build_commit << "\"}\n";
            Smoke smoke(o, session, transport, clock);
            smoke.run();
            transport.abort("smoke completed");
            require(session.close());
        } catch (...) {
            // Abort before destroying caller-owned buffers or streams. Failure is
            // terminal: no retry and no passing result may follow this path.
            transport.abort("smoke failed");
            session.abort("smoke failed");
            throw;
        }
        return 0;
    } catch (const std::exception& error) {
        // Errors are fixed labels or FailureCode names, never bootstrap/credential.
        std::cerr << "smoke failed: " << error.what() << '\n';
        std::cout << "{\"event\":\"failure\",\"status\":\"failed\",\"rank\":" << rank
                  << ",\"failure_code\":\"" << last_failure_code << "\"}\n";
        return 1;
    }
}
