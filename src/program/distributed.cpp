// Opt-in, uncaptured C1 teacher-forcing driver. Every rank reads identical
// token IDs; the control protocol agrees them before any model operation.
#include "distributed/nccl_transport.cpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/distributed/expert_model.hpp"
#include "strata/distributed/model_worker.hpp"
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
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
using namespace strata;
using namespace strata::distributed;
void cuda_check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void require(bool ok, const std::string& err) { if (!ok) throw std::runtime_error(err); }
void require(Failure f) { if (f) throw std::runtime_error(f.detail); }
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
        const std::set<std::string> allowed = {"rank", "rank0-address", "port", "session-uuid",
            "library-path", "nccl-version", "library-provenance", "model-hash", "config-fingerprint",
            "pack", "native", "ple", "context", "logits-dir", "timeout-ms",
            "state-dir", "state-positions", "state-max-bytes"};
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
        if (rank >= 3 || port < 1024 || port > 65535 || timeout < 100 || timeout > 600000 || context < 4 || context > 32768)
            throw std::runtime_error("invalid rank, port, deadline or context");
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
        require(weights.load(pack, weight_memory.pointer, weight_bytes, err, &skip), err);
        require(core::check_all(weights, geometry, err), err);
        core::NativeDense dense;
        require(dense.load(shards, weights, err, true), err);
        core::NativeHead head;
        core::NativeEmbed embedding;
        require(head.load(shards, geometry.n_embd, geometry.n_vocab, err), err);
        require(embedding.load(shards, geometry.n_embd, geometry.n_vocab, err), err);
        core::set_native_embed(&embedding);
        core::SessionState state;
        Allocation state_memory, ple_embedding, ple_scratch;
        state_memory.allocate(core::session_bytes(geometry, context, geometry.n_expert_used));
        require(core::session_init(geometry, context, geometry.n_expert_used, state_memory.pointer, state) != 0,
                "cannot initialize the full model session");
        SessionLifetime state_lifetime{state};
        kernels::PleTable table;
        kernels::PleIoOptions ple_io; ple_io.mode = kernels::PleIo::Mmap; ple_io.cache_rows = 0;
        require(table.open(ple, err, ple_io), err);
        std::vector<float> ple_host;
        wire_ple(weights, state, table, ple_embedding, ple_scratch, ple_host);

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
        join.backend_identity = "nccl"; join.nccl_version_expected = version; join.library_provenance = provenance;
        join.nccl_bootstrap_id = std::string(256, '0');
        if (rank == 0) require(make_nccl_bootstrap_id(nccl_config.library_path, join.nccl_bootstrap_id, err), err);
        TcpChannelConfig tcp;
        tcp.rank = rank; tcp.world = 3; tcp.self = {get("rank0-address"), uint16_t(port)};
        tcp.rank0 = tcp.self; tcp.credential = credential.secret();
        auto channel = make_tcp_control_channel(tcp, clock, err);
        require(bool(channel), err);
        SessionConfig config;
        config.rank = rank; config.session_uuid = join.session_uuid; config.credential = credential;
        config.approved_nccl_version = version; config.approved_library_provenance = provenance;
        config.budgets = {uint32_t(timeout), uint32_t(timeout), uint32_t(timeout), uint32_t(timeout)};
        Session session(config, join, *channel, clock);
        require(session.start());
        require(session.bootstrap_id(0, nccl_config.bootstrap_id));
        nccl_config.initialize_ms = std::min(config.budgets.initialize_ms, session.roster().budgets.initialize_ms);
        NcclBackend nccl_backend(std::move(api), nccl_config, {version, provenance});
        GuardedTransport transport(nccl_backend, clock);
        require(nccl_backend.init());
        try {
            SessionCollectives collectives(session, transport, clock, uint32_t(timeout));
            NativeRoutedExperts experts(pack, native, geometry, int(rank), collectives,
                                         {256, 192, 192}, uint32_t(timeout));
            require(experts.initialize(err), err);
            ModelForward model(weights, geometry, state, int(rank), collectives, experts);
            model.native_head(&head);
            WorkerConfig worker_config{join.session_uuid, join.model_hash, uint32_t(rank)};
            ModelWorkerBackend execution({worker_config, stream.value, 0}, model, collectives, session, transport);
            Worker worker(worker_config, execution);
            uint64_t step = 1;
            uint64_t request_sequence = 0;
            auto execute = [&](StepPhase phase, std::vector<int32_t> tokens = {}, std::vector<uint64_t> positions = {}) {
                const std::string request = phase == StepPhase::init ? std::string{} :
                    "teacher-forcing-" + std::to_string(request_sequence);
                StepCommand command{join.session_uuid, request, 0, step++, phase,
                                    std::move(tokens), std::move(positions), join.model_hash, {}};
                require(execution.plan(command, command.expected_collective_schedule));
                auto result = worker.execute(command);
                if (result.status != StepStatus::completed) throw std::runtime_error(result.failure.detail);
                return result;
            };
            execute(StepPhase::init);
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
                      << " fused_gr=0 fused_gdn=0 fast_attn=0 flash_short=0 shared_early=1\n";
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "QUIT") { if (state_last) dump_state(false); break; }
                if (line == "RESET") {
                    if (state_last) dump_state(false);
                    execute(StepPhase::reset); ++request_sequence;
                    dumped_positions.clear(); dump_state(true);
                    std::cout << "RESET_OK\n"; continue;
                }
                std::istringstream input(line);
                std::string verb; input >> verb;
                if (verb != "GEN") throw std::runtime_error("expected GEN <teacher-forced token IDs>, RESET or QUIT");
                int64_t id;
                bool any = false;
                while (input >> id) {
                    if (id < 0 || id >= geometry.n_vocab) throw std::runtime_error("invalid token ID");
                    any = true;
                    const auto pos = model.next_position();
                    execute(StepPhase::prefill, {int32_t(id)}, {uint64_t(pos)});
                    const auto result = execute(StepPhase::commit);
                    if (state_positions.contains(uint64_t(pos))) dump_state(false);
                    if (rank == 0 && result.output_ready) {
                        const auto* logits = execution.committed_logits();
                        require(logits != nullptr, "commit did not publish logits");
                        const auto top = std::max_element(logits, logits + geometry.n_vocab) - logits;
                        if (args.contains("logits-dir")) {
                            const auto directory = std::filesystem::path(args.at("logits-dir")) /
                                                   ("request-" + std::to_string(request_sequence));
                            std::filesystem::create_directories(directory);
                            const auto path = directory / ("logits-" + std::to_string(pos) + ".f32");
                            std::ofstream file(path, std::ios::binary);
                            file.write(reinterpret_cast<const char*>(logits), geometry.n_vocab * sizeof(float));
                            require(bool(file), "cannot write logits artifact");
                        }
                        std::cout << "TOKEN position=" << pos << " input=" << id << " argmax=" << top << '\n';
                    }
                }
                require(any && input.eof(), "GEN needs valid integer token IDs");
            }
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
