// Independent, uncaptured C1 teacher-forcing oracle. No distributed model kernels.
#include "strata/distributed/reference.hpp"
#include "strata/distributed/capture_positions.hpp"
#include "strata/distributed/parity_trace_writer.hpp"
#include "strata/distributed/state_evidence.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
namespace {
using namespace strata;
using namespace strata::distributed;
void cuda_check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void require(bool ok, const std::string& err) { if (!ok) throw std::runtime_error(err); }
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
    if (key.ne0 != kernels::NG_N_EMBD || key.ne1 != kernels::NG_HC_DIM ||
        (!key.quantized() && (key.kind != core::WeightKind::Bf16InF32 ||
                              key.bytes != uint64_t(key.elements) * 2)))
        throw std::runtime_error("PLE key has an incompatible shape or storage");
    const auto& value = get("ple_value.weight");
    if (value.ne0 != kernels::NG_N_EMBD || value.ne1 != kernels::NG_N_EMBD ||
        value.kind != core::WeightKind::Bf16InF32 || value.bytes != uint64_t(value.elements) * 2)
        throw std::runtime_error("PLE value must retain the BF16 square projection");
    for (const char* name : {"ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight"}) {
        const auto& norm = get(name);
        if (norm.elements != kernels::NG_HC_DIM || norm.kind != core::WeightKind::F32 ||
            norm.bytes != uint64_t(kernels::NG_HC_DIM) * sizeof(float))
            throw std::runtime_error("PLE norm has an incompatible shape or storage");
    }
    if (conv.elements != kernels::NG_HC_DIM * kernels::PLE_CONV_KERNEL)
        throw std::runtime_error("PLE convolution has an incompatible shape");
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
} // namespace

int main(int argc, char** argv) {
    using namespace strata;
    using namespace strata::distributed;
    std::cout.setf(std::ios::unitbuf);
    try {
        std::map<std::string, std::string> args;
        const std::set<std::string> allowed = {"pack", "native", "ple", "context", "logits-dir", "logits-positions",
            "state-dir", "state-positions", "state-max-bytes", "trace-dir"};
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i + 1 == argc || !key.starts_with("--") || !allowed.contains(key.substr(2)) ||
                args.contains(key.substr(2)) || !*argv[i + 1])
                throw std::runtime_error("unknown, duplicate, empty or incomplete argument");
            args[key.substr(2)] = argv[i + 1];
        }
        auto get = [&](const char* key) -> const std::string& {
            if (!args.contains(key)) throw std::runtime_error(std::string("missing --") + key);
            return args.at(key);
        };
        uint64_t context = 256;
        if (args.contains("context")) {
            const auto& text = args.at("context");
            const auto [end, status] = std::from_chars(text.data(), text.data() + text.size(), context);
            require(status == std::errc{} && end == text.data() + text.size(), "invalid --context");
        }
        require(context >= 4 && context <= 1048576, "context must be in [4, 1048576]");
        const bool selected_logits = args.contains("logits-positions");
        require(!selected_logits || args.contains("logits-dir"), "--logits-positions requires --logits-dir");
        const auto logits_positions = selected_logits ?
            CapturePositions::parse(args.at("logits-positions"), context) : CapturePositions{};
        require(!args.contains("state-positions") || args.contains("state-dir"),
                "--state-positions requires --state-dir");
        require(!args.contains("state-max-bytes") || args.contains("state-dir"),
                "--state-max-bytes requires --state-dir");
        std::set<uint64_t> state_positions;
        bool state_last = true;
        std::string positions = args.contains("state-positions") ? args.at("state-positions") : "0,3,last";
        if (args.contains("state-positions")) state_last = false;
        std::istringstream position_input(positions);
        std::string position_word;
        require(positions.back() != ',', "invalid --state-positions");
        while (std::getline(position_input, position_word, ',')) {
            if (position_word == "last") { state_last = true; continue; }
            uint64_t value = 0;
            const auto [end, status] = std::from_chars(position_word.data(),
                position_word.data() + position_word.size(), value);
            require(status == std::errc{} && end == position_word.data() + position_word.size() &&
                    value < context && state_positions.insert(value).second, "invalid --state-positions");
        }
        uint64_t state_max_bytes = uint64_t(4) << 30;
        if (args.contains("state-max-bytes")) {
            const auto& text = args.at("state-max-bytes");
            const auto [end, status] = std::from_chars(text.data(), text.data() + text.size(), state_max_bytes);
            require(status == std::errc{} && end == text.data() + text.size() && state_max_bytes > 0,
                    "invalid --state-max-bytes");
        }
        // These shortcuts bypass the ordinary grouped v1 path before its setters.
        for (const char* name : {"STRATA_EXPERT_V2", "STRATA_EXPERT_V2K"}) {
            const char* value = std::getenv(name);
            require(!value || value[0] != '1', std::string("oracle requires ") + name + " disabled");
        }
        const auto pack = get("pack"), native = get("native"), ple = get("ple");
        const auto shards = gguf_split_paths(native);
        const GgufFile metadata(shards.front());
        core::ModelGeometry geometry;
        const auto geometry_error = read_geometry(metadata, geometry);
        require(geometry_error.empty(), geometry_error);
        const auto unsupported = geometry_unsupported(geometry);
        require(unsupported.empty(), unsupported);
        require(geometry.n_vocab == 248320, "oracle requires 248320 vocabulary rows");
        if (args.contains("logits-dir")) std::filesystem::create_directories(args.at("logits-dir"));
        cuda_check(cudaSetDevice(0));
        Stream stream;
        kernels::gr_set_fp32_activations(true); kernels::gr_set_native_mmvf(true);
        core::layer_set_native_bf16(true); core::layer_set_fused_gr(false);
        core::layer_set_fused_gdn(false); core::layer_set_fast_attn(false);
        core::layer_set_native_flash_attn_short(false); core::layer_set_shared_early(true);
        // Block selection supports contexts above the legacy 32768-cell top-k limit.
        core::layer_set_fast_select(true);
        kernels::native_gdn_set_enabled(true); kernels::native_router_set_enabled(true);
        kernels::native_qsa_set_enabled(true); kernels::native_qsa_indexer_set_enabled(true);
        kernels::native_rope_set_enabled(true); kernels::native_moe_combine_set_enabled(true);
        kernels::ple_set_native_bf16(true); kernels::ple_set_native_postops(true);
        kernels::shared_expert_set_native_bf16(true);
        kernels::native_grouped_set_v1(true); kernels::native_expert_set_expdec(0);
        kernels::native_expert_set_mode(0, 0); kernels::iq_set_old_kernels(true);
        kernels::native_mmvq_set_multi_exact(true); kernels::native_mmvq_set_b6_rows(0);

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
        struct EmbedReset { ~EmbedReset() { core::set_native_embed(nullptr); } } embed_reset;
        core::SessionState state;
        Allocation state_memory, ple_embedding, ple_scratch, logits;
        state_memory.allocate(core::session_bytes(geometry, int64_t(context), geometry.n_expert_used));
        SessionLifetime state_lifetime{state};
        require(core::session_init(geometry, int64_t(context), geometry.n_expert_used,
                                   state_memory.pointer, state) != 0, "session initialization failed");
        kernels::PleTable table;
        kernels::PleIoOptions ple_io; ple_io.mode = kernels::PleIo::Mmap; ple_io.cache_rows = 0;
        require(table.open(ple, err, ple_io), err);
        std::vector<float> ple_host;
        wire_ple(weights, state, table, ple_embedding, ple_scratch, ple_host);
        MonolithicReference model(weights, geometry, state, pack, native);
        model.native_head(&head);
        std::unique_ptr<ParityTraceWriter> trace;
        if (args.contains("trace-dir")) {
            trace = std::make_unique<ParityTraceWriter>(args.at("trace-dir"));
            require(trace->request("request-0", err), err);
            model.parity_observer([&](const ParityTraceView& view, void* s, std::string& error) {
                return view.position != 0 || trace->capture(view, s, error);
            });
        }
        require(model.initialize(err), err);
        require(model.reset(stream.value, err), err);
        logits.allocate(size_t(geometry.n_vocab) * sizeof(float));
        std::vector<float> host_logits(size_t(geometry.n_vocab));
        uint64_t request_sequence = 0;
        std::set<uint64_t> dumped_positions;
        auto request_id = [&] { return "request-" + std::to_string(request_sequence); };
        auto dump_state = [&](bool reset) {
            if (!args.contains("state-dir")) return;
            const auto count = uint64_t(model.next_position());
            if (!reset && (count == 0 || !dumped_positions.insert(count - 1).second)) return;
            const StateReadback readback = [](void* dst, const void* src, size_t bytes, std::string& error) {
                const auto status = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
                if (status != cudaSuccess) { error = cudaGetErrorString(status); return false; }
                return true;
            };
            std::vector<StateEvidenceArray> arrays;
            require(describe_session_state(state, geometry, count, readback, {}, arrays, err), err);
            StateEvidenceRequest request;
            request.request_id = request_id(); request.token_count = count;
            request.position = count ? count - 1 : 0; request.max_bytes = state_max_bytes;
            request.reset = reset;
            request.required = state_evidence_inventory(arrays);
            const auto directory = std::filesystem::path(args.at("state-dir")) / request_id() /
                (reset ? "reset" : "position-" + std::to_string(count - 1));
            require(write_state_evidence(directory.string(), request, arrays, err), err);
            std::cout << "STATE request=" << request_id() << " tokens=" << count
                      << " directory=" << directory.string() << '\n';
        };
        dump_state(true);
        std::cout << "READY rank=0 tp=1 pp=1 layers=48 c=1 oracle=monolithic staging=selected-experts"
                     " gdn=native-unfused qsa=gather-reference select=block gr=native-bf16-mmvf"
                     " expert-activation=Q8_1 grouped=v1 iq-kernels=old expert-mode=0 expdec=0"
                     " benchmark=unsupported";
        if (selected_logits) std::cout << " logits_positions=" << args.at("logits-positions")
                                       << " logits_last=gen-segment";
        std::cout << '\n';
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            if (line == "RESET") {
                if (state_last) dump_state(false);
                require(model.reset(stream.value, err), err);
                ++request_sequence;
                if (trace) require(trace->request("request-" + std::to_string(request_sequence), err), err);
                dumped_positions.clear(); dump_state(true);
                std::cout << "RESET_OK\n"; continue;
            }
            std::istringstream input(line);
            std::string word; input >> word;
            require(word == "GEN", "expected GEN <teacher-forced token IDs>, RESET or QUIT");
            std::vector<int32_t> tokens;
            const auto position = uint64_t(model.next_position());
            require(position <= context, "position exceeds context");
            while (input >> word) {
                int32_t id = -1;
                const auto [end, status] = std::from_chars(word.data(), word.data() + word.size(), id);
                require(status == std::errc{} && end == word.data() + word.size() && id >= 0 &&
                        id < geometry.n_vocab, "invalid token ID");
                require(tokens.size() < context - position, "GEN exceeds remaining context");
                tokens.push_back(id);
            }
            require(!tokens.empty() && tokens.size() <= size_t(context - model.next_position()),
                    "GEN requires token IDs fitting the remaining context");
            size_t token_index = 0;
            for (const auto id : tokens) {
                ++token_index;
                const auto pos = model.next_position();
                require(model.token(id, pos, 0, static_cast<float*>(logits.pointer), stream.value, err), err);
                // token() has completed its bounded event; this copy has no pending device producers.
                cuda_check(cudaMemcpy(host_logits.data(), logits.pointer, host_logits.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                require(std::all_of(host_logits.begin(), host_logits.end(), [](float x) { return std::isfinite(x); }),
                        "oracle produced non-finite logits");
                const auto top = std::max_element(host_logits.begin(), host_logits.end()) - host_logits.begin();
                if (args.contains("logits-dir") && logits_positions.contains(uint64_t(pos), token_index == tokens.size())) {
                    const auto directory = std::filesystem::path(args.at("logits-dir")) / request_id();
                    std::filesystem::create_directories(directory);
                    const auto path = directory / ("logits-" + std::to_string(pos) + ".f32");
                    std::ofstream file(path, std::ios::binary);
                    file.write(reinterpret_cast<const char*>(host_logits.data()),
                               std::streamsize(host_logits.size() * sizeof(float)));
                    file.close();
                    require(bool(file), "cannot write logits artifact");
                }
                if (state_positions.contains(uint64_t(pos))) dump_state(false);
                std::cout << "TOKEN position=" << pos << " input=" << id << " argmax=" << top << '\n';
            }
        }
        if (state_last) dump_state(false);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "distributed reference: " << e.what() << '\n';
        return 1;
    }
}
