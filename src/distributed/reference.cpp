#include "strata/distributed/reference.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

// The original session implementation owns this state-selection operation.
namespace strata::core {
void gdn_point_at(const ModelGeometry&, int64_t, SessionState&);
}

namespace strata::distributed {
namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    uint8_t* bytes() const { return static_cast<uint8_t*>(data); }
    float* floats() const { return static_cast<float*>(data); }
};
struct HostBuffer {
    void* data = nullptr;
    ~HostBuffer() { if (data) cudaFreeHost(data); }
    void allocate(size_t bytes) { check(cudaHostAlloc(&data, bytes, cudaHostAllocDefault)); }
    uint8_t* bytes() const { return static_cast<uint8_t*>(data); }
};
} // namespace

struct MonolithicReference::Impl : core::DistributedLayerHooks {
    struct Layer {
        kernels::NativeExpertLayout format;
        const uint8_t* role[3]{};
        size_t role_bytes[3]{};
    };
    const core::WeightTable& weights;
    core::ModelGeometry geometry;
    core::SessionState& state;
    std::string pack, native;
    uint32_t timeout_ms;
    std::map<std::string, std::unique_ptr<GgufFile>> files;
    std::vector<Layer> layers;
    DeviceBuffer blobs, metadata, activation, scratch, embedding, parts;
    HostBuffer host_blobs, host_metadata;
    cudaEvent_t event = nullptr;
    size_t slot_bytes = 0;
    unsigned long long* pointers = nullptr;
    int32_t *starts = nullptr, *groups = nullptr, *destinations = nullptr, *tokens = nullptr, *ids = nullptr;
    const core::NativeHead* head = nullptr;
    bool ready = false, failed = false;
    int64_t position = 0;

    Impl(const core::WeightTable& w, const core::ModelGeometry& g, core::SessionState& s,
         std::string p, std::string n, uint32_t ms)
        : weights(w), geometry(g), state(s), pack(std::move(p)), native(std::move(n)), timeout_ms(ms) {}
    ~Impl() { if (event) cudaEventDestroy(event); }

    void complete(void* stream) {
        check(cudaGetLastError());
        check(cudaEventRecord(event, static_cast<cudaStream_t>(stream)));
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const auto result = cudaEventQuery(event);
            if (result == cudaSuccess) return;
            if (result != cudaErrorNotReady) check(result);
            if (std::chrono::steady_clock::now() >= end)
                throw std::runtime_error("TP1 reference: CUDA event completion deadline exceeded");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void initialize() {
        require(!ready && layers.empty(), "TP1 reference: initialize called twice");
        require(timeout_ms >= 100 && timeout_ms <= 600000 && geometry.n_vocab == 248320 &&
                geometry.n_layers == 48 && geometry.n_expert_used == 10 && geometry.n_expert == 512 &&
                geometry.n_embd == 2560 && geometry.n_ff == 640 && state.k == 10 &&
                state.layer_lo == 0 && state.layer_hi == geometry.n_layers && state.max_cells > 0 &&
                state.max_cells <= std::numeric_limits<int32_t>::max() && state.ple.ready(),
                "TP1 reference: requires full Qwen4Exp state and real PLE");
        require(std::filesystem::exists(std::filesystem::path(pack) / "native_experts.txt"),
                "TP1 reference: native expert manifest is missing");
        std::string err;
        if (!kernels::cpu::expert_layout_load(pack, geometry.n_layers, geometry.n_expert, err))
            throw std::runtime_error(err);
        const auto layout = kernels::cpu::expert_layout();
        require(layout.native && layout.fmt.size() == size_t(geometry.n_layers) &&
                layout.bytes.size() == size_t(geometry.n_layers) &&
                layout.gguf_off.size() == size_t(3 * geometry.n_layers),
                "TP1 reference: incomplete original GGUF role manifest");
        for (int64_t l = 0; l < geometry.n_layers; ++l) {
            const auto& fmt = layout.fmt[size_t(l)];
            require(fmt.n_embd == geometry.n_embd && fmt.n_ff == geometry.n_ff && !fmt.tail_off &&
                    !fmt.had_block && !fmt.had_gu && !fmt.had_d &&
                    kernels::native_expert_supported(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff),
                    "TP1 reference: unsupported original expert geometry or codec");
            Layer layer;
            layer.format = kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
            const auto& f = layer.format;
            require(!f.gu_nib && !f.tail_off && !f.had_block && f.gu_row == fmt.gu_row &&
                    f.d_row == fmt.d_row && f.up_off == fmt.up_off && f.down_off == fmt.down_off &&
                    f.bytes == fmt.bytes && f.bytes == layout.bytes[size_t(l)] &&
                    f.down_off == 2 * f.up_off && f.up_off == size_t(geometry.n_ff) * f.gu_row &&
                    f.bytes - f.down_off == size_t(geometry.n_embd) * f.d_row,
                    "TP1 reference: original CPU/CUDA blob layouts differ");
            for (int role = 0; role < 3; ++role) {
                const auto index = size_t(3 * l + role);
                std::filesystem::path path(native);
                if (index < layout.gguf_file.size() && !layout.gguf_file[index].empty())
                    path = path.parent_path() / layout.gguf_file[index];
                auto& file = files[path.string()];
                if (!file) file = std::make_unique<GgufFile>(path.string());
                static constexpr const char* names[] = {"gate", "up", "down"};
                const auto name = "blk." + std::to_string(l) + ".ffn_" + names[role] + "_exps.weight";
                const auto* tensor = file->find(name);
                const auto width = role == 2 ? geometry.n_ff : geometry.n_embd;
                const auto rows = role == 2 ? geometry.n_embd : geometry.n_ff;
                const auto type = role == 2 ? f.d_type : f.gu_type;
                const auto bytes = role == 2 ? f.bytes - f.down_off : f.up_off;
                require(tensor && tensor->shape == std::vector<uint64_t>{uint64_t(width), uint64_t(rows),
                            uint64_t(geometry.n_expert)} && tensor->type == uint32_t(type),
                        "TP1 reference: original GGUF tensor identity differs");
                const uint64_t total = uint64_t(bytes) * geometry.n_expert;
                require(file->data_start() <= file->file_size() &&
                        tensor->offset <= file->file_size() - file->data_start() &&
                        total <= file->file_size() - file->data_start() - tensor->offset &&
                        file->data_start() + tensor->offset == layout.gguf_off[index] &&
                        tensor_payload_bytes(*tensor) == total,
                        "TP1 reference: original GGUF role span differs");
                layer.role[role] = file->tensor_data(*tensor);
                layer.role_bytes[role] = bytes;
            }
            slot_bytes = std::max(slot_bytes, (f.bytes + 255) & ~size_t(255));
            layers.push_back(layer);
        }
        const size_t k = size_t(state.k);
        blobs.allocate(k * slot_bytes); host_blobs.allocate(k * slot_bytes);
        // Matching host/device offsets: pointers, starts, count, destinations,
        // token indices, and the routed IDs copied back before staging.
        const size_t meta_bytes = k * sizeof(unsigned long long) + (4 * k + 2) * sizeof(int32_t);
        metadata.allocate(meta_bytes); host_metadata.allocate(meta_bytes);
        pointers = reinterpret_cast<unsigned long long*>(metadata.data);
        starts = reinterpret_cast<int32_t*>(pointers + k); groups = starts + k + 1;
        destinations = groups + 1; tokens = destinations + k; ids = tokens + k;
        activation.allocate(size_t(geometry.n_embd / 32) * 36);
        scratch.allocate(kernels::native_expert_scratch_bytes(state.k, geometry.n_ff));
        embedding.allocate(size_t(geometry.n_embd) * sizeof(float));
        parts.allocate(k * size_t(geometry.n_embd) * sizeof(float));
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        ready = true;
    }

    ParityObserver observer;

    bool mixer(int64_t layer, int64_t pos, int32_t base, const float* input,
               float* output, void* stream, std::string& err) override {
        if (!observer({"mixer_input", layer, pos, input, size_t(geometry.n_embd)}, stream, err)) return false;
        const bool ok = core::is_qsa_layer(geometry, layer)
            ? core::qsa_layer(weights, geometry, layer, pos, base,
                              state.qsa_states[layer / geometry.qsa_interval], state.qsa_bufs,
                              input, output, stream, err)
            : core::gdn_layer(weights, geometry, layer, state.gdn, input, output, stream, err);
        return ok && observer({"mixer_output", layer, pos, output, size_t(geometry.n_embd)}, stream, err);
    }

    void experts(int64_t layer_index, void* stream) {
        const size_t k = size_t(state.k);
        const auto ids_offset = reinterpret_cast<uint8_t*>(ids) - metadata.bytes();
        auto* selected = reinterpret_cast<int32_t*>(host_metadata.bytes() + ids_offset);
        check(cudaMemcpyAsync(selected, state.moe.ids, k * sizeof(int32_t), cudaMemcpyDeviceToHost,
                              static_cast<cudaStream_t>(stream)));
        complete(stream);
        auto* ptr = reinterpret_cast<unsigned long long*>(host_metadata.data);
        auto* start = reinterpret_cast<int32_t*>(ptr + k);
        auto* count = start + k + 1;
        auto* dst = count + 1;
        auto* tok = dst + k;
        const auto& layer = layers[size_t(layer_index)];
        const size_t offsets[] = {0, layer.format.up_off, layer.format.down_off};
        for (size_t i = 0; i < k; ++i) {
            require(selected[i] >= 0 && selected[i] < geometry.n_expert,
                    "TP1 reference: routed expert ID outside original model");
            for (size_t j = 0; j < i; ++j)
                require(selected[i] != selected[j], "TP1 reference: duplicate routed expert ID");
            for (int role = 0; role < 3; ++role)
                std::memcpy(host_blobs.bytes() + i * slot_bytes + offsets[role],
                            layer.role[role] + size_t(selected[i]) * layer.role_bytes[role],
                            layer.role_bytes[role]);
            ptr[i] = reinterpret_cast<unsigned long long>(blobs.bytes() + i * slot_bytes);
            start[i] = int32_t(i); dst[i] = int32_t(i); tok[i] = 0;
        }
        start[k] = int32_t(k); *count = int32_t(k);
        check(cudaMemcpyAsync(metadata.data, host_metadata.data, size_t(ids_offset), cudaMemcpyHostToDevice,
                              static_cast<cudaStream_t>(stream)));
        for (size_t i = 0; i < k; ++i)
            check(cudaMemcpyAsync(blobs.bytes() + i * slot_bytes, host_blobs.bytes() + i * slot_bytes,
                                  layer.format.bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
        kernels::quantize_q8_1_rows(state.block.mixed, 1, geometry.n_embd, activation.data, stream);
        kernels::native_expert_grouped(layer.format, pointers, starts, groups, destinations, tokens,
                                       state.k, state.k, activation.data, scratch.data, parts.floats(), stream);
    }
};

MonolithicReference::MonolithicReference(const core::WeightTable& w, const core::ModelGeometry& g,
                                         core::SessionState& s, std::string pack, std::string native, uint32_t ms)
    : impl_(std::make_unique<Impl>(w, g, s, std::move(pack), std::move(native), ms)) {}
MonolithicReference::~MonolithicReference() = default;
bool MonolithicReference::initialize(std::string& err) {
    try { impl_->initialize(); return true; }
    catch (const std::exception& e) { err = e.what(); return false; }
}
void MonolithicReference::native_head(const core::NativeHead* h) { impl_->head = h; }
void MonolithicReference::parity_observer(ParityObserver observer) { impl_->observer = std::move(observer); }
int64_t MonolithicReference::next_position() const { return impl_->position; }
bool MonolithicReference::reset(void* stream, std::string& err) {
    auto& p = *impl_;
    if (!p.ready || p.failed || !stream) { err = "TP1 reference: invalid reset state"; return false; }
    try {
        core::session_zero(p.state, p.geometry, nullptr, stream);
        for (int64_t j = 0; j < p.state.qsa_alloc; ++j) {
            auto& q = p.state.qsa_states[p.state.qsa_ord0 + j];
            std::memset(q.host_step, 0, kernels::qsa_step_bytes() + sizeof(int32_t));
            std::memset(q.host_pos, 0, size_t(p.geometry.n_head) * sizeof(int32_t));
            check(cudaMemsetAsync(q.step, 0, kernels::qsa_step_bytes(), static_cast<cudaStream_t>(stream)));
            check(cudaMemsetAsync(q.pos_dev, 0, size_t(p.geometry.n_head) * sizeof(int32_t),
                                  static_cast<cudaStream_t>(stream)));
            check(cudaMemsetAsync(q.idx_block_pos, 0, sizeof(int32_t), static_cast<cudaStream_t>(stream)));
            check(cudaMemsetAsync(q.attention_status, 0, sizeof(int32_t), static_cast<cudaStream_t>(stream)));
        }
        p.complete(stream); p.position = 0; return true;
    } catch (const std::exception& e) { p.failed = true; err = e.what(); return false; }
}
bool MonolithicReference::token(int32_t token, int64_t pos, int32_t base, float* logits,
                                void* stream, std::string& err) {
    auto& p = *impl_;
    if (!p.ready || p.failed || !stream || !logits || token < 0 || token >= p.geometry.n_vocab ||
        pos != p.position || pos < 0 || base < 0 || int64_t(base) + pos >= p.state.max_cells) {
        err = "TP1 reference: requires valid consecutive token position and output"; return false;
    }
    try {
        p.state.ple_token = token;
        if (!core::ple_stage_token(p.state.ple, stream, err) ||
            !core::embed_row(p.weights, p.geometry, token, p.embedding.floats(), stream, err))
            throw std::runtime_error(err);
        for (int64_t c = 0; c < p.geometry.hc; ++c)
            check(cudaMemcpyAsync(p.state.R + c * p.geometry.n_embd, p.embedding.data,
                                  size_t(p.geometry.n_embd) * sizeof(float), cudaMemcpyDeviceToDevice,
                                  static_cast<cudaStream_t>(stream)));
        int64_t qsa_index = p.state.qsa_ord0;
        for (int64_t layer = 0; layer < p.geometry.n_layers; ++layer) {
            if (p.observer && !p.observer({"residual_pre", layer, pos, p.state.R,
                                          size_t(p.geometry.hc * p.geometry.n_embd)}, stream, err))
                throw std::runtime_error(err);
            core::gdn_point_at(p.geometry, layer, p.state);
            const bool qsa = core::is_qsa_layer(p.geometry, layer);
            auto& qst = p.state.qsa_states[qsa ? qsa_index : p.state.qsa_primary()];
            if (!core::block_layer_pre(p.weights, p.geometry, layer, pos, base, p.state.gdn, qst,
                                       p.state.qsa_bufs, p.state.moe, p.state.k, p.state.block,
                                       stream, err, nullptr, &p.state.ple, 0, 0, p.observer ? &p : nullptr))
                throw std::runtime_error("layer " + std::to_string(layer) + " pre: " + err);
            if (p.observer && !p.observer({"expert_input", layer, pos, p.state.block.mixed,
                                          size_t(p.geometry.n_embd)}, stream, err))
                throw std::runtime_error(err);
            p.experts(layer, stream);
            if (p.observer && !p.observer({"expert_output", layer, pos, p.parts.floats(),
                                          size_t(p.state.k * p.geometry.n_embd)}, stream, err))
                throw std::runtime_error(err);
            if (!core::block_layer_post(p.weights, p.geometry, layer, p.state.k, p.state.moe,
                                        p.state.block, p.parts.floats(), stream, err))
                throw std::runtime_error("layer " + std::to_string(layer) + " post: " + err);
            if (p.observer && !p.observer({"residual_post", layer, pos, p.state.R,
                                          size_t(p.geometry.hc * p.geometry.n_embd)}, stream, err))
                throw std::runtime_error(err);
            if (qsa) ++qsa_index;
        }
        if (p.head ? (!core::lm_head_mix(p.weights, p.geometry, p.state.block, stream, err) ||
                       !p.head->run(p.state.block.mixed, logits, stream, err))
                   : !core::lm_head(p.weights, p.geometry, p.state.block, logits, stream, err))
            throw std::runtime_error(err);
        p.complete(stream);
        p.state.ple_prev[0] = p.state.ple_prev[1]; p.state.ple_prev[1] = token;
        ++p.position; return true;
    } catch (const std::exception& e) { p.failed = true; err = e.what(); return false; }
}

} // namespace strata::distributed
