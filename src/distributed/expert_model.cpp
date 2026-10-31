#include "strata/distributed/expert_model.hpp"
#include "strata/distributed/cuda_segment_graph.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_upload.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/tp_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::distributed {
namespace {
namespace tp = kernels::tp::cuda;
struct UploadBackend {
    using Error = cudaError_t;
    static constexpr Error success = cudaSuccess;
    Error allocate(void** p, size_t bytes) { return cudaMallocHost(p, bytes); }
    void release(void* p) { (void)cudaFreeHost(p); }
    Error copy(void* dst, const void* src, size_t bytes) {
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
};
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Buffer {
    void* data = nullptr;
    std::size_t bytes = 0;
    Buffer() = default;
    explicit Buffer(std::size_t n) : bytes(n) { check(cudaMalloc(&data, n)); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data(other.data), bytes(other.bytes) { other.data = nullptr; }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data; bytes = other.bytes; other.data = nullptr;
        }
        return *this;
    }
    uint8_t* byte() const { return static_cast<uint8_t*>(data); }
    float* floats() const { return static_cast<float*>(data); }
};
struct RouteReadback {
    int* ids = nullptr;
    cudaEvent_t complete = nullptr;
    ~RouteReadback() {
        if (complete) cudaEventDestroy(complete);
        if (ids) cudaFreeHost(ids);
    }
    void initialize(std::size_t count) {
        check(cudaMallocHost(reinterpret_cast<void**>(&ids), count * sizeof(int)));
        check(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
    }
};
const char* codec(int type) {
    switch (type) {
        case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 18: return "IQ3_XXS";
        case 21: return "IQ3_S"; case 22: return "IQ2_S"; case 23: return "IQ4_XS";
        case 20: return "IQ4_NL"; case 42: return "Q2_0";
        default: throw std::runtime_error("TP expert model: unsupported original GGUF codec");
    }
}
int block(int type) { return type == 20 ? 32 : type == 42 ? 64 : 256; }
tp::MatrixShard shard(int type, int64_t width, int64_t output, int axis, int rank,
                     int64_t begin, int64_t end, const Buffer& buffer, std::size_t offset,
                     std::size_t row_stride, std::size_t bytes) {
    TensorSlice slice;
    slice.global_shape = {width, output}; slice.local_shape = slice.global_shape;
    slice.local_shape[axis] = end - begin; slice.axis = axis; slice.rank = rank;
    slice.range_start = begin; slice.range_end = end;
    slice.valid_lanes.assign(3, 0); slice.valid_lanes[rank] = end - begin;
    slice.codec = codec(type); slice.block_size = block(type);
    slice.scale_kind = "fp16-scale-plus-original-codes";
    slice.byte_row_stride = int64_t(kernels::iq_row_bytes(type, width));
    return {std::move(slice), type, buffer.byte() + offset, row_stride, bytes};
}
}  // namespace

struct NativeRoutedExperts::Impl {
    struct Layer {
        kernels::NativeExpertLayout layout;
        Buffer gate, up, down;
        tp::MatrixShard gate_view, up_view, down_view;
        std::size_t gu_bytes = 0, down_bytes = 0, down_row = 0;
    };
    struct FinishGraphs {
        Buffer output;
        std::array<CudaSegmentGraph, 2> codecs;
    };
    std::string pack, native;
    core::ModelGeometry geometry;
    int rank;
    LayerCollectives& collective;
    std::array<uint64_t, 3> valid;
    uint64_t begin = 0, end = 0, owned = 0;
    std::vector<Layer> layers;
    Buffer input_q, gate, up, local_hidden, wire, hidden, hidden_q, partial;
    Buffer lane_terms, lane_wire;
    Buffer batch_input_q, batch_packed_input_q, batch_gate, batch_up;
    Buffer batch_gu_hidden, batch_hidden;
    Buffer batch_packed_hidden_q;
    Buffer batch_hidden_local, batch_hidden_wire, batch_lane_group, batch_lane_local;
    Buffer batch_lane_wire, batch_parts;
    std::vector<int> batch_expert_starts;
    std::unique_ptr<FinishGraphs> finish_graphs;
    ExpertDownPolicy down_policy;
    ExpertCollectivePolicy collective_policy;
    bool compact_operands;
    bool graph_finish;
    bool graph_failed = false;
    ActivationContract input_contract, hidden_contract;
    RouteReadback readback;
    uint32_t readback_timeout_ms;
    bool readback_failed = false;
    bool initialized = false;
    bool batch_enabled = false;

    Impl(std::string pack_dir, std::string gguf, const core::ModelGeometry& g, int r,
         LayerCollectives& c, std::array<uint64_t, 3> counts, uint32_t timeout_ms, ExpertDownPolicy policy,
         ExpertCollectivePolicy collect_policy, bool enable_graph_finish, bool use_compact_operands)
        : pack(std::move(pack_dir)), native(std::move(gguf)), geometry(g), rank(r), collective(c), valid(counts),
          down_policy(policy), collective_policy(collect_policy), compact_operands(use_compact_operands),
          graph_finish(enable_graph_finish),
          readback_timeout_ms(timeout_ms) {
        require(timeout_ms > 0 && timeout_ms <= 3600000,
                "TP expert model: readback timeout must be in [1,3600000] ms");
        require(policy == ExpertDownPolicy::ExactLaneReference || policy == ExpertDownPolicy::PartialReduce,
                "TP expert model: unknown down arithmetic policy");
        require(!use_compact_operands || policy == ExpertDownPolicy::ExactLaneReference,
                "TP compact expert operands require exact lane arithmetic");
        require(collect_policy == ExpertCollectivePolicy::Sequential || collect_policy == ExpertCollectivePolicy::Batched,
                "TP expert model: unknown collective policy");
        require(collect_policy != ExpertCollectivePolicy::Batched || policy == ExpertDownPolicy::ExactLaneReference,
                "TP batched experts require exact lane arithmetic");
        require(!graph_finish || (collect_policy == ExpertCollectivePolicy::Batched &&
                                  policy == ExpertDownPolicy::ExactLaneReference),
                "TP finish graphs require batched exact experts");
        require(!graph_finish || (geometry.n_embd == 2560 && geometry.n_ff == 640 &&
                                  geometry.n_expert_used == 10 && valid == std::array<uint64_t, 3>{256, 192, 192}),
                "TP finish graphs require native TP3 geometry (2560/640/K10; 256/192/192)");
    }

    void initialize() {
        require(!initialized && layers.empty(), "TP expert model: initialize called twice");
        require(rank >= 0 && rank < 3 && geometry.n_embd > 0 && geometry.n_embd % 256 == 0 &&
                geometry.n_layers > 0 && geometry.n_expert > 0 && geometry.n_expert_used > 0 &&
                geometry.n_expert_used <= 15, "TP expert model: unsupported geometry/rank");
        require(std::accumulate(valid.begin(), valid.end(), uint64_t(0)) == uint64_t(geometry.n_ff),
                "TP expert model: FF ownership does not cover original model");
        for (int r = 0; r < 3; ++r) {
            require(valid[r] > 0 && valid[r] % 64 == 0, "TP expert model: FF cut splits original down block");
            if (r < rank) begin += valid[r];
        }
        end = begin + valid[rank];
        const auto slots = *std::max_element(valid.begin(), valid.end());
        std::string why;
        require(std::filesystem::exists(std::filesystem::path(pack) / "native_experts.txt"),
                "TP expert model requires original native_experts.txt");
        if (!kernels::cpu::expert_layout_load(pack, geometry.n_layers, geometry.n_expert, why))
            throw std::runtime_error(why);
        const auto layout = kernels::cpu::expert_layout();
        require(layout.native && layout.gguf_off.size() == std::size_t(3 * geometry.n_layers),
                "TP expert model requires original GGUF role offsets");
        require(layout.fmt.size() == std::size_t(geometry.n_layers) &&
                layout.bytes.size() == std::size_t(geometry.n_layers), "TP expert model: incomplete native layer table");
        // Validate every codec and the aggregate before any large allocation.
        uint64_t planned = 0;
        std::size_t compact_lane_width = 0;
        UploadBackend upload_backend;
        core::NativeUpload<UploadBackend> upload(upload_backend);
        std::vector<uint8_t> compact_down;
        for (int64_t index = 0; index < geometry.n_layers; ++index) {
            const auto& fmt = layout.fmt[std::size_t(index)];
            if (down_policy == ExpertDownPolicy::ExactLaneReference)
                require(geometry.n_ff <= 1024 && geometry.n_ff / block(fmt.d_type) * 2 <= 64,
                        "TP exact lane model requires at most two calls per warp lane");
            require(fmt.n_embd == geometry.n_embd && fmt.n_ff == geometry.n_ff && !fmt.tail_off &&
                    !fmt.had_block && !fmt.had_gu && !fmt.had_d,
                    "TP expert model: scaled/rotated expert is unsupported");
            (void)codec(fmt.gu_type); (void)codec(fmt.d_type);
            require(fmt.gu_type != 20 && fmt.gu_type != 42 && (fmt.d_type == 20 || fmt.d_type == 42),
                    "TP expert model: unsupported codec roles");
            if (compact_operands)
                compact_lane_width = std::max(compact_lane_width,
                    std::size_t(tp::compact_down_payload_width(fmt.d_type, int(geometry.n_ff), valid)));
            const uint64_t gu = valid[rank] * fmt.gu_row;
            const uint64_t down = uint64_t(geometry.n_embd) * kernels::iq_row_bytes(fmt.d_type, int64_t(valid[rank]));
            require(gu <= UINT64_MAX / 2 && 2 * gu <= UINT64_MAX - down &&
                    2 * gu + down <= (UINT64_MAX - planned) / uint64_t(geometry.n_expert),
                    "TP expert model: owned weight size overflow");
            planned += (2 * gu + down) * geometry.n_expert;
        }
        uint64_t planned_batch_scratch = 0;
        if (batch_enabled && collective_policy == ExpertCollectivePolicy::Batched &&
            compact_operands && down_policy == ExpertDownPolicy::ExactLaneReference) {
            const uint64_t entries = 8 * uint64_t(geometry.n_expert_used);
            const uint64_t hidden_slots = uint64_t(*std::max_element(valid.begin(), valid.end()));
            auto add_product = [&](uint64_t a, uint64_t b) {
                require(!b || a <= (UINT64_MAX - planned_batch_scratch) / b,
                        "TP expert model: routed batch scratch size overflow");
                planned_batch_scratch += a * b;
            };
            add_product(8, tp::global_q81_bytes(1, int(geometry.n_embd)));
            add_product(entries, tp::global_q81_bytes(1, int(geometry.n_embd)));
            add_product(entries, tp::global_q81_bytes(1, int(geometry.n_ff)));
            add_product(3 * entries * valid[rank], sizeof(float)); // gate, up and local hidden
            add_product(entries * uint64_t(geometry.n_ff), sizeof(float));
            add_product(entries * hidden_slots, sizeof(float));
            add_product(3 * entries * hidden_slots, sizeof(float));
            add_product(2 * entries * uint64_t(geometry.n_embd) * 32, sizeof(float));
            add_product(3 * entries * uint64_t(geometry.n_embd) * 32, sizeof(float));
            add_product(entries * uint64_t(geometry.n_embd), sizeof(float));
            require(planned <= UINT64_MAX - planned_batch_scratch,
                    "TP expert model: routed batch total size overflow");
            planned += planned_batch_scratch;
        }
        std::size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        require(planned < free_bytes, "TP expert model: owned weights and routed batch scratch exceed currently free device memory");
        std::map<std::string, std::unique_ptr<GgufFile>> files;
        auto source = [&](int64_t layer, int role, int type, int64_t width, int64_t rows,
                          std::size_t expert_bytes) -> const uint8_t* {
            const auto i = std::size_t(3 * layer + role);
            std::filesystem::path path(native);
            if (i < layout.gguf_file.size() && !layout.gguf_file[i].empty())
                path = path.parent_path() / layout.gguf_file[i];
            auto& file = files[path.string()];
            if (!file) file = std::make_unique<GgufFile>(path.string());
            static constexpr const char* roles[] = {"gate", "up", "down"};
            const auto name = "blk." + std::to_string(layer) + ".ffn_" + roles[role] + "_exps.weight";
            const auto* tensor = file->find(name);
            require(tensor && tensor->type == uint32_t(type) && tensor->shape.size() == 3 &&
                    tensor->shape[0] == uint64_t(width) && tensor->shape[1] == uint64_t(rows) &&
                    tensor->shape[2] == uint64_t(geometry.n_expert), "TP expert model: GGUF tensor identity mismatch");
            const uint64_t bytes = uint64_t(expert_bytes) * geometry.n_expert;
            require(file->data_start() + tensor->offset == layout.gguf_off[i] &&
                    tensor_payload_bytes(*tensor) == bytes && file->data_start() <= file->file_size() &&
                    tensor->offset <= file->file_size() - file->data_start() &&
                    bytes <= file->file_size() - file->data_start() - tensor->offset,
                    "TP expert model: GGUF role span differs from native pack");
            return file->tensor_data(*tensor);
        };
        for (int64_t index = 0; index < geometry.n_layers; ++index) {
            const auto& fmt = layout.fmt[std::size_t(index)];
            require(fmt.n_embd == geometry.n_embd && fmt.n_ff == geometry.n_ff && fmt.tail_off == 0 &&
                    fmt.had_block == 0, "TP expert model: unsupported scaled/rotated or changed geometry");
            (void)codec(fmt.gu_type); (void)codec(fmt.d_type);
            require(fmt.gu_type != 20 && fmt.gu_type != 42 && (fmt.d_type == 20 || fmt.d_type == 42),
                    "TP expert model: unsupported gate/up/down codec roles");
            Layer layer;
            layer.layout = kernels::native_expert_layout(fmt.gu_type, fmt.d_type, geometry.n_embd, geometry.n_ff);
            require(layer.layout.bytes == layout.bytes[std::size_t(index)] &&
                    layer.layout.gu_row == fmt.gu_row && layer.layout.d_row == fmt.d_row,
                    "TP expert model: native CPU/CUDA weight layout mismatch");
            layer.gu_bytes = std::size_t(valid[rank]) * fmt.gu_row;
            layer.down_row = kernels::iq_row_bytes(fmt.d_type, int64_t(valid[rank]));
            layer.down_bytes = std::size_t(geometry.n_embd) * layer.down_row;
            layer.gate = Buffer(layer.gu_bytes * geometry.n_expert);
            layer.up = Buffer(layer.gu_bytes * geometry.n_expert);
            layer.down = Buffer(layer.down_bytes * geometry.n_expert);
            const auto* gate_source = source(index, 0, fmt.gu_type, geometry.n_embd, geometry.n_ff, fmt.up_off);
            const auto* up_source = source(index, 1, fmt.gu_type, geometry.n_embd, geometry.n_ff, fmt.up_off);
            const auto down_expert_bytes = std::size_t(geometry.n_embd) * fmt.d_row;
            const auto* down_source = source(index, 2, fmt.d_type, geometry.n_ff, geometry.n_embd, down_expert_bytes);
            compact_down.resize(layer.down_bytes);
            for (int64_t expert = 0; expert < geometry.n_expert; ++expert) {
                const auto gu_offset = std::size_t(expert) * fmt.up_off + std::size_t(begin) * fmt.gu_row;
                check(upload.copy(layer.gate.byte() + expert * layer.gu_bytes, gate_source + gu_offset,
                                  layer.gu_bytes));
                check(upload.copy(layer.up.byte() + expert * layer.gu_bytes, up_source + gu_offset,
                                  layer.gu_bytes));
                const auto down_offset = std::size_t(expert) * down_expert_bytes + kernels::iq_row_bytes(fmt.d_type, int64_t(begin));
                // Pack unchanged input blocks on the CPU; the GPU sees only pinned source memory.
                for (int64_t row = 0; row < geometry.n_embd; ++row)
                    std::memcpy(compact_down.data() + size_t(row) * layer.down_row,
                                down_source + down_offset + size_t(row) * fmt.d_row, layer.down_row);
                check(upload.copy(layer.down.byte() + expert * layer.down_bytes,
                                  compact_down.data(), compact_down.size()));
            }
            std::fprintf(stderr, "expert shards loaded: layer=%lld rank=%d bytes=%llu\n",
                         static_cast<long long>(index), rank,
                         static_cast<unsigned long long>(layer.gate.bytes + layer.up.bytes + layer.down.bytes));
            owned += layer.gate.bytes + layer.up.bytes + layer.down.bytes;
            const auto& l = layer.layout;
            layer.gate_view = shard(l.gu_type, geometry.n_embd, geometry.n_ff, 1, rank, int64_t(begin), int64_t(end),
                                    layer.gate, 0, l.gu_row, layer.gu_bytes);
            layer.up_view = shard(l.gu_type, geometry.n_embd, geometry.n_ff, 1, rank, int64_t(begin), int64_t(end),
                                  layer.up, 0, l.gu_row, layer.gu_bytes);
            layer.down_view = shard(l.d_type, geometry.n_ff, geometry.n_embd, 0, rank, int64_t(begin), int64_t(end),
                                    layer.down, 0, layer.down_row, layer.down_bytes);
            layers.push_back(std::move(layer));
        }
        input_q = Buffer(tp::global_q81_bytes(1, int(geometry.n_embd)));
        const auto scratch_experts = collective_policy == ExpertCollectivePolicy::Batched ?
                                     std::size_t(geometry.n_expert_used) : std::size_t(1);
        gate = Buffer(scratch_experts * valid[rank] * sizeof(float)); up = Buffer(gate.bytes);
        local_hidden = Buffer(scratch_experts * slots * sizeof(float)); wire = Buffer(3 * local_hidden.bytes);
        hidden = Buffer(scratch_experts * geometry.n_ff * sizeof(float));
        hidden_q = Buffer(scratch_experts * tp::global_q81_bytes(1, int(geometry.n_ff)));
        partial = Buffer(geometry.n_embd * sizeof(float));
        if (down_policy == ExpertDownPolicy::ExactLaneReference) {
            const auto lane_width = compact_operands ? compact_lane_width :
                                    std::size_t(tp::down_lane_payload_width);
            lane_terms = Buffer(scratch_experts * geometry.n_embd * lane_width * sizeof(float));
            lane_wire = Buffer(3 * lane_terms.bytes);
        }
        if (batch_enabled && collective_policy == ExpertCollectivePolicy::Batched && compact_operands &&
            down_policy == ExpertDownPolicy::ExactLaneReference) {
            const auto entries = std::size_t(8 * geometry.n_expert_used);
            const auto q_input_bytes = tp::global_q81_bytes(8, int(geometry.n_embd));
            batch_input_q = Buffer(q_input_bytes);
            batch_packed_input_q = Buffer(tp::global_q81_bytes(int(entries), int(geometry.n_embd)));
            batch_packed_hidden_q = Buffer(tp::global_q81_bytes(int(entries), int(geometry.n_ff)));
            batch_gate = Buffer(entries * valid[rank] * sizeof(float));
            batch_up = Buffer(batch_gate.bytes);
            batch_gu_hidden = Buffer(batch_gate.bytes);
            batch_hidden = Buffer(entries * std::size_t(geometry.n_ff) * sizeof(float));
            batch_hidden_local = Buffer(entries * std::size_t(slots) * sizeof(float));
            batch_hidden_wire = Buffer(3 * entries * std::size_t(slots) * sizeof(float));
            const auto max_lane_width = std::size_t(32);
            batch_lane_group = Buffer(entries * std::size_t(geometry.n_embd) * max_lane_width * sizeof(float));
            batch_lane_local = Buffer(entries * std::size_t(geometry.n_embd) * max_lane_width * sizeof(float));
            batch_lane_wire = Buffer(3 * batch_lane_local.bytes);
            batch_parts = Buffer(entries * std::size_t(geometry.n_embd) * sizeof(float));
            batch_expert_starts.resize(std::size_t(geometry.n_expert + 1));
        }
        readback.initialize(std::size_t(batch_enabled ? 8 : 1) * std::size_t(geometry.n_expert_used));
        input_contract.consumer = "expert_gate_up"; input_contract.gather = "none";
        input_contract.quantizer = "Q8_1"; input_contract.block_size = 32;
        input_contract.reference = "global_reference"; input_contract.backend_policy = "cuda_native_expert_grouped";
        input_contract.layout = "contiguous";
        hidden_contract = input_contract; hidden_contract.consumer = "expert_down";
        hidden_contract.gather = "all_gather_padded"; hidden_contract.wire_slots = int64_t(slots);
        hidden_contract.valid_counts.assign(valid.begin(), valid.end());
        if (graph_finish) {
            finish_graphs = std::make_unique<FinishGraphs>();
            finish_graphs->output = Buffer(scratch_experts * geometry.n_embd * sizeof(float));
            for (std::size_t c = 0; c < finish_graphs->codecs.size(); ++c) {
                const int type = c == 0 ? 20 : 42;
                bool warmup = true;
                const auto enqueue = [&](void* capture_stream) {
                    // Only the uncaptured warmup initializes operands. Capture
                    // sees precisely the existing finalizer and fixed pointers.
                    if (warmup) {
                        check(cudaMemsetAsync(lane_wire.data, 0, lane_wire.bytes,
                                              static_cast<cudaStream_t>(capture_stream)));
                        warmup = false;
                    }
                    if (compact_operands) {
                        const auto lane_width = tp::compact_down_payload_width(type, int(geometry.n_ff), valid);
                        tp::finish_down_lanes_compact(lane_wire.floats(), type, int(geometry.n_ff),
                                                      int(geometry.n_embd), int(geometry.n_expert_used), valid,
                                                      lane_width, finish_graphs->output.floats(), capture_stream);
                    } else {
                        tp::finish_down_lanes(lane_wire.floats(), type, int(geometry.n_ff),
                                              int(geometry.n_embd), int(geometry.n_expert_used), valid,
                                              finish_graphs->output.floats(), capture_stream);
                    }
                };
                if (!finish_graphs->codecs[c].initialize(enqueue, readback_timeout_ms, why))
                    throw std::runtime_error("TP expert finish graph: " + why);
            }
        }
        initialized = true;
    }

    bool forward(int64_t index, const float* input, const int* ids, int64_t k, float* parts, void* stream,
                 std::string& err) {
        require(initialized && !readback_failed && !graph_failed && index >= 0 && index < int64_t(layers.size()) && k == geometry.n_expert_used &&
                input && ids && parts && stream, "TP expert model: invalid forward state/arguments");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(readback_timeout_ms);
        // An incomplete/failed transfer must never reuse its host buffer or
        // launch expert work. The caller aborts the request on this failure.
        readback_failed = true;
        check(cudaMemcpyAsync(readback.ids, ids, std::size_t(k) * sizeof(int), cudaMemcpyDeviceToHost,
                              static_cast<cudaStream_t>(stream)));
        check(cudaEventRecord(readback.complete, static_cast<cudaStream_t>(stream)));
        for (;;) {
            require(std::chrono::steady_clock::now() < deadline,
                    "TP expert model: routed-id readback deadline expired");
            const auto status = cudaEventQuery(readback.complete);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) check(status);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        require(std::chrono::steady_clock::now() < deadline,
                "TP expert model: routed-id readback completed after deadline");
        readback_failed = false;
        for (int64_t i = 0; i < k; ++i)
            require(readback.ids[i] >= 0 && readback.ids[i] < geometry.n_expert,
                    "TP expert model: routed id outside original experts");
        const auto q = tp::quantize_global(input_contract, input, 1, int(geometry.n_embd), input_q.data, input_q.bytes, stream);
        auto& layer = layers[std::size_t(index)];
        if (collective_policy == ExpertCollectivePolicy::Batched) {
            const auto slots = uint64_t(hidden_contract.wire_slots);
            const auto qbytes = tp::global_q81_bytes(1, int(geometry.n_ff));
            const auto lane_width = compact_operands ?
                tp::compact_down_payload_width(layer.layout.d_type, int(geometry.n_ff), valid) :
                tp::down_lane_payload_width;
            const auto lane_count = uint64_t(geometry.n_embd) * uint64_t(lane_width);
            check(cudaMemsetAsync(local_hidden.data, 0, local_hidden.bytes, static_cast<cudaStream_t>(stream)));
            for (int64_t i = 0; i < k; ++i) {
                const auto expert = std::size_t(readback.ids[i]);
                layer.gate_view.data = layer.gate.byte() + expert * layer.gu_bytes;
                layer.up_view.data = layer.up.byte() + expert * layer.gu_bytes;
                tp::gate_up(layer.gate_view, layer.up_view, input_contract, q,
                            gate.floats() + i * valid[rank], up.floats() + i * valid[rank],
                            local_hidden.floats() + i * slots, stream);
            }
            // Uniform transport counts include the holes at the end of each expert's local row.
            const auto hidden_count = uint64_t(k) * slots;
            if (!collective.gather(index, "expert_hidden_batch", local_hidden.floats(), wire.floats(),
                                   {hidden_count, hidden_count, hidden_count}, stream, err)) return false;
            tp::unpack_expert_batch(hidden_contract, wire.floats(), int(k), hidden.floats(), stream);
            for (int64_t i = 0; i < k; ++i) {
                const auto hq = tp::quantize_global(hidden_contract, hidden.floats() + i * geometry.n_ff,
                    1, int(geometry.n_ff), hidden_q.byte() + i * qbytes, qbytes, stream);
                layer.down_view.data = layer.down.byte() + std::size_t(readback.ids[i]) * layer.down_bytes;
                if (compact_operands)
                    tp::down_lanes_compact(layer.down_view, hidden_contract, hq,
                                           lane_terms.floats() + i * lane_count, lane_width, stream);
                else
                    tp::down_lanes(layer.down_view, hidden_contract, hq,
                                   lane_terms.floats() + i * lane_count, stream);
            }
            const auto count = uint64_t(k) * lane_count;
            if (!collective.gather(index, "expert_lane_terms_batch", lane_terms.floats(), lane_wire.floats(),
                                   {count, count, count}, stream, err)) return false;
            // The existing exact finalizer consumes [rank][row][output][operand]; expert is row here.
            if (finish_graphs) {
                auto& graph = finish_graphs->codecs[layer.layout.d_type == 20 ? 0 : 1];
                if (!graph.launch(stream, err)) return false;
                check(cudaMemcpyAsync(parts, finish_graphs->output.data, finish_graphs->output.bytes,
                                      cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
            } else {
                if (compact_operands)
                    tp::finish_down_lanes_compact(lane_wire.floats(), layer.layout.d_type, int(geometry.n_ff),
                                                  int(geometry.n_embd), int(k), valid, lane_width, parts, stream);
                else
                    tp::finish_down_lanes(lane_wire.floats(), layer.layout.d_type, int(geometry.n_ff),
                                          int(geometry.n_embd), int(k), valid, parts, stream);
            }
            return true;
        }
        for (int64_t i = 0; i < k; ++i) {
            const auto expert = std::size_t(readback.ids[i]);
            layer.gate_view.data = layer.gate.byte() + expert * layer.gu_bytes;
            layer.up_view.data = layer.up.byte() + expert * layer.gu_bytes;
            layer.down_view.data = layer.down.byte() + expert * layer.down_bytes;
            check(cudaMemsetAsync(local_hidden.data, 0, local_hidden.bytes, static_cast<cudaStream_t>(stream)));
            tp::gate_up(layer.gate_view, layer.up_view, input_contract, q, gate.floats(), up.floats(), local_hidden.floats(), stream);
            char stage[48];
            std::snprintf(stage, sizeof(stage), "expert_hidden_%lld", static_cast<long long>(i));
            if (!collective.gather(index, stage, local_hidden.floats(), wire.floats(), valid, stream, err)) return false;
            tp::unpack_gathered(hidden_contract, wire.floats(), 1, hidden.floats(), stream);
            const auto hq = tp::quantize_global(hidden_contract, hidden.floats(), 1, int(geometry.n_ff),
                                                hidden_q.data, hidden_q.bytes, stream);
            if (down_policy == ExpertDownPolicy::ExactLaneReference) {
                const auto lane_width = compact_operands ?
                    tp::compact_down_payload_width(layer.layout.d_type, int(geometry.n_ff), valid) :
                    tp::down_lane_payload_width;
                if (compact_operands)
                    tp::down_lanes_compact(layer.down_view, hidden_contract, hq, lane_terms.floats(),
                                           lane_width, stream);
                else
                    tp::down_lanes(layer.down_view, hidden_contract, hq, lane_terms.floats(), stream);
                const uint64_t count = uint64_t(geometry.n_embd) * uint64_t(lane_width);
                std::snprintf(stage, sizeof(stage), "expert_lane_terms_%lld", static_cast<long long>(i));
                if (!collective.gather(index, stage, lane_terms.floats(), lane_wire.floats(),
                                       {count, count, count}, stream, err)) return false;
                if (compact_operands)
                    tp::finish_down_lanes_compact(lane_wire.floats(), layer.layout.d_type, int(geometry.n_ff),
                                                  int(geometry.n_embd), 1, valid, lane_width,
                                                  parts + i * geometry.n_embd, stream);
                else
                    tp::finish_down_lanes(lane_wire.floats(), layer.layout.d_type, int(geometry.n_ff),
                                          int(geometry.n_embd), 1, valid, parts + i * geometry.n_embd, stream);
            } else {
                tp::down(layer.down_view, hidden_contract, hq, partial.floats(), stream);
                std::snprintf(stage, sizeof(stage), "expert_output_%lld", static_cast<long long>(i));
                if (!collective.reduce(index, stage, partial.floats(),
                                       parts + i * geometry.n_embd, std::size_t(geometry.n_embd), stream, err)) return false;
            }
        }
        return true;
    }

    bool forward_batch(int64_t index, int64_t rows, const float* inputs, const int* ids, int64_t k,
                       float* parts, void* stream, std::string& err) {
        require(initialized && !readback_failed && !graph_failed &&
                collective_policy == ExpertCollectivePolicy::Batched &&
                down_policy == ExpertDownPolicy::ExactLaneReference && compact_operands &&
                index >= 0 && index < int64_t(layers.size()) && rows >= 2 && rows <= 8 &&
                k == geometry.n_expert_used && inputs && ids && parts && stream,
                "TP expert model: invalid or unsupported routed batch arguments");
        const int64_t entries = rows * k;
        require(batch_enabled && entries <= 80 && readback.ids,
                "TP expert model: routed batch exceeds eight rows or was not enabled");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(readback_timeout_ms);
        readback_failed = true;
        check(cudaMemcpyAsync(readback.ids, ids, std::size_t(entries) * sizeof(int), cudaMemcpyDeviceToHost,
                              static_cast<cudaStream_t>(stream)));
        check(cudaEventRecord(readback.complete, static_cast<cudaStream_t>(stream)));
        for (;;) {
            require(std::chrono::steady_clock::now() < deadline,
                    "TP expert model: routed batch readback deadline expired");
            const auto status = cudaEventQuery(readback.complete);
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady) check(status);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        readback_failed = false;
        for (int64_t entry = 0; entry < entries; ++entry)
            require(readback.ids[entry] >= 0 && readback.ids[entry] < geometry.n_expert,
                    "TP expert model: routed batch id outside original experts");

        auto& layer = layers[std::size_t(index)];
        const int max_lane_width = tp::compact_down_payload_width(layer.layout.d_type, int(geometry.n_ff), valid);
        require(std::size_t(entries) * std::size_t(geometry.n_embd) *
                    std::size_t(max_lane_width) * sizeof(float) <= std::size_t(80) *
                    std::size_t(geometry.n_embd) * std::size_t(32) * sizeof(float),
                "TP expert model: routed batch operand buffer limit exceeded");
        (void)tp::quantize_global(input_contract, inputs, int(rows), int(geometry.n_embd),
                                  batch_input_q.data, batch_input_q.bytes, stream);
        const auto slots = std::size_t(*std::max_element(valid.begin(), valid.end()));
        std::array<int, 8 * 15> grouped_entries{};
        auto& expert_starts = batch_expert_starts;
        int grouped_count = 0;
        for (int expert = 0; expert < geometry.n_expert; ++expert) {
            expert_starts[size_t(expert)] = grouped_count;
            for (int64_t entry = 0; entry < entries; ++entry)
                if (readback.ids[entry] == expert) grouped_entries[size_t(grouped_count++)] = int(entry);
        }
        expert_starts[size_t(geometry.n_expert)] = grouped_count;
        check(cudaMemsetAsync(batch_hidden_local.data, 0, batch_hidden_local.bytes,
                              static_cast<cudaStream_t>(stream)));
        const auto q_input_row_bytes = tp::global_q81_bytes(1, int(geometry.n_embd));
        const auto q_hidden_row_bytes = tp::global_q81_bytes(1, int(geometry.n_ff));
        for (int expert = 0; expert < geometry.n_expert; ++expert) {
            const int start = expert_starts[size_t(expert)];
            const int count = expert_starts[size_t(expert + 1)] - start;
            if (count == 0) continue;
            for (int row = 0; row < count; ++row) {
                const auto entry = std::size_t(grouped_entries[size_t(start + row)]);
                const auto token = entry / std::size_t(k);
                check(cudaMemcpyAsync(static_cast<uint8_t*>(batch_packed_input_q.data) +
                                          std::size_t(start + row) * q_input_row_bytes,
                                      static_cast<const uint8_t*>(batch_input_q.data) + token * q_input_row_bytes,
                                      q_input_row_bytes, cudaMemcpyDeviceToDevice,
                                      static_cast<cudaStream_t>(stream)));
            }
            const auto* qrows = static_cast<const uint8_t*>(batch_packed_input_q.data) +
                                std::size_t(start) * q_input_row_bytes;
            tp::GlobalQ81 q{qrows, int(geometry.n_embd), count, q_input_row_bytes,
                            std::size_t(count) * q_input_row_bytes};
            layer.gate_view.data = layer.gate.byte() + std::size_t(expert) * layer.gu_bytes;
            layer.up_view.data = layer.up.byte() + std::size_t(expert) * layer.gu_bytes;
            auto* gate_rows = batch_gate.floats();
            auto* up_rows = batch_up.floats();
            auto* hidden_rows = batch_gu_hidden.floats();
            tp::gate_up(layer.gate_view, layer.up_view, input_contract, q,
                        gate_rows, up_rows, hidden_rows, stream);
            for (int row = 0; row < count; ++row)
                check(cudaMemcpyAsync(batch_hidden_local.floats() + std::size_t(start + row) * slots,
                                      hidden_rows + std::size_t(row) * valid[rank],
                                      std::size_t(valid[rank]) * sizeof(float), cudaMemcpyDeviceToDevice,
                                      static_cast<cudaStream_t>(stream)));
        }
        const auto hidden_count = uint64_t(entries) * uint64_t(slots);
        if (!collective.gather(index, "expert_hidden_rows", batch_hidden_local.floats(),
                               batch_hidden_wire.floats(), {hidden_count, hidden_count, hidden_count},
                               stream, err)) return false;
        tp::unpack_expert_batch(hidden_contract, batch_hidden_wire.floats(), int(entries),
                                batch_hidden.floats(), stream);
        for (int packed = 0; packed < grouped_count; ++packed)
            tp::quantize_global(hidden_contract, batch_hidden.floats() + std::size_t(packed) * geometry.n_ff,
                                1, int(geometry.n_ff),
                                static_cast<uint8_t*>(batch_packed_hidden_q.data) + std::size_t(packed) * q_hidden_row_bytes,
                                q_hidden_row_bytes, stream);

        const auto lane_width = std::size_t(max_lane_width);
        const auto lane_count = std::size_t(geometry.n_embd) * lane_width;
        check(cudaMemsetAsync(batch_lane_local.data, 0, batch_lane_local.bytes,
                              static_cast<cudaStream_t>(stream)));
        for (int expert = 0; expert < geometry.n_expert; ++expert) {
            const int start = expert_starts[size_t(expert)];
            const int count = expert_starts[size_t(expert + 1)] - start;
            if (count == 0) continue;
            const auto* qrows = static_cast<const uint8_t*>(batch_packed_hidden_q.data) +
                                std::size_t(start) * q_hidden_row_bytes;
            tp::GlobalQ81 hq{qrows, int(geometry.n_ff), count, q_hidden_row_bytes,
                             std::size_t(count) * q_hidden_row_bytes};
            layer.down_view.data = layer.down.byte() + std::size_t(expert) * layer.down_bytes;
            tp::down_lanes_compact(layer.down_view, hidden_contract, hq,
                                   batch_lane_group.floats(),
                                   max_lane_width, stream);
            for (int row = 0; row < count; ++row) {
                const auto entry = std::size_t(grouped_entries[size_t(start + row)]);
                check(cudaMemcpyAsync(batch_lane_local.floats() + entry * lane_count,
                                      batch_lane_group.floats() + std::size_t(row) * lane_count,
                                      lane_count * sizeof(float), cudaMemcpyDeviceToDevice,
                                      static_cast<cudaStream_t>(stream)));
            }
        }
        const auto lane_wire_count = uint64_t(entries) * uint64_t(geometry.n_embd) * lane_width;
        if (!collective.gather(index, "expert_lane_terms_rows", batch_lane_local.floats(),
                               batch_lane_wire.floats(),
                               {lane_wire_count, lane_wire_count, lane_wire_count}, stream, err)) return false;
        tp::finish_down_lanes_compact(batch_lane_wire.floats(), layer.layout.d_type, int(geometry.n_ff),
                                      int(geometry.n_embd), int(entries), valid, max_lane_width,
                                      batch_parts.floats(), stream);
        for (int64_t entry = 0; entry < entries; ++entry)
            check(cudaMemcpyAsync(parts + entry * geometry.n_embd, batch_parts.floats() + entry * geometry.n_embd,
                                  std::size_t(geometry.n_embd) * sizeof(float), cudaMemcpyDeviceToDevice,
                                  static_cast<cudaStream_t>(stream)));
        return true;
    }

    void poison_graphs() noexcept {
        if (!graph_finish) return;
        graph_failed = true;
        if (finish_graphs) for (auto& graph : finish_graphs->codecs) graph.poison();
    }
};

NativeRoutedExperts::NativeRoutedExperts(std::string pack, std::string native, const core::ModelGeometry& geometry,
                                       int rank, LayerCollectives& collective, std::array<uint64_t, 3> valid,
                                       uint32_t readback_timeout_ms, ExpertDownPolicy down_policy,
                                       ExpertCollectivePolicy collective_policy, bool graph_finish,
                                       bool compact_operands)
    : impl_(std::make_unique<Impl>(std::move(pack), std::move(native), geometry, rank, collective, valid,
                                   readback_timeout_ms, down_policy, collective_policy, graph_finish,
                                   compact_operands)) {}
NativeRoutedExperts::~NativeRoutedExperts() {
    // A bounded CUDA failure cannot be followed by cudaFree on captured buffers
    // that the GPU may still use. The worker owner must _Exit in this state.
    if (requires_process_exit()) (void)impl_.release();
}
bool NativeRoutedExperts::configure_prefill_batch(bool enabled, std::string& err) {
    if (impl_->initialized || impl_->readback.ids) {
        err = "TP prefill expert batch must be configured before expert initialization";
        return false;
    }
    impl_->batch_enabled = enabled;
    return true;
}
bool NativeRoutedExperts::initialize(std::string& err) {
    try { impl_->initialize(); return true; }
    catch (const std::exception& e) { impl_->poison_graphs(); err = e.what(); return false; }
}
bool NativeRoutedExperts::supports_batch() const noexcept {
    return impl_->batch_enabled && impl_->collective_policy == ExpertCollectivePolicy::Batched &&
           impl_->down_policy == ExpertDownPolicy::ExactLaneReference && impl_->compact_operands;
}
bool NativeRoutedExperts::forward(int64_t layer, const float* input, const int* ids, int64_t k,
                                  float* parts, void* stream, std::string& err) {
    try {
        const bool success = impl_->forward(layer, input, ids, k, parts, stream, err);
        if (!success) impl_->poison_graphs();
        return success;
    } catch (const std::exception& e) { impl_->poison_graphs(); err = e.what(); return false; }
}
bool NativeRoutedExperts::forward_batch(int64_t layer, int64_t rows, const float* inputs, const int* ids,
                                        int64_t k, float* parts, void* stream, std::string& err) {
    try {
        const bool success = impl_->forward_batch(layer, rows, inputs, ids, k, parts, stream, err);
        if (!success) impl_->poison_graphs();
        return success;
    } catch (const std::exception& e) { impl_->poison_graphs(); err = e.what(); return false; }
}
uint64_t NativeRoutedExperts::owned_weight_bytes() const { return impl_->owned; }
std::array<uint64_t, 3> NativeRoutedExperts::valid_counts() const { return impl_->valid; }
std::vector<int> NativeRoutedExperts::down_codecs() const {
    std::vector<int> result;
    result.reserve(impl_->layers.size());
    for (const auto& layer : impl_->layers) result.push_back(layer.layout.d_type);
    return result;
}
ExpertFinishGraphStats NativeRoutedExperts::finish_graph_stats() const {
    ExpertFinishGraphStats stats;
    if (impl_->finish_graphs) {
        stats.output_buffer_bytes = impl_->finish_graphs->output.bytes;
        for (const auto& graph : impl_->finish_graphs->codecs) {
            stats.graph_count += graph.ready() ? 1 : 0;
            stats.node_count += graph.node_count();
        }
    }
    return stats;
}
bool NativeRoutedExperts::requires_process_exit() const noexcept {
    if (impl_->finish_graphs)
        for (const auto& graph : impl_->finish_graphs->codecs)
            if (graph.requires_process_exit()) return true;
    return false;
}
}  // namespace strata::distributed
