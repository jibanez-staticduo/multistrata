#include "strata/distributed/model.hpp"
#include "strata/distributed/qsa_model.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/tp_gdn.hpp"
#include "model_state.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <vector>

namespace strata::distributed {
namespace {
void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct DeviceMemory {
    void* data = nullptr;
    ~DeviceMemory() { if (data) cudaFree(data); }
    void allocate(size_t bytes) { checked(cudaMalloc(&data, bytes)); }
};
// A null counter takes no timestamps, keeping disabled profiling off the clock path.
struct ProfileSpan {
    uint64_t* counter;
    std::chrono::steady_clock::time_point start{};
    explicit ProfileSpan(uint64_t* value) : counter(value) {
        if (counter) start = std::chrono::steady_clock::now();
    }
    ~ProfileSpan() {
        if (counter) *counter += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    }
};
#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
struct CheckpointIo {
    CudaCheckpointTransfer transfer;
    ~CheckpointIo() {
        // A worker with pending DMA exits without unwinding borrowed session
        // owners. Never free staging that a failed CUDA operation may still use.
        if (transfer.pending || transfer.poisoned) return;
        if (transfer.completion_event) cudaEventDestroy(static_cast<cudaEvent_t>(transfer.completion_event));
        if (transfer.pinned_staging) cudaFreeHost(transfer.pinned_staging);
    }
    bool prepare(std::string& err) {
        if (transfer.poisoned || transfer.pending) { err = "checkpoint CUDA transfer is poisoned"; return false; }
        if (transfer.pinned_staging && transfer.completion_event) return true;
        constexpr size_t bytes = 1024 * 1024;
        auto status = cudaSuccess;
        if (!transfer.pinned_staging) {
            status = cudaMallocHost(&transfer.pinned_staging, bytes);
            if (status != cudaSuccess) { err = cudaGetErrorString(status); return false; }
            transfer.staging_bytes = bytes;
        }
        cudaEvent_t event = nullptr;
        status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (status != cudaSuccess) { err = cudaGetErrorString(status); return false; }
        transfer.completion_event = event;
        return true;
    }
    bool copy(void* host, void* device, size_t bytes, bool to_device, void* stream,
              CheckpointDeadline deadline, std::string& err) {
        for (size_t offset = 0; offset < bytes;) {
            const auto count = std::min(bytes - offset, transfer.staging_bytes);
            if (!cuda_checkpoint_copy(transfer, static_cast<uint8_t*>(host) + offset,
                                      static_cast<uint8_t*>(device) + offset, count,
                                      to_device, stream, deadline, err)) return false;
            offset += count;
        }
        return true;
    }
};
#endif
}

SessionCollectives::SessionCollectives(Session& s, GuardedTransport& t, const Clock& c, uint32_t ms)
    : session_(s), transport_(t), clock_(c), timeout_ms_(ms) {}

bool SessionCollectives::agree_routes(int64_t layer, const int* ids, int64_t k,
                                     void* stream, std::string& err) {
    if (!ids || k != 10 || !stream) { err = "invalid TP route agreement"; return false; }
    std::array<int, 10> host{};
    Deadline deadline;
    if (const auto f = session_.operation_deadline(deadline)) { err = f.detail; return false; }
    deadline.end = std::min(deadline.end, clock_.now() + std::chrono::milliseconds(timeout_ms_));
    auto* stats = profiling_ && layer >= 0 && uint64_t(layer) < profile_.size() ? &profile_[size_t(layer)] : nullptr;
    if (stats) ++stats->route_calls;
    {
        ProfileSpan timing(stats ? &stats->route_readback_ns : nullptr);
        auto status = cudaMemcpyAsync(host.data(), ids, sizeof(host), cudaMemcpyDeviceToHost,
                                       static_cast<cudaStream_t>(stream));
        if (status != cudaSuccess) { err = cudaGetErrorString(status); return false; }
        for (;;) {
            status = cudaStreamQuery(static_cast<cudaStream_t>(stream));
            if (status == cudaSuccess) break;
            if (status != cudaErrorNotReady || deadline_passed(clock_, deadline)) {
                err = "TP route readback failed or exceeded deadline";
                session_.abort(err); transport_.abort(err); return false;
            }
            if (stats) ++stats->route_pending_polls;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ProfileSpan timing(stats ? &stats->route_agreement_ns : nullptr);
    std::string payload = std::to_string(step_) + ":" + std::to_string(layer) + ":";
    for (const int id : host) payload += std::to_string(id) + ":";
    if (const auto f = session_.agree_control_payload("routes-v1", payload, deadline)) {
        err = f.detail; transport_.abort(err); return false;
    }
    return true;
}

bool SessionCollectives::execute(CollectiveDescriptor d, const char* stage, const float* in, float* out,
                                 void* stream, std::string& err) {
    std::string actual, planned;
    if (observed_.size() >= expected_.size() ||
        canonical_collective(d, actual) ||
        canonical_collective(expected_[observed_.size()].descriptor, planned) ||
        actual != planned || expected_[observed_.size()].route != stage) {
        err = "actual TP operation differs from agreed command schedule";
        session_.abort(err); transport_.abort(err); return false;
    }
    Deadline deadline;
    if (const auto f = session_.operation_deadline(deadline)) { err = f.detail; return false; }
    deadline.end = std::min(deadline.end, clock_.now() + std::chrono::milliseconds(timeout_ms_));
    auto* stats = profiling_ && d.operation.layer < profile_.size() ? &profile_[size_t(d.operation.layer)] : nullptr;
    if (stats) {
        ++stats->collective_calls;
        stats->input_bytes += d.elements * sizeof(float);
        stats->output_bytes += d.elements * sizeof(float) *
                               (d.kind == CollectiveKind::all_gather_padded ? 3 : 1);
    }
    if (!preagreed_) {
        ProfileSpan timing(stats ? &stats->stage_metadata_ns : nullptr);
        const std::string stage_payload = actual + ":" + std::to_string(std::char_traits<char>::length(stage)) + ":" + stage;
        if (const auto f = session_.agree_control_payload("model-stage-v1", stage_payload, deadline)) {
            err = f.detail; transport_.abort(err); return false;
        }
    }
    {
        ProfileSpan timing(stats ? &stats->operation_agreement_ns : nullptr);
        const auto f = preagreed_ ? session_.agree_preagreed_operation(d, stage, deadline) :
                                   session_.agree_operation(d, deadline);
        if (f) { err = f.detail; return false; }
    }
    Failure f;
    {
        ProfileSpan timing(stats ? &stats->enqueue_ns : nullptr);
        if (d.kind == CollectiveKind::all_reduce_sum)
            f = transport_.all_reduce_sum({const_cast<float*>(in), d.elements, ScalarType::fp32},
                                         {out, d.elements, ScalarType::fp32}, d.operation,
                                         static_cast<cudaStream_t>(stream));
        else
            f = transport_.all_gather_padded({const_cast<float*>(in), d.elements, ScalarType::fp32},
                                            {out, 3 * d.elements, ScalarType::fp32}, d.valid_counts,
                                            d.operation, static_cast<cudaStream_t>(stream));
    }
    if (!f) {
        ProfileSpan timing(stats ? &stats->wait_ns : nullptr);
        f = transport_.wait(deadline);
    }
    if (f) { err = f.detail; session_.abort(f.detail); return false; }
    if (const auto complete = session_.complete_operation(d.operation)) { err = complete.detail; return false; }
    ++sequence_;
    if (stats) ++stats->completed_collectives;
    return true;
}

bool SessionCollectives::reduce(int64_t layer, const char* stage, const float* in, float* out,
                                size_t n, void* stream, std::string& err) {
    CollectiveDescriptor d;
    d.operation = {sequence_, static_cast<uint32_t>(layer), step_};
    d.elements = n;
    if (!execute(d, stage, in, out, stream, err)) return false;
    observed_.push_back({d, stage});
    return true;
}

bool SessionCollectives::gather(int64_t layer, const char* stage, const float* in, float* out,
                                std::array<uint64_t, 3> valid, void* stream, std::string& err) {
    CollectiveDescriptor d;
    d.operation = {sequence_, static_cast<uint32_t>(layer), step_};
    d.kind = CollectiveKind::all_gather_padded;
    d.elements = *std::max_element(valid.begin(), valid.end());
    d.valid_counts.assign(valid.begin(), valid.end());
    if (!execute(d, stage, in, out, stream, err)) return false;
    observed_.push_back({d, stage});
    return true;
}

struct ModelForward::Impl {
    const core::WeightTable& weights;
    core::ModelGeometry geometry;
    core::SessionState& state;
    int rank;
    LayerCollectives& collectives;
    RoutedExperts& experts;
    DeviceMemory memory;
    DeviceMemory batch_memory;
    std::array<kernels::tp::cuda::GdnTpMap, 3> maps;
    float *qkv = nullptr, *conv = nullptr, *alpha = nullptr, *beta = nullptr;
    float *dt = nullptr, *a = nullptr, *z = nullptr, *gate = nullptr, *local_y = nullptr;
    float *conv_raw = nullptr, *norm_qkv = nullptr, *readout = nullptr;
    float *wire = nullptr, *global_y = nullptr, *embedding = nullptr, *parts = nullptr;
    float *batch_input = nullptr, *batch_qkv = nullptr, *batch_z = nullptr;
    float *batch_alpha = nullptr, *batch_beta = nullptr, *batch_projected = nullptr;
    void* batch_q8_1 = nullptr;
    std::array<core::BlockBuffers, 8> batch_blocks{};
    bool batch_projections_active = false;
    int batch_projection_row = 0;
    bool batch_enabled = false;
    float* states = nullptr;
    float* qsa_partial = nullptr;
    float* qsa_global = nullptr;
    const core::NativeHead* head = nullptr;
    ParityObserver observer;
    bool ready = false;
    bool failed = false;
    int64_t next_position = 0;
    int32_t checkpoint_position_base = 0;
    bool checkpoint_exit = false;
#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
    CheckpointIo checkpoint_io;
#endif

    Impl(const core::WeightTable& w, const core::ModelGeometry& g, core::SessionState& s,
         int r, LayerCollectives& c, RoutedExperts& e)
        : weights(w), geometry(g), state(s), rank(r), collectives(c), experts(e) {
        const int starts[4] = {0, 6, 11, 16};
        for (int i = 0; i < 3; ++i) { maps[i].key_begin = starts[i]; maps[i].key_end = starts[i + 1]; }
    }

    const core::WeightRef& weight(int64_t layer, const char* name) const {
        const auto* result = core::LayerView(weights, layer).get(name);
        if (!result) throw std::runtime_error("missing layer tensor " + std::string(name));
        return *result;
    }

    void project(const core::WeightRef& w, const float* x, float* y,
                 int row_begin, int rows, void* stream) {
        if (w.native_data && w.native_q8_1) {
            const auto stride = kernels::native_mmvq_weight_bytes(w.native_type, int(w.ne0), 1);
            kernels::native_quantize_q8_1(x, w.native_q8_1, int(w.ne0), 1, stream);
            kernels::native_mmvq(w.native_type, static_cast<const uint8_t*>(w.native_data) + row_begin * stride,
                                w.native_q8_1, y, int(w.ne0), rows, 1, stream);
        } else if (w.kind == core::WeightKind::Bf16InF32 && w.data) {
            kernels::bf16_gemv_fp32_mmvf(x, static_cast<const uint16_t*>(w.data) + row_begin * w.ne0,
                                        y, w.ne0, rows, stream);
        } else throw std::runtime_error("TP projection requires original native GGUF or BF16 weights");
    }

    void project_multi(const core::WeightRef& w, const float* x, void* q8_1, float* compact,
                       float* output, int output_stride, int output_offset, int row_begin,
                       int rows, int ncols, void* stream) {
        if (w.ne0 <= 0 || w.ne0 > INT_MAX || rows <= 0 || ncols < 2 || ncols > 8)
            throw std::runtime_error("invalid exact multirow projection geometry");
        if (w.native_data && q8_1) {
            const auto stride = kernels::native_mmvq_weight_bytes(w.native_type, int(w.ne0), 1);
            kernels::native_quantize_q8_1(x, q8_1, int(w.ne0), ncols, stream);
            kernels::native_mmvq(w.native_type,
                static_cast<const uint8_t*>(w.native_data) + size_t(row_begin) * stride,
                q8_1, compact, int(w.ne0), rows, ncols, stream);
        } else if (w.kind == core::WeightKind::Bf16InF32 && w.data) {
            kernels::bf16_gemv_fp32_mmvf_multi(x, w.ne0,
                static_cast<const uint16_t*>(w.data) + size_t(row_begin) * size_t(w.ne0),
                compact, rows, w.ne0, rows, ncols, stream);
        } else throw std::runtime_error("TP batch projection requires original native GGUF or BF16 weights");
        for (int row = 0; row < ncols; ++row)
            checked(cudaMemcpyAsync(output + size_t(row) * output_stride + output_offset,
                                    compact + size_t(row) * rows, size_t(rows) * sizeof(float),
                                    cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
    }

    void project_gdn_batch(int64_t layer, int ncols, void* stream) {
        using namespace kernels::tp::cuda;
        const auto& m = maps[rank];
        const int hk = m.key_heads(), hv = m.value_heads(), s = m.state_size;
        const int channels = m.channels();
        const auto& q = weight(layer, "attn_qkv.weight");
        const auto& zw = weight(layer, "attn_gate.weight");
        const auto& aw = weight(layer, "ssm_alpha.weight");
        const auto& bw = weight(layer, "ssm_beta.weight");
        float* temp = batch_projected;
        auto projection = [&](const core::WeightRef& w, int first, int count, float* destination,
                              int stride, int offset) {
            project_multi(w, batch_input, batch_q8_1, temp, destination, stride, offset,
                          first, count, ncols, stream);
        };
        projection(q, m.key_begin * s, hk * s, batch_qkv, channels, 0);
        projection(q, (16 + m.key_begin) * s, hk * s, batch_qkv, channels, hk * s);
        for (int run = 0; run < 3; ++run) {
            const int first = run * 16 + m.key_begin;
            projection(q, (32 + first) * s, hk * s, batch_qkv, channels, (2 * hk + run * hk) * s);
            projection(zw, first * s, hk * s, batch_z, hv * s, run * hk * s);
            projection(aw, first, hk, batch_alpha, 3 * hk, run * hk);
            projection(bw, first, hk, batch_beta, 3 * hk, run * hk);
        }
    }

    bool gdn_projected(int64_t layer, const float* x, float* output, const float* qkv_input,
                       const float* z_input, const float* alpha_input, const float* beta_input,
                       void* stream, std::string& err) {
        using namespace kernels::tp::cuda;
        const auto& m = maps[rank];
        const int hk = m.key_heads(), hv = m.value_heads(), s = m.state_size;
        tp_gdn_compact(m, GdnTpLayout::Qkv, 1, 4,
                       {static_cast<const float*>(weight(layer, "ssm_conv1d.weight").data), size_t(10240 * 4)},
                       {conv, size_t(m.channels() * 4)}, stream);
        tp_gdn_compact(m, GdnTpLayout::Values, 1, 1,
                       {static_cast<const float*>(weight(layer, "ssm_dt.bias").data), 48}, {dt, size_t(hv)}, stream);
        tp_gdn_compact(m, GdnTpLayout::Values, 1, 1,
                       {static_cast<const float*>(weight(layer, "ssm_a").data), 48}, {a, size_t(hv)}, stream);
        float* beta_values = const_cast<float*>(beta_input);
        if (beta_values != beta) {
            checked(cudaMemcpyAsync(beta, beta_values, size_t(hv) * sizeof(float), cudaMemcpyDeviceToDevice,
                                    static_cast<cudaStream_t>(stream)));
            beta_values = beta;
        }
        kernels::native_gdn_beta_gate(beta_values, hv, stream);
        kernels::native_gdn_gate(alpha_input, dt, a, gate, hv, stream);
        const size_t one_state = m.state_elements() + m.history_elements();
        const int64_t ordinal = layer - layer / geometry.qsa_interval;
        GdnTpState st{{states + ordinal * one_state, m.state_elements()},
                      {states + ordinal * one_state + m.state_elements(), m.history_elements()}};
        tp_gdn_token(m, GdnTpArithmetic::Native, st,
                     {qkv_input, size_t(m.channels())}, {conv, size_t(m.channels() * 4)},
                     {gate, size_t(hv)}, {beta_values, size_t(hv)}, {z_input, size_t(hv * s)},
                     {static_cast<const float*>(weight(layer, "ssm_norm.weight").data), size_t(s)},
                     float(geometry.rms_eps), {{conv_raw, size_t(m.channels())},
                     {norm_qkv, size_t(m.channels())}, {readout, size_t(hv * s)}},
                     {local_y, size_t(hv * s)}, stream);
        // Equal wire slots are zero padded; the quantizer sees reconstructed
        // logical head order, including the three interleaved value ranges.
        checked(cudaMemsetAsync(local_y + hv * s, 0, size_t((18 - hv) * s) * sizeof(float),
                                static_cast<cudaStream_t>(stream)));
        if (!collectives.gather(layer, "gdn_values", local_y, wire,
                                {18 * 128, 15 * 128, 15 * 128}, stream, err)) return false;
        GdnTpMap all_maps[3] = {maps[0], maps[1], maps[2]};
        tp_gdn_scatter_gathered(all_maps, 1, 18 * 128, {wire, 3 * 18 * 128}, {global_y, 6144}, stream);
        if (rank == 0) project(weight(layer, "ssm_out.weight"), global_y, output, 0, int(geometry.n_embd), stream);
        else checked(cudaMemsetAsync(output, 0, size_t(geometry.n_embd) * sizeof(float),
                                     static_cast<cudaStream_t>(stream)));
        return collectives.reduce(layer, "mixer", output, output, size_t(geometry.n_embd), stream, err);
    }

    bool gdn(int64_t layer, const float* x, float* output, void* stream, std::string& err) {
        const auto& m = maps[rank];
        const int hk = m.key_heads(), hv = m.value_heads(), s = m.state_size;
        const auto& q = weight(layer, "attn_qkv.weight");
        const auto& zw = weight(layer, "attn_gate.weight");
        const auto& aw = weight(layer, "ssm_alpha.weight");
        const auto& bw = weight(layer, "ssm_beta.weight");
        project(q, x, qkv, m.key_begin * s, hk * s, stream);
        project(q, x, qkv + hk * s, (16 + m.key_begin) * s, hk * s, stream);
        for (int run = 0; run < 3; ++run) {
            const int first = run * 16 + m.key_begin;
            project(q, x, qkv + (2 * hk + run * hk) * s, (32 + first) * s, hk * s, stream);
            project(zw, x, z + run * hk * s, first * s, hk * s, stream);
            project(aw, x, alpha + run * hk, first, hk, stream);
            project(bw, x, beta + run * hk, first, hk, stream);
        }
        return gdn_projected(layer, x, output, qkv, z, alpha, beta, stream, err);
    }
};

ModelForward::ModelForward(const core::WeightTable& w, const core::ModelGeometry& g,
                           core::SessionState& s, int rank, LayerCollectives& c, RoutedExperts& e)
    : impl_(std::make_unique<Impl>(w, g, s, rank, c, e)) {}
ModelForward::~ModelForward() = default;
const core::ModelGeometry& ModelForward::geometry() const { return impl_->geometry; }
int64_t ModelForward::next_position() const { return impl_->next_position; }
std::vector<GdnStateSource> ModelForward::state_sources() const {
    const auto& p = *impl_;
    if (!p.ready || p.failed) throw std::runtime_error("GDN evidence requires a live initialized model");
    const auto& map = p.maps[p.rank];
    const auto stride = map.state_elements() + map.history_elements();
    std::vector<uint64_t> heads, channels;
    for (int run = 0; run < 3; ++run)
        for (int key = map.key_begin; key < map.key_end; ++key)
            heads.push_back(uint64_t(run * map.global_key_heads + key));
    for (int run = 0; run < 2; ++run)
        for (int key = map.key_begin; key < map.key_end; ++key)
            for (int lane = 0; lane < map.state_size; ++lane)
                channels.push_back(uint64_t((run * map.global_key_heads + key) * map.state_size + lane));
    for (const auto head : heads)
        for (int lane = 0; lane < map.state_size; ++lane)
            channels.push_back(uint64_t(2 * map.global_key_heads * map.state_size) + head * map.state_size + lane);
    std::vector<GdnStateSource> result;
    size_t ordinal = 0;
    for (int64_t layer = 0; layer < p.geometry.n_layers; ++layer) {
        if (core::is_qsa_layer(p.geometry, layer)) continue;
        auto* recurrent = p.states + ordinal++ * stride;
        result.push_back({layer, recurrent, recurrent + map.state_elements(), heads, channels});
    }
    return result;
}
void ModelForward::invalidate() noexcept { impl_->failed = true; }
bool ModelForward::requires_process_exit() const noexcept { return impl_->checkpoint_exit; }

bool ModelForward::capture_checkpoint(ModelCheckpoint& checkpoint, ModelCheckpointIdentity id,
                                      uint64_t max_bytes, void* stream, CheckpointDeadline deadline,
                                      std::string& err) {
    checkpoint.clear();
    auto& p = *impl_;
    if (!p.ready || p.failed || !stream || id.rank != uint32_t(p.rank) ||
        id.max_context != uint64_t(p.state.max_cells) ||
        (p.next_position && id.position_base != p.checkpoint_position_base) ||
        std::chrono::steady_clock::now() >= deadline) {
        p.failed = true; err = "checkpoint capture requires live model, matching rank/capacity and deadline"; return false;
    }
#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
    try {
        if (!p.checkpoint_io.prepare(err)) { p.failed = true; return false; }
        const detail::ModelStateCopy copy = [&](void* host, void* device, size_t bytes, bool to_device, std::string& error) {
            return p.checkpoint_io.copy(host, device, bytes, to_device, stream, deadline, error);
        };
        detail::ModelStateArrays arrays;
        if (detail::describe_checkpoint_state(p.state, p.geometry, state_sources(), uint64_t(p.next_position), copy, arrays, err) &&
            checkpoint.capture({std::move(id), uint64_t(p.next_position), max_bytes,
                                state_evidence_inventory(arrays.sources)}, arrays.sources, deadline, err)) return true;
    } catch (const std::exception& ex) { err = std::string("model checkpoint capture: ") + ex.what(); }
      catch (...) { err = "model checkpoint capture failed"; }
    p.failed = true;
    p.checkpoint_exit = p.checkpoint_io.transfer.pending;
    if (p.checkpoint_exit) p.memory.data = nullptr;
    return false;
#else
    (void)max_bytes;
    p.failed = true; err = "model checkpoint CUDA support is not compiled"; return false;
#endif
}

bool ModelForward::restore_checkpoint(const ModelCheckpoint& checkpoint, ModelCheckpointIdentity id,
                                      void* stream, CheckpointDeadline deadline, std::string& err) {
    auto& p = *impl_;
    if (!p.ready || p.failed || !stream || id.rank != uint32_t(p.rank) ||
        id.max_context != uint64_t(p.state.max_cells) || std::chrono::steady_clock::now() >= deadline) {
        p.failed = true; err = "checkpoint restore requires live model, matching rank/capacity and deadline"; return false;
    }
#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
    try {
        if (!checkpoint.complete()) { p.failed = true; err = "cannot restore incomplete model checkpoint"; return false; }
        if (!p.checkpoint_io.prepare(err)) { p.failed = true; return false; }
        const detail::ModelStateCopy copy = [&](void* host, void* device, size_t bytes, bool to_device, std::string& error) {
            return p.checkpoint_io.copy(host, device, bytes, to_device, stream, deadline, error);
        };
        detail::ModelStateArrays arrays;
        if (detail::describe_checkpoint_state(p.state, p.geometry, state_sources(), checkpoint.next_position(), copy, arrays, err)) {
            const auto base = id.position_base;
            ModelCheckpointDestination destination{std::move(id), state_evidence_inventory(arrays.sources),
                std::move(arrays.destinations),
                [&, base](uint64_t position, std::string&) {
                    p.next_position = int64_t(position); p.checkpoint_position_base = base; return true;
                },
                [&] { p.failed = true; }};
            if (checkpoint.restore(destination, deadline, err)) return true;
        }
    } catch (const std::exception& ex) { err = std::string("model checkpoint restore: ") + ex.what(); }
      catch (...) { err = "model checkpoint restore failed"; }
    p.failed = true;
    p.checkpoint_exit = p.checkpoint_io.transfer.pending;
    if (p.checkpoint_exit) p.memory.data = nullptr;
    return false;
#else
    (void)checkpoint;
    p.failed = true; err = "model checkpoint CUDA support is not compiled"; return false;
#endif
}
void ModelForward::native_head(const core::NativeHead* head) { impl_->head = head; }
void ModelForward::parity_observer(ParityObserver observer) { impl_->observer = std::move(observer); }

bool ModelForward::initialize(std::string& err) {
    auto& p = *impl_;
    if (p.failed || p.ready) { err = "TP model is failed or already initialized"; return false; }
    if (p.rank < 0 || p.rank >= 3 || p.geometry.n_layers != 48 || p.geometry.n_embd != 2560 ||
        p.geometry.ssm_k_heads != 16 || p.geometry.ssm_v_heads != 48 || p.geometry.ssm_state_size != 128 ||
        p.geometry.ssm_d_conv != 4 || p.geometry.ssm_conv_channels != 10240 || p.geometry.ssm_value_dim != 6144 ||
        p.geometry.qsa_interval != 4 || p.geometry.n_head != 24 || p.geometry.n_head_kv != 2 ||
        p.geometry.head_dim != 256 || p.geometry.n_ff != 640 || p.geometry.n_expert_used != 10 ||
        p.geometry.hc != 4 || p.geometry.hc_lr != 320 || p.state.k != 10 || p.state.layer_lo != 0 ||
        p.state.layer_hi != 48 || !p.state.qsa_states || !p.state.R || !p.state.ple.ready()) {
        err = "TP3 C1 requires the full 48-layer Qwen geometry, full session and initialized PLE";
        return false;
    }
    try {
        const auto& m = p.maps[p.rank];
        const size_t recurrent = (m.state_elements() + m.history_elements()) * 36;
        const size_t scratch = 10240 * 12 + 6144 * 5 + 2560 * 2 + size_t(p.state.k) * 2560;
        p.memory.allocate((recurrent + scratch) * sizeof(float));
        float* next = static_cast<float*>(p.memory.data);
        auto take = [&next](size_t n) { auto* result = next; next += n; return result; };
        p.states = take(recurrent);
        p.qkv = take(10240); p.conv = take(10240 * 4);
        p.alpha = take(48); p.beta = take(48); p.dt = take(48); p.a = take(48); p.gate = take(48);
        p.z = take(6144); p.local_y = take(18 * 128); p.conv_raw = take(10240);
        p.norm_qkv = take(10240); p.readout = take(6144);
        p.wire = take(3 * 18 * 128); p.global_y = take(6144);
        p.qsa_partial = take(2560); p.qsa_global = take(6144);
        p.embedding = take(2560); p.parts = take(size_t(p.state.k) * 2560);
        if (p.batch_enabled) {
            const auto block_bytes = core::block_buffers_bytes(p.geometry);
            const size_t q8_bytes = kernels::native_q8_1_bytes(int(p.geometry.n_embd), 8);
            const size_t batch_bytes = size_t(8) * block_bytes +
                size_t(8 * p.geometry.n_embd) * sizeof(float) +
                size_t(8 * m.channels()) * sizeof(float) +
                size_t(8 * m.value_heads() * m.state_size) * sizeof(float) +
                size_t(8 * 3 * m.key_heads()) * sizeof(float) * 2 +
                size_t(8 * 2304) * sizeof(float) + q8_bytes + 256;
            p.batch_memory.allocate(batch_bytes);
            uint8_t* batch_next = static_cast<uint8_t*>(p.batch_memory.data);
            for (auto& block : p.batch_blocks) {
                const auto used = core::block_buffers_init(p.geometry, batch_next, block);
                batch_next += (used + 15) & ~size_t(15);
            }
            auto take_batch = [&batch_next](size_t count) {
                auto* result = reinterpret_cast<float*>(batch_next);
                batch_next += count * sizeof(float);
                return result;
            };
            p.batch_input = take_batch(size_t(8) * size_t(p.geometry.n_embd));
            p.batch_qkv = take_batch(size_t(8) * size_t(m.channels()));
            p.batch_z = take_batch(size_t(8) * size_t(m.value_heads() * m.state_size));
            p.batch_alpha = take_batch(size_t(8) * size_t(3 * m.key_heads()));
            p.batch_beta = take_batch(size_t(8) * size_t(3 * m.key_heads()));
            p.batch_projected = take_batch(size_t(8 * 2304));
            batch_next = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(batch_next) + 15) & ~uintptr_t(15));
            p.batch_q8_1 = batch_next;
        }
        p.ready = true;
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

bool ModelForward::configure_prefill_batch_exactnative(bool enabled, std::string& err) {
    auto& p = *impl_;
    if (p.ready || p.failed) {
        err = "prefill batch mode must be configured before model initialization";
        return false;
    }
    p.batch_enabled = enabled;
    return true;
}

bool ModelForward::reset(void* stream, std::string& err) {
    auto& p = *impl_;
    if (!p.ready || p.failed || !stream) { err = "TP model is not ready for reset"; return false; }
    try {
        const auto& m = p.maps[p.rank];
        checked(cudaMemsetAsync(p.states, 0, 36 * (m.state_elements() + m.history_elements()) * sizeof(float),
                                static_cast<cudaStream_t>(stream)));
        checked(cudaMemsetAsync(p.embedding, 0, size_t(p.geometry.n_embd) * sizeof(float),
                                static_cast<cudaStream_t>(stream)));
        core::session_zero(p.state, p.geometry, p.embedding, stream);
        for (int64_t i = 0; i < p.state.qsa_alloc; ++i) {
            auto& q = p.state.qsa_states[p.state.qsa_ord0 + i];
            checked(cudaMemsetAsync(q.step, 0, kernels::qsa_step_bytes(), static_cast<cudaStream_t>(stream)));
            checked(cudaMemsetAsync(q.pos_dev, 0, size_t(p.geometry.n_head) * sizeof(int32_t),
                                    static_cast<cudaStream_t>(stream)));
            checked(cudaMemsetAsync(q.idx_block_pos, 0, sizeof(int32_t), static_cast<cudaStream_t>(stream)));
            if (q.attention_status)
                checked(cudaMemsetAsync(q.attention_status, 0, sizeof(int32_t), static_cast<cudaStream_t>(stream)));
            std::fill_n(q.host_step, kernels::kStepCount + 1, int32_t(0));
            std::fill_n(q.host_pos, p.geometry.n_head, int32_t(0));
        }
        p.next_position = 0;
        p.checkpoint_position_base = 0;
        return true;
    } catch (const std::exception& e) { p.failed = true; err = e.what(); return false; }
}

bool ModelForward::mixer(int64_t layer, int64_t pos, int32_t base, const float* input,
                         float* output, void* stream, std::string& err) {
    auto& p = *impl_;
    if (p.observer && !p.observer({"mixer_input", layer, pos, input, size_t(p.geometry.n_embd)}, stream, err))
        return false;
    bool ok;
    if (core::is_qsa_layer(p.geometry, layer)) {
        ok = qsa_layer_tp(p.weights, p.geometry, layer, pos, base,
                            p.state.qsa_states[layer / 4], p.state.qsa_bufs,
                            input, output, {p.rank, QsaAttentionPolicy::GatherReference, QsaOutputPolicy::RankZeroReference},
                            {p.qsa_global, 6144, p.qsa_partial, 2560}, p.collectives, stream, err);
    } else if (p.batch_projections_active) {
        const size_t row = size_t(p.batch_projection_row);
        const auto& map = p.maps[p.rank];
        ok = p.gdn_projected(layer, input, output,
                             p.batch_qkv + row * size_t(map.channels()),
                             p.batch_z + row * size_t(map.value_heads() * map.state_size),
                             p.batch_alpha + row * size_t(3 * map.key_heads()),
                             p.batch_beta + row * size_t(3 * map.key_heads()),
                             stream, err);
    } else ok = p.gdn(layer, input, output, stream, err);
    return ok && (!p.observer || p.observer({"mixer_output", layer, pos, output, size_t(p.geometry.n_embd)}, stream, err));
}

bool ModelForward::token(int32_t token, int64_t pos, int32_t base,
                         float* logits, void* stream, std::string& err) {
    auto& p = *impl_;
    if (!p.ready || p.failed || !stream || !logits || pos != p.next_position ||
        token < 0 || token >= p.geometry.n_vocab || pos >= p.state.max_cells) {
        err = "TP token requires ready C1 state, consecutive position and valid token/output/stream";
        return false;
    }
    try {
        for (int64_t j = 0; j < p.geometry.n_qsa_layers(); ++j) {
            auto& q = p.state.qsa_states[j];
            kernels::qsa_step_fill(q.host_step, pos, core::qsa_shapes(p.geometry));
            for (int64_t h = 0; h < p.geometry.n_head; ++h) q.host_pos[h] = int32_t(base + pos);
        }
        p.state.ple_token = token;
        if (!core::ple_stage_token(p.state.ple, stream, err) ||
            !core::embed_row(p.weights, p.geometry, token, p.embedding, stream, err)) {
            p.failed = true; return false;
        }
        for (int64_t c = 0; c < p.geometry.hc; ++c)
            checked(cudaMemcpyAsync(p.state.R + c * p.geometry.n_embd, p.embedding,
                                    size_t(p.geometry.n_embd) * sizeof(float), cudaMemcpyDeviceToDevice,
                                    static_cast<cudaStream_t>(stream)));
        for (int64_t layer = 0; layer < p.geometry.n_layers; ++layer) {
            const auto& qs = p.state.qsa_states[layer / p.geometry.qsa_interval];
            if ((p.observer && !p.observer({"residual_pre", layer, pos, p.state.R,
                                            size_t(p.geometry.hc * p.geometry.n_embd)}, stream, err)) ||
                !core::block_layer_pre(p.weights, p.geometry, layer, pos, base, p.state.gdn,
                                       qs, p.state.qsa_bufs, p.state.moe, p.state.k, p.state.block,
                                       stream, err, nullptr, &p.state.ple, 0, 0, this) ||
                (p.observer && !p.observer({"expert_input", layer, pos, p.state.block.mixed,
                                            size_t(p.geometry.n_embd)}, stream, err)) ||
                !p.collectives.agree_routes(layer, p.state.moe.ids, p.state.k, stream, err) ||
                !p.experts.forward(layer, p.state.block.mixed, p.state.moe.ids,
                                   p.state.k, p.parts, stream, err) ||
                (p.observer && !p.observer({"expert_output", layer, pos, p.parts,
                                            size_t(p.state.k * p.geometry.n_embd)}, stream, err)) ||
                !core::block_layer_post(p.weights, p.geometry, layer, p.state.k, p.state.moe,
                                        p.state.block, p.parts, stream, err, this) ||
                (p.observer && !p.observer({"residual_post", layer, pos, p.state.R,
                                            size_t(p.geometry.hc * p.geometry.n_embd)}, stream, err))) {
                p.failed = true; err = "layer " + std::to_string(layer) + ": " + err; return false;
            }
        }
        if (p.head ? (!core::lm_head_mix(p.weights, p.geometry, p.state.block, stream, err) ||
                       !p.head->run(p.state.block.mixed, logits, stream, err))
                   : !core::lm_head(p.weights, p.geometry, p.state.block, logits, stream, err)) {
            p.failed = true; return false;
        }
        // The worker polls completion with its deadline and validates logits
        // before commit. A CUDA stream synchronization here could wait forever
        // after a peer failure and bypass that bounded completion path.
        p.state.ple_prev[0] = p.state.ple_prev[1]; p.state.ple_prev[1] = token;
        ++p.next_position;
        p.checkpoint_position_base = base;
        return true;
    } catch (const std::exception& e) { p.failed = true; err = e.what(); return false; }
}

bool ModelForward::tokens(const std::vector<int32_t>& tokens, int64_t pos, int32_t base,
                          float* logits, void* stream, std::string& err) {
    auto& p = *impl_;
    if (!p.ready || p.failed || !p.batch_enabled || !stream || !logits ||
        !kernels::native_mmvq_multi_exact() || tokens.size() < 2 || tokens.size() > 8 ||
        pos != p.next_position || pos < 0 || pos + int64_t(tokens.size()) > p.state.max_cells ||
        base < 0 || base % 4 || int64_t(base) + pos + int64_t(tokens.size()) > p.state.max_cells) {
        err = "TP exact-native prefill requires enabled model, exact MMVQ, 2..8 rows and valid context";
        return false;
    }
    for (size_t row = 0; row < tokens.size(); ++row) {
        if (tokens[row] < 0 || tokens[row] >= p.geometry.n_vocab) {
            err = "TP exact-native prefill contains an invalid token";
            return false;
        }
    }
    try {
        const size_t rows = tokens.size();
        const size_t residual_floats = size_t(p.geometry.hc * p.geometry.n_embd);
        for (size_t row = 0; row < rows; ++row) {
            auto& block = p.batch_blocks[row];
            if (!core::embed_row(p.weights, p.geometry, tokens[row], p.embedding, stream, err)) {
                p.failed = true;
                return false;
            }
            for (int64_t c = 0; c < p.geometry.hc; ++c)
                checked(cudaMemcpyAsync(block.R + c * p.geometry.n_embd, p.embedding,
                                        size_t(p.geometry.n_embd) * sizeof(float), cudaMemcpyDeviceToDevice,
                                        static_cast<cudaStream_t>(stream)));
        }

        for (int64_t layer = 0; layer < p.geometry.n_layers; ++layer) {
            const bool qsa = core::is_qsa_layer(p.geometry, layer);
            for (size_t row = 0; row < rows; ++row) {
                auto& block = p.batch_blocks[row];
                const int64_t row_pos = pos + int64_t(row);
                if (p.observer && !p.observer({"residual_pre", layer, row_pos, block.R,
                                               residual_floats}, stream, err)) {
                    p.failed = true;
                    err = "layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                if (layer == 1) {
                    p.state.ple_token = tokens[row];
                    if (!core::ple_stage_token(p.state.ple, stream, err)) {
                        p.failed = true;
                        err = "layer 1: " + err;
                        return false;
                    }
                }
                const auto& qs = p.state.qsa_states[layer / p.geometry.qsa_interval];
                if (!core::block_layer_prefill_stage(
                        p.weights, p.geometry, layer, row_pos, base, p.state.gdn, qs,
                        p.state.qsa_bufs, p.state.moe, p.state.k, block, stream, err, this,
                        layer == 1 ? &p.state.ple : nullptr, core::DistributedPrefillStage::attention_read)) {
                    p.failed = true;
                    err = "layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                if (layer == 1) {
                    p.state.ple_prev[0] = p.state.ple_prev[1];
                    p.state.ple_prev[1] = tokens[row];
                }
            }

            if (!qsa) {
                for (size_t row = 0; row < rows; ++row)
                    checked(cudaMemcpyAsync(p.batch_input + row * size_t(p.geometry.n_embd),
                                            p.batch_blocks[row].mixed,
                                            size_t(p.geometry.n_embd) * sizeof(float),
                                            cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
                p.project_gdn_batch(layer, int(rows), stream);
            }

            for (size_t row = 0; row < rows; ++row) {
                auto& block = p.batch_blocks[row];
                const int64_t row_pos = pos + int64_t(row);
                p.batch_projection_row = int(row);
                p.batch_projections_active = !qsa;
                if (qsa) {
                    auto& q = p.state.qsa_states[layer / p.geometry.qsa_interval];
                    kernels::qsa_step_fill(q.host_step, row_pos, core::qsa_shapes(p.geometry));
                    for (int64_t h = 0; h < p.geometry.n_head; ++h)
                        q.host_pos[h] = int32_t(base + row_pos);
                }
                if (!core::block_layer_prefill_stage(
                        p.weights, p.geometry, layer, row_pos, base, p.state.gdn,
                        p.state.qsa_states[layer / p.geometry.qsa_interval], p.state.qsa_bufs,
                        p.state.moe, p.state.k, block, stream, err, this, nullptr,
                        core::DistributedPrefillStage::mixer) ||
                    !core::block_layer_prefill_stage(
                        p.weights, p.geometry, layer, row_pos, base, p.state.gdn,
                        p.state.qsa_states[layer / p.geometry.qsa_interval], p.state.qsa_bufs,
                        p.state.moe, p.state.k, block, stream, err, this, nullptr,
                        core::DistributedPrefillStage::attention_write) ||
                    !core::block_layer_prefill_stage(
                        p.weights, p.geometry, layer, row_pos, base, p.state.gdn,
                        p.state.qsa_states[layer / p.geometry.qsa_interval], p.state.qsa_bufs,
                        p.state.moe, p.state.k, block, stream, err, this, nullptr,
                        core::DistributedPrefillStage::ffn_read) ||
                    !core::block_layer_prefill_stage(
                        p.weights, p.geometry, layer, row_pos, base, p.state.gdn,
                        p.state.qsa_states[layer / p.geometry.qsa_interval], p.state.qsa_bufs,
                        p.state.moe, p.state.k, block, stream, err, this, nullptr,
                        core::DistributedPrefillStage::route) ||
                    (p.observer && !p.observer({"expert_input", layer, row_pos, block.mixed,
                                                size_t(p.geometry.n_embd)}, stream, err)) ||
                    !p.collectives.agree_routes(layer, p.state.moe.ids, p.state.k, stream, err) ||
                    !p.experts.forward(layer, block.mixed, p.state.moe.ids, p.state.k, p.parts,
                                       stream, err) ||
                    (p.observer && !p.observer({"expert_output", layer, row_pos, p.parts,
                                                size_t(p.state.k * p.geometry.n_embd)}, stream, err)) ||
                    !core::block_layer_post(p.weights, p.geometry, layer, p.state.k, p.state.moe,
                                            block, p.parts, stream, err, this) ||
                    (p.observer && !p.observer({"residual_post", layer, row_pos, block.R,
                                                residual_floats}, stream, err))) {
                    p.failed = true;
                    err = "layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                p.batch_projections_active = false;
            }
        }

        auto& final_block = p.batch_blocks[rows - 1];
        checked(cudaMemcpyAsync(p.state.R, final_block.R, residual_floats * sizeof(float),
                                cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
        if (p.head ? (!core::lm_head_mix(p.weights, p.geometry, final_block, stream, err) ||
                      !p.head->run(final_block.mixed, logits, stream, err))
                   : !core::lm_head(p.weights, p.geometry, final_block, logits, stream, err)) {
            p.failed = true;
            return false;
        }
        checked(cudaMemcpyAsync(p.state.block.mixed, final_block.mixed,
                                size_t(p.geometry.n_embd) * sizeof(float), cudaMemcpyDeviceToDevice,
                                static_cast<cudaStream_t>(stream)));
        checked(cudaMemcpyAsync(p.state.block.block_out, final_block.block_out,
                                size_t(p.geometry.n_embd) * sizeof(float), cudaMemcpyDeviceToDevice,
                                static_cast<cudaStream_t>(stream)));
        checked(cudaMemcpyAsync(p.state.block.inject, final_block.inject,
                                size_t(p.geometry.hc) * sizeof(float), cudaMemcpyDeviceToDevice,
                                static_cast<cudaStream_t>(stream)));
        checked(cudaMemcpyAsync(p.state.block.inject2, final_block.inject2,
                                size_t(p.geometry.hc) * sizeof(float), cudaMemcpyDeviceToDevice,
                                static_cast<cudaStream_t>(stream)));
        p.state.ple_token = tokens.back();
        p.state.ple_prev[0] = tokens.size() > 1 ? tokens[tokens.size() - 2] : p.state.ple_prev[1];
        p.state.ple_prev[1] = tokens.back();
        p.next_position += int64_t(rows);
        p.checkpoint_position_base = base;
        return true;
    } catch (const std::exception& e) {
        p.batch_projections_active = false;
        p.failed = true;
        err = std::string("TP exact-native prefill: ") + e.what();
        return false;
    }
}

} // namespace strata::distributed
