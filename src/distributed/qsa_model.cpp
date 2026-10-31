#include "strata/distributed/qsa_model.hpp"

#include "strata/distributed/model.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/tp_qsa.hpp"

#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <exception>
#include <limits>
#include <stdexcept>

namespace strata::distributed {
namespace {
constexpr int kLocalHeads = 8;
constexpr int kDimension = 256;
constexpr int kLocalChannels = kLocalHeads * kDimension;
constexpr int kGlobalChannels = 3 * kLocalChannels;
bool fail(std::string& err, const std::string& message) {
    err = "qsa_layer_tp: " + message;
    return false;
}
bool geometry(const core::ModelGeometry& g, std::string& err) {
    if (g.qsa_interval <= 0 || g.n_head != 24 || g.n_head_kv != 2 || g.head_dim != kDimension || g.n_rot != 64 ||
        g.idx_q_heads != 4 || g.idx_key_dim != 128 || g.idx_block != 4 ||
        g.n_embd <= 0 || g.n_embd > INT_MAX || g.n_embd % 256 || g.idx_top_k <= 0 || g.idx_top_k > 4093)
        return fail(err, "requires global Q24/KV2/dim256/rot64/indexer4x128/block4 geometry");
    return true;
}
bool native_matrix(const core::WeightRef* w, int64_t input, int64_t output,
                     const char* name, std::string& err) {
    if (!w || !w->native_data || !w->native_q8_1 || w->ne0 != input || w->ne1 != output)
        return fail(err, std::string(name) + " requires original NativeDense matrix and Q8_1 scratch");
    try {
        (void) kernels::native_mmvq_weight_bytes(w->native_type, int(input), int(output));
    } catch (const std::exception& e) {
        return fail(err, std::string(name) + ": " + e.what());
    }
    return true;
}
bool matrices(const core::WeightTable& t, const core::ModelGeometry& g, int64_t layer,
                std::string& err) {
    if (!geometry(g, err) || layer < 0 || layer >= g.n_layers || !core::is_qsa_layer(g, layer))
        return fail(err, "invalid QSA layer or geometry");
    const core::LayerView v(t, layer);
    if (!native_matrix(v.get("attn_q.weight"), g.n_embd, 2 * kGlobalChannels, "attn_q", err) ||
        !native_matrix(v.get("attn_k.weight"), g.n_embd, 512, "attn_k", err) ||
        !native_matrix(v.get("attn_v.weight"), g.n_embd, 512, "attn_v", err) ||
        !native_matrix(v.get("attn_output.weight"), kGlobalChannels, g.n_embd, "attn_output", err))
        return false;
    const std::array<std::pair<const char*, int64_t>, 2> projections{{
        {"indexer.k_proj.weight", 128}, {"indexer.q_proj.weight", 512}}};
    for (const auto& spec : projections) {
        const auto* w = v.get(spec.first);
        if (!w || !w->data || w->kind != core::WeightKind::Bf16InF32 ||
            w->ne0 != g.n_embd || w->ne1 != spec.second)
            return fail(err, std::string(spec.first) + " requires replicated original BF16 matrix");
    }
    const std::array<std::pair<const char*, int64_t>, 4> norms{{
        {"attn_q_norm.weight", 256}, {"attn_k_norm.weight", 256},
        {"indexer.q_norm.weight", 128}, {"indexer.k_norm.weight", 128}}};
    for (const auto& spec : norms) {
        const auto* w = v.get(spec.first);
        if (!w || !w->data || w->kind != core::WeightKind::F32 || w->elements != spec.second)
            return fail(err, std::string(spec.first) + " requires replicated original FP32 norm");
    }
    return true;
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
void checked(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

class Stages final : public core::QsaTpStages {
public:
    Stages(const core::ModelGeometry& g, int rank, int64_t layer,
           const QsaModelScratch& scratch, LayerCollectives& c)
        : g_(g), rank_(rank), layer_(layer), scratch_(scratch), collectives_(c) {
        shard_.global = core::qsa_shapes(g);
        shard_.local_heads = kLocalHeads;
        shard_.head_offset = rank * kLocalHeads;
    }
    int local_heads() const override { return kLocalHeads; }
    bool query(const core::WeightRef& w, const float* x, float* output,
                void* stream, std::string& err) override {
        try {
            kernels::native_quantize_q8_1(x, w.native_q8_1, int(g_.n_embd), 1, stream);
            const size_t stride = kernels::native_mmvq_weight_bytes(w.native_type, int(w.ne0), 1);
            const auto* first = static_cast<const uint8_t*>(w.native_data) +
                                size_t(rank_ * 2 * kLocalChannels) * stride;
            kernels::native_mmvq(w.native_type, first, w.native_q8_1, output,
                                  int(w.ne0), 2 * kLocalChannels, 1, stream);
            return true;
        } catch (const std::exception& e) { return fail(err, e.what()); }
    }
    bool attend(const core::QsaState& st, const core::QsaBuffers& b, int64_t cap,
                 void* stream, std::string& err) override {
        try {
            kernels::tp::cuda::qsa_shard_attend(b.qcur, core::qsa_attn_pools(st), b.ids,
                st.step, int(cap), shard_, b.attn, nullptr, 1, stream);
            return true;
        } catch (const std::exception& e) { return fail(err, e.what()); }
    }
    bool output(const core::WeightRef& w, const float* gated, float* output,
                 void* stream, std::string& err) override {
        // Uniform TP3 heads need no padding removal: gathered rank order is
        // exactly the original contiguous global activation before Q8_1.
        if (!collectives_.gather(layer_, "qsa_gated_reference", gated, scratch_.global_gated,
                                  {kLocalChannels, kLocalChannels, kLocalChannels}, stream, err)) return false;
        try {
            if (rank_ == 0) {
                kernels::native_quantize_q8_1(scratch_.global_gated, w.native_q8_1, kGlobalChannels, 1, stream);
                kernels::native_mmvq(w.native_type, w.native_data, w.native_q8_1, scratch_.partial,
                                      kGlobalChannels, int(g_.n_embd), 1, stream);
            } else {
                checked(cudaMemsetAsync(scratch_.partial, 0, size_t(g_.n_embd) * sizeof(float),
                                         static_cast<cudaStream_t>(stream)));
            }
        } catch (const std::exception& e) { return fail(err, e.what()); }
        return collectives_.reduce(layer_, "qsa_output_rank_zero_reference", scratch_.partial,
                                     output, size_t(g_.n_embd), stream, err);
    }
private:
    const core::ModelGeometry& g_;
    int rank_;
    int64_t layer_;
    QsaModelScratch scratch_;
    LayerCollectives& collectives_;
    kernels::tp::cuda::QsaShard shard_;
};
} // namespace

bool qsa_model_budget(const core::WeightTable& tables, const core::ModelGeometry& g,
                       int64_t layer, int64_t max_cells, QsaModelBudget& budget,
                       std::string& err) {
    if (!matrices(tables, g, layer, err)) return false;
    if (max_cells <= 0 || max_cells > INT_MAX) return fail(err, "invalid context capacity");
    QsaModelBudget result;
    const core::LayerView v(tables, layer);
    try {
        for (const char* name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"}) {
            const auto& w = *v.get(name);
            result.replicated_native_weight_bytes_lower_bound +=
                kernels::native_mmvq_weight_bytes(w.native_type, int(w.ne0), int(w.ne1));
        }
        for (const char* name : {"indexer.k_proj.weight", "indexer.q_proj.weight", "attn_q_norm.weight",
                                 "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"})
            result.replicated_canonical_weight_bytes_lower_bound += v.get(name)->bytes;
        result.extra_scratch_bytes = uint64_t(kGlobalChannels + g.n_embd) * sizeof(float);
        result.original_state_bytes = core::qsa_state_bytes(g, max_cells);
        result.original_buffer_bytes = core::qsa_buffers_bytes(g, max_cells);
    } catch (const std::exception& e) { return fail(err, e.what()); }
    budget = result;
    return true;
}

bool qsa_layer_tp(const core::WeightTable& tables, const core::ModelGeometry& g,
                   int64_t layer, int64_t pos, int32_t pos_base, const core::QsaState& st,
                   const core::QsaBuffers& b, const float* x, float* out, const QsaModelOptions& options,
                   const QsaModelScratch& scratch, LayerCollectives& collectives,
                   void* stream, std::string& err) {
    if (options.rank < 0 || options.rank > 2 ||
        options.attention != QsaAttentionPolicy::GatherReference ||
        options.output != QsaOutputPolicy::RankZeroReference)
        return fail(err, "requires explicit gather-reference attention and rank-zero-reference output policies");
    if (!matrices(tables, g, layer, err)) return false;
    if (!kernels::native_qsa_enabled() || !kernels::native_rope_enabled() ||
        !kernels::native_qsa_indexer_enabled())
        return fail(err, "native QSA, RoPE and indexer switches must be configured before the call");
    if (!x || !out || !stream || !scratch.global_gated || !scratch.partial ||
        scratch.global_gated_floats < kGlobalChannels || scratch.partial_floats < size_t(g.n_embd))
        return fail(err, "missing ordered stream, output or caller-owned reference scratch");
    if (reinterpret_cast<uintptr_t>(scratch.global_gated) % alignof(float) ||
        reinterpret_cast<uintptr_t>(scratch.partial) % alignof(float) ||
        overlaps(scratch.global_gated, kGlobalChannels * sizeof(float), scratch.partial, size_t(g.n_embd) * sizeof(float)) ||
        overlaps(out, size_t(g.n_embd) * sizeof(float), scratch.partial, size_t(g.n_embd) * sizeof(float)) ||
        overlaps(b.attn32, kLocalChannels * sizeof(float), scratch.global_gated, kGlobalChannels * sizeof(float)))
        return fail(err, "reference scratch must be aligned and disjoint from local gated values/output");
    if (st.max_cells <= 0 || st.max_cells > INT_MAX || pos < 0 || pos >= st.max_cells ||
        pos_base < 0 || pos_base % 4 || int64_t(pos_base) + st.max_cells > INT_MAX ||
        int64_t(pos_base) + pos >= st.max_cells || st.kv_mode == 2 ||
        st.idx_pooled_rows < st.max_cells / g.idx_block + 1 || !st.step || !st.pos_dev ||
        !st.host_step || !st.host_pos || !st.page_table || !st.idx_tail || !st.idx_dead ||
        !st.idx_pooled || !st.idx_block_pos || !b.q_full || !b.qcur || !b.kcur || !b.vcur ||
        !b.idx_raw || !b.q_idx || !b.cell_scores || !b.ids || !b.attn || !b.attn32)
        return fail(err, "incomplete original QSA state/buffers, unsupported ring, or position outside capacity");
    Stages stages(g, options.rank, layer, scratch, collectives);
    return core::qsa_layer_staged(tables, g, layer, pos, pos_base, st, b, x, out, stages, stream, err);
}
} // namespace strata::distributed
