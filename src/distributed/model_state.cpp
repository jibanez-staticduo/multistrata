#include "model_state.hpp"

#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>

namespace strata::distributed::detail {
namespace {
bool fail(std::string& err, const char* message) { err = message; return false; }
}

bool describe_checkpoint_state(core::SessionState& state, const core::ModelGeometry& g,
                               const std::vector<GdnStateSource>& gdn, uint64_t tokens,
                               const ModelStateCopy& copy, ModelStateArrays& output,
                               std::string& err) {
    output = {};
    if (!copy || state.max_cells <= 0 || state.max_cells > std::numeric_limits<int32_t>::max() ||
        tokens > uint64_t(state.max_cells) ||
        state.layer_lo != 0 || state.layer_hi != g.n_layers || g.n_layers <= 0 ||
        g.qsa_interval <= 0 || g.ssm_state_size <= 0 || g.ssm_d_conv <= 1 ||
        g.ssm_v_heads <= 0 || g.ssm_conv_channels <= 0 || g.n_head <= 0 || g.hc <= 0 ||
        g.n_embd <= 0 || g.hc > kernels::NG_HC_DIM ||
        g.n_embd != kernels::NG_HC_DIM / g.hc ||
        g.hc * g.n_embd != kernels::NG_HC_DIM || !state.R || !state.ple_hist ||
        state.qsa_ord0 != 0 || state.qsa_alloc != g.n_qsa_layers() || !state.qsa_states)
        return fail(err, "unsupported or incomplete full checkpoint session");
    ModelStateArrays result;
    auto add = [&](StateEvidenceDescriptor d, void* ptr, bool host = false) {
        result.sources.push_back({d, [d, ptr, host, copy](void* dst, std::size_t n, std::string& error) {
            uint64_t expected = 0;
            if (!state_evidence_bytes(d, expected, error) || n != expected || (n && (!ptr || !dst)))
                return fail(error, "invalid checkpoint state reader");
            if (!n) return true;
            if (host) { std::memcpy(dst, ptr, n); return true; }
            return copy(dst, ptr, n, false, error);
        }});
        result.destinations.push_back({d, [d, ptr, host, copy](const void* src, std::size_t n, std::string& error) {
            uint64_t expected = 0;
            if (!state_evidence_bytes(d, expected, error) || n != expected || (n && (!ptr || !src)))
                return fail(error, "invalid checkpoint state writer");
            if (!n) return true;
            if (host) { std::memcpy(ptr, src, n); return true; }
            return copy(const_cast<void*>(src), ptr, n, true, error);
        }});
    };
    auto plain = [&](const char* kind, const std::string& name, const char* dtype,
                     std::vector<uint64_t> shape, int64_t layer, void* ptr, bool host = false) {
        add({kind, name, dtype, std::move(shape), layer, {}, -1, {}}, ptr, host);
    };
    std::set<int64_t> used;
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        const std::string prefix = "layer." + std::to_string(layer) + ".";
        if (!core::is_qsa_layer(g, layer)) {
            const auto found = std::find_if(gdn.begin(), gdn.end(), [&](const auto& x) { return x.layer == layer; });
            if (found == gdn.end() || !found->recurrent || !found->convolution ||
                found->value_heads.empty() || found->channels.empty())
                return fail(err, "checkpoint requires every rank-local GDN state and ownership map");
            used.insert(layer);
            const auto S = uint64_t(g.ssm_state_size);
            add({"gdn", prefix + "recurrent", "<f4", {S, found->value_heads.size(), S}, layer,
                 {S, uint64_t(g.ssm_v_heads), S}, 1, found->value_heads}, const_cast<float*>(found->recurrent));
            add({"gdn", prefix + "convolution", "<f4", {found->channels.size(), uint64_t(g.ssm_d_conv - 1)}, layer,
                 {uint64_t(g.ssm_conv_channels), uint64_t(g.ssm_d_conv - 1)}, 0, found->channels},
                 const_cast<float*>(found->convolution));
            continue;
        }
        auto& q = state.qsa_states[layer / g.qsa_interval];
        const auto s = core::qsa_shapes(g);
        if (q.kv_mode != 0 || q.kv_int8 || q.kv_q4 || q.kv_hybrid || q.kv_rot || q.kv_elastic >= 0 ||
            q.max_cells != state.max_cells || q.n_pages <= 0 || q.n_pages != q.n_slots ||
            q.n_pages > std::numeric_limits<int32_t>::max() || s.page_size <= 0 ||
            s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_block <= 1 || s.idx_dim <= 0 ||
            q.n_pages != (state.max_cells + s.page_size - 1) / s.page_size ||
            q.idx_pooled_rows != state.max_cells / s.idx_block + 2 ||
            !q.page_table || !q.k_pool || !q.v_pool || !q.step || !q.pos_dev || !q.host_step ||
            !q.host_pos || !q.attention_status || !q.idx_tail || !q.idx_dead || !q.idx_pooled || !q.idx_block_pos)
            return fail(err, "checkpoint requires complete resident identity FP16 QSA/indexer state");
        std::vector<int32_t> table(static_cast<std::size_t>(q.n_pages));
        if (!copy(table.data(), q.page_table, table.size() * sizeof(int32_t), false, err)) return false;
        for (std::size_t i = 0; i < table.size(); ++i)
            if (table[i] != int32_t(i)) return fail(err, "checkpoint rejects nonidentity QSA page mapping");

        for (int kv = 0; kv < 2; ++kv) {
            void* pool = kv ? static_cast<void*>(q.v_pool) : q.k_pool;
            StateEvidenceDescriptor d{"kv", prefix + (kv ? "v" : "k"), "<f2",
                                      {tokens, uint64_t(s.n_head_kv), uint64_t(s.head_dim)}, layer, {}, -1, {}};
            // Read/write only loaded logical rows. At most one head's page
            // fragment is transferred at a time; unused physical capacity is absent.
            auto transfer = [d, pool, s, tokens, copy](void* host, std::size_t n, bool to_device, std::string& error) {
                uint64_t expected = 0;
                if (!state_evidence_bytes(d, expected, error) || n != expected || (n && !host))
                    return fail(error, "checkpoint KV byte count mismatch");
                const uint64_t row = uint64_t(s.head_dim) * 2;
                std::vector<uint8_t> fragment(static_cast<std::size_t>(uint64_t(s.page_size) * row));
                for (uint64_t page_start = 0; page_start < tokens; page_start += uint64_t(s.page_size)) {
                    const auto count = std::min(uint64_t(s.page_size), tokens - page_start);
                    for (uint64_t head = 0; head < uint64_t(s.n_head_kv); ++head) {
                        auto* dev = static_cast<uint8_t*>(pool) +
                            (page_start / uint64_t(s.page_size) * uint64_t(s.n_head_kv) + head) * uint64_t(s.page_size) * row;
                        if (!to_device && !copy(fragment.data(), dev, static_cast<std::size_t>(count * row), false, error)) return false;
                        for (uint64_t i = 0; i < count; ++i) {
                            auto* h = static_cast<uint8_t*>(host) + ((page_start + i) * uint64_t(s.n_head_kv) + head) * row;
                            if (to_device) std::memcpy(fragment.data() + i * row, h, static_cast<std::size_t>(row));
                            else std::memcpy(h, fragment.data() + i * row, static_cast<std::size_t>(row));
                        }
                        if (to_device && !copy(fragment.data(), dev, static_cast<std::size_t>(count * row), true, error)) return false;
                    }
                }
                return true;
            };
            result.sources.push_back({d, [transfer](void* dst, std::size_t n, std::string& error) { return transfer(dst, n, false, error); }});
            result.destinations.push_back({d, [transfer](const void* src, std::size_t n, std::string& error) {
                return transfer(const_cast<void*>(src), n, true, error);
            }});
        }
        // Page mapping is an immutable contract. Still store/check its exact
        // bytes so every captured logical array has explicit restore coverage.
        StateEvidenceDescriptor pd{"metadata", prefix + "page_table", "<i4", {uint64_t(q.n_pages)}, layer, {}, -1, {}};
        result.sources.push_back({pd, [table](void* dst, std::size_t n, std::string& error) {
            if (n != table.size() * sizeof(int32_t)) return fail(error, "checkpoint page table byte mismatch");
            std::memcpy(dst, table.data(), n); return true;
        }});
        result.destinations.push_back({pd, [table](const void* src, std::size_t n, std::string& error) {
            return (n == table.size() * sizeof(int32_t) && !std::memcmp(src, table.data(), n)) ||
                   fail(error, "checkpoint page mapping differs from resident identity");
        }});
        plain("metadata", prefix + "step", "<i4", {uint64_t(kernels::kStepCount)}, layer, q.step);
        plain("metadata", prefix + "position", "<i4", {uint64_t(g.n_head)}, layer, q.pos_dev);
        plain("metadata", prefix + "host_step", "<i4", {uint64_t(kernels::kStepCount + 1)}, layer, q.host_step, true);
        plain("metadata", prefix + "host_position", "<i4", {uint64_t(g.n_head)}, layer, q.host_pos, true);
        plain("metadata", prefix + "attention_status", "<i4", {1}, layer, q.attention_status);
        plain("indexer", prefix + "tail", "<f4", {uint64_t(s.idx_block - 1), uint64_t(s.idx_dim)}, layer, q.idx_tail);
        plain("indexer", prefix + "dead", "<f4", {uint64_t(s.idx_dim)}, layer, q.idx_dead);
        plain("indexer", prefix + "pooled", "<f4", {uint64_t(q.idx_pooled_rows), uint64_t(s.idx_dim)}, layer, q.idx_pooled);
        plain("metadata", prefix + "indexer_block_pos", "<i4", {1}, layer, q.idx_block_pos);
    }
    if (used.size() != gdn.size()) return fail(err, "duplicate or unknown checkpoint GDN state");
    plain("ple", "ple.history", "<f4", {uint64_t(kernels::NG_HC_DIM), uint64_t(kernels::NG_HIST)}, 1, state.ple_hist);
    StateEvidenceDescriptor window{"ple", "ple.token_window", "<i4", {3}, 1, {}, -1, {}};
    result.sources.push_back({window, [&state](void* dst, std::size_t n, std::string& error) {
        if (n != 3 * sizeof(int32_t)) return fail(error, "checkpoint PLE window byte mismatch");
        const int32_t values[3] = {state.ple_prev[0], state.ple_prev[1], state.ple_token};
        std::memcpy(dst, values, n); return true;
    }});
    result.destinations.push_back({window, [&state](const void* src, std::size_t n, std::string& error) {
        if (n != 3 * sizeof(int32_t)) return fail(error, "checkpoint PLE window byte mismatch");
        int32_t values[3]; std::memcpy(values, src, n);
        state.ple_prev[0] = values[0]; state.ple_prev[1] = values[1]; state.ple_token = values[2];
        return true;
    }});
    plain("residual", "residual.R", "<f4", {uint64_t(g.hc), uint64_t(g.n_embd)}, -1, state.R);
    for (const auto& a : result.sources) {
        uint64_t ignored = 0;
        if (!state_evidence_bytes(a.descriptor, ignored, err)) return false;
    }
    output = std::move(result);
    return true;
}
} // namespace strata::distributed::detail
