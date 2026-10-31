#pragma once

#include "strata/core/layer.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::distributed {
class LayerCollectives;

// Explicit baseline: attention uses the gather-reference FP16 rounding even
// with compressed pools. It never silently selects fast FP32 dequantization.
enum class QsaAttentionPolicy { GatherReference };
// Temporary full output projection on rank 0, after gathering the LOCAL gated
// heads. Query projection and attention are sharded; output projection is not.
enum class QsaOutputPolicy { RankZeroReference };

struct QsaModelOptions {
    int rank;
    QsaAttentionPolicy attention;
    QsaOutputPolicy output;
};

struct QsaModelScratch {
    float* global_gated = nullptr;  // [24*256], contiguous rank0/rank1/rank2
    size_t global_gated_floats = 0;
    float* partial = nullptr;       // [n_embd], disjoint from final output
    size_t partial_floats = 0;
};

struct QsaModelBudget {
    uint64_t extra_scratch_bytes = 0;
    uint64_t replicated_native_weight_bytes_lower_bound = 0;
    uint64_t replicated_canonical_weight_bytes_lower_bound = 0;
    uint64_t original_state_bytes = 0;
    uint64_t original_buffer_bytes = 0;
};

// No allocation or device access. Replicated weights are intentional for this
// adapter: NativeDense stays alive, with original ne0=input/ne1=output.
// The lower bound excludes allocator granules, shared Q8_1 scratch and weights
// outside this layer. State sizes reflect the configured original KV policy.
bool qsa_model_budget(const core::WeightTable&, const core::ModelGeometry&,
                       int64_t layer, int64_t max_cells, QsaModelBudget&,
                       std::string& err);

// One sequential decode/reference-prefill token, using caller-owned original
// state and full QsaBuffers capacity (only eight query heads are written).
// Requires native QSA norm/gate, native RoPE, native indexer, and native BF16
// projections configured before session capture; this call changes no flags.
// All four cache formats, original tail pooling/top-k and streaming resolution
// stay in qsa_layer. The adapter is uncaptured because it calls collectives.
// Rank offsets are 0,8,16; KV2 and all four indexer query heads are replicated.
// No projection of the other 16 query heads, no full attention compute/crop.
bool qsa_layer_tp(const core::WeightTable&, const core::ModelGeometry&,
                   int64_t layer, int64_t pos, int32_t pos_base,
                   const core::QsaState&, const core::QsaBuffers&,
                   const float* x, float* out, const QsaModelOptions&,
                   const QsaModelScratch&, LayerCollectives&,
                   void* stream, std::string& err);

} // namespace strata::distributed
