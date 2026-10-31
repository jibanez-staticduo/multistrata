#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <array>
#include <cstdint>

namespace strata::kernels::tp::cuda {

// Keep global geometry here: changing n_head to the local count changes GQA.
// q/k/v and indexer projections are supplied by the caller. KV heads and all
// four indexer heads remain replicated, including their original cache state.
struct QsaShard {
    QsaShapes global;
    int local_heads = 0;
    int head_offset = 0;
    // When explicit_heads is true, query_heads[h] replaces head_offset + h.
    bool explicit_heads = false;
    std::array<int32_t, 32> query_heads{};
};

void validate_qsa_shard(const QsaShard& shard);
int qsa_shard_kv_head(const QsaShard& shard, int local_head);

// All pointers are device pointers. q/attn [rows, local_heads, head_dim],
// ids [rows, cap], steps [rows, kStepCount]. Optional weights have fixed stride
// [rows, local_heads, cap]; only the first step[Width] entries are written.
// Reads replicated pools directly; never computes or crops other query heads.
// FP16, Q8, Q4 and K8V4 use the original gather's fp16 dequantization rounding.
// ids must name resident pages; causal cells beyond step[Pos] receive zero
// weight. Invalid device widths or missing resident pages produce NaN output.
// cap is 1..4096. No allocation/synchronization; nonnull stream required.
// Batch is a per-query reference prefill path, not a tensor-core GEMM path.
void qsa_shard_attend(const float* q, const QsaAttnPools& pools,
                      const int32_t* ids, const int32_t* steps, int cap,
                      const QsaShard& shard, float* attn, float* weights,
                      int rows, void* stream);

// Local [q|gate] heads, q_norm broadcast per channel, positions [rows*heads].
// Uses the original native RMSNorm/RoPE arithmetic. A caller using rotated KV
// must also apply the original fwht256 to q before attention and to attn after.
void qsa_shard_prepare_query(const float* q_full, const float* q_norm, float* q,
                             const int32_t* positions, const QsaShard& shard,
                             int rows, const RopeScaling& rope, void* stream);
void qsa_shard_gate(const float* attn, const float* q_full, float* output,
                    const QsaShard& shard, int rows, void* stream);

// Original replicated indexer scores and exact top-k. Call with global shapes;
// buffers/state are caller-owned and may be snapshotted/reset by byte copies.
// No per-rank indexer-head split and no local-query-head geometry is used.
void qsa_shard_select(const float* pooled, const float* q_idx, const float* bias,
                      const int32_t* step, int64_t max_blocks, int64_t cap,
                      const QsaShard& shard, float* scores, int32_t* ids,
                      void* stream);

}  // namespace strata::kernels::tp::cuda
