#include "strata/kernels/tp_qsa.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels::tp::cuda {
namespace {
void need(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(std::string("tp qsa: ") + message);
}
void launched() {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("tp qsa: ") + cudaGetErrorString(e));
}
struct Heads {
    int kv[32];
};
int pool_mode(const QsaAttnPools& p) {
    need(p.page_table != nullptr, "page table is null");
    if (p.k_q4) {
        need(p.v_q4 && !p.k_q && !p.k_pool && !p.v_pool && !p.v_q,
             "inconsistent Q4 pools");
        return 2;
    }
    if (p.k_q) {
        need(p.k_scale && !p.k_pool && !p.v_pool, "inconsistent Q8 key pools");
        if (p.v_q4) {
            need(!p.v_q, "ambiguous K8V4 value pools");
            return 3;
        }
        need(p.v_q && p.v_scale, "Q8 value pools are missing");
        return 1;
    }
    need(p.k_pool && p.v_pool && !p.v_q4 && !p.v_q, "FP16 pools are missing");
    return 0;
}

template<int Mode>
__device__ __forceinline__ float load(const QsaAttnPools& p, bool value, long long row, int d, int dim) {
    if constexpr (Mode == 0) return f32_from_f16((value ? p.v_pool : p.k_pool)[row * dim + d]);
    if constexpr (Mode == 2 || Mode == 3) {
        if (Mode == 2 || value) {
            const auto* blocks = reinterpret_cast<const block_q4_0*>(value ? p.v_q4 : p.k_q4);
            const block_q4_0& b = blocks[row * (dim / 32) + d / 32];
            const int i = d % 32;
            const int code = ((i < 16 ? b.qs[i] & 15 : b.qs[i - 16] >> 4)) - 8;
            return f32_from_f16(f16_from_f32(float(code) * f32_from_f16(b.d)));
        }
    }
    const int8_t code = (value ? p.v_q : p.k_q)[row * dim + d];
    const float scale = f32_from_f16((value ? p.v_scale : p.k_scale)[row * (dim / 64) + d / 64]);
    return f32_from_f16(f16_from_f32(float(code) * scale));
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Same score, softmax and value accumulation order as qsa_attend_kernel; only
// the head mapping and direct pool read differ. Work is local_heads, not 24.
template<int Mode>
__global__ void attend(const float* q, QsaAttnPools p, const int32_t* ids, const int32_t* steps,
                        int cap, int heads, int kv_heads, int dim, int page_size, Heads map,
                        float* output, float* weights) {
    const int h = blockIdx.x, z = blockIdx.y, d = threadIdx.x;
    const int32_t* step = steps + z * kStepCount;
    const int n = step[kStepWidth];
    const size_t out_base = (size_t(z) * heads + h) * dim;
    if (n < 0 || n > cap) { output[out_base + d] = nanf(""); return; }
    if (n == 0) { output[out_base + d] = 0; return; }
    ids += size_t(z) * cap;
    q += out_base;
    extern __shared__ float sm[];
    float* red = sm;
    float* w = sm + 32;
    const int kv = map.kv[h];
    const int lane = d & 31, wid = d >> 5, nwarp = dim / 32;
    const float scale = 1.0f / sqrtf(float(dim));
    __shared__ int invalid;
    if (d == 0) invalid = 0;
    __syncthreads();
    for (int j = d; j < n; j += dim) {
        const int cell = ids[j];
        if (cell < 0 || cell >= step[kStepNKv]) {
            atomicExch(&invalid, 1); w[j] = -CUDART_INF_F; continue;
        }
        if (cell > step[kStepPos]) { w[j] = -CUDART_INF_F; continue; }
        const int page = p.page_table[cell / page_size];
        if (page < 0) { atomicExch(&invalid, 1); w[j] = -CUDART_INF_F; continue; }
        const long long row = (static_cast<long long>(page) * kv_heads + kv) * page_size + cell % page_size;
        float acc = 0;
        for (int i = 0; i < dim; ++i) acc += load<Mode>(p, false, row, i, dim) * q[i];
        w[j] = acc * scale;
    }
    __syncthreads();
    if (invalid) { output[out_base + d] = nanf(""); return; }
    float mx = -FLT_MAX;
    for (int j = d; j < n; j += dim) mx = fmaxf(mx, w[j]);
    mx = warp_max(mx);
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        mx = warp_max(lane < nwarp ? red[lane] : -FLT_MAX);
        if (lane == 0) red[0] = mx;
    }
    __syncthreads();
    mx = red[0];
    __syncthreads();
    float sum = 0;
    for (int j = d; j < n; j += dim) {
        const float e = expf(w[j] - mx);
        w[j] = e; sum += e;
    }
    sum = warp_sum(sum);
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = warp_sum(lane < nwarp ? red[lane] : 0);
        if (lane == 0) red[0] = sum > 0 ? 1.0f / sum : 0;
    }
    __syncthreads();
    const float inv = red[0];
    __syncthreads();
    float acc = 0;
    for (int j = 0; j < n; ++j) {
        const int cell = ids[j];
        if (cell > step[kStepPos]) continue;
        const long long row = (static_cast<long long>(p.page_table[cell / page_size]) * kv_heads + kv)
                               * page_size + cell % page_size;
        acc += (w[j] * inv) * load<Mode>(p, true, row, d, dim);
    }
    output[out_base + d] = acc;
    if (weights)
        for (int j = d; j < n; j += dim) weights[(size_t(z) * heads + h) * cap + j] = w[j] * inv;
}
}  // namespace

void validate_qsa_shard(const QsaShard& s) {
    need(s.global.n_head > 0 && s.global.n_head_kv > 0 && s.global.n_head % s.global.n_head_kv == 0,
         "invalid global GQA geometry");
    need(s.global.head_dim == 256 && s.global.page_size > 0 && s.global.page_size <= INT_MAX,
         "requires head_dim=256 and a positive int page size");
    need(s.local_heads > 0 && s.local_heads <= 32, "local head count must be 1..32");
    need(s.global.n_head <= INT_MAX && s.global.n_head_kv <= INT_MAX, "head geometry overflows int");
    for (int h = 0; h < s.local_heads; ++h) {
        const int64_t g = s.explicit_heads ? s.query_heads[h] : int64_t(s.head_offset) + h;
        need(g >= 0 && g < s.global.n_head, "query head outside global geometry");
        for (int j = 0; j < h; ++j)
            need(g != (s.explicit_heads ? s.query_heads[j] : int64_t(s.head_offset) + j), "duplicate query head");
    }
}
int qsa_shard_kv_head(const QsaShard& s, int local_head) {
    validate_qsa_shard(s);
    need(local_head >= 0 && local_head < s.local_heads, "local head index outside shard");
    const int h = s.explicit_heads ? s.query_heads[local_head] : s.head_offset + local_head;
    return int(h / (s.global.n_head / s.global.n_head_kv));
}
void qsa_shard_attend(const float* q, const QsaAttnPools& p, const int32_t* ids,
                      const int32_t* steps, int cap, const QsaShard& s, float* output,
                      float* weights, int rows, void* stream) {
    validate_qsa_shard(s);
    need(q && ids && steps && output && stream, "null buffer or stream");
    need(cap > 0 && cap <= 4096 && rows > 0 && rows <= 65535, "invalid capacity or row count");
    Heads map{};
    for (int h = 0; h < s.local_heads; ++h) map.kv[h] = qsa_shard_kv_head(s, h);
    const dim3 grid(unsigned(s.local_heads), unsigned(rows));
    const size_t shared = size_t(cap + 32) * sizeof(float);
#define TP_QSA_LAUNCH(mode) attend<mode><<<grid, 256, shared, static_cast<cudaStream_t>(stream)>>>( \
    q, p, ids, steps, cap, s.local_heads, int(s.global.n_head_kv), 256, int(s.global.page_size), map, output, weights)
    switch (pool_mode(p)) {
        case 0: TP_QSA_LAUNCH(0); break;
        case 1: TP_QSA_LAUNCH(1); break;
        case 2: TP_QSA_LAUNCH(2); break;
        case 3: TP_QSA_LAUNCH(3); break;
    }
#undef TP_QSA_LAUNCH
    launched();
}
void qsa_shard_prepare_query(const float* full, const float* norm, float* q,
                             const int32_t* positions, const QsaShard& s, int rows,
                             const RopeScaling& rope, void* stream) {
    validate_qsa_shard(s);
    need(rows > 0 && rows <= 65535 / s.local_heads, "invalid query row count");
    need(s.global.n_rot == 64, "native query preparation requires n_rot=64");
    native_qsa_rms_norm_rope(full, 512, norm, q, rows * s.local_heads, 256,
                            int(s.global.n_rot), qsa_rms_eps(), rope, positions, stream);
}
void qsa_shard_gate(const float* attn, const float* full, float* output,
                    const QsaShard& s, int rows, void* stream) {
    validate_qsa_shard(s);
    need(rows > 0 && rows <= INT_MAX / (s.local_heads * 512), "invalid gate row count");
    native_qsa_gate_apply(attn, full, output, rows * s.local_heads, 256, stream);
}
void qsa_shard_select(const float* pooled, const float* q_idx, const float* bias,
                      const int32_t* step, int64_t max_blocks, int64_t cap,
                      const QsaShard& s, float* scores, int32_t* ids, void* stream) {
    validate_qsa_shard(s);
    need(stream != nullptr, "stream is null");
    qsa_index_step(pooled, q_idx, bias, s.global, step, max_blocks, scores, stream);
    topk_512_step(scores, s.global, cap, step, ids, stream);
}
}  // namespace strata::kernels::tp::cuda
