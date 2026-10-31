#include "contracts.cuh"
#include "strata/kernels/dp4a.hpp"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::kernels::tp::cuda {
namespace {
static_assert(sizeof(block_q8_1) == 36);

// The two block dots retain the integer accumulation and scale application of
// iq_kernels.cu (pinned ggml vecdotq.cuh, MIT, third_party/ggml/LICENSE).
__device__ int load2(const void* p, int i) {
    const auto* q = static_cast<const uint16_t*>(p);
    return int(q[2 * i]) | (int(q[2 * i + 1]) << 16);
}
__device__ int2 table16(int q) {
    const auto* t = reinterpret_cast<const uint32_t*>(kvalues_iq4nl);
    uint32_t tmp[2];
    const uint32_t select = 0x32103210 | ((q & 0x88888888) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const auto lo = __byte_perm(t[0], t[1], q >> (16 * i));
        const auto hi = __byte_perm(t[2], t[3], q >> (16 * i));
        tmp[i] = __byte_perm(lo, hi, select >> (16 * i));
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
template<int T> __device__ float2 operands(const uint8_t*, const block_q8_1*, int, int);
template<> __device__ float2 operands<20>(const uint8_t* row, const block_q8_1* a, int block, int part) {
    const auto* w = reinterpret_cast<const block_iq4_nl*>(row) + block;
    const int iqs = 2 * part;
    const auto* q8 = reinterpret_cast<const int*>(a->qs) + iqs;
    int sum = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int2 v = table16(load2(w->qs, iqs + l));
        sum = STRATA_DP4A(v.x, q8[l], sum);
        sum = STRATA_DP4A(v.y, q8[l + 4], sum);
    }
    return make_float2(__half2float(w->d) * __low2float(a->ds), float(sum));
}
template<> __device__ float2 operands<42>(const uint8_t* row, const block_q8_1* a, int block, int part) {
    const auto* w = reinterpret_cast<const block_q2_0*>(row) + block;
    const auto* qs = reinterpret_cast<const int16_t*>(w->qs) + part * 4;
    const auto* q8 = reinterpret_cast<const int*>(a[part].qs);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        sum = STRATA_DP4A(q8[2 * j], int(__byte_perm(qe, qo, 0x5140)), sum);
        sum = STRATA_DP4A(q8[2 * j + 1], int(__byte_perm(qe, qo, 0x7362)), sum);
    }
    return make_float2(__half2float(w->d) * __low2float(a[part].ds), float(sum));
}

template<int T>
__global__ void down_kernel(const uint8_t* weights, std::size_t stride, const uint8_t* q81,
                            std::size_t activation_stride, float* y, int nout,
                            int global_width, int begin, int end) {
    constexpr int qk = T == 20 ? 32 : 64;
    const int out = int(blockIdx.x) * 4 + int(threadIdx.y);
    if (out >= nout) return;
    const int lane = int(threadIdx.x);
    const auto* a = reinterpret_cast<const block_q8_1*>(q81 + blockIdx.y * activation_stride);
    const auto* row = weights + std::size_t(out) * stride;
    float sum = 0.f;
    // Keep the monolithic call-to-lane assignment even at a shard boundary.
    for (int call = lane; call < global_width / qk * 2; call += 32) {
        const int global_block = call / 2;
        if (global_block < begin / qk || global_block >= end / qk) continue;
        const float2 op = operands<T>(row, a + global_block * (qk / 32), global_block - begin / qk, call % 2);
        sum += op.x * op.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) y[std::size_t(blockIdx.y) * nout + out] = sum;
}
template<int T>
__global__ void down_lanes_kernel(const uint8_t* weights, std::size_t stride, const uint8_t* q81,
                                  std::size_t activation_stride, float* payload, int nout,
                                  int width, int begin, int end) {
    constexpr int qk = T == 20 ? 32 : 64;
    const int out = int(blockIdx.x) * 4 + int(threadIdx.y), lane = int(threadIdx.x);
    if (out >= nout) return;
    const auto* a = reinterpret_cast<const block_q8_1*>(q81 + blockIdx.y * activation_stride);
    const auto* row = weights + std::size_t(out) * stride;
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        const int call = lane + 32 * slot, global_block = call / 2;
        float2 op = make_float2(0.f, 0.f);
        if (call < width / qk * 2 && global_block >= begin / qk && global_block < end / qk)
            op = operands<T>(row, a + global_block * (qk / 32), global_block - begin / qk, call % 2);
        const auto i = ((std::size_t(blockIdx.y) * nout + out) * 2 + slot) * 32 + lane;
        reinterpret_cast<float2*>(payload)[i] = op;
    }
}
__global__ void finish_lanes_kernel(const float2* payload, float* output, int qk, int width,
                                    int nout, int rows, int end0, int end1) {
    const int out = int(blockIdx.x) * 4 + int(threadIdx.y), lane = int(threadIdx.x);
    if (out >= nout) return;
    float sum = 0.f;
    for (int call = lane; call < width / qk * 2; call += 32) {
        const int input = (call / 2) * qk;
        const int rank = input < end0 ? 0 : input < end1 ? 1 : 2;
        const auto i = ((std::size_t(rank) * rows + blockIdx.y) * nout + out) * 64 + (call / 32) * 32 + lane;
        const auto op = payload[i];
        sum = __fmaf_rn(op.x, op.y, sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if (lane == 0) output[std::size_t(blockIdx.y) * nout + out] = sum;
}
template<int T>
__global__ void down_lanes_compact_kernel(const uint8_t* weights, std::size_t stride, const uint8_t* q81,
                                          std::size_t activation_stride, float2* payload, int nout,
                                          int begin, int owned_calls, int slots) {
    constexpr int qk = T == 20 ? 32 : 64;
    const int out = int(blockIdx.x) * 4 + int(threadIdx.y);
    if (out >= nout) return;
    const auto* a = reinterpret_cast<const block_q8_1*>(q81 + blockIdx.y * activation_stride);
    const auto* row = weights + std::size_t(out) * stride;
    for (int local_call = int(threadIdx.x); local_call < slots; local_call += 32) {
        float2 op = make_float2(0.f, 0.f);
        if (local_call < owned_calls) {
            const int call = begin / qk * 2 + local_call, global_block = call / 2;
            op = operands<T>(row, a + global_block * (qk / 32), local_call / 2, call % 2);
        }
        payload[(std::size_t(blockIdx.y) * nout + out) * slots + local_call] = op;
    }
}
template<int T>
__global__ void down_lanes_compact_routed_kernel(const uint8_t* weights, std::size_t expert_stride,
                                                 std::size_t row_stride, const int32_t* expert_ids,
                                                 int expert_count, const uint8_t* q81,
                                                 std::size_t activation_stride, float2* payload, int nout,
                                                 int begin, int owned_calls, int slots, int* status) {
    constexpr int qk = T == 20 ? 32 : 64;
    const int out = int(blockIdx.x) * 4 + int(threadIdx.y), lane = int(threadIdx.x);
    const int route = int(blockIdx.y);
    if (out >= nout) return;
    const int expert = expert_ids[route];
    const std::size_t payload_offset = (std::size_t(route) * nout + out) * slots;
    if (expert < 0 || expert >= expert_count) {
        if (lane == 0) atomicExch(status, 1);
        for (int call = lane; call < slots; call += 32) payload[payload_offset + call] = make_float2(0.f, 0.f);
        return;
    }
    const auto* a = reinterpret_cast<const block_q8_1*>(q81 + std::size_t(route) * activation_stride);
    const auto* row = weights + std::size_t(expert) * expert_stride + std::size_t(out) * row_stride;
    for (int local_call = lane; local_call < slots; local_call += 32) {
        float2 op = make_float2(0.f, 0.f);
        if (local_call < owned_calls) {
            const int call = begin / qk * 2 + local_call, global_block = call / 2;
            op = operands<T>(row, a + global_block * (qk / 32), local_call / 2, call % 2);
        }
        payload[payload_offset + local_call] = op;
    }
}
__global__ void routed_swiglu_kernel(const float* gate, const float* up, float* h, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) h[i] = (gate[i] / (1.0f + __expf(-gate[i]))) * up[i];
}
__global__ void expand_q81_topk_kernel(const uint8_t* input, uint8_t* output,
                                       std::size_t input_stride, std::size_t output_row_bytes,
                                       int top_k, std::size_t bytes) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) {
        const auto row = i / output_row_bytes;
        const auto token = row / std::size_t(top_k);
        output[i] = input[token * input_stride + i % output_row_bytes];
    }
}
__global__ void finish_lanes_compact_kernel(const float2* payload, float* output, int calls,
                                            int nout, int rows, int end0, int end1, int slots) {
    const int out = int(blockIdx.x) * 4 + int(threadIdx.y), lane = int(threadIdx.x);
    if (out >= nout) return;
    float sum = 0.f;
    for (int call = lane; call < calls; call += 32) {
        const int rank = call < end0 ? 0 : call < end1 ? 1 : 2;
        const int local_call = call - (rank == 0 ? 0 : rank == 1 ? end0 : end1);
        const auto i = ((std::size_t(rank) * rows + blockIdx.y) * nout + out) * slots + local_call;
        const auto op = payload[i];
        sum = __fmaf_rn(op.x, op.y, sum);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    if (lane == 0) output[std::size_t(blockIdx.y) * nout + out] = sum;
}
}  // namespace

std::size_t global_q81_bytes(int rows, int width) {
    detail::require(rows > 0 && rows <= 65535 && width > 0 && width % 32 == 0, "TP CUDA: bad Q8_1 dimensions");
    return std::size_t(rows) * std::size_t(width / 32) * sizeof(block_q8_1);
}

GlobalQ81 quantize_global(const distributed::ActivationContract& c, const float* logical,
                         int rows, int width, void* q81, std::size_t capacity, void* stream) {
    detail::contract(c);
    const auto bytes = global_q81_bytes(rows, width);
    if (c.gather == "all_gather_padded") {
        int64_t sum = 0;
        detail::require(c.wire_slots > 0 && !c.valid_counts.empty(), "TP CUDA: missing gather layout");
        for (const auto count : c.valid_counts) {
            detail::require(count >= 0 && count <= c.wire_slots && count <= width - sum,
                            "TP CUDA: invalid global quantizer gather counts");
            sum += count;
        }
        detail::require(sum == width, "TP CUDA: gathered quantizer width differs from logical width");
    }
    detail::require(logical && q81 && capacity >= bytes && stream, "TP CUDA: missing global quantizer buffers/stream");
    detail::require(reinterpret_cast<std::uintptr_t>(logical) % alignof(float) == 0 &&
                    reinterpret_cast<std::uintptr_t>(q81) % alignof(block_q8_1) == 0,
                    "TP CUDA: unaligned global quantizer buffer");
    quantize_q8_1_rows(logical, rows, width, q81, stream);
    return {q81, width, rows, std::size_t(width / 32) * sizeof(block_q8_1), bytes};
}

GlobalQ81 expand_global_q81_topk(const GlobalQ81& input, int top_k, void* expanded_q81,
                                 std::size_t capacity, void* stream) {
    detail::image(input, input.width);
    detail::require(top_k > 0 && input.rows <= std::numeric_limits<int>::max() / top_k &&
                    expanded_q81 && stream,
                    "TP CUDA: invalid Q8_1 top-k expansion arguments");
    const int rows = input.rows * top_k;
    const auto input_row_bytes = std::size_t(input.width / 32) * sizeof(block_q8_1);
    const auto bytes = global_q81_bytes(rows, input.width);
    const auto input_begin = reinterpret_cast<std::uintptr_t>(input.data);
    const auto output_begin = reinterpret_cast<std::uintptr_t>(expanded_q81);
    detail::require(capacity >= bytes &&
                    reinterpret_cast<std::uintptr_t>(expanded_q81) % alignof(block_q8_1) == 0,
                    "TP CUDA: Q8_1 top-k expansion output is too small or unaligned");
    detail::require(input.bytes <= std::numeric_limits<std::uintptr_t>::max() - input_begin &&
                    bytes <= std::numeric_limits<std::uintptr_t>::max() - output_begin &&
                    (input_begin + input.bytes <= output_begin || output_begin + bytes <= input_begin),
                    "TP CUDA: Q8_1 top-k expansion buffers overlap");
    const auto total = bytes;
    constexpr int threads = 256;
    expand_q81_topk_kernel<<<unsigned((total + threads - 1) / threads), threads, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        static_cast<const uint8_t*>(input.data), static_cast<uint8_t*>(expanded_q81),
        input.row_stride, input_row_bytes, top_k, total);
    detail::check(cudaGetLastError());
    return {expanded_q81, input.width, rows, input_row_bytes, bytes};
}

void down(const MatrixShard& w, const distributed::ActivationContract& c, const GlobalQ81& a,
          float* partial, void* stream) {
    detail::contract(c);
    detail::matrix(w, 0);
    detail::require(w.ggml_type == 20 || w.ggml_type == 42, "TP CUDA: down codec must be IQ4_NL or Q2_0");
    detail::image(a, int(w.slice.global_shape[0]));
    detail::require(partial && stream, "TP CUDA: missing output or ordered stream");
    const int nout = int(w.slice.global_shape[1]);
    const dim3 grid((nout + 3) / 4, a.rows), block(32, 4);
    const auto* p = static_cast<const uint8_t*>(w.data);
    const auto* q = static_cast<const uint8_t*>(a.data);
    if (w.ggml_type == 20)
        down_kernel<20><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            p, w.row_stride, q, a.row_stride, partial, nout, a.width,
            int(w.slice.range_start), int(w.slice.range_end));
    else
        down_kernel<42><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            p, w.row_stride, q, a.row_stride, partial, nout, a.width,
            int(w.slice.range_start), int(w.slice.range_end));
    detail::check(cudaGetLastError());
}

void gate_up_routed(const MatrixShard& g, const MatrixShard& u, const distributed::ActivationContract& c,
                    const GlobalQ81& a, const int32_t* ids, int expert_count,
                    std::size_t gate_bytes_per_expert, std::size_t up_bytes_per_expert,
                    int routed_rows, bool broadcast_input, float* gate_out, float* up_out, float* h,
                    int* status, void* stream) {
    detail::contract(c);
    detail::matrix(g, 1); detail::matrix(u, 1);
    detail::require(g.ggml_type != 20 && g.ggml_type != 42 && g.ggml_type == u.ggml_type &&
                    (g.ggml_type == 16 || g.ggml_type == 17 || g.ggml_type == 18 ||
                     g.ggml_type == 21 || g.ggml_type == 22 || g.ggml_type == 23) &&
                    g.slice.global_shape == u.slice.global_shape &&
                    g.slice.range_start == u.slice.range_start && g.slice.range_end == u.slice.range_end,
                    "TP routed gate/up: ownership or codec mismatch");
    detail::image(a, int(g.slice.global_shape[0]));
    const int nout = int(g.slice.range_end - g.slice.range_start);
    const auto rb = iq_row_bytes(g.ggml_type, a.width);
    detail::require(ids && expert_count > 0 && routed_rows > 0 &&
                    (broadcast_input ? a.rows == 1 : a.rows == routed_rows) &&
                    gate_out && up_out && h && status && stream &&
                    gate_out != up_out && h != gate_out && h != up_out &&
                    nout <= std::numeric_limits<int>::max() / routed_rows &&
                    gate_bytes_per_expert >= std::size_t(nout) * rb &&
                    up_bytes_per_expert >= std::size_t(nout) * rb &&
                    gate_bytes_per_expert <= std::numeric_limits<std::size_t>::max() / std::size_t(expert_count) &&
                    up_bytes_per_expert <= std::numeric_limits<std::size_t>::max() / std::size_t(expert_count) &&
                    g.bytes >= gate_bytes_per_expert * std::size_t(expert_count) &&
                    u.bytes >= up_bytes_per_expert * std::size_t(expert_count) &&
                    g.row_stride == rb && u.row_stride == rb &&
                    reinterpret_cast<std::uintptr_t>(ids) % alignof(int32_t) == 0 &&
                    reinterpret_cast<std::uintptr_t>(status) % alignof(int) == 0,
                    "TP routed gate/up: invalid expert planes or buffers");
    native_expert_gate_up_routed(g.ggml_type, g.data, u.data, gate_bytes_per_expert, up_bytes_per_expert,
                                 ids, expert_count, a.data, a.row_stride, routed_rows, broadcast_input,
                                 a.width, nout, gate_out, up_out, status, stream);
    const int64_t count = int64_t(routed_rows) * nout;
    routed_swiglu_kernel<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        gate_out, up_out, h, count);
    detail::check(cudaGetLastError());
}

void gate_up_swiglu_routed_padded(const MatrixShard& g, const MatrixShard& u,
                                  const distributed::ActivationContract& c, const GlobalQ81& a,
                                  const int32_t* ids, int expert_count,
                                  std::size_t gate_bytes_per_expert, std::size_t up_bytes_per_expert,
                                  int routed_rows, bool broadcast_input, int output_stride,
                                  float* h, int* status, void* stream) {
    detail::contract(c);
    detail::matrix(g, 1); detail::matrix(u, 1);
    detail::require(g.ggml_type != 20 && g.ggml_type != 42 && g.ggml_type == u.ggml_type &&
                    (g.ggml_type == 16 || g.ggml_type == 17 || g.ggml_type == 18 ||
                     g.ggml_type == 21 || g.ggml_type == 22 || g.ggml_type == 23) &&
                    g.slice.global_shape == u.slice.global_shape &&
                    g.slice.range_start == u.slice.range_start && g.slice.range_end == u.slice.range_end,
                    "TP routed fused gate/up: ownership or codec mismatch");
    detail::image(a, int(g.slice.global_shape[0]));
    const int nout = int(g.slice.range_end - g.slice.range_start);
    const auto rb = iq_row_bytes(g.ggml_type, a.width);
    detail::require(ids && expert_count > 0 && routed_rows > 0 &&
                    (broadcast_input ? a.rows == 1 : a.rows == routed_rows) &&
                    output_stride >= nout && h && status && stream &&
                    output_stride <= std::numeric_limits<int>::max() / routed_rows &&
                    gate_bytes_per_expert >= std::size_t(nout) * rb &&
                    up_bytes_per_expert >= std::size_t(nout) * rb &&
                    gate_bytes_per_expert <= std::numeric_limits<std::size_t>::max() / std::size_t(expert_count) &&
                    up_bytes_per_expert <= std::numeric_limits<std::size_t>::max() / std::size_t(expert_count) &&
                    g.bytes >= gate_bytes_per_expert * std::size_t(expert_count) &&
                    u.bytes >= up_bytes_per_expert * std::size_t(expert_count) &&
                    g.row_stride == rb && u.row_stride == rb &&
                    reinterpret_cast<std::uintptr_t>(ids) % alignof(int32_t) == 0 &&
                    reinterpret_cast<std::uintptr_t>(h) % alignof(float) == 0 &&
                    reinterpret_cast<std::uintptr_t>(status) % alignof(int) == 0,
                    "TP routed fused gate/up: invalid expert planes or buffers");
    native_expert_gate_up_swiglu_routed_padded(g.ggml_type, g.data, u.data,
                                               gate_bytes_per_expert, up_bytes_per_expert,
                                               ids, expert_count, a.data, a.row_stride, routed_rows,
                                               broadcast_input, a.width, nout, output_stride, h, status, stream);
}

void down_lanes(const MatrixShard& w, const distributed::ActivationContract& c, const GlobalQ81& a,
                float* payload, void* stream) {
    detail::contract(c); detail::matrix(w, 0);
    detail::require(w.ggml_type == 20 || w.ggml_type == 42, "TP exact lanes: unsupported down codec");
    detail::image(a, int(w.slice.global_shape[0]));
    const int qk = detail::block(w.ggml_type);
    detail::require(a.width <= 1024 && a.width / qk * 2 <= 64 && payload && stream &&
                    reinterpret_cast<std::uintptr_t>(payload) % alignof(float2) == 0,
                    "TP exact lanes: unsupported call geometry or unaligned payload");
    const int nout = int(w.slice.global_shape[1]);
    const dim3 grid((nout + 3) / 4, a.rows), threads(32, 4);
    if (w.ggml_type == 20)
        down_lanes_kernel<20><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(w.data), w.row_stride, static_cast<const uint8_t*>(a.data),
            a.row_stride, payload, nout, a.width, int(w.slice.range_start), int(w.slice.range_end));
    else
        down_lanes_kernel<42><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(w.data), w.row_stride, static_cast<const uint8_t*>(a.data),
            a.row_stride, payload, nout, a.width, int(w.slice.range_start), int(w.slice.range_end));
    detail::check(cudaGetLastError());
}
void finish_down_lanes(const float* payload, int type, int width, int nout, int rows,
                       std::array<uint64_t, 3> valid, float* output, void* stream) {
    detail::require(type == 20 || type == 42, "TP exact lanes: unsupported codec");
    const int qk = detail::block(type);
    detail::require(width > 0 && width <= 1024 && width % qk == 0 && width / qk * 2 <= 64 &&
                    nout > 0 && rows > 0 && rows <= 65535 && payload && output && stream && payload != output &&
                    reinterpret_cast<std::uintptr_t>(payload) % alignof(float2) == 0,
                    "TP exact lanes: invalid finish geometry/buffers");
    uint64_t sum = 0;
    for (auto count : valid) {
        detail::require(count > 0 && count % qk == 0 && count <= uint64_t(width) - sum,
                        "TP exact lanes: invalid ownership cuts");
        sum += count;
    }
    detail::require(sum == uint64_t(width), "TP exact lanes: incomplete ownership");
    finish_lanes_kernel<<<dim3((nout + 3) / 4, rows), dim3(32, 4), 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float2*>(payload), output, qk, width, nout, rows, int(valid[0]), int(valid[0] + valid[1]));
    detail::check(cudaGetLastError());
}

int compact_down_payload_width(int type, int width, std::array<uint64_t, 3> valid) {
    detail::require(type == 20 || type == 42, "TP compact exact v2: unsupported codec");
    const int qk = detail::block(type);
    detail::require(width > 0 && width <= 1024 && width % qk == 0 && width / qk * 2 <= 64,
                    "TP compact exact v2: unsupported call geometry");
    uint64_t sum = 0, largest = 0;
    for (const auto count : valid) {
        detail::require(count > 0 && count % qk == 0 && count <= uint64_t(width) - sum,
                        "TP compact exact v2: invalid ownership cuts");
        sum += count;
        if (count > largest) largest = count;
    }
    detail::require(sum == uint64_t(width), "TP compact exact v2: incomplete ownership");
    return int(largest / qk) * 4;
}

void down_lanes_compact(const MatrixShard& w, const distributed::ActivationContract& c, const GlobalQ81& a,
                        float* payload, int slot_width, void* stream) {
    detail::contract(c); detail::matrix(w, 0);
    detail::image(a, int(w.slice.global_shape[0]));
    detail::require(c.valid_counts.size() == 3, "TP compact exact v2: three ownership counts required");
    std::array<uint64_t, 3> valid{};
    bool owned_cut = false;
    int64_t begin = 0;
    for (int rank = 0; rank < 3; ++rank) {
        detail::require(c.valid_counts[rank] > 0 && c.valid_counts[rank] <= a.width - begin,
                        "TP compact exact v2: invalid activation ownership");
        valid[rank] = uint64_t(c.valid_counts[rank]);
        owned_cut |= w.slice.range_start == begin && w.slice.range_end == begin + c.valid_counts[rank];
        begin += c.valid_counts[rank];
    }
    detail::require(slot_width == compact_down_payload_width(w.ggml_type, a.width, valid) && owned_cut &&
                    payload && stream && reinterpret_cast<std::uintptr_t>(payload) % alignof(float2) == 0,
                    "TP compact exact v2: wrong payload width, shard cut or buffers");
    const int qk = detail::block(w.ggml_type), nout = int(w.slice.global_shape[1]);
    const int calls = int(w.slice.range_end - w.slice.range_start) / qk * 2;
    const dim3 grid((nout + 3) / 4, a.rows), threads(32, 4);
    if (w.ggml_type == 20)
        down_lanes_compact_kernel<20><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(w.data), w.row_stride, static_cast<const uint8_t*>(a.data),
            a.row_stride, reinterpret_cast<float2*>(payload), nout, int(w.slice.range_start), calls, slot_width / 2);
    else
        down_lanes_compact_kernel<42><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(w.data), w.row_stride, static_cast<const uint8_t*>(a.data),
            a.row_stride, reinterpret_cast<float2*>(payload), nout, int(w.slice.range_start), calls, slot_width / 2);
    detail::check(cudaGetLastError());
}

void down_lanes_compact_routed(const MatrixShard& w, const distributed::ActivationContract& c,
                               const GlobalQ81& a, const int32_t* ids, int expert_count,
                               std::size_t bytes_per_expert, float* payload, int slot_width,
                               int* status, void* stream) {
    detail::contract(c); detail::matrix(w, 0); detail::image(a, int(w.slice.global_shape[0]));
    detail::require(c.valid_counts.size() == 3, "TP compact routed: three ownership counts required");
    std::array<uint64_t, 3> valid{};
    bool owned_cut = false;
    int64_t begin = 0;
    for (int rank = 0; rank < 3; ++rank) {
        detail::require(c.valid_counts[rank] > 0 && c.valid_counts[rank] <= a.width - begin,
                        "TP compact routed: invalid activation ownership");
        valid[rank] = uint64_t(c.valid_counts[rank]);
        owned_cut |= w.slice.range_start == begin && w.slice.range_end == begin + c.valid_counts[rank];
        begin += c.valid_counts[rank];
    }
    const int qk = detail::block(w.ggml_type);
    const int nout = int(w.slice.global_shape[1]);
    detail::require(nout <= std::numeric_limits<int>::max() / a.rows,
                    "TP compact routed: routed row geometry overflows");
    const auto row_bytes = iq_row_bytes(w.ggml_type, int(w.slice.local_shape[0]));
    const auto owned_calls = int(w.slice.range_end - w.slice.range_start) / qk * 2;
    detail::require((w.ggml_type == 20 || w.ggml_type == 42) &&
                    slot_width == compact_down_payload_width(w.ggml_type, a.width, valid) &&
                    owned_cut && ids && expert_count > 0 && bytes_per_expert >= std::size_t(nout) * row_bytes &&
                    bytes_per_expert <= std::numeric_limits<std::size_t>::max() / std::size_t(expert_count) &&
                    w.bytes >= bytes_per_expert * std::size_t(expert_count) &&
                    payload && status && stream &&
                    reinterpret_cast<std::uintptr_t>(ids) % alignof(int32_t) == 0 &&
                    reinterpret_cast<std::uintptr_t>(payload) % alignof(float2) == 0 &&
                    reinterpret_cast<std::uintptr_t>(status) % alignof(int) == 0,
                    "TP compact routed: invalid payload, expert planes, or ownership");
    const dim3 grid((nout + 3) / 4, a.rows), threads(32, 4);
    if (w.ggml_type == 20)
        down_lanes_compact_routed_kernel<20><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(w.data), bytes_per_expert, w.row_stride, ids, expert_count,
            static_cast<const uint8_t*>(a.data), a.row_stride, reinterpret_cast<float2*>(payload), nout,
            int(w.slice.range_start), owned_calls, slot_width / 2, status);
    else
        down_lanes_compact_routed_kernel<42><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint8_t*>(w.data), bytes_per_expert, w.row_stride, ids, expert_count,
            static_cast<const uint8_t*>(a.data), a.row_stride, reinterpret_cast<float2*>(payload), nout,
            int(w.slice.range_start), owned_calls, slot_width / 2, status);
    detail::check(cudaGetLastError());
}

void finish_down_lanes_compact(const float* payload, int type, int width, int nout, int rows,
                               std::array<uint64_t, 3> valid, int slot_width, float* output, void* stream) {
    detail::require(slot_width == compact_down_payload_width(type, width, valid),
                    "TP compact exact v2: wrong uniform payload width");
    detail::require(nout > 0 && nout <= std::numeric_limits<int>::max() - 3 && rows > 0 && rows <= 65535 &&
                    payload && output && stream && reinterpret_cast<std::uintptr_t>(payload) % alignof(float2) == 0 &&
                    reinterpret_cast<std::uintptr_t>(output) % alignof(float) == 0,
                    "TP compact exact v2: invalid finish geometry/buffers");
    const auto payload_begin = reinterpret_cast<std::uintptr_t>(payload);
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output);
    const auto payload_bytes = std::size_t(3) * rows * nout * slot_width * sizeof(float);
    const auto output_bytes = std::size_t(rows) * nout * sizeof(float);
    detail::require(payload_begin <= std::numeric_limits<std::uintptr_t>::max() - payload_bytes &&
                    output_begin <= std::numeric_limits<std::uintptr_t>::max() - output_bytes &&
                    (payload_begin + payload_bytes <= output_begin || output_begin + output_bytes <= payload_begin),
                    "TP compact exact v2: overlapping finish buffers");
    const int qk = detail::block(type);
    finish_lanes_compact_kernel<<<dim3((nout + 3) / 4, rows), dim3(32, 4), 0, static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float2*>(payload), output, width / qk * 2, nout, rows,
        int(valid[0]) / qk * 2, int(valid[0] + valid[1]) / qk * 2, slot_width / 2);
    detail::check(cudaGetLastError());
}
}  // namespace strata::kernels::tp::cuda
