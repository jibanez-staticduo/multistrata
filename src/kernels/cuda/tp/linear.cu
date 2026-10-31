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
template<int T> __device__ float dot(const uint8_t*, const block_q8_1*, int, int);
template<> __device__ float dot<20>(const uint8_t* row, const block_q8_1* a, int block, int part) {
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
    return (__half2float(w->d) * __low2float(a->ds)) * sum;
}
template<> __device__ float dot<42>(const uint8_t* row, const block_q8_1* a, int block, int part) {
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
    return (__half2float(w->d) * __low2float(a[part].ds)) * sum;
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
        sum += dot<T>(row, a + global_block * (qk / 32), global_block - begin / qk, call % 2);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) y[std::size_t(blockIdx.y) * nout + out] = sum;
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
}  // namespace strata::kernels::tp::cuda
