#include "strata/kernels/fast_gdn_batch.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels::tp::cuda {
namespace {
constexpr int kState = 128;
constexpr int kMaxGroups = 8;
constexpr int kMaxRows = strata::kernels::kWindowMaxT;

__device__ __forceinline__ int group_for_row(const FastGdnBatchGroups& groups, int row) {
    for (int i = 0; i < groups.n; ++i) {
        if (row >= groups.first[i] && row < groups.first[i] + groups.count[i]) return i;
    }
    return -1;
}

__global__ void conv_kernel(FastGdnBatchGroups groups, const float* qkv, const float* weights, float* h,
                            int channels, int key_heads, int row_extent) {
    const int row = blockIdx.y;
    const int channel = blockIdx.x * 128 + threadIdx.x;
    if (row >= row_extent || channel >= channels) return;
    const int gi = group_for_row(groups, row);
    if (gi < 0) return;
    const int local = row - groups.first[gi];
    const float* history = groups.history[gi];

    float value = 0.0f;
    for (int tap = 0; tap < 4; ++tap) {
        const int p = local + tap - 3;
        const float x = p < 0 ? history[channel * 3 + p + 3]
                              : qkv[size_t(groups.first[gi] + p) * channels + channel];
        value += weights[channel * 4 + tap] * x;
    }
    value = value / (1.0f + expf(-value));
    if (blockIdx.x < 2 * key_heads) {
        __shared__ float sums[4];
        float sum = value * value;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = sum;
        __syncthreads();
        const float total = sums[0] + sums[1] + sums[2] + sums[3];
        value *= rsqrtf(total + 1.0e-6f);
    }
    h[size_t(row) * channels + channel] = value;
}

__global__ void recurrence_kernel(FastGdnBatchGroups groups, const float* h, const float* gate, const float* beta,
                                  const float* z, const float* norm, float* out, int key_heads, int value_heads,
                                  float eps) {
    constexpr int kRowGroups = 4;
    constexpr int kRowsPerGroup = kState / kRowGroups;
    const int gi = blockIdx.y;
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    if (gi >= groups.n || head >= value_heads) return;

    const int first = groups.first[gi];
    const int rows = groups.count[gi];
    const int channels = (2 * key_heads + value_heads) * kState;
    const int key = head % key_heads;
    const int value_width = value_heads * kState;
    float* state = groups.state[gi];
    float s[kRowsPerGroup];
    float* base = state + ((size_t(rg * kRowsPerGroup) * value_heads + head) * kState + col);
    const size_t row_stride = size_t(value_heads) * kState;
#pragma unroll
    for (int r = 0; r < kRowsPerGroup; ++r) s[r] = base[size_t(r) * row_stride];

    __shared__ float sk[2][kState], sq[2][kState];
    __shared__ float red_kv[kRowGroups][kState];
    __shared__ float red_o[kRowGroups][kState];
    __shared__ float wsum[kState / 32];
    const float gam = rg == 0 ? norm[col] : 0.0f;

    for (int t = 0; t < rows; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const int row = first + t;
        const float* ht = h + size_t(row) * channels;
        const float* zt = z + size_t(row) * value_width;
        if (t == 0) {
            if (rg == 0) sk[0][col] = ht[key_heads * kState + key * kState + col];
            else if (rg == 1) sq[0][col] = ht[key * kState + col];
        }
        __syncthreads();
        if (t + 1 < rows) {
            const float* next_h = ht + channels;
            if (rg == 0) sk[nxt][col] = next_h[key_heads * kState + key * kState + col];
            else if (rg == 1) sq[nxt][col] = next_h[key * kState + col];
        }
        const float g = __expf(gate[size_t(row) * value_heads + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < kRowsPerGroup; ++r) kv = fmaf(s[r], sk[cur][rg * kRowsPerGroup + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * key_heads * kState + head * kState + col] -
                             g * kv_col) * beta[size_t(row) * value_heads + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < kRowsPerGroup; ++r) {
            s[r] = fmaf(g, s[r], sk[cur][rg * kRowsPerGroup + r] * delta);
            o = fmaf(s[r], sq[cur][rg * kRowsPerGroup + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, square = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf(float(kState));
            square = oc * oc;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            square += __shfl_xor_sync(0xffffffffu, square, offset);
        if (rg == 0 && (col & 31) == 0) wsum[col >> 5] = square;
        __syncthreads();
        if (rg == 0) {
            const float sum = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(sum / float(kState) + eps);
            const float zv = zt[head * kState + col];
            const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zv)));
            out[size_t(row) * value_width + head * kState + col] = yv;
        }
        if (groups.snapshot[gi] && t + 1 < rows) {
            float* prefix = groups.state_prefix[gi] + size_t(t) * kState * value_heads * kState;
#pragma unroll
            for (int r = 0; r < kRowsPerGroup; ++r) {
                prefix[(size_t(rg * kRowsPerGroup + r) * value_heads + head) * kState + col] = s[r];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kRowsPerGroup; ++r) base[size_t(r) * row_stride] = s[r];
}

__global__ void history_commit_kernel(FastGdnBatchGroups groups, const float* qkv, int channels) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int gi = blockIdx.y;
    if (gi >= groups.n || channel >= channels) return;
    const int first = groups.first[gi];
    const int rows = groups.count[gi];
    float* history = groups.history[gi];
    float tail[3] = {history[channel * 3], history[channel * 3 + 1], history[channel * 3 + 2]};
    for (int t = 0; t < rows; ++t) {
        const float* row = qkv + size_t(first + t) * channels + channel;
        tail[0] = tail[1];
        tail[1] = tail[2];
        tail[2] = *row;
        if (groups.snapshot[gi] && t + 1 < rows) {
            float* prefix = groups.history_prefix[gi] + size_t(t) * channels * 3;
            prefix[channel * 3] = tail[0];
            prefix[channel * 3 + 1] = tail[1];
            prefix[channel * 3 + 2] = tail[2];
        }
    }
    history[channel * 3] = tail[0];
    history[channel * 3 + 1] = tail[1];
    history[channel * 3 + 2] = tail[2];
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void validate(const FastGdnBatchGroups& groups, const float* qkv, const float* conv_w, float* h,
              const float* gate, const float* beta, const float* z, const float* norm, float* out,
              int key_heads, float eps, void* stream) {
    if (!stream || !qkv || !conv_w || !h || !gate || !beta || !z || !norm || !out ||
        (key_heads != 5 && key_heads != 6) || !std::isfinite(eps) || eps < 0.0f ||
        groups.n < 1 || groups.n > kMaxGroups) {
        throw std::invalid_argument("fast GDN batch requires valid pointers, geometry, epsilon, groups and stream");
    }
    int total_rows = 0;
    int row_extent = 0;
    for (int i = 0; i < groups.n; ++i) {
        if (!groups.state[i] || !groups.history[i] || groups.count[i] < 1 || groups.count[i] > 5 ||
            groups.first[i] < 0 || groups.first[i] > kMaxRows - groups.count[i] ||
            (groups.snapshot[i] && (!groups.state_prefix[i] || !groups.history_prefix[i]))) {
            throw std::invalid_argument("fast GDN batch group has invalid state, range or snapshot storage");
        }
        total_rows += groups.count[i];
        row_extent = std::max(row_extent, groups.first[i] + groups.count[i]);
        for (int j = 0; j < i; ++j) {
            const int a0 = groups.first[i], a1 = a0 + groups.count[i];
            const int b0 = groups.first[j], b1 = b0 + groups.count[j];
            if (a0 < b1 && b0 < a1) throw std::invalid_argument("fast GDN batch row ranges overlap");
        }
    }
    if (total_rows > kMaxRows) throw std::invalid_argument("fast GDN batch exceeds the configured window limit");
}
}  // namespace

void fast_gdn_batch(const FastGdnBatchGroups& groups, const float* qkv, const float* conv_w, float* h,
                    const float* gate, const float* beta, const float* z, const float* norm, float* out,
                    int key_heads, float eps, void* stream) {
    validate(groups, qkv, conv_w, h, gate, beta, z, norm, out, key_heads, eps, stream);
    int row_extent = 0;
    for (int i = 0; i < groups.n; ++i) row_extent = std::max(row_extent, groups.first[i] + groups.count[i]);
    const int channels = key_heads * 5 * kState;
    const int value_heads = key_heads * 3;
    conv_kernel<<<dim3(channels / kState, row_extent), kState, 0, static_cast<cudaStream_t>(stream)>>>(
        groups, qkv, conv_w, h, channels, key_heads, row_extent);
    check_launch();
    recurrence_kernel<<<dim3(value_heads, groups.n), dim3(kState, 4), 0, static_cast<cudaStream_t>(stream)>>>(
        groups, h, gate, beta, z, norm, out, key_heads, value_heads, eps);
    check_launch();
    history_commit_kernel<<<dim3((channels + 127) / 128, groups.n), 128, 0,
                            static_cast<cudaStream_t>(stream)>>>(groups, qkv, channels);
    check_launch();
}
}  // namespace strata::kernels::tp::cuda
