#include "contracts.cuh"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/swiglu.cuh"

#include <numeric>

namespace strata::kernels::tp::cuda {
namespace {
__global__ void swiglu(const float* gate, const float* up, float* h, int64_t n,
                       const float* gs, const float* us) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gs ? gate[i] * *gs : gate[i];
        const float u = us ? up[i] * *us : up[i];
        h[i] = swilu_fast(g, u); // native_expert_grouped's unfused expression
    }
}
__global__ void finish(float* y, int64_t n, int width, const float* scale, const float* bias) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = y[i];
        if (scale) v = __fmul_rn(v, *scale);
        if (bias) v = __fadd_rn(v, bias[i % width]);
        y[i] = v;
    }
}
__global__ void scale_parts(const float* x, const float* scales, float* y, int64_t n, int width, int k) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * scales[(i / width) % k];
}
}  // namespace

void gate_up(const MatrixShard& g, const MatrixShard& u, const distributed::ActivationContract& c,
             const GlobalQ81& a, float* gate_out, float* up_out, float* h, void* stream,
             const float* gs, const float* us) {
    detail::contract(c);
    detail::matrix(g, 1); detail::matrix(u, 1);
    detail::require(g.ggml_type != 20 && g.ggml_type != 42 && g.ggml_type == u.ggml_type &&
                    g.slice.global_shape == u.slice.global_shape &&
                    g.slice.range_start == u.slice.range_start && g.slice.range_end == u.slice.range_end,
                    "TP CUDA: gate/up ownership or codec mismatch");
    detail::image(a, int(g.slice.global_shape[0]));
    detail::require(gate_out && up_out && h && stream && gate_out != up_out && h != gate_out && h != up_out,
                    "TP CUDA: missing/overlapping gate/up buffers or ordered stream");
    const auto rb = iq_row_bytes(g.ggml_type, a.width);
    detail::require(g.row_stride == rb && u.row_stride == rb, "TP CUDA: gate/up rows must be contiguous");
    const int nout = int(g.slice.range_end - g.slice.range_start);
    if (a.row_stride == std::size_t(a.width / 32) * 36) {
        iq_mmvq(g.ggml_type, g.data, a.data, gate_out, a.width, nout, a.rows, stream);
        iq_mmvq(u.ggml_type, u.data, a.data, up_out, a.width, nout, a.rows, stream);
    } else {
        for (int row = 0; row < a.rows; ++row) {
            const auto* x = static_cast<const uint8_t*>(a.data) + std::size_t(row) * a.row_stride;
            iq_mmvq(g.ggml_type, g.data, x, gate_out + std::size_t(row) * nout, a.width, nout, 1, stream);
            iq_mmvq(u.ggml_type, u.data, x, up_out + std::size_t(row) * nout, a.width, nout, 1, stream);
        }
    }
    const int64_t n = int64_t(a.rows) * nout;
    swiglu<<<unsigned((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(gate_out, up_out, h, n, gs, us);
    detail::check(cudaGetLastError());
}

void unpack_gathered(const distributed::ActivationContract& c, const float* wire, int rows,
                     float* logical, void* stream) {
    detail::contract(c);
    detail::require(c.gather == "all_gather_padded" && c.wire_slots > 0 && !c.valid_counts.empty() &&
                    rows > 0 && wire && logical && wire != logical && stream, "TP CUDA: invalid padded gather");
    int64_t width = 0;
    for (auto n : c.valid_counts) {
        detail::require(n >= 0 && n <= c.wire_slots, "TP CUDA: invalid gather valid count");
        width += n;
    }
    detail::require(width > 0, "TP CUDA: empty gather");
    int64_t offset = 0;
    for (std::size_t rank = 0; rank < c.valid_counts.size(); ++rank) {
        const auto n = c.valid_counts[rank];
        if (n) detail::check(cudaMemcpy2DAsync(logical + offset, std::size_t(width) * sizeof(float),
            wire + rank * c.wire_slots, c.valid_counts.size() * std::size_t(c.wire_slots) * sizeof(float),
            std::size_t(n) * sizeof(float), rows, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
        offset += n;
    }
}

void finish_projection(float* reduced, int rows, int width, const float* scale, const float* bias, void* stream) {
    detail::require(reduced && rows > 0 && rows <= 65535 && width > 0 && stream,
                    "TP CUDA: invalid finish buffers/shape");
    const int64_t n = int64_t(rows) * width;
    finish<<<unsigned((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(reduced, n, width, scale, bias);
    detail::check(cudaGetLastError());
}

void combine(const float* parts, const float* weights, const float* scales, const float* shared,
             float* output, int rows, int width, int k, float* scratch, void* stream) {
    detail::require(parts && weights && output && rows > 0 && rows <= 65535 && width > 0 && k > 0 && k <= 15 && stream &&
                    output != parts && output != weights && output != shared,
                    "TP CUDA: invalid native combine");
    if (scales) {
        detail::require(scratch && scratch != parts && scratch != output && scratch != shared &&
                        scratch != weights && scratch != scales,
                        "TP CUDA: missing distinct scale scratch");
        const int64_t n = int64_t(rows) * width * k;
        scale_parts<<<unsigned((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(parts, scales, scratch, n, width, k);
        detail::check(cudaGetLastError());
        parts = scratch;
    }
    native_moe_combine_multi(parts, weights, shared, output, width, k, rows, stream);
}
}  // namespace strata::kernels::tp::cuda
