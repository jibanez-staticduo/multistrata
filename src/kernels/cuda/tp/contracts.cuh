#pragma once

#include "strata/kernels/tp_cuda.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::tp::cuda::detail {
inline void require(bool ok, const char* why) {
    if (!ok) throw std::invalid_argument(why);
}
inline void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
inline bool span(std::size_t capacity, std::size_t rows, std::size_t stride, std::size_t width) {
    return rows > 0 && width > 0 && stride >= width && capacity >= width &&
           (rows == 1 || stride <= (capacity - width) / (rows - 1));
}
inline void contract(const distributed::ActivationContract& c) {
    require(c.quantizer == "Q8_1" && c.block_size == 32 && c.reference == "global_reference",
            "TP CUDA requires an explicit global native Q8_1 contract");
    require(c.backend_policy == "cuda_native_expert_grouped" && c.layout == "contiguous",
            "TP CUDA requires the native CUDA expert backend and contiguous logical order");
    require(c.gather == "none" || c.gather == "all_gather_padded" || c.gather == "all_reduce_sum",
            "TP CUDA: unknown activation gather");
}
inline int block(int t) {
    switch (t) {
        case 16: case 17: case 18: case 21: case 22: case 23: return 256;
        case 20: return 32;
        case 42: return 64;
        default: throw std::invalid_argument("TP CUDA: unsupported original codec");
    }
}
inline const char* codec(int t) {
    switch (t) {
        case 16: return "IQ2_XXS"; case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS"; case 20: return "IQ4_NL";
        case 21: return "IQ3_S"; case 22: return "IQ2_S";
        case 23: return "IQ4_XS"; case 42: return "Q2_0";
        default: return "";
    }
}
inline void matrix(const MatrixShard& w, int axis) {
    const auto& s = w.slice;
    require(s.axis == axis && (s.global_shape.size() == 2 || s.global_shape.size() == 3) &&
            s.local_shape.size() == s.global_shape.size(), "TP CUDA: wrong projection axis/shape");
    for (std::size_t i = 0; i < s.global_shape.size(); ++i) {
        require(s.global_shape[i] > 0 && s.global_shape[i] <= std::numeric_limits<int>::max(),
                "TP CUDA: projection dimension out of range");
        require(s.local_shape[i] == (int(i) == axis ? s.range_end - s.range_start : s.global_shape[i]),
                "TP CUDA: local shape disagrees with ownership");
    }
    require(s.range_start >= 0 && s.range_end > s.range_start && s.range_end <= s.global_shape[axis],
            "TP CUDA: ownership out of range");
    require(s.rank >= 0 && std::size_t(s.rank) < s.valid_lanes.size() &&
            s.valid_lanes[std::size_t(s.rank)] == s.range_end - s.range_start,
            "TP CUDA: padded or missing owned valid-lane count");
    const int b = block(w.ggml_type);
    require(s.codec == codec(w.ggml_type) && s.block_size == b && !s.scale_kind.empty() && s.scale_ref.empty(),
            "TP CUDA: codec or embedded scale contract mismatch");
    require(s.global_shape[0] % b == 0 && (axis != 0 || (s.range_start % b == 0 && s.range_end % b == 0)),
            "TP CUDA: cut splits an original weight block");
    const auto global_rb = iq_row_bytes(w.ggml_type, s.global_shape[0]);
    require(s.byte_row_stride == int64_t(global_rb), "TP CUDA: original row stride mismatch");
    const auto rb = iq_row_bytes(w.ggml_type, s.local_shape[0]);
    require(w.row_stride >= rb, "TP CUDA: weight row too short");
    require(reinterpret_cast<std::uintptr_t>(w.data) % 2 == 0 && w.row_stride % 2 == 0,
            "TP CUDA: original weight blocks require two-byte alignment");
    const auto nout = std::size_t(s.local_shape[1]);
    require(span(w.bytes, nout, w.row_stride, rb) && w.data, "TP CUDA: weight span too short/null");
}
inline void image(const GlobalQ81& a, int width) {
    require(a.width == width && a.rows > 0 && a.rows <= 65535 && width > 0 && width % 32 == 0,
            "TP CUDA: activation must contain the full logical width");
    const auto rb = std::size_t(width / 32) * 36;
    require(reinterpret_cast<std::uintptr_t>(a.data) % 4 == 0 && a.row_stride % 4 == 0,
            "TP CUDA: Q8_1 blocks require four-byte alignment");
    require(a.data && span(a.bytes, std::size_t(a.rows), a.row_stride, rb),
            "TP CUDA: activation span too short/null");
}
}  // namespace strata::kernels::tp::cuda::detail
