#pragma once

#include "strata/distributed/shard_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <array>

namespace strata::kernels::tp::cuda {

// One expert's original GGUF plane. data starts at the owned output row (axis 1)
// or owned input block (axis 0). row_stride may retain the original GGUF stride
// or describe compact rank-pack rows. No weight/scale conversion is performed.
// data is one expert plane even when TensorSlice retains its third expert axis.
// The owned valid count must cover the local range; wire padding is not weight.
struct MatrixShard {
    distributed::TensorSlice slice;
    int ggml_type = -1;
    const void* data = nullptr;
    std::size_t row_stride = 0;
    std::size_t bytes = 0;
};

struct GlobalQ81 {
    const void* data = nullptr;
    int width = 0;
    int rows = 0;
    std::size_t row_stride = 0;
    std::size_t bytes = 0;
};

// This backend follows native_expert_grouped's CUDA Q8_1 contract, not the
// CPU Q8_K/Q8_0 contract recorded by the format inventory. Callers must select
// Q8_1 / block 32 / global_reference / cuda_native_expert_grouped with
// contiguous logical layout explicitly; other contracts are rejected.
std::size_t global_q81_bytes(int rows, int width);
GlobalQ81 quantize_global(const distributed::ActivationContract& contract,
                         const float* logical, int rows, int width, void* q81,
                         std::size_t capacity, void* stream);

// Original input quantization is shared by both projections. gate/up/h contain
// rows * owned_output floats, token-major. Scales, when supplied, are device
// scalars and multiply gate/up before the original native SwiGLU expression.
void gate_up(const MatrixShard& gate, const MatrixShard& up,
             const distributed::ActivationContract& contract, const GlobalQ81& input,
             float* gate_out, float* up_out, float* h, void* stream,
             const float* gate_scale = nullptr, const float* up_scale = nullptr);

// Input-sharded IQ4_NL / Q2_0. Reads owned blocks from a GLOBAL activation image
// and writes rows * global_output floats. Sum rank partials before any bias,
// expert scale or router weighting. Prefill uses the same MMVQ arithmetic per
// token; this is a component reference path, not a tensor-core GEMM backend.
void down(const MatrixShard& weights, const distributed::ActivationContract& contract,
          const GlobalQ81& activation, float* partial, void* stream);

// Exact ordinary native-MMVQ control: store the scale and integer-dot operand
// for each of at most two calls per warp lane, NOT rounded products/sums (the
// native loop contracts its product+accumulation to FMA). Local payload layout
// [rows][nout][call_slot=2][lane=32][scale,integer_dot=2], 128 floats/output.
// Gather three uniform payloads in rank-major order, then finish once. Only
// original widths <=1024 with <=64 dot calls and whole weight cuts qualify.
constexpr int down_lane_payload_width = 128;
void down_lanes(const MatrixShard&, const distributed::ActivationContract&,
                const GlobalQ81&, float* operands, void* stream);
void finish_down_lanes(const float* rank_major_operands, int ggml_type,
                       int input_width, int output_width, int rows,
                       std::array<uint64_t, 3> valid_input, float* output, void* stream);

// Gathered wire has [token][rank][wire_slots], h [token][sum(valid_counts)].
// Padding is discarded before the global quantizer; buffers must not overlap.
void unpack_gathered(const distributed::ActivationContract& contract, const float* wire,
                     int rows, float* logical, void* stream);

// Add one global scale and optional bias AFTER the all-reduce of partials.
void finish_projection(float* reduced, int rows, int width, const float* scale,
                       const float* bias, void* stream);

// Native top-k combine (k <= 15), optional shared contribution added exactly
// once. Expert scales multiply each part before router weights. Zero routes
// must provide a zero part, matching the monolithic native contract.
// All input/output spans and scaled_parts must be disjoint; no partial aliasing.
void combine(const float* parts, const float* router_weights, const float* expert_scales,
             const float* shared, float* output, int rows, int width, int k,
             float* scaled_parts, void* stream);

}  // namespace strata::kernels::tp::cuda
