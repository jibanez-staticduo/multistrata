#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::tp::cuda {

struct GdnTpMap {
    int state_size = 128;
    int global_key_heads = 16;
    int global_value_heads = 48;
    int conv_taps = 4;
    int key_begin = 0;
    int key_end = 0;

    int key_heads() const { return key_end - key_begin; }
    int value_heads() const { return key_heads() * (global_value_heads / global_key_heads); }
    int channels() const { return (2 * key_heads() + value_heads()) * state_size; }
    int global_channels() const { return (2 * global_key_heads + global_value_heads) * state_size; }
    std::size_t state_elements() const { return std::size_t(state_size) * value_heads() * state_size; }
    std::size_t history_elements() const { return std::size_t(channels()) * (conv_taps - 1); }
};

struct GdnConstBuffer { const float* data; std::size_t elements; };
struct GdnBuffer { float* data; std::size_t elements; };
struct GdnTpState { GdnBuffer recurrent; GdnBuffer convolution; };

// All operations require valid maps, nonnull ordered streams, aligned device
// spans of the documented sizes, and disjoint source/destination buffers.
// No operation allocates, synchronizes, communicates, or owns any buffer.
// Invalid host contracts throw; CUDA launch failures throw runtime_error.
void tp_gdn_validate_map(const GdnTpMap& map);

// Qkv: [rows, channels, inner], q then k then v. inner=1 for activations,
// inner=4 for conv weights, inner=3 for native/fused channel-major history.
// Values: [rows, value_heads, inner], inner=1 for gate/beta/dt/ssm_a,
// inner=128 for z/output, or projection row width for value-headed weights.
// Recurrent: [128, value_heads, 128], rows=1, inner=128.
// Local values are [k0:k1], [HK+k0:HK+k1], [2*HK+k0:2*HK+k1].
// Thus key_begin + local_value % local_HK == global_value % global_HK.
enum class GdnTpLayout { Qkv, Values, Recurrent };
void tp_gdn_compact(const GdnTpMap& map, GdnTpLayout layout, int rows, int inner,
                    GdnConstBuffer global, GdnBuffer local, void* stream);
// Writes only owned global entries. Other ranks' entries are preserved.
void tp_gdn_scatter(const GdnTpMap& map, GdnTpLayout layout, int rows, int inner,
                    GdnConstBuffer local, GdnBuffer global, void* stream);

// Input [rows,3,wire_slots] from an equal-count padded all-gather. Padding
// is ignored. Output [rows,48,128] in global order, required before global
// Q8_1 quantization and the replicated ssm_out projection. Maps are rank order.
void tp_gdn_scatter_gathered(const GdnTpMap (&maps)[3], int rows, int wire_slots,
                            GdnConstBuffer gathered, GdnBuffer global, void* stream);

// Replicated ssm_out fallback: project the reconstructed global activation;
// only rank 0 contributes that complete result to the residual all-reduce.
// This copies rank 0's projection or zeros other ranks' contribution, without
// reading their projection. It never sums three complete projection outputs.
void tp_gdn_replicated_output_contribution(int rank, GdnConstBuffer projection,
                                          GdnBuffer contribution, int elements, void* stream);

void tp_gdn_reset(const GdnTpMap& map, GdnTpState state, void* stream);
void tp_gdn_copy_state(const GdnTpMap& map, GdnTpState destination,
                       const GdnTpState& source, void* stream); // snapshot or restore

enum class GdnTpArithmetic { Native, FusedSigmoid, FusedSilu };
struct GdnTpWorkspace { GdnBuffer conv_raw; GdnBuffer normalized_qkv; GdnBuffer readout; };

// One token BETWEEN the projections. qkv is compact local [channels], gate
// and beta are already preprocessed local [value_heads], z [value_heads,128],
// conv_weights [channels,4], gamma [128]. State/history are updated in place.
// Native uses native_gdn_* operators. Fused paths use fused_gdn_* operators;
// arithmetic is selected explicitly, never by process-global flags.
void tp_gdn_token(const GdnTpMap& map, GdnTpArithmetic arithmetic,
                   GdnTpState state, GdnConstBuffer qkv, GdnConstBuffer conv_weights,
                   GdnConstBuffer gate, GdnConstBuffer beta, GdnConstBuffer z,
                   GdnConstBuffer gamma, float epsilon, GdnTpWorkspace workspace,
                   GdnBuffer output, void* stream);

} // namespace strata::kernels::tp::cuda
