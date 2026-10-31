#include "strata/kernels/tp_gdn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::kernels::tp::cuda {
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void span(GdnConstBuffer b, std::size_t n) {
    const auto p = reinterpret_cast<std::uintptr_t>(b.data);
    require(b.data && p % alignof(float) == 0 && b.elements >= n && n <= (UINTPTR_MAX - p) / sizeof(float),
            "TP GDN: invalid or undersized aligned device span");
}
GdnConstBuffer view(GdnBuffer b) { return {b.data, b.elements}; }
void separate(GdnConstBuffer a, std::size_t na, GdnConstBuffer b, std::size_t nb) {
    span(a, na); span(b, nb);
    const auto ap = reinterpret_cast<std::uintptr_t>(a.data), bp = reinterpret_cast<std::uintptr_t>(b.data);
    require(ap >= bp + nb * sizeof(float) || bp >= ap + na * sizeof(float), "TP GDN: overlapping device spans");
}
void ordered(const GdnTpMap& m, void* stream) { tp_gdn_validate_map(m); require(stream, "TP GDN: ordered stream required"); }
__device__ int global_value(GdnTpMap m, int head) {
    const int nk = m.key_end - m.key_begin;
    return (head / nk) * m.global_key_heads + m.key_begin + head % nk;
}
__device__ int global_channel(GdnTpMap m, int channel) {
    const int nk = m.key_end - m.key_begin, s = m.state_size;
    if (channel < 2 * nk * s)
        return (channel / (nk * s)) * m.global_key_heads * s + m.key_begin * s + channel % (nk * s);
    const int v = channel - 2 * nk * s;
    return 2 * m.global_key_heads * s + global_value(m, v / s) * s + v % s;
}
__global__ void transfer(GdnTpMap m, GdnTpLayout layout, int inner, std::size_t count,
                          const float* source, float* destination, bool scatter) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int nk = m.key_end - m.key_begin, nv = nk * (m.global_value_heads / m.global_key_heads);
    const int local_width = layout == GdnTpLayout::Qkv ? (2 * nk + nv) * m.state_size : nv;
    const int global_width = layout == GdnTpLayout::Qkv ? (2 * m.global_key_heads + m.global_value_heads) * m.state_size : m.global_value_heads;
    const auto item = i / inner;
    const int head = int(item % local_width);
    const int mapped = layout == GdnTpLayout::Qkv ? global_channel(m, head) : global_value(m, head);
    const auto gi = (item / local_width * global_width + mapped) * inner + i % inner;
    if (scatter) destination[gi] = source[i]; else destination[i] = source[gi];
}
std::size_t extent(const GdnTpMap& m, GdnTpLayout layout, int rows, int inner, bool local) {
    require(rows > 0 && inner > 0, "TP GDN: positive transfer shape required");
    require(layout == GdnTpLayout::Qkv || layout == GdnTpLayout::Values || layout == GdnTpLayout::Recurrent,
            "TP GDN: unknown transfer layout");
    if (layout == GdnTpLayout::Recurrent) require(rows == 1 && inner == m.state_size, "TP GDN: recurrent transfer shape");
    const std::size_t width = layout == GdnTpLayout::Qkv ? (local ? m.channels() : m.global_channels()) :
                             (local ? m.value_heads() : m.global_value_heads);
    const std::size_t outer = layout == GdnTpLayout::Recurrent ? m.state_size : rows;
    require(width <= std::numeric_limits<std::size_t>::max() / outer / std::size_t(inner), "TP GDN: transfer overflow");
    const auto result = outer * width * inner;
    require(result <= std::size_t(std::numeric_limits<int>::max()), "TP GDN: transfer exceeds launch bound");
    return result;
}
void move(const GdnTpMap& m, GdnTpLayout layout, int rows, int inner, GdnConstBuffer source,
          GdnBuffer destination, bool scatter, void* stream) {
    ordered(m, stream);
    const auto ln = extent(m, layout, rows, inner, true), gn = extent(m, layout, rows, inner, false);
    separate(source, scatter ? ln : gn, view(destination), scatter ? gn : ln);
    transfer<<<unsigned((ln + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        m, layout, inner, ln, source.data, destination.data, scatter);
    check(cudaGetLastError());
}
struct Maps { GdnTpMap map[3]; };
__global__ void unpack(Maps maps, int wire, std::size_t count, const float* gathered, float* global) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto m = maps.map[0];
    const int col = int(i % m.state_size), head = int(i / m.state_size % m.global_value_heads);
    const auto row = i / (m.state_size * m.global_value_heads);
    const int key = head % m.global_key_heads;
    int rank = key < maps.map[0].key_end ? 0 : key < maps.map[1].key_end ? 1 : 2;
    const auto r = maps.map[rank];
    const int lh = (head / m.global_key_heads) * (r.key_end - r.key_begin) + key - r.key_begin;
    global[i] = gathered[(row * 3 + rank) * wire + lh * m.state_size + col];
}
void state_spans(const GdnTpMap& m, GdnTpState s) {
    separate(view(s.recurrent), m.state_elements(), view(s.convolution), m.history_elements());
}
}

void tp_gdn_validate_map(const GdnTpMap& m) {
    require(m.state_size == 128 && m.global_key_heads == 16 && m.global_value_heads == 48 && m.conv_taps == 4 &&
            m.key_begin >= 0 && m.key_end > m.key_begin && m.key_end <= m.global_key_heads,
            "TP GDN: expected global S128 HK16 HV48 conv4 and nonempty key range");
}
void tp_gdn_compact(const GdnTpMap& m, GdnTpLayout l, int rows, int inner, GdnConstBuffer g, GdnBuffer local, void* s) {
    move(m, l, rows, inner, g, local, false, s);
}
void tp_gdn_scatter(const GdnTpMap& m, GdnTpLayout l, int rows, int inner, GdnConstBuffer local, GdnBuffer g, void* s) {
    move(m, l, rows, inner, local, g, true, s);
}
void tp_gdn_scatter_gathered(const GdnTpMap (&maps)[3], int rows, int wire, GdnConstBuffer in, GdnBuffer out, void* s) {
    int end = 0;
    for (const auto& m : maps) {
        ordered(m, s);
        require(m.key_begin == end && wire >= m.value_heads() * m.state_size, "TP GDN: invalid rank order or gather padding");
        end = m.key_end;
    }
    require(end == 16 && rows > 0 && wire > 0, "TP GDN: gather must cover all global heads");
    const auto n = extent(maps[0], GdnTpLayout::Values, rows, 128, false);
    const auto ni = std::size_t(rows) * 3 * wire;
    separate(in, ni, view(out), n);
    Maps by_value{{maps[0], maps[1], maps[2]}};
    unpack<<<unsigned((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(s)>>>(by_value, wire, n, in.data, out.data);
    check(cudaGetLastError());
}
void tp_gdn_replicated_output_contribution(int rank, GdnConstBuffer projection, GdnBuffer output, int n, void* s) {
    require(rank >= 0 && rank < 3 && n > 0 && s, "TP GDN: invalid replicated output contribution");
    span(view(output), n);
    if (rank == 0) {
        separate(projection, n, view(output), n);
        check(cudaMemcpyAsync(output.data, projection.data, std::size_t(n) * sizeof(float), cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(s)));
    } else check(cudaMemsetAsync(output.data, 0, std::size_t(n) * sizeof(float), static_cast<cudaStream_t>(s)));
}
void tp_gdn_reset(const GdnTpMap& m, GdnTpState state, void* s) {
    ordered(m, s); state_spans(m, state);
    check(cudaMemsetAsync(state.recurrent.data, 0, m.state_elements() * sizeof(float), static_cast<cudaStream_t>(s)));
    check(cudaMemsetAsync(state.convolution.data, 0, m.history_elements() * sizeof(float), static_cast<cudaStream_t>(s)));
}
void tp_gdn_copy_state(const GdnTpMap& m, GdnTpState dst, const GdnTpState& src, void* s) {
    ordered(m, s); state_spans(m, dst); state_spans(m, src);
    const GdnBuffer destinations[] = {dst.recurrent, dst.convolution};
    const GdnBuffer sources[] = {src.recurrent, src.convolution};
    const std::size_t sizes[] = {m.state_elements(), m.history_elements()};
    for (int d = 0; d < 2; ++d)
        for (int r = 0; r < 2; ++r) separate(view(destinations[d]), sizes[d], view(sources[r]), sizes[r]);
    check(cudaMemcpyAsync(dst.recurrent.data, src.recurrent.data, m.state_elements() * sizeof(float), cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(s)));
    check(cudaMemcpyAsync(dst.convolution.data, src.convolution.data, m.history_elements() * sizeof(float), cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(s)));
}
void tp_gdn_token(const GdnTpMap& m, GdnTpArithmetic arithmetic, GdnTpState state,
                   GdnConstBuffer qkv, GdnConstBuffer weights, GdnConstBuffer gate, GdnConstBuffer beta,
                   GdnConstBuffer z, GdnConstBuffer gamma, float eps, GdnTpWorkspace w, GdnBuffer output, void* s) {
    ordered(m, s);
    require(std::isfinite(eps) && eps > 0 && (arithmetic == GdnTpArithmetic::Native || arithmetic == GdnTpArithmetic::FusedSigmoid ||
            arithmetic == GdnTpArithmetic::FusedSilu), "TP GDN: invalid epsilon/arithmetic");
    const std::size_t c = m.channels(), v = m.value_heads() * 128;
    const GdnConstBuffer writable[] = {view(state.recurrent), view(state.convolution), view(w.conv_raw), view(w.normalized_qkv), view(w.readout), view(output)};
    const std::size_t sizes[] = {m.state_elements(), m.history_elements(), c, c, v, v};
    const GdnConstBuffer inputs[] = {qkv, weights, gate, beta, z, gamma};
    const std::size_t input_sizes[] = {c, c * 4, std::size_t(m.value_heads()), std::size_t(m.value_heads()), v, 128};
    for (int i = 0; i < 6; ++i) {
        span(writable[i], sizes[i]); span(inputs[i], input_sizes[i]);
        for (int j = 0; j < i; ++j) separate(writable[i], sizes[i], writable[j], sizes[j]);
        for (int j = 0; j < 6; ++j) separate(writable[i], sizes[i], inputs[j], input_sizes[j]);
    }
    auto* h = w.normalized_qkv.data;
    const auto* k = h + m.key_heads() * 128;
    const auto* values = k + m.key_heads() * 128;
    if (arithmetic == GdnTpArithmetic::Native) {
        native_gdn_conv_silu(state.convolution.data, qkv.data, weights.data, w.conv_raw.data, h, c, 4, s);
        native_gdn_l2_norm(h, 2 * m.key_heads(), 128, eps, s);
        native_gdn_step(state.recurrent.data, h, k, values, gate.data, beta.data, w.readout.data,
                        {128, m.key_heads(), m.value_heads()}, s);
        native_gdn_out_norm(w.readout.data, z.data, gamma.data, output.data, m.value_heads(), 128, eps, s);
    } else {
        fused_gdn_conv_l2(state.convolution.data, qkv.data, weights.data, h, int(c), 2 * m.key_heads(), eps, s);
        if (arithmetic == GdnTpArithmetic::FusedSilu)
            fused_gdn_step_norm_silu(state.recurrent.data, h, k, values, gate.data, beta.data, z.data, gamma.data, eps,
                                     output.data, m.key_heads(), m.value_heads(), s);
        else fused_gdn_step_norm(state.recurrent.data, h, k, values, gate.data, beta.data, z.data, gamma.data, eps,
                                 output.data, m.key_heads(), m.value_heads(), s);
    }
}
} // namespace strata::kernels::tp::cuda
