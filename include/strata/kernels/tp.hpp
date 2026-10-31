// include/strata/kernels/tp.hpp - the shard-aware projection primitives of the TP3 reference path.
//
// What one rank computes when a projection is cut across ranks, in the smallest readable CPU scalar form, so
// that the partition rules can be checked where there is no GPU.  Four rules hold it together, and each comes
// out of the contract rather than out of a local choice:
//
//   * the shape comes from the TensorSlice.  Nothing here names 640, 2560 or a head count as a local constant:
//     a slice of another shape computes, and a slice whose cut would split a quantized weight block is
//     refused instead of being requantized to fit.
//   * the activation comes from the original ActivationContract.  A projection consumes the FULL logical input
//     width, so a rank that hands it its own fragment is refused, not silently answered.
//   * a down projection writes a FULL-WIDTH partial, and the partials are summed exactly once, before the next
//     replicated operation.  A replicated term that rides on every partial before the sum is counted `world`
//     times, and nothing downstream can tell.
//   * an activation is quantized once, on the grid anchored at lane 0 of the logical vector, after the gather.
//     The same lane coded against a rank-local anchor is a different number.
//
// Shapes follow the inventory, which follows GGUF: axis 0 is the contiguous axis, the projection's INPUT and the
// quantization blocks run along it, and outputs are the later axis.  A weight plane is therefore
// `plane[out * row_stride + in]`, and `row_stride` is the ORIGINAL stride: a shard keeps the pack's row and
// drops the lanes it does not own.
//
// gate/up keeps the monolithic SwiGLU, not a distributed lookalike of it:
//     ff[t][r] = (g / (1.f + std::exp(-g))) * u
// which is `src/kernels/cpu/native_expert.cpp` L107, L426 and L511, and the same expression the device
// projections share in `include/strata/kernels/swiglu.cuh` (`swilu_fast`, which differs only in the device
// intrinsics it may use).  The monolithic K-quant path folds the row scales in front of the silu
// (`native_expert.cpp` L509-510: `g *= sg; u *= su;`); with FP32 slabs those scales are 1, so this reference
// leaves them out rather than inventing a third form.
//
// The caller owns every buffer and passes the execution stream; the CPU reference path ignores it.  No
// allocation, no global state, no device dependency.
#pragma once

#include "strata/distributed/shard_plan.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace strata::kernels::tp {

using strata::distributed::ActivationContract;
using strata::distributed::TensorSlice;

/// Why a primitive refused to read a buffer it cannot trust.  A refusal is never a partial result.
enum class Refusal {
    None = 0,
    NotAProjection,    // not a two-axis weight slice
    ShapeMismatch,     // local_shape does not follow from global_shape and the owned range
    CutOnWeightBlock,  // an input cut that would split a quantized weight block
    BadRange,          // an empty or out-of-range owned range
    SlabTooNarrow,     // a weight plane row shorter than the projection's input width
    ActivationWidth,   // the activation image is not the width this projection consumes
    ContractMismatch,  // the activation image does not match the contract it came through
    BadContract,       // a contract or block width that cannot be laid out as given
    NullBuffer,        // a required pointer is missing
};

inline const char* refusal_text(Refusal r) {
    switch (r) {
        case Refusal::None: return "none";
        case Refusal::NotAProjection: return "the slice is not a two-axis projection";
        case Refusal::ShapeMismatch: return "local_shape disagrees with global_shape and the owned range";
        case Refusal::CutOnWeightBlock: return "the cut would split a quantized weight block";
        case Refusal::BadRange: return "the owned range is empty or outside the global shape";
        case Refusal::SlabTooNarrow: return "a weight plane row is narrower than the slice requires";
        case Refusal::ActivationWidth: return "the activation is not the width this projection consumes";
        case Refusal::ContractMismatch: return "the activation image does not match its contract";
        case Refusal::BadContract: return "the contract has no layout that fits the vector";
        case Refusal::NullBuffer: return "a required buffer is null";
    }
    return "unknown";
}

/// What a slice says about its projection, read out of the slice itself.
struct ProjectionView {
    int64_t n_in = 0;          // the FULL logical input width (global axis 0): every rank needs all of it
    int64_t n_out = 0;         // the FULL output width (global axis 1)
    int64_t first = 0;         // first owned lane on the cut axis
    int64_t last = 0;          // one past the last owned lane
    int64_t owned = 0;         // last - first
    bool cuts_output = false;  // true: gate/up owns output rows; false: down owns input columns
    int64_t block_size = 0;    // weight block width along the contiguous axis; 0 = not block-quantized
};

/// Read a projection out of a slice.  An output cut (axis 1) copies whole input rows, so it cannot split a
/// weight block; an input cut (axis 0) runs along the block axis and has to land on block boundaries.
inline Refusal describe(const TensorSlice& slice, ProjectionView& out) {
    if (slice.global_shape.size() != 2 || slice.local_shape.size() != 2) return Refusal::NotAProjection;
    if (slice.axis != 0 && slice.axis != 1) return Refusal::NotAProjection;
    const int64_t n_in = slice.global_shape[0];
    const int64_t n_out = slice.global_shape[1];
    if (n_in <= 0 || n_out <= 0) return Refusal::NotAProjection;

    ProjectionView v;
    v.n_in = n_in;
    v.n_out = n_out;
    v.cuts_output = slice.axis == 1;
    v.first = slice.range_start;
    v.last = slice.range_end;
    v.block_size = slice.block_size > 1 ? slice.block_size : 0;
    if (v.first < 0 || v.last <= v.first) return Refusal::BadRange;
    if (v.last > (v.cuts_output ? n_out : n_in)) return Refusal::BadRange;
    v.owned = v.last - v.first;
    if (slice.local_shape[0] != (v.cuts_output ? n_in : v.owned) ||
        slice.local_shape[1] != (v.cuts_output ? v.owned : n_out))
        return Refusal::ShapeMismatch;
    // Only an input cut runs along the block axis.  An output cut copies whole rows, so it carries whole
    // blocks with them and cannot split one.
    if (!v.cuts_output && v.block_size && (v.first % v.block_size || v.last % v.block_size))
        return Refusal::CutOnWeightBlock;
    out = v;
    return Refusal::None;
}

/// One shard's weight plane, in GGUF order: `plane[out * row_stride + in]`.  `data` points at the shard's own
/// bytes, so an output cut starts at its first owned row and an input cut at lane `first` of every row.
struct Weights {
    const float* data = nullptr;
    int64_t row_stride = 0;  // the ORIGINAL stride in floats, never a packed one
};

/// One caller-owned activation image.  `width` is the buffer's row width and `valid` the number of lanes that
/// belong to the logical vector: a gather buffer is sized by the caller, so its tail is workspace, and a lane
/// must not reach the model through it.
struct Activation {
    const float* data = nullptr;
    int64_t width = 0;
    int64_t valid = 0;
    int64_t tokens = 0;
    int64_t stride = 0;  // floats between token rows
};

namespace detail {

inline float dot(const float* a, const float* b, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// ggml's `nearest_int`: round-half-to-EVEN through the 12582912.0f magic, as transcribed in
// `src/kernels/quantize_act_parity.cpp` L139.  `roundf` rounds half AWAY from zero; the two differ only on
// exact ties, which is exactly why a wrong one survives any amount of end-to-end testing.
inline int nearest_int(float fval) {
    const float val = fval + 12582912.0f;
    int i = 0;
    std::memcpy(&i, &val, sizeof i);
    return (i & 0x007fffff) - 0x00400000;
}

// The SwiGLU of the monolithic path, one copy, same coefficients.  See the file header for the references.
inline float swiglu(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }

// Where rank r's fragment begins in the logical vector.  The ranges tile the axis in increasing order - the
// planner's own invariant - so it is the sum of the valid counts before it.
inline int64_t logical_offset(const ActivationContract& c, int r) {
    int64_t s = 0;
    for (int i = 0; i < r; ++i) s += c.valid_counts[(size_t)i];
    return s;
}

// The image width a projection is allowed to read: the full logical input, and nothing smaller.
inline Refusal check_activation(const ProjectionView& v, const ActivationContract& c, const Activation& a) {
    if (!a.data) return Refusal::NullBuffer;
    if (a.valid != v.n_in) return Refusal::ActivationWidth;
    if (a.width < a.valid) return Refusal::ActivationWidth;
    if (a.stride < a.valid) return Refusal::ActivationWidth;
    if (a.tokens <= 0) return Refusal::ActivationWidth;
    const int64_t world = (int64_t)c.valid_counts.size();
    if (world <= 0) return Refusal::BadContract;
    if (c.gather == "all_gather_padded") {
        // The image is the fragments concatenated in rank order, so its valid lanes are the contract's counts
        // summed.  A rank that hands over its own piece fails here rather than answering for a third of the
        // vector and looking plausible.
        int64_t sum = 0;
        for (const int64_t n : c.valid_counts) {
            if (n <= 0 || n > c.wire_slots) return Refusal::BadContract;
            sum += n;
        }
        if (sum != a.valid) return Refusal::ContractMismatch;
    } else if (c.gather == "none") {
        // A vector that needs no gather is whole on every rank, so every count is the whole width and there is
        // no padding to carry.
        for (const int64_t n : c.valid_counts)
            if (n != a.valid) return Refusal::ContractMismatch;
    } else {
        return Refusal::BadContract;  // all_reduce_sum is a reduction, not a way to widen a vector
    }
    return Refusal::None;
}

}  // namespace detail

/// gate/up over owned output rows, with the SwiGLU the monolithic path uses.
///
/// Both slices must own the same rows: a gate row multiplied by another row's `up` is not a row of anything.
/// `ff` receives `owned` values per token, `ff_stride` apart - the rank's piece of the logical intermediate,
/// which the next primitive gathers rather than re-derives.
inline Refusal gate_up_swiglu(const TensorSlice& gate_slice, const Weights& gate, const TensorSlice& up_slice,
                             const Weights& up, const ActivationContract& contract, const Activation& act,
                             float* ff, int64_t ff_stride, void* /*stream*/ = nullptr) {
    ProjectionView g, u;
    Refusal r = describe(gate_slice, g);
    if (r == Refusal::None) r = describe(up_slice, u);
    if (r == Refusal::None && (g.first != u.first || g.last != u.last ||
                              g.n_in != u.n_in || g.n_out != u.n_out ||
                              g.cuts_output != u.cuts_output))
        r = Refusal::ShapeMismatch;  // gate and up must own the same rows, or the product is not one row
    if (r == Refusal::None && !g.cuts_output) r = Refusal::NotAProjection;
    if (r == Refusal::None && (!gate.data || !up.data || !ff)) r = Refusal::NullBuffer;
    if (r == Refusal::None && (gate.row_stride < g.n_in || up.row_stride < u.n_in))
        r = Refusal::SlabTooNarrow;
    if (r == Refusal::None && ff_stride < g.owned) r = Refusal::ActivationWidth;
    if (r == Refusal::None) r = detail::check_activation(g, contract, act);
    if (r != Refusal::None) return r;
    for (int64_t t = 0; t < act.tokens; ++t) {
        const float* x = act.data + t * act.stride;
        float* row = ff + t * ff_stride;
        for (int64_t i = 0; i < g.owned; ++i) {
            const float gv = detail::dot(gate.data + i * gate.row_stride, x, g.n_in);
            const float uv = detail::dot(up.data + i * up.row_stride, x, g.n_in);
            row[i] = detail::swiglu(gv, uv);
        }
    }
    return Refusal::None;
}

/// down over owned input columns, writing a FULL-WIDTH partial.
///
/// The partial is `n_out` wide although the rank owns only `owned` of the input lanes, so adding the ranks'
/// partials is the same arithmetic as one whole projection.  `act` is the one global activation image: the
/// rank reads lanes [first,last) of it, never a re-quantized copy of its own piece.
inline Refusal down_partial(const TensorSlice& slice, const Weights& w, const ActivationContract& contract,
                           const Activation& act, float* partial, int64_t partial_stride,
                           void* /*stream*/ = nullptr) {
    ProjectionView d;
    Refusal r = describe(slice, d);
    if (r == Refusal::None && d.cuts_output) r = Refusal::NotAProjection;
    if (r == Refusal::None && (!w.data || !partial)) r = Refusal::NullBuffer;
    if (r == Refusal::None && w.row_stride < d.n_in) r = Refusal::SlabTooNarrow;
    if (r == Refusal::None && partial_stride < d.n_out) r = Refusal::ActivationWidth;
    if (r == Refusal::None) r = detail::check_activation(d, contract, act);
    if (r != Refusal::None) return r;
    for (int64_t t = 0; t < act.tokens; ++t) {
        const float* x = act.data + t * act.stride;
        float* row = partial + t * partial_stride;
        for (int64_t j = 0; j < d.n_out; ++j) {
            const float* wr = w.data + j * w.row_stride + d.first;
            row[j] = detail::dot(wr, x + d.first, d.owned);
        }
    }
    return Refusal::None;
}

/// The one reduction: out[j] = sum over ranks of partial_rank[j].  It is the only place a TP projection is
/// allowed to add its pieces together, and every rank runs it, or the collective deadlocks.
inline Refusal reduce_partials(const float* const* partials, int world, int64_t width, float* out,
                               void* /*stream*/ = nullptr) {
    if (!partials || world <= 0 || width <= 0 || !out) return Refusal::NullBuffer;
    for (int r = 0; r < world; ++r)
        if (!partials[r]) return Refusal::NullBuffer;
    for (int64_t j = 0; j < width; ++j) {
        float s = 0.0f;
        for (int r = 0; r < world; ++r) s += partials[r][j];
        out[j] = s;
    }
    return Refusal::None;
}

/// A replicated operation applied ONCE, to the reduced row: out[j] = (sum[j] + bias[j]) * scale[j].  Either
/// term may be null.  This is where a global bias or a global scale belongs.
inline Refusal apply_replicated(const float* sum, const float* bias, const float* scale, int64_t width,
                                float* out, void* /*stream*/ = nullptr) {
    if (!sum || !out || width <= 0) return Refusal::NullBuffer;
    for (int64_t j = 0; j < width; ++j) {
        float v = sum[j];
        if (bias) v += bias[j];
        if (scale) v *= scale[j];
        out[j] = v;
    }
    return Refusal::None;
}

/// Assemble the original logical vector from the ranks' fragments.
///
/// Unequal shards (256/192/192) do not fit a uniform collective, so the contract names one max-sized WIRE slot
/// width (`wire_slots`) and the explicit valid counts: that is what travels.  What the projections consume is
/// the concatenation, in the order the ranges tile the axis - the planner's own invariant - because a lane of
/// the logical vector has to sit at its logical index for the block grid and for a rank's own [first,last)
/// window to mean the same thing.  So the padding is a wire detail and this writes no gaps.
///
/// The caller sizes `out`, and the reference path sizes it to a whole number of quantizer blocks so the
/// quantizer has a grid to sit on (a 640-wide vector in a 768-wide buffer, say); from `logical_width` up to
/// `buffer_width` everything is zeroed, because that tail is workspace and a stale lane must not set a scale.
inline Refusal gather_padded(const ActivationContract& c, const float* const* frags, float* out,
                             int64_t buffer_width, int64_t& logical_width, void* /*stream*/ = nullptr) {
    const int world = (int)c.valid_counts.size();
    if (!frags || !out || world <= 0 || c.wire_slots <= 0 || buffer_width <= 0) return Refusal::NullBuffer;
    int64_t total = 0;
    for (int r = 0; r < world; ++r) {
        const int64_t n = c.valid_counts[(size_t)r];
        if (n <= 0 || n > c.wire_slots) return Refusal::BadContract;
        if (!frags[r]) return Refusal::NullBuffer;
        total += n;
    }
    if (total > buffer_width) return Refusal::BadContract;
    for (int r = 0; r < world; ++r) {
        const int64_t n = c.valid_counts[(size_t)r];
        float* dst = out + detail::logical_offset(c, r);
        std::memcpy(dst, frags[r], (size_t)n * sizeof(float));
    }
    for (int64_t i = total; i < buffer_width; ++i) out[i] = 0.0f;
    logical_width = total;
    return Refusal::None;
}

/// Quantize an activation ONCE, on the grid anchored at lane 0, and hand back the values the dot consumes.
///
/// That anchor is the whole point.  With the agreed FFN layout the block [256,512) is completed by lanes two
/// ranks away, so a quantizer anchored at a rank's own first lane reads a different maximum and codes the same
/// lane with a different number: not a smaller version of the same answer, a different answer.  `width` is the
/// buffer and `valid` the logical lanes; `width` must be a whole number of blocks (the same structural rule
/// `quantize_act.hpp` states for Q8_0 and Q8_K) and only lanes below `valid` set a block's scale, so padding
/// stays workspace.  The rule is ggml's: the scale comes from the largest magnitude, `-127` and not `-128`, the
/// maximum is the SIGNED value at the position of that magnitude, and the code is `nearest_int(iscale * x)`.
inline Refusal quantize_activation(const ActivationContract& c, const float* x, int64_t width, int64_t valid,
                                   float* out, void* /*stream*/ = nullptr) {
    if (!x || !out || width <= 0) return Refusal::NullBuffer;
    if (valid <= 0 || valid > width) return Refusal::ActivationWidth;
    if (c.quantizer == "none" || c.block_size <= 1) {
        for (int64_t i = 0; i < valid; ++i) out[i] = x[i];
        for (int64_t i = valid; i < width; ++i) out[i] = 0.0f;
        return Refusal::None;
    }
    if (width % c.block_size) return Refusal::BadContract;
    for (int64_t b = 0; b * c.block_size < width; ++b) {
        const int64_t first = b * c.block_size;
        const int64_t last = first + c.block_size;
        float max = 0.0f, amax = 0.0f;
        for (int64_t i = first; i < last && i < valid; ++i) {
            const float ax = std::fabs(x[i]);
            if (ax > amax) { amax = ax; max = x[i]; }  // strictly greater: ties keep the FIRST maximum
        }
        if (amax == 0.0f) {
            for (int64_t i = first; i < last; ++i) out[i] = 0.0f;
            continue;
        }
        const float iscale = -127.0f / max;
        const float d = 1.0f / iscale;
        for (int64_t i = first; i < last; ++i)
            out[i] = i < valid ? (float)detail::nearest_int(iscale * x[i]) * d : 0.0f;
    }
    return Refusal::None;
}

}  // namespace strata::kernels::tp
