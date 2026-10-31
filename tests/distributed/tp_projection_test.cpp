// tests/distributed/tp_projection_test.cpp - checks for the shard-aware projection primitives of the TP3
// reference path, in their CPU scalar form: gate/up with the monolithic SwiGLU, a down projection that writes
// a full-width partial, the one reduction, a replicated term applied once, a padded gather, and an activation
// quantized once on the global grid.
//
// The oracle is the monolithic path, never the planner.  Every expected number here is either written out by
// hand or computed by a whole-projection reference written independently in this file, so a planner bug and a
// primitive bug cannot agree with each other.  The references are:
//   * the SwiGLU of the monolithic path: `src/kernels/cpu/native_expert.cpp` L107, L426 and L511,
//     `ff[t][r] = (g / (1.f + std::exp(-g))) * u`.
//   * the activation quantizer: `src/kernels/quantize_act_parity.cpp` L139, the host transcription of
//     `quantize_row_q8_K_ref` (ggml-quants.c L2768) - the scale comes from the SIGNED value at the first
//     position of largest magnitude, `-127` and not `-128`, `nearest_int` rounds half to EVEN, and a block with
//     no lane to read codes as all zeros.
//   * the shapes and contracts: the Task 2 fixture in `tests/distributed/shard_plan_test.cpp` - FFN 640 wide
//     with the world-3 cuts at 256 and 448, `gate_up` Q8_K/256 over the whole residual, `down` Q8_0/32, and
//     the same down rerouted to the global reference path when its quantizer is Q8_K/256.
//
// Two things this does NOT cover, stated so nobody reads more into the green: the CUDA kernels themselves and
// NCCL.  These are the same arithmetic in scalar form on one core, so they say the partition rules are
// consistent, not that the device path implements them.  The CPU expression is also not bit-identical to the
// device one for every codec: `include/strata/kernels/swiglu.cuh` L27 computes its silu in double
// (`swilu_legacy`) and L37 uses `__fdividef` with `__expf` (`swilu_native`), so device rows may round
// differently from these.  No framework, no GPU: plain checks, printed results, non-zero exit if any fails.
#include "strata/kernels/tp.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using strata::distributed::ActivationContract;
using strata::distributed::TensorFamily;
using strata::distributed::TensorSlice;
using strata::kernels::tp::Activation;
using strata::kernels::tp::apply_replicated;
using strata::kernels::tp::describe;
using strata::kernels::tp::down_partial;
using strata::kernels::tp::gather_padded;
using strata::kernels::tp::gate_up_swiglu;
using strata::kernels::tp::ProjectionView;
using strata::kernels::tp::quantize_activation;
using strata::kernels::tp::reduce_partials;
using strata::kernels::tp::refusal_text;
using strata::kernels::tp::Refusal;
using strata::kernels::tp::Weights;

int g_checks = 0;
int g_failed = 0;
std::string g_test = "setup";

void begin(const std::string& name) {
    g_test = name;
    std::cout << "== " << name << "\n";
}

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << "\n";
}

template <typename A, typename B>
void check_eq(const A& got, const B& want, const std::string& what) {
    ++g_checks;
    if (got == want) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << ": got " << got << ", want " << want << "\n";
}

void check_refusal(Refusal got, Refusal want, const std::string& what) {
    ++g_checks;
    if (got == want) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << ": got " << refusal_text(got) << ", want "
              << refusal_text(want) << "\n";
}

// Two floats are the same answer here only when every bit matches.  A tolerance hides the exact thing these
// primitives are meant to get right, so this file uses one, in the single place where the summation order
// genuinely differs, and writes it out in full.
void check_bits(float got, float want, const std::string& what) {
    ++g_checks;
    uint32_t a = 0, b = 0;
    std::memcpy(&a, &got, sizeof a);
    std::memcpy(&b, &want, sizeof b);
    if (a == b) return;
    ++g_failed;
    std::cout << "FAIL [" << g_test << "] " << what << ": got " << got << " (0x" << std::hex << a
              << std::dec << "), want " << want << " (0x" << std::hex << b << std::dec << ")\n";
}

void check_close(float got, float want, float tol, const std::string& what) {
    ++g_checks;
    if (!(std::fabs(got - want) <= tol)) {
        ++g_failed;
        std::cout << "FAIL [" << g_test << "] " << what << ": got " << got << ", want " << want
                  << " (tol " << tol << ")\n";
        return;
    }
}

void note(const std::string& text) { std::cout << "  " << text << "\n"; }

void note_val(const std::string& label, float v) { std::cout << "  " << label << v << "\n"; }

// ------------------------------------------------------ the two references, written independently

// The monolithic dot: one accumulator over the whole input, in lane order.  This is what a projection is
// before it is cut, and it is what the partitioned result has to answer.
float ref_dot(const std::vector<float>& plane, int64_t row, int64_t row_stride, const float* x, int64_t n) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) s += plane[(size_t)(row * row_stride + i)] * x[i];
    return s;
}

// `src/kernels/cpu/native_expert.cpp` L511, one copy.
float ref_swiglu(float g, float u) { return (g / (1.f + std::exp(-g))) * u; }

// ggml's `nearest_int`, the same transcription as `src/kernels/quantize_act_parity.cpp` L139, used here only
// to read a code back out of a dequantized value.
int ref_nearest_int(float fval) {
    const float val = fval + 12582912.0f;
    int i = 0;
    std::memcpy(&i, &val, sizeof i);
    return (i & 0x007fffff) - 0x00400000;
}

// The scale a block gets under the ggml rule, from the maximum the scan is expected to find.  Written here so
// the expected dequantized value comes from a maximum named by hand rather than from the code under test.
float ref_scale(float signed_max) { return 1.0f / (-127.0f / signed_max); }

// ------------------------------------------------------ fixture builders

// A slice cut on the output axis (gate/up): whole input rows, so a slab is the owned rows at the ORIGINAL
// row stride.
TensorSlice output_cut(int64_t n_in, int64_t n_out, int64_t first, int64_t last, int64_t block,
                       const std::vector<int64_t>& valid_lanes) {
    TensorSlice s;
    s.tensor = "blk.0.ffn_gate_shexp.weight";
    s.family = TensorFamily::GateUp;
    s.global_shape = {n_in, n_out};
    s.local_shape = {n_in, last - first};
    s.axis = 1;
    s.range_start = first;
    s.range_end = last;
    s.cut_unit = 1;
    s.head_start = first;
    s.head_end = last;
    s.codec = block > 1 ? "Q8_K" : "F32";
    s.block_size = block;
    s.scale_kind = block > 1 ? "fp16-superblock-scale-plus-sub-scales" : "";
    s.valid_lanes = valid_lanes;
    return s;
}

// A slice cut on the input axis (down): the block axis, so the cut has to land on block boundaries.  A slab
// keeps the pack's rows at the ORIGINAL stride and the primitive reads its own window of them.
TensorSlice input_cut(int64_t n_in, int64_t n_out, int64_t first, int64_t last, int64_t block,
                      const std::vector<int64_t>& valid_lanes) {
    TensorSlice s;
    s.tensor = "blk.0.ffn_down_shexp.weight";
    s.family = TensorFamily::Down;
    s.global_shape = {n_in, n_out};
    s.local_shape = {last - first, n_out};
    s.axis = 0;
    s.range_start = first;
    s.range_end = last;
    s.cut_unit = 1;
    s.head_start = first;
    s.head_end = last;
    s.codec = block > 1 ? "Q8_0" : "F32";
    s.block_size = block;
    s.scale_kind = block > 1 ? "fp16-scale-plus-codebook-codes" : "";
    s.valid_lanes = valid_lanes;
    return s;
}

ActivationContract contract(const std::string& consumer, const std::string& gather, int64_t wire_slots,
                            const std::vector<int64_t>& valid_counts, const std::string& quantizer,
                            int64_t block, bool crosses, const std::string& reference) {
    ActivationContract c;
    c.consumer = consumer;
    c.gather = gather;
    c.wire_slots = wire_slots;
    c.valid_counts = valid_counts;
    c.quantizer = quantizer;
    c.block_size = block;
    c.crosses_rank_boundary = crosses;
    c.reference = reference;
    return c;
}

// The world-3 FFN cuts of the Task 2 fixture, for the tests that want the real geometry.
constexpr int64_t kHidden = 2560;
constexpr int64_t kFfn = 640;
constexpr int64_t kCut1 = 256;
constexpr int64_t kCut2 = 448;
constexpr int64_t kWireSlots = 256;  // the largest owned range, padded up to a uniform slot
const std::vector<int64_t> kValid{256, 192, 192};

// The gathered image of a 640-wide vector through three 256-wide slots: the width every consumer of the
// reference path sees, and the reason a block grid cannot be anchored anywhere but lane 0.
constexpr int64_t kSlotsWidth = kWireSlots * 3;

// ------------------------------------------------------ 1. the shape comes out of the slice

void test_describe() {
    begin("describe reads the projection out of the slice, and refuses what it cannot lay out");

    ProjectionView v{};
    check_refusal(describe(output_cut(kHidden, kFfn, kCut1, kCut2, 0, kValid), v), Refusal::None,
                  "a rank-1 gate/up slice is a projection");
    check_eq(v.n_in, kHidden, "gate/up reads the full residual width");
    check_eq(v.n_out, kFfn, "gate/up writes into the 640-wide intermediate");
    check_eq(v.first, kCut1, "rank 1 starts at the agreed cut");
    check_eq(v.last, kCut2, "rank 1 ends at the agreed cut");
    check_eq(v.owned, kCut2 - kCut1, "rank 1 owns 192 rows");
    check(v.cuts_output, "an axis-1 slice cuts the output");

    check_refusal(describe(input_cut(kFfn, kHidden, kCut1, kCut2, 32, kValid), v), Refusal::None,
                  "a rank-1 down slice is a projection");
    check_eq(v.n_in, kFfn, "down consumes the whole 640-wide intermediate");
    check_eq(v.n_out, kHidden, "down answers at the residual width");
    check(!v.cuts_output, "an axis-0 slice cuts the input");
    check_eq(v.block_size, (int64_t)32, "the block width travels with the slice");

    // A 32-block divides 256 and 448, so that cut is clean.  A 256-block does not divide 448 and would split
    // the block [256,512): that is the case the planner answers with halos and a global quantizer, and the
    // primitive refuses it rather than requantizing to make it fit.
    check_refusal(describe(input_cut(kFfn, kHidden, kCut1, kCut2, 256, kValid), v),
                  Refusal::CutOnWeightBlock, "an input cut through a 256-block is refused");
    // An output cut copies whole rows, so the same 256-block is no obstacle there.
    check_refusal(describe(output_cut(kHidden, kFfn, kCut1, kCut2, 256, kValid), v), Refusal::None,
                  "an output cut cannot split a block");
    check_eq(v.block_size, (int64_t)256, "and the block width is still reported");

    const TensorSlice empty = input_cut(kFfn, kHidden, 128, 128, 32, kValid);
    check_refusal(describe(empty, v), Refusal::BadRange, "an empty owned range is refused");
    const TensorSlice past = input_cut(kFfn, kHidden, 608, 704, 32, kValid);
    check_refusal(describe(past, v), Refusal::BadRange, "a range past the global shape is refused");

    TensorSlice wrong = output_cut(kHidden, kFfn, 0, kCut1, 0, kValid);
    wrong.local_shape = {kHidden, kCut1 + 1};
    check_refusal(describe(wrong, v), Refusal::ShapeMismatch,
                  "a local_shape that does not follow from the range is refused");
    TensorSlice wrong2 = input_cut(kFfn, kHidden, 0, kCut1, 32, kValid);
    wrong2.local_shape = {kCut1, kHidden - 1};
    check_refusal(describe(wrong2, v), Refusal::ShapeMismatch,
                  "a down local_shape that contradicts the output width is refused");

    TensorSlice one_axis = output_cut(kHidden, kFfn, 0, kCut1, 0, kValid);
    one_axis.global_shape = {kHidden};
    one_axis.local_shape = {kCut1};
    check_refusal(describe(one_axis, v), Refusal::NotAProjection, "a 1-axis slice is not a projection");
    TensorSlice three_axis = output_cut(kHidden, kFfn, 0, kCut1, 0, kValid);
    three_axis.global_shape = {kHidden, kFfn, 512};
    three_axis.local_shape = {kHidden, kCut1};
    check_refusal(describe(three_axis, v), Refusal::NotAProjection,
                  "a 3-axis expert tensor is not a single projection");
    TensorSlice other_axis = output_cut(kHidden, kFfn, 0, kCut1, 0, kValid);
    other_axis.axis = 2;
    check_refusal(describe(other_axis, v), Refusal::NotAProjection, "an axis outside 0 and 1 is refused");
}

// ------------------------------------------------------ 2. gate/up keeps the monolithic SwiGLU

// Everything below is computed by hand:
//   x = [3, -1]
//   gate rows [1,2] [3,-1] [0,5] [-2,4]     up rows [2,1] [1,3] [-1,2] [4,0]
//   g = 1, 10, -5, -10                      u = 5, 0, -5, 12
//   ff = g/(1+e^-g)*u
// The two rows that carry the point are row 0, whose silu sits in the middle of the curve, and row 3, where
// the gate is negative enough that the product is a hair below zero.
//
// A second token rides the same fixture at a wider stride, because a prefill is what the projection really
// sees and the token stride is arithmetic of its own: lanes past `valid` of a row are the next row's business
// or nobody's, and must not be read.
void test_gate_up_matches_the_monolithic_swiglu() {
    begin("gate/up answers with the monolithic SwiGLU, row for row");

    const int64_t n_in = 2, n_out = 4;
    const std::vector<float> xbuf{3.0f, -1.0f};
    // A second token, four floats apart, with two lanes of junk between them that a striding bug would read.
    //   token 1 = [-2, 0.5]:
    //     row 0: g = 1*(-2) + 2*0.5  = -1     u = 2*(-2) + 1*0.5  = -3.5
    //     row 1: g = 3*(-2) + (-1)*0.5 = -6.5 u = 1*(-2) + 3*0.5  = -0.5
    const std::vector<float> two_tokens{3.0f, -1.0f, 99.0f, -99.0f, -2.0f, 0.5f, 99.0f, -99.0f};
    const std::vector<float> gate{1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 5.0f, -2.0f, 4.0f};
    const std::vector<float> up{2.0f, 1.0f, 1.0f, 3.0f, -1.0f, 2.0f, 4.0f, 0.0f};

    // Two ranks, rows 0-1 and rows 2-3.  A slab is its own rows at the original stride of 2.
    const std::vector<int64_t> halves{2, 2};
    const TensorSlice g0 = output_cut(n_in, n_out, 0, 2, 0, halves);
    const TensorSlice g1 = output_cut(n_in, n_out, 2, 4, 0, halves);
    const TensorSlice u0 = output_cut(n_in, n_out, 0, 2, 0, halves);
    const TensorSlice u1 = output_cut(n_in, n_out, 2, 4, 0, halves);
    const ActivationContract whole_residual =
        contract("gate_up", "none", 0, halves, "none", 1, false, "global_reference");
    const Activation image{xbuf.data(), n_in, n_in, 1, n_in};

    std::vector<float> ff(4, -1.0f);
    const Weights gw0{gate.data(), n_in};
    const Weights uw0{up.data(), n_in};
    const Weights gw1{gate.data() + 2 * n_in, n_in};
    const Weights uw1{up.data() + 2 * n_in, n_in};
    check_refusal(gate_up_swiglu(g0, gw0, u0, uw0, whole_residual, image, ff.data(), 2), Refusal::None,
                  "rank 0 computes its two rows");
    check_refusal(gate_up_swiglu(g1, gw1, u1, uw1, whole_residual, image, ff.data() + 2, 2), Refusal::None,
                  "rank 1 computes its two rows");

    // The same fixture with two tokens in one image.  Token 1 by hand, computed independently of the f32 path:
    //   row 0: -1/(1+e^1)     * -3.5 =  0.9412949747949828
    //   row 1: -6.5/(1+e^6.5) * -0.5 =  0.004878842334395223
    //   row 2:  2.5/(1+e^-2.5) * 3.0 =  6.931063649840674
    //   row 3:  6/(1+e^-6)     * -8   = -47.88131408848154
    std::vector<float> ff2((size_t)4 * 2, -1.0f);
    const Activation two{two_tokens.data(), 4, n_in, 2, 4};
    check_refusal(gate_up_swiglu(g0, gw0, u0, uw0, whole_residual, two, ff2.data(), 4), Refusal::None,
                  "rank 0 computes two rows for two tokens");
    check_refusal(gate_up_swiglu(g1, gw1, u1, uw1, whole_residual, two, ff2.data() + 2, 4), Refusal::None,
                  "rank 1 does the same");
    for (int64_t t = 0; t < 2; ++t)
        for (int64_t r = 0; r < n_out; ++r) {
            const float g = ref_dot(gate, r, n_in, two_tokens.data() + t * 4, n_in);
            const float u = ref_dot(up, r, n_in, two_tokens.data() + t * 4, n_in);
            check_bits(ff2[(size_t)(t * 4 + r)], ref_swiglu(g, u),
                       "token " + std::to_string(t) + " row " + std::to_string(r));
        }

    // Bit for bit what the monolithic path produces for the same rows.
    for (int64_t r = 0; r < n_out; ++r) {
        const float g = ref_dot(gate, r, n_in, xbuf.data(), n_in);
        const float u = ref_dot(up, r, n_in, xbuf.data(), n_in);
        check_bits(ff[(size_t)r], ref_swiglu(g, u), "row " + std::to_string(r) + " of the partition");
    }

    // And the hand values.  The tolerance is for the person, not the arithmetic: these are the numbers a
    // calculator gives, not the f32 expression, so they can differ in the last bit or two.
    const float hand[4] = {3.6552928931500244f, 0.0f, 0.16732127295236472f, -0.0054477442442921f};
    for (int64_t r = 0; r < n_out; ++r)
        check_close(ff[(size_t)r], hand[r], 1e-7f, "row " + std::to_string(r) + " against the hand value");
    const float hand_t1[4] = {0.9412949747949828f, 0.0048788423f, 6.93106365f, -47.8813141f};
    for (int64_t r = 0; r < n_out; ++r)
        check_close(ff2[(size_t)(4 + r)], hand_t1[r], 1e-5f,
                    "token 1 row " + std::to_string(r) + " against the hand value");
    note_val("row 0 ff = ", ff[0]);
    note_val("row 3 ff = ", ff[3]);
}

// ------------------------------------------------------ 3. down writes a full-width partial

// Integer probes: with whole numbers every partial and the whole sum are exactly representable in f32, so
// "the ranks add up to the monolithic answer" is an exact claim rather than a tolerance.
//
//   down is 16 wide in and 8 wide out, cut at 6 and 11, so the ranks own 6, 5 and 5 lanes;
//   the global weight of row j at lane i is (i - first_of_its_rank + 1), so every rank's own slab reads 1,2,3...
//   x[i] = i + 1.
//     rank 0: 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6  =  91
//     rank 1: 1*7 + 2*8 + 3*9 + 4*10 + 5*11      = 145
//     rank 2: 1*12 + 2*13 + 3*14 + 4*15 + 5*16   = 220
//   monolithic                                      = 456
// A rank that read its lanes from its own lane 0 instead of the global `first` would answer 91 where 145 is
// wanted, which is the whole failure mode of a partitioned down projection.
void test_down_partial_full_width_and_one_sum() {
    begin("a down partial is full width, and the ranks add up to the monolithic answer exactly");

    const int64_t n_in = 16, n_out = 8;
    const int64_t firsts[3] = {0, 6, 11};
    const int64_t lasts[3] = {6, 11, 16};
    const std::vector<int64_t> counts{6, 5, 5};

    std::vector<float> full((size_t)n_out * n_in, 0.0f);
    for (int64_t r = 0; r < 3; ++r)
        for (int64_t j = 0; j < n_out; ++j)
            for (int64_t i = firsts[r]; i < lasts[r]; ++i)
                full[(size_t)(j * n_in + i)] = (float)(i - firsts[r] + 1);
    std::vector<float> x((size_t)n_in, 0.0f);
    for (int64_t i = 0; i < n_in; ++i) x[(size_t)i] = (float)(i + 1);

    // The reference: one dot over all 16 lanes.
    std::vector<float> mono((size_t)n_out, 0.0f);
    for (int64_t j = 0; j < n_out; ++j) mono[(size_t)j] = ref_dot(full, j, n_in, x.data(), n_in);
    check_bits(mono[0], 456.0f, "the monolithic answer is the hand-computed 456");

    // The reference path always hands down the whole logical vector, through the contract's padded slots.
    const ActivationContract gathered =
        contract("down", "all_gather_padded", 16, counts, "none", 1, false, "global_reference");
    std::vector<float> frag0, frag1, frag2;
    for (int64_t r = 0; r < 3; ++r) {
        std::vector<float>* dst = r == 0 ? &frag0 : r == 1 ? &frag1 : &frag2;
        for (int64_t i = firsts[r]; i < lasts[r]; ++i) dst->push_back(x[(size_t)i]);
    }
    const float* frags[3] = {frag0.data(), frag1.data(), frag2.data()};
    std::vector<float> slots((size_t)32, -7.0f);  // the caller sizes the buffer; the tail is workspace
    int64_t logical = 0;
    check_refusal(gather_padded(gathered, frags, slots.data(), 32, logical), Refusal::None,
                  "three fragments assemble into the logical vector");
    check_eq(logical, n_in, "the logical width is the sum of the valid counts");
    // The concatenation, not the wire slots: a logical lane has to sit at its logical index, or a rank's own
    // [first,last) window and the quantizer's block grid mean different things from the contract.
    check_bits(slots[0], 1.0f, "the image starts at lane 0");
    check_bits(slots[6], 7.0f, "rank 1's fragment follows rank 0's range, with no gap for the padded slot");
    check_bits(slots[11], 12.0f, "rank 2's fragment follows rank 1's range");
    for (int64_t i = 16; i < 32; ++i)
        check_bits(slots[(size_t)i], 0.0f, "workspace lane " + std::to_string(i) + " is zeroed");

    const Activation image{slots.data(), 32, 16, 1, 32};
    std::vector<float> partials[3];
    std::vector<float*> pptrs;
    for (int64_t r = 0; r < 3; ++r) {
        partials[r].assign((size_t)n_out, -1.0f);
        pptrs.push_back(partials[r].data());
        const TensorSlice sl = input_cut(n_in, n_out, firsts[r], lasts[r], 0, counts);
        const Weights w{full.data(), n_in};
        check_refusal(down_partial(sl, w, gathered, image, partials[r].data(), n_out), Refusal::None,
                      "rank " + std::to_string(r) + " writes its partial");
        const float want = r == 0 ? 91.0f : r == 1 ? 145.0f : 220.0f;
        // Full width: a partial that stopped at the owned count would leave a -1 in some lane.
        for (int64_t j = 0; j < n_out; ++j)
            check_bits(partials[r][(size_t)j], want,
                       "rank " + std::to_string(r) + " partial at output lane " + std::to_string(j));
    }
    note("partials 91 + 145 + 220");

    std::vector<float> sum((size_t)n_out, 0.0f);
    check_refusal(reduce_partials(pptrs.data(), 3, n_out, sum.data()), Refusal::None,
                  "one reduction over the three partials");
    for (int64_t j = 0; j < n_out; ++j)
        check_bits(sum[(size_t)j], mono[(size_t)j], "output lane " + std::to_string(j) + " vs monolithic");
    check_bits(sum[0], 456.0f, "the sum is 456");
}

// ------------------------------------------------------ 4. a replicated term rides once

// A global bias is one vector that every rank holds.  If each rank folds its copy into its own partial before
// the sum, the bias lands `world` times and nothing downstream can see it: the answer is merely wrong.
void test_replicated_term_applied_once() {
    begin("a replicated bias and scale are applied to the sum, once");

    const int64_t n_in = 16, n_out = 8;
    const int64_t firsts[3] = {0, 6, 11};
    const int64_t lasts[3] = {6, 11, 16};
    const std::vector<int64_t> counts{6, 5, 5};

    std::vector<float> full((size_t)n_out * n_in, 0.0f);
    for (int64_t r = 0; r < 3; ++r)
        for (int64_t j = 0; j < n_out; ++j)
            for (int64_t i = firsts[r]; i < lasts[r]; ++i)
                full[(size_t)(j * n_in + i)] = (float)(i - firsts[r] + 1);
    std::vector<float> x((size_t)n_in, 0.0f);
    for (int64_t i = 0; i < n_in; ++i) x[(size_t)i] = (float)(i + 1);

    std::vector<float> bias((size_t)n_out, 0.0f);
    std::vector<float> scale((size_t)n_out, 0.0f);
    for (int64_t j = 0; j < n_out; ++j) {
        bias[(size_t)j] = (float)(10 + j);
        scale[(size_t)j] = 2.0f;
    }

    const ActivationContract gathered =
        contract("down", "all_gather_padded", 16, counts, "none", 1, false, "global_reference");
    std::vector<float> frag0, frag1, frag2;
    for (int64_t r = 0; r < 3; ++r) {
        std::vector<float>* dst = r == 0 ? &frag0 : r == 1 ? &frag1 : &frag2;
        for (int64_t i = firsts[r]; i < lasts[r]; ++i) dst->push_back(x[(size_t)i]);
    }
    const float* frags[3] = {frag0.data(), frag1.data(), frag2.data()};
    std::vector<float> slots((size_t)32, -7.0f);
    int64_t logical = 0;
    check_refusal(gather_padded(gathered, frags, slots.data(), 32, logical), Refusal::None,
                  "the activation image is assembled");
    check_bits(slots[15], 16.0f, "the last logical lane is the last one gathered");
    for (int64_t i = 16; i < 32; ++i)
        check_bits(slots[(size_t)i], 0.0f, "workspace lane " + std::to_string(i) + " is zeroed");
    const Activation image{slots.data(), 32, 16, 1, 32};

    std::vector<float> partials[3];
    std::vector<float*> pptrs;
    std::vector<float> naive[3];
    std::vector<float*> nptrs;
    for (int64_t r = 0; r < 3; ++r) {
        partials[r].assign((size_t)n_out, 0.0f);
        pptrs.push_back(partials[r].data());
        naive[r].assign((size_t)n_out, 0.0f);
        nptrs.push_back(naive[r].data());
        const TensorSlice sl = input_cut(n_in, n_out, firsts[r], lasts[r], 0, counts);
        const Weights w{full.data(), n_in};
        check_refusal(down_partial(sl, w, gathered, image, partials[r].data(), n_out), Refusal::None,
                      "rank " + std::to_string(r) + " writes its clean partial");
        check_refusal(down_partial(sl, w, gathered, image, naive[r].data(), n_out), Refusal::None,
                      "and the same partial for the wrong way round");
        for (int64_t j = 0; j < n_out; ++j) naive[r][(size_t)j] += bias[(size_t)j];
    }

    std::vector<float> sum((size_t)n_out, 0.0f);
    std::vector<float> naive_sum((size_t)n_out, 0.0f);
    check_refusal(reduce_partials(pptrs.data(), 3, n_out, sum.data()), Refusal::None, "the one sum");
    check_refusal(reduce_partials(nptrs.data(), 3, n_out, naive_sum.data()), Refusal::None,
                  "the sum of the bias-carrying partials");

    std::vector<float> out((size_t)n_out, 0.0f);
    check_refusal(apply_replicated(sum.data(), bias.data(), scale.data(), n_out, out.data()), Refusal::None,
                  "the replicated term is applied to the sum");

    // (456 + bias) * 2, exactly: integers throughout, so any difference is the bug and not the rounding.
    for (int64_t j = 0; j < n_out; ++j) {
        const float want = (456.0f + bias[(size_t)j]) * scale[(size_t)j];
        check_bits(out[(size_t)j], want, "lane " + std::to_string(j) + " biased once and scaled");
        const float thrice = (456.0f + 3.0f * bias[(size_t)j]) * scale[(size_t)j];
        check(naive_sum[(size_t)j] != sum[(size_t)j],
              "lane " + std::to_string(j) + ": the per-rank bias changes the answer");
        check_bits(naive_sum[(size_t)j] * scale[(size_t)j], thrice,
                   "lane " + std::to_string(j) + ": and it changes it by the bias counted three times");
    }
    note_val("lane 0 correct (once): ", out[0]);
    note_val("lane 0 each-rank bias: ", naive_sum[0] * 2.0f);

    // Either term on its own is legal, and neither rides twice.
    std::vector<float> bias_only((size_t)n_out, 0.0f);
    check_refusal(apply_replicated(sum.data(), bias.data(), nullptr, n_out, bias_only.data()), Refusal::None,
                  "a bias with no scale");
    check_bits(bias_only[0], 466.0f, "the bias rides once");
    std::vector<float> scale_only((size_t)n_out, 0.0f);
    check_refusal(apply_replicated(sum.data(), nullptr, scale.data(), n_out, scale_only.data()),
                  Refusal::None, "a scale with no bias");
    check_bits(scale_only[0], 912.0f, "the scale rides once");
    std::vector<float> pass((size_t)n_out, 0.0f);
    check_refusal(apply_replicated(sum.data(), nullptr, nullptr, n_out, pass.data()), Refusal::None,
                  "and with neither term");
    check_bits(pass[0], 456.0f, "the sum passes through unchanged");
}

// ------------------------------------------------------ 5. the whole FFN against the monolithic

// Deterministic values in [-1,1), so a failure is reproducible from the printed seed alone.
struct Lcg {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t m = (uint32_t)((s >> 40) & 0xFFFFFFU);
        return ((float)m - 8388608.0f) / 8388608.0f;
    }
};

void test_ffn_parity_vs_monolithic() {
    begin("the partitioned FFN answers like the monolithic one");

    const int64_t h = 48, f = 64;  // residual width, intermediate width
    const int64_t firsts[3] = {0, 24, 44};
    const int64_t lasts[3] = {24, 44, 64};
    const std::vector<int64_t> counts{24, 20, 20};

    Lcg rng;
    std::vector<float> gate((size_t)f * h), up((size_t)f * h), down((size_t)h * f), x((size_t)h);
    for (auto& v : gate) v = rng.next();
    for (auto& v : up) v = rng.next();
    for (auto& v : down) v = rng.next();
    for (auto& v : x) v = rng.next();

    // Monolithic, the way the single-GPU path does it: each row over the whole input, then the SwiGLU, then a
    // whole 64-lane dot per output.
    std::vector<float> mono_ff((size_t)f, 0.0f);
    for (int64_t r = 0; r < f; ++r) {
        const float g = ref_dot(gate, r, h, x.data(), h);
        const float u = ref_dot(up, r, h, x.data(), h);
        mono_ff[(size_t)r] = ref_swiglu(g, u);
    }
    std::vector<float> mono_out((size_t)h, 0.0f);
    for (int64_t j = 0; j < h; ++j) mono_out[(size_t)j] = ref_dot(down, j, f, mono_ff.data(), f);

    // Partitioned.  The residual is whole on every rank, so gate/up needs no gather; the intermediate is cut,
    // so down gets the gathered image.
    const ActivationContract gu_contract =
        contract("gate_up", "none", 0, {48, 48, 48}, "none", 1, false, "global_reference");
    const ActivationContract down_contract =
        contract("down", "all_gather_padded", 24, counts, "none", 1, false, "global_reference");
    const Activation residual{x.data(), h, h, 1, h};

    std::vector<float> frag[3];
    for (int64_t r = 0; r < 3; ++r) {
        frag[r].assign((size_t)(lasts[r] - firsts[r]), 0.0f);
        const TensorSlice gs = output_cut(h, f, firsts[r], lasts[r], 0, counts);
        const TensorSlice us = output_cut(h, f, firsts[r], lasts[r], 0, counts);
        const Weights gw{gate.data() + (size_t)firsts[r] * h, h};
        const Weights uw{up.data() + (size_t)firsts[r] * h, h};
        check_refusal(gate_up_swiglu(gs, gw, us, uw, gu_contract, residual, frag[r].data(),
                                     lasts[r] - firsts[r]),
                      Refusal::None, "rank " + std::to_string(r) + " gate/up");
        // An output cut is a set of rows and the dot runs in the same order, so this is the same bits rather
        // than a near miss: the SwiGLU of the partition cannot drift from the monolithic one.
        for (int64_t i = firsts[r]; i < lasts[r]; ++i)
            check_bits(frag[r][(size_t)(i - firsts[r])], mono_ff[(size_t)i],
                       "intermediate lane " + std::to_string(i) + " is the monolithic row");
    }

    std::vector<float> slots((size_t)64, -3.0f);
    const float* fptrs[3] = {frag[0].data(), frag[1].data(), frag[2].data()};
    int64_t logical = 0;
    check_refusal(gather_padded(down_contract, fptrs, slots.data(), 64, logical), Refusal::None,
                  "the fragments gather into the 64-lane logical vector");
    check_eq(logical, f, "the logical width is the 64 intermediate lanes");
    check_bits(slots[24], frag[1][0], "rank 1's fragment begins where rank 0's range ends");
    check_bits(slots[44], frag[2][0], "rank 2's fragment begins where rank 1's range ends");
    const Activation image{slots.data(), 64, f, 1, 64};

    std::vector<float> partial[3];
    std::vector<float*> pptrs;
    for (int64_t r = 0; r < 3; ++r) {
        partial[r].assign((size_t)h, 0.0f);
        pptrs.push_back(partial[r].data());
        const TensorSlice ds = input_cut(f, h, firsts[r], lasts[r], 0, counts);
        const Weights w{down.data(), f};
        check_refusal(down_partial(ds, w, down_contract, image, partial[r].data(), h), Refusal::None,
                      "rank " + std::to_string(r) + " down partial");
    }
    std::vector<float> sum((size_t)h, 0.0f);
    check_refusal(reduce_partials(pptrs.data(), 3, h, sum.data()), Refusal::None, "the one all-reduce");

    // The one tolerance in this file, and the reason for it.  The monolithic row adds 64 products into one
    // accumulator in lane order; the partition adds three of 24, 20 and 20 and then adds those three.  Float
    // addition is not associative, so the two cannot be bit-identical; what has to hold is that the gap stays
    // at the size of the rounding it costs, and that is what this bound says.
    const float kTol = 1e-5f;
    float worst = 0.0f;
    int64_t worst_at = 0;
    for (int64_t j = 0; j < h; ++j) {
        const float d = std::fabs(sum[(size_t)j] - mono_out[(size_t)j]);
        if (d > worst) {
            worst = d;
            worst_at = j;
        }
    }
    note("largest |partition - monolithic| at lane " + std::to_string(worst_at) + ":");
    note_val("  ", worst);
    note_val("declared tolerance: ", kTol);
    check(worst <= kTol, "every lane is within the declared tolerance of the monolithic answer");
    for (int64_t j = 0; j < h; ++j)
        check_close(sum[(size_t)j], mono_out[(size_t)j], kTol, "lane " + std::to_string(j));
}

// ------------------------------------------------------ 6. quantized once, on the global grid

// The case the design note counts on: the 640-wide intermediate, cut at 256 and 448, with a Q8_K/256
// quantizer.  Block [256,512) is completed by lanes two ranks away, so a quantizer anchored at a rank's own
// first lane reads a different maximum and codes the same lane with a different number.
//
//   lane   value   global grid (anchor 0)        rank-1 grid (anchor 256)
//   300    0.315   code -5                       code -127
//   260    0.1     code -2                       code -40
//   460    0.25    code -4
//   500    8.0     code -127 (the block maximum, and why the others are so small)
//   700    12.5    padding: never read for a scale, coded as zero
void test_quantize_once_on_the_global_grid() {
    begin("an activation is quantized once, on the grid anchored at lane 0 of the logical vector");

    const ActivationContract global_c =
        contract("down", "all_gather_padded", kWireSlots, kValid, "Q8_K", 256, true, "global_reference");

    std::vector<float> x((size_t)kSlotsWidth, 0.1f);
    for (int64_t i = kCut2; i < kFfn; ++i) x[(size_t)i] = 0.25f;
    x[300] = 0.315f;
    x[500] = 8.0f;
    for (int64_t i = kFfn; i < kSlotsWidth; ++i) x[(size_t)i] = 12.5f;  // stale workspace

    std::vector<float> q((size_t)kSlotsWidth, -1.0f);
    check_refusal(quantize_activation(global_c, x.data(), kSlotsWidth, kFfn, q.data()), Refusal::None,
                  "a 768-wide image is 3 blocks of 256");

    const float dg = ref_scale(8.0f);  // block [256,512): the signed maximum is the +8.0 at lane 500
    check_bits(q[300], (float)(-5) * dg, "lane 300 against the global scale");
    check_bits(q[260], (float)(-2) * dg, "a 0.1 lane in the same block");
    check_bits(q[460], (float)(-4) * dg, "a 0.25 lane in the same block");
    check_bits(q[500], (float)(-127) * dg, "the maximum itself codes as -127, not -128");
    check_close(q[500], 8.0f, 1e-6f, "and dequantizes back to 8");

    // The neighbouring blocks scale off their own maxima, and the padding beyond lane 640 never sets one.
    const float d0 = ref_scale(0.1f);
    check_bits(q[0], (float)(-127) * d0, "block 0 scales off its own 0.1");
    const float d2 = ref_scale(0.25f);
    check_bits(q[600], (float)(-127) * d2,
               "block 2 scales off the 0.25 inside the valid lanes, not the 12.5 beyond them");
    for (int64_t i = kFfn; i < kSlotsWidth; ++i)
        check_bits(q[(size_t)i], 0.0f, "padding lane " + std::to_string(i) + " codes as zero");

    // The same vector through a rank-local grid: rank 1 anchors at its own first lane, 256, and holds only
    // 192 of the block, so its maximum is 0.315 and lane 300 becomes that block's -127.
    std::vector<float> local((size_t)256, 12.5f);
    for (int64_t e = 0; e < 192; ++e) local[(size_t)e] = x[(size_t)(kCut1 + e)];
    std::vector<float> ql((size_t)256, -1.0f);
    const ActivationContract local_c =
        contract("down", "none", kWireSlots, kValid, "Q8_K", 256, true, "local");
    check_refusal(quantize_activation(local_c, local.data(), 256, 192, ql.data()), Refusal::None,
                  "rank 1 quantizes its own 192 lanes in a 256-wide block");
    const float dl = ref_scale(0.315f);
    check_bits(ql[44], (float)(-127) * dl, "lane 300 in rank 1's own grid");
    check_bits(ql[4], (float)(-40) * dl, "a 0.1 lane in rank 1's own grid");

    const int code_global = ref_nearest_int(q[300] / dg);
    const int code_local = ref_nearest_int(ql[44] / dl);
    note("lane 300: global code " + std::to_string(code_global) + ", rank-local code " +
         std::to_string(code_local) + " - the same lane, two answers, so the gather has to come first");
    check_eq(code_global, -5, "the global code is -5");
    check_eq(code_local, -127, "the local code is -127");

    // Ties keep the FIRST maximum and the maximum keeps its sign: two lanes of equal magnitude with opposite
    // signs pick their scale by position, and a scale taken off the absolute value would flip every code in
    // the block.
    std::vector<float> tie((size_t)256, 0.02f);
    tie[10] = 0.5f;
    tie[200] = -0.5f;
    std::vector<float> qt((size_t)256, -1.0f);
    const ActivationContract tie_c =
        contract("down", "none", 256, {256, 0, 0}, "Q8_K", 256, false, "global_reference");
    check_refusal(quantize_activation(tie_c, tie.data(), 256, 256, qt.data()), Refusal::None,
                  "a block with two equal magnitudes");
    const float dt = ref_scale(0.5f);
    check_bits(qt[10], (float)(-127) * dt, "the first of the two is the scale, with its + sign");
    check_bits(qt[200], (float)127 * dt, "the other codes as +127");

    // A block with no valid lane codes as all zeros rather than dividing by a zero maximum.
    std::vector<float> tail((size_t)512, 3.0f);
    for (int64_t i = 128; i < 512; ++i) tail[(size_t)i] = 4.0f;
    std::vector<float> qtail((size_t)512, -1.0f);
    const ActivationContract tail_c =
        contract("down", "all_gather_padded", 256, {128, 0, 0}, "Q8_K", 256, false, "global_reference");
    check_refusal(quantize_activation(tail_c, tail.data(), 512, 128, qtail.data()), Refusal::None,
                  "a block lying entirely past the valid lanes");
    for (int64_t i = 256; i < 512; ++i)
        check_bits(qtail[(size_t)i], 0.0f, "zero-block lane " + std::to_string(i));
    const float dtail = ref_scale(3.0f);
    check_bits(qtail[0], (float)(-127) * dtail, "the block that does have valid lanes still scales");
}

// ------------------------------------------------------ 7. refusals, never a partial answer

void test_refusals() {
    begin("every refusal is a refusal, and nothing is written on the way out");

    const int64_t h = 48, f = 64;
    Lcg rng;
    std::vector<float> gate((size_t)f * h), up((size_t)f * h), down((size_t)h * f), x((size_t)h);
    for (auto& v : gate) v = rng.next();
    for (auto& v : up) v = rng.next();
    for (auto& v : down) v = rng.next();
    for (auto& v : x) v = rng.next();
    const std::vector<int64_t> counts{24, 20, 20};

    // A rank handing over its own fragment instead of the gathered vector.
    const TensorSlice gs = output_cut(h, f, 0, 24, 0, counts);
    const TensorSlice us = output_cut(h, f, 0, 24, 0, counts);
    const Weights gw{gate.data(), h};
    const Weights uw{up.data(), h};
    const ActivationContract whole_residual =
        contract("gate_up", "none", 0, {48, 48, 48}, "none", 1, false, "global_reference");
    std::vector<float> fragment((size_t)24, 0.1f);
    std::vector<float> ff(24, 7.0f);
    const Activation frag_image{fragment.data(), 24, 24, 1, 24};
    check_refusal(gate_up_swiglu(gs, gw, us, uw, whole_residual, frag_image, ff.data(), 24),
                  Refusal::ActivationWidth, "gate/up over a 24-lane fragment of a 48-lane residual");
    for (int64_t i = 0; i < 24; ++i) check_bits(ff[(size_t)i], 7.0f, "output lane " + std::to_string(i));

    // gate and up must own the same rows: a gate row times another row's up is not a row of anything.
    const TensorSlice up_other = output_cut(h, f, 24, 44, 0, counts);
    const Activation residual{x.data(), h, h, 1, h};
    check_refusal(gate_up_swiglu(gs, gw, up_other, uw, whole_residual, residual, ff.data(), 24),
                  Refusal::ShapeMismatch, "gate rows 0-23 against up rows 24-43");

    // A gather contract, and an image that cannot have come through it.
    const ActivationContract padded =
        contract("down", "all_gather_padded", 256, kValid, "Q8_K", 256, true, "global_reference");
    std::vector<float> image640((size_t)kFfn, 0.1f);
    const TensorSlice ds = input_cut(f, h, 0, 24, 0, counts);
    const Weights dw{down.data(), f};
    std::vector<float> p(h, 0.0f);
    // The image is the width this projection consumes, but the contract's fragments come to 60 of the 64
    // lanes, so whatever it is, it is not the gathered vector of that contract.
    const ActivationContract short_sum =
        contract("down", "all_gather_padded", 24, {24, 20, 16}, "none", 1, false, "global_reference");
    std::vector<float> image64((size_t)f, 0.1f);
    const Activation whole64{image64.data(), f, f, 1, f};
    check_refusal(down_partial(ds, dw, short_sum, whole64, p.data(), h), Refusal::ContractMismatch,
                  "fragments that come to 60 lanes of a 64-lane projection");
    // A vector that needs no gather is whole on every rank, so a contract saying otherwise does not describe
    // the image it arrived with.
    const ActivationContract ragged_none =
        contract("gate_up", "none", 0, {h, h, 24}, "none", 1, false, "global_reference");
    std::vector<float> ff3(24, 7.0f);
    check_refusal(gate_up_swiglu(gs, gw, us, uw, ragged_none, residual, ff3.data(), 24),
                  Refusal::ContractMismatch, "a no-gather contract with one rank short of the whole width");
    check_bits(ff3[0], 7.0f, "untouched");

    // A slab narrower than the projection's input cannot be read at the stride the slice needs.
    const Weights narrow{gate.data(), h - 1};
    std::vector<float> ff2(24, 7.0f);
    check_refusal(gate_up_swiglu(gs, narrow, us, uw, whole_residual, residual, ff2.data(), 24),
                  Refusal::SlabTooNarrow, "a slab one lane short of the residual");
    check_bits(ff2[0], 7.0f, "untouched");

    const ActivationContract no_gather =
        contract("down", "none", 0, {64, 64, 64}, "none", 1, false, "global_reference");
    check_refusal(down_partial(gs, dw, no_gather, whole64, p.data(), h), Refusal::NotAProjection,
                  "down handed an output-cut slice");
    // The wrong primitive for the wrong axis.
    check_refusal(gate_up_swiglu(ds, dw, ds, dw, whole_residual, residual, ff2.data(), 24),
                  Refusal::NotAProjection, "gate/up handed an input-cut slice");

    // Nulls.
    check_refusal(gate_up_swiglu(gs, Weights{nullptr, h}, us, uw, whole_residual, residual, ff2.data(), 24),
                  Refusal::NullBuffer, "a null gate slab");
    check_refusal(gate_up_swiglu(gs, gw, us, uw, whole_residual, residual, nullptr, 24), Refusal::NullBuffer,
                  "a null output");
    const Activation nullact{nullptr, 0, 0, 0, 0};
    check_refusal(gate_up_swiglu(gs, gw, us, uw, whole_residual, nullact, ff2.data(), 24),
                  Refusal::NullBuffer, "a null activation");
    std::vector<float*> pptrs(3, p.data());
    pptrs[1] = nullptr;
    std::vector<float> sum(h, 0.0f);
    check_refusal(reduce_partials(pptrs.data(), 3, h, sum.data()), Refusal::NullBuffer,
                  "a reduction with a missing partial");
    check_refusal(apply_replicated(nullptr, nullptr, nullptr, h, sum.data()), Refusal::NullBuffer,
                  "a replicated term with no sum");

    // A width that is not a whole number of blocks has no layout on the block grid at all, which is the same
    // structural rule `include/strata/kernels/quantize_act.hpp` states for Q8_0 and Q8_K.
    std::vector<float> q(700, -1.0f);
    check_refusal(quantize_activation(padded, image640.data(), 700, kFfn, q.data()), Refusal::BadContract,
                  "700 lanes is not a whole number of 256-blocks");
    check_bits(q[0], -1.0f, "and nothing was written");
    check_refusal(quantize_activation(padded, image640.data(), kFfn, 700, q.data()),
                  Refusal::ActivationWidth, "valid past the buffer width");

    // A gather whose slot is narrower than the count it has to carry.
    const ActivationContract too_small =
        contract("down", "all_gather_padded", 16, counts, "none", 1, false, "global_reference");
    std::vector<float> slots(48, 0.0f);
    std::vector<float> f24(24, 0.5f), f20(20, 0.5f);
    const float* frags[3] = {f24.data(), f20.data(), f20.data()};
    int64_t logical = 0;
    check_refusal(gather_padded(too_small, frags, slots.data(), 48, logical), Refusal::BadContract,
                  "a 24-lane count in a 16-lane slot");
    check_eq(logical, (int64_t)0, "and the logical width is not half-filled in");
    // The buffer has to hold the whole logical vector, not the wire's padded slots.
    const ActivationContract tight_buffer =
        contract("down", "all_gather_padded", 24, counts, "none", 1, false, "global_reference");
    check_refusal(gather_padded(tight_buffer, frags, slots.data(), 32, logical), Refusal::BadContract,
                  "64 logical lanes do not fit a 32-lane buffer");
}

void test_invalid_projection_geometry() {
    begin("malformed geometry is refused before reading weights");
    ProjectionView view;
    check_refusal(describe(input_cut(64, 8, -32, 0, 32, {32,32,32}), view),
                  Refusal::BadRange, "a negative input offset cannot point before the weight slab");
    const auto gate = output_cut(4, 3, 0, 3, 0, {3});
    const auto up = output_cut(5, 3, 0, 3, 0, {3});
    std::vector<float> gw(12, 1), uw(15, 1), x(4, 1), out(3, 91);
    const auto c = contract("gate_up", "none", 0, {4}, "none", 1, false, "global_reference");
    check_refusal(gate_up_swiglu(gate, {gw.data(),4}, up, {uw.data(),5}, c,
                               {x.data(),4,4,1,4}, out.data(),3),
                  Refusal::ShapeMismatch, "gate and up cannot consume different input widths");
    check_bits(out[0], 91, "a refused projection leaves the output untouched");
    float raw[8] = {1,2,3,4,99,99,99,99};
    float copy[8] = {};
    check_refusal(quantize_activation(c, raw, 8, 4, copy), Refusal::None,
                  "an unquantized activation still has a logical length");
    check_bits(copy[7], 0, "unused lanes are workspace rather than stale activations");
}

}  // namespace

int main() {
    test_describe();
    test_gate_up_matches_the_monolithic_swiglu();
    test_down_partial_full_width_and_one_sum();
    test_replicated_term_applied_once();
    test_ffn_parity_vs_monolithic();
    test_quantize_once_on_the_global_grid();
    test_refusals();
    test_invalid_projection_geometry();
    std::cout << "\nchecks: " << (g_checks - g_failed) << " passed, " << g_failed << " failed\n";
    return g_failed == 0 ? 0 : 1;
}
