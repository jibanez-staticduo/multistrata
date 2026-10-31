// Synthetic component qualification only. No weights, transport, layer
// projection, full-model tolerance, or GPU performance claim is covered here.
#include "strata/kernels/tp_gdn.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tp = strata::kernels::tp::cuda;
namespace k = strata::kernels;
namespace {
constexpr int S = 128, HK = 16, HV = 48, C = (2 * HK + HV) * S, V = HV * S;
const tp::GdnTpMap maps[3] = {{128,16,48,4,0,6}, {128,16,48,4,6,11}, {128,16,48,4,11,16}};
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char* what) { if (!b) throw std::runtime_error(what); }
struct Stream {
    cudaStream_t p{};
    Stream() { ck(cudaStreamCreate(&p)); }
    ~Stream() { if (p) cudaStreamDestroy(p); }
};
struct Device {
    float* p{};
    std::size_t n;
    explicit Device(std::size_t count) : n(count) { ck(cudaMalloc(&p, n * sizeof(float))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    tp::GdnBuffer buffer() { return {p, n}; }
    tp::GdnConstBuffer input() const { return {p, n}; }
    void upload(const std::vector<float>& v) {
        require(v.size() == n, "upload size"); ck(cudaMemcpy(p, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> read() const {
        std::vector<float> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost)); return v;
    }
    void zero(cudaStream_t s) { ck(cudaMemsetAsync(p, 0, n * sizeof(float), s)); }
};
struct Error {
    double abs = 0, rel = 0;
    std::size_t unequal = 0;
    void see(float a, float b) {
        if (!std::isfinite(a) || !std::isfinite(b)) { abs = rel = std::numeric_limits<double>::infinity(); ++unequal; return; }
        const double d = std::fabs(double(a) - b);
        abs = std::max(abs, d); rel = std::max(rel, d / (std::fabs(double(b)) + 1e-12));
        unequal += std::memcmp(&a, &b, sizeof(float)) != 0;
    }
};
void equal(const char* label, const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "comparison size");
    Error e;
    for (std::size_t i = 0; i < actual.size(); ++i) e.see(actual[i], expected[i]);
    std::printf("%-51s max_abs=%.9g max_rel=%.9g unequal=%zu\n", label, e.abs, e.rel, e.unequal);
    // Identical kernels and arithmetic order: these component comparisons
    // demand exact bits, without inventing an error budget for a full model.
    require(e.unequal == 0, label);
}
std::vector<float> signal(std::size_t n, int salt, float scale) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto x = (i * 1664525ULL + std::size_t(salt) * 1013904223ULL) % 1009;
        v[i] = scale * (float(x) / 1009.0f - 0.5f);
    }
    return v;
}
int value_head(const tp::GdnTpMap& m, int h) { return h / m.key_heads() * HK + m.key_begin + h % m.key_heads(); }
int channel(const tp::GdnTpMap& m, int c) {
    const int qk = m.key_heads() * S;
    if (c < 2 * qk) return c / qk * HK * S + m.key_begin * S + c % qk;
    c -= 2 * qk;
    return 2 * HK * S + value_head(m, c / S) * S + c % S;
}
void mapping(tp::GdnTpLayout layout, int rows, int inner, cudaStream_t s) {
    const bool qkv = layout == tp::GdnTpLayout::Qkv, recurrent = layout == tp::GdnTpLayout::Recurrent;
    const int outer = recurrent ? S : rows, width = qkv ? C : HV;
    Device global(std::size_t(outer) * width * inner), reconstructed(global.n);
    auto input = signal(global.n, 11, 1.0f); global.upload(input); reconstructed.zero(s);
    for (const auto& m : maps) {
        const int local_width = qkv ? m.channels() : m.value_heads();
        Device local(std::size_t(outer) * local_width * inner);
        tp::tp_gdn_compact(m, layout, rows, inner, global.input(), local.buffer(), s);
        ck(cudaStreamSynchronize(s));
        std::vector<float> expected(local.n);
        for (int r = 0; r < outer; ++r) for (int h = 0; h < local_width; ++h) for (int j = 0; j < inner; ++j)
            expected[(std::size_t(r) * local_width + h) * inner + j] =
                input[(std::size_t(r) * width + (qkv ? channel(m, h) : value_head(m, h))) * inner + j];
        equal("compact vs independent host index map", local.read(), expected);
        tp::tp_gdn_scatter(m, layout, rows, inner, local.input(), reconstructed.buffer(), s);
    }
    ck(cudaStreamSynchronize(s)); equal("scatter reconstruction", reconstructed.read(), input);
}
struct Rank {
    tp::GdnTpMap map;
    Device state, history, snapshot, conv_snapshot, qkv, weights, gates, betas, z, raw, normalized, readout, output;
    Rank(tp::GdnTpMap m, int tokens) : map(m), state(m.state_elements()), history(m.history_elements()),
        snapshot(state.n), conv_snapshot(history.n), qkv(std::size_t(tokens) * m.channels()), weights(m.channels() * 4),
        gates(tokens * m.value_heads()), betas(gates.n), z(std::size_t(tokens) * m.value_heads() * S),
        raw(m.channels()), normalized(m.channels()), readout(m.value_heads() * S), output(std::size_t(tokens) * m.value_heads() * S) {}
    tp::GdnTpState live() { return {state.buffer(), history.buffer()}; }
    tp::GdnTpState saved() { return {snapshot.buffer(), conv_snapshot.buffer()}; }
};
void native_global(tp::GdnTpArithmetic arithmetic, Device& state, Device& history, Device& weights,
                   Device& gamma, Device& raw, Device& normalized, Device& readout, const float* qkv,
                   const float* gate, const float* beta, const float* z, float* output, cudaStream_t s) {
    const float eps = 1e-6f;
    auto* h = normalized.p;
    if (arithmetic == tp::GdnTpArithmetic::Native) {
        k::native_gdn_conv_silu(history.p, qkv, weights.p, raw.p, h, C, 4, s);
        k::native_gdn_l2_norm(h, 2 * HK, S, eps, s);
        k::native_gdn_step(state.p, h, h + HK * S, h + 2 * HK * S, gate, beta, readout.p, {S,HK,HV}, s);
        k::native_gdn_out_norm(readout.p, z, gamma.p, output, HV, S, eps, s);
    } else {
        k::fused_gdn_conv_l2(history.p, qkv, weights.p, h, C, 2 * HK, eps, s);
        if (arithmetic == tp::GdnTpArithmetic::FusedSilu)
            k::fused_gdn_step_norm_silu(state.p, h, h + HK * S, h + 2 * HK * S, gate, beta, z, gamma.p, eps, output, HK, HV, s);
        else k::fused_gdn_step_norm(state.p, h, h + HK * S, h + 2 * HK * S, gate, beta, z, gamma.p, eps, output, HK, HV, s);
    }
}
void invalid_contracts(cudaStream_t stream) {
    int rejected = 0;
    try { tp::tp_gdn_validate_map({128,16,48,4,6,6}); } catch (const std::invalid_argument&) { ++rejected; }
    Device a(128), b(128);
    try { tp::tp_gdn_compact(maps[0], tp::GdnTpLayout::Values, 1, 128, a.input(), b.buffer(), stream); }
    catch (const std::invalid_argument&) { ++rejected; }
    Device full(V);
    try { tp::tp_gdn_compact(maps[0], tp::GdnTpLayout::Values, 1, 128, full.input(), full.buffer(), stream); }
    catch (const std::invalid_argument&) { ++rejected; }
    auto wrong = maps[0]; wrong.global_value_heads = 54;
    try { tp::tp_gdn_validate_map(wrong); } catch (const std::invalid_argument&) { ++rejected; }
    try { tp::tp_gdn_compact(maps[0], tp::GdnTpLayout::Values, 1, 1, full.input(), b.buffer(), nullptr); }
    catch (const std::invalid_argument&) { ++rejected; }
    require(rejected == 5, "invalid contracts were accepted");
    std::printf("invalid map/span/alias/stream contracts: %d rejected\n", rejected);
}
void projection_contributions(cudaStream_t s) {
    Device projection(257), contribution(257);
    const auto expected = signal(257, 30, 1.0f); projection.upload(expected);
    for (int rank = 0; rank < 3; ++rank) {
        tp::tp_gdn_replicated_output_contribution(rank, rank ? tp::GdnConstBuffer{nullptr,0} : projection.input(),
                                                  contribution.buffer(), 257, s);
        ck(cudaStreamSynchronize(s));
        equal("replicated projection rank contribution", contribution.read(), rank ? std::vector<float>(257,0) : expected);
    }
}
void run(int tokens, tp::GdnTpArithmetic arithmetic, cudaStream_t s) {
    std::printf("\nT=%d arithmetic=%d chunks=[1,4,60,64] truncated at T\n", tokens, int(arithmetic));
    Device qkv(std::size_t(tokens) * C), gate(tokens * HV), beta(tokens * HV), z(std::size_t(tokens) * V),
        weights(C * 4), gamma(S), state(std::size_t(S) * V), history(C * 3), raw(C), normalized(C), readout(V), output(std::size_t(tokens) * V),
        recovered_state(state.n), recovered_history(history.n), gathered(std::size_t(tokens) * 3 * 18 * S), recovered_output(output.n);
    auto x = signal(qkv.n, 3, 0.8f), g = signal(gate.n, 7, 0.1f), b = signal(beta.n, 13, 0.3f),
        zz = signal(z.n, 17, 2.0f), ww = signal(weights.n, 23, 0.7f), ga = signal(S, 29, 0.3f),
        st = signal(state.n, 31, 0.02f), hist = signal(history.n, 37, 0.2f);
    // Nonuniform head offsets prevent repeated or accidentally permuted heads
    // from matching; every global key and each of its three values differ.
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < 2 * HK + HV; ++h) for (int j = 0; j < S; ++j)
            x[std::size_t(t) * C + h * S + j] += 0.007f * h;
        for (int h = 0; h < HV; ++h) { g[t * HV + h] -= 0.15f + 0.002f * h; b[t * HV + h] += 0.5f; }
    }
    for (auto& v : ga) v += 1.0f;
    qkv.upload(x); gate.upload(g); beta.upload(b); z.upload(zz); weights.upload(ww); gamma.upload(ga); state.upload(st); history.upload(hist);
    Rank r0(maps[0],tokens), r1(maps[1],tokens), r2(maps[2],tokens);
    Rank* ranks[] = {&r0,&r1,&r2};
    for (auto* r : ranks) {
        const auto m = r->map;
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Qkv,tokens,1,qkv.input(),r->qkv.buffer(),s);
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Qkv,1,4,weights.input(),r->weights.buffer(),s);
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Qkv,1,3,history.input(),r->history.buffer(),s);
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Values,tokens,1,gate.input(),r->gates.buffer(),s);
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Values,tokens,1,beta.input(),r->betas.buffer(),s);
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Values,tokens,S,z.input(),r->z.buffer(),s);
        tp::tp_gdn_compact(m,tp::GdnTpLayout::Recurrent,1,S,state.input(),r->state.buffer(),s);
    }
    // Poison padded slots. The gather-to-global reorder must not read them.
    gathered.upload(std::vector<float>(gathered.n, std::numeric_limits<float>::quiet_NaN()));
    int boundary = 1;
    for (int t = 0; t < tokens; ++t) {
        native_global(arithmetic,state,history,weights,gamma,raw,normalized,readout,
                       qkv.p + std::size_t(t)*C,gate.p+t*HV,beta.p+t*HV,z.p+std::size_t(t)*V,output.p+std::size_t(t)*V,s);
        for (int rank = 0; rank < 3; ++rank) {
            auto& r = *ranks[rank]; const auto m = r.map;
            const int c = m.channels(), v = m.value_heads()*S;
            tp::tp_gdn_token(m,arithmetic,r.live(),{r.qkv.p+std::size_t(t)*c,std::size_t(c)},r.weights.input(),
                              {r.gates.p+t*m.value_heads(),std::size_t(m.value_heads())},
                              {r.betas.p+t*m.value_heads(),std::size_t(m.value_heads())},
                              {r.z.p+std::size_t(t)*v,std::size_t(v)},gamma.input(),1e-6f,
                              {r.raw.buffer(),r.normalized.buffer(),r.readout.buffer()},
                              {r.output.p+std::size_t(t)*v,std::size_t(v)},s);
            ck(cudaMemcpyAsync(gathered.p+(std::size_t(t)*3+rank)*18*S,r.output.p+std::size_t(t)*v,
                               v*sizeof(float),cudaMemcpyDeviceToDevice,s));
        }
        if (t + 1 == boundary || t + 1 == tokens) {
            recovered_state.zero(s); recovered_history.zero(s);
            for (auto* r : ranks) {
                tp::tp_gdn_scatter(r->map,tp::GdnTpLayout::Recurrent,1,S,r->state.input(),recovered_state.buffer(),s);
                tp::tp_gdn_scatter(r->map,tp::GdnTpLayout::Qkv,1,3,r->history.input(),recovered_history.buffer(),s);
            }
            ck(cudaStreamSynchronize(s));
            std::printf("boundary=%d\n", t+1);
            equal("recurrent state vs same-arithmetic global",recovered_state.read(),state.read());
            equal("conv continuation vs same-arithmetic global",recovered_history.read(),history.read());
            // Snapshot, reset, and restore at every chunk boundary; subsequent
            // tokens must produce the same trajectory as the unbroken global run.
            for (auto* r : ranks) {
                tp::tp_gdn_copy_state(r->map,r->saved(),r->live(),s);
                tp::tp_gdn_reset(r->map,r->live(),s);
                ck(cudaStreamSynchronize(s));
                equal("reset recurrent",r->state.read(),std::vector<float>(r->state.n,0));
                equal("reset convolution",r->history.read(),std::vector<float>(r->history.n,0));
                tp::tp_gdn_copy_state(r->map,r->live(),r->saved(),s);
                ck(cudaStreamSynchronize(s));
                equal("snapshot restore recurrent",r->state.read(),r->snapshot.read());
                equal("snapshot restore convolution",r->history.read(),r->conv_snapshot.read());
            }
            boundary = boundary == 1 ? 5 : boundary == 5 ? 65 : 129;
        }
    }
    tp::tp_gdn_scatter_gathered(maps,tokens,18*S,gathered.input(),recovered_output.buffer(),s);
    ck(cudaStreamSynchronize(s));
    equal("padded gather vs same-arithmetic global output",recovered_output.read(),output.read());
    // Negative sensitivity check: rank-major concatenation is not global order.
    const auto all = gathered.read(), expected = output.read();
    bool differs = false;
    for (int i = 6*S; i < 18*S; ++i) if (all[i] != expected[i]) { differs = true; break; }
    require(differs,"test cannot distinguish erroneous contiguous-value gather");
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1],"--selftest") != 0)) {
        std::fprintf(stderr,"usage: tp_gdn_parity [--selftest]\n"); return 2;
    }
    int count = 0;
    const auto available = cudaGetDeviceCount(&count);
    if (available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver || (available == cudaSuccess && count == 0)) {
        std::puts("tp_gdn_parity SKIP: no CUDA device/driver"); return 77;
    }
    try {
        ck(available); Stream stream;
        invalid_contracts(stream.p);
        mapping(tp::GdnTpLayout::Qkv,4,1,stream.p);
        mapping(tp::GdnTpLayout::Qkv,1,3,stream.p);
        mapping(tp::GdnTpLayout::Qkv,1,4,stream.p);
        mapping(tp::GdnTpLayout::Values,4,1,stream.p);
        mapping(tp::GdnTpLayout::Values,4,S,stream.p);
        mapping(tp::GdnTpLayout::Recurrent,1,S,stream.p);
        projection_contributions(stream.p);
        for (const auto arithmetic : {tp::GdnTpArithmetic::Native,tp::GdnTpArithmetic::FusedSigmoid,tp::GdnTpArithmetic::FusedSilu})
            for (const int tokens : {1,4,65,129}) run(tokens,arithmetic,stream.p);
        std::puts("tp_gdn_parity OK: component only; no full-layer or full-model validation"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"tp_gdn_parity FAIL: %s\n",e.what()); return 1; }
}
