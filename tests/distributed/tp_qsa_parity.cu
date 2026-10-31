// Single-GPU emulation of TP ranks against original native QSA components.
// This is a per-query prefill reference, not a tensor-core GEMM benchmark.
// --inspect-contract needs no CUDA device. No GPU work is done until main's
// other path. --measure-only reports fast-kernel errors and returns 77;
// --atol X --rtol Y additionally gates those errors with caller-chosen bounds.
#include "strata/kernels/tp_qsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/rope.hpp"

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

namespace k = strata::kernels;
namespace tp = strata::kernels::tp::cuda;
namespace {
void need(bool v, const char* why) { if (!v) throw std::runtime_error(why); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Stream {
    cudaStream_t p{};
    Stream() { ck(cudaStreamCreate(&p)); }
    ~Stream() { if (p) cudaStreamDestroy(p); }
};
template<class T> struct Device {
    T* p{};
    size_t n{};
    explicit Device(size_t count) : n(count) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& v) {
        need(v.size() <= n, "upload overflows device buffer");
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
    void zero() { ck(cudaMemset(p, 0, n * sizeof(T))); }
};
template<class T> void exact(const std::vector<T>& a, const std::vector<T>& b, const char* message) {
    need(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0, message);
}
float sample(size_t i, float phase = 0) {
    return 0.32f * std::sin(float(i) * 0.031f + phase) + 0.11f * std::cos(float(i) * 0.107f);
}
tp::QsaShard shard(const k::QsaShapes& shape, int offset, int count = 8) {
    tp::QsaShard s;
    s.global = shape; s.local_heads = count; s.head_offset = offset;
    return s;
}
int global_head(const tp::QsaShard& s, int h) {
    return s.explicit_heads ? s.query_heads[h] : s.head_offset + h;
}
void contract() {
    const auto g = k::qsa_real_shapes();
    const int expected[3][8] = {{0,0,0,0,0,0,0,0}, {0,0,0,0,1,1,1,1}, {1,1,1,1,1,1,1,1}};
    for (int r = 0; r < 3; ++r) {
        auto s = shard(g, 8 * r);
        for (int h = 0; h < 8; ++h) need(tp::qsa_shard_kv_head(s, h) == expected[r][h], "wrong global GQA map");
    }
    auto s = shard(g, 10, 8);
    need(tp::qsa_shard_kv_head(s, 1) == 0 && tp::qsa_shard_kv_head(s, 2) == 1, "cross-boundary map");
    s.explicit_heads = true;
    for (int h = 0; h < 8; ++h) s.query_heads[h] = h % 2 ? h : 23 - h;
    for (int h = 0; h < 8; ++h)
        need(tp::qsa_shard_kv_head(s, h) == s.query_heads[h] / 12, "explicit global GQA map");
    s.query_heads[1] = s.query_heads[0];
    bool rejected = false;
    try { tp::validate_qsa_shard(s); } catch (const std::invalid_argument&) { rejected = true; }
    need(rejected, "duplicate global heads accepted");
    std::puts("contract: TP3 KV distribution 8/0, 4/4, 0/8; boundary and explicit maps passed");
}

struct Options {
    bool inspect = false, measure = true;
    double atol = 0, rtol = 0;
};
Options options(int argc, char** argv) {
    Options o;
    bool a = false, r = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--inspect-contract") { o.inspect = true; continue; }
        if (arg == "--measure-only") { o.measure = true; continue; }
        need(i + 1 < argc, "option needs a value");
        const double v = std::stod(argv[++i]);
        need(std::isfinite(v) && v >= 0, "tolerance must be finite and nonnegative");
        if (arg == "--atol") { o.atol = v; a = true; }
        else if (arg == "--rtol") { o.rtol = v; r = true; }
        else throw std::runtime_error("unknown option");
    }
    need(a == r, "provide both atol and rtol");
    if (a && r) o.measure = false;
    return o;
}

// Pools retain original page/head/token/channel layout. Snapshot restoration
// exercises the same caller-owned cache contract as the integrated layer.
struct Cache {
    int mode;
    k::QsaShapes s;
    Device<uint16_t> kh, vh, ks, vs;
    Device<int8_t> k8, v8;
    Device<uint8_t> k4, v4;
    Device<int32_t> table;
    explicit Cache(int m, k::QsaShapes shape, int cells)
        : mode(m), s(shape), kh(cells * 512), vh(cells * 512), ks(cells * 8), vs(cells * 8),
          k8(cells * 512), v8(cells * 512), k4(cells * 288), v4(cells * 288), table(cells / shape.page_size) {
        std::vector<int32_t> pages(table.n);
        for (size_t i = 0; i < pages.size(); ++i) pages[i] = int(pages.size() - i - 1);
        table.put(pages);
        kh.zero(); vh.zero(); ks.zero(); vs.zero(); k8.zero(); v8.zero(); k4.zero(); v4.zero();
    }
    k::QsaAttnPools pools() const {
        k::QsaAttnPools p;
        p.page_table = table.p;
        if (mode == 0) { p.k_pool = kh.p; p.v_pool = vh.p; }
        if (mode == 1 || mode == 3) { p.k_q = k8.p; p.k_scale = ks.p; }
        if (mode == 1) { p.v_q = v8.p; p.v_scale = vs.p; }
        if (mode == 2) p.k_q4 = k4.p;
        if (mode == 2 || mode == 3) p.v_q4 = v4.p;
        return p;
    }
    void append(const int32_t* step, const float* key, const float* value, void* stream) {
        if (mode == 0) k::kv_append_step(kh.p, vh.p, table.p, step, key, value, s, stream);
        if (mode == 1) k::kv_append_q8_step(k8.p, v8.p, ks.p, vs.p, table.p, step, key, value, s, stream);
        if (mode == 2) k::kv_append_q4_step(k4.p, v4.p, table.p, step, key, value, s, stream);
        if (mode == 3) {
            k::kv_append_q8_step(k8.p, k8.p, ks.p, ks.p, table.p, step, key, key, s, stream);
            k::kv_append_q4_step(v4.p, v4.p, table.p, step, value, value, s, stream);
        }
    }
    void gather(const int32_t* ids, const int32_t* step, int cap, uint16_t* key, uint16_t* value, void* stream) {
        if (mode == 0) k::kv_gather_step(kh.p, vh.p, table.p, ids, step, cap, s, key, value, stream);
        if (mode == 1) k::kv_gather_q8_step(k8.p, v8.p, ks.p, vs.p, table.p, ids, step, cap, s, key, value, stream);
        if (mode == 2) k::kv_gather_q4_step(k4.p, v4.p, table.p, ids, step, cap, s, key, value, stream);
        if (mode == 3) {
            k::kv_gather_q8_step(k8.p, k8.p, ks.p, ks.p, table.p, ids, step, cap, s, key, key, stream);
            k::kv_gather_q4_step(v4.p, v4.p, table.p, ids, step, cap, s, value, value, stream);
        }
    }
    std::vector<uint8_t> snapshot() const {
        std::vector<uint8_t> bytes;
        auto add = [&](const auto& d) {
            const auto host = d.get();
            const auto* first = reinterpret_cast<const uint8_t*>(host.data());
            bytes.insert(bytes.end(), first, first + host.size() * sizeof(host[0]));
        };
        add(kh); add(vh); add(ks); add(vs); add(k8); add(v8); add(k4); add(v4); add(table);
        return bytes;
    }
    void restore(const std::vector<uint8_t>& bytes) {
        size_t at = 0;
        auto set = [&](auto& d) {
            const size_t count = d.n * sizeof(*d.p);
            need(at + count <= bytes.size(), "cache snapshot too short");
            ck(cudaMemcpy(d.p, bytes.data() + at, count, cudaMemcpyHostToDevice));
            at += count;
        };
        set(kh); set(vh); set(ks); set(vs); set(k8); set(v8); set(k4); set(v4); set(table);
        need(at == bytes.size(), "cache snapshot too long");
    }
    void zero() { kh.zero(); vh.zero(); ks.zero(); vs.zero(); k8.zero(); v8.zero(); k4.zero(); v4.zero(); }
};

void norm_gate(const k::QsaShapes& s, void* stream) {
    constexpr int rows = 3, heads = 24, dim = 256;
    std::vector<float> full(rows * heads * dim * 2), norm(dim, 1.0f), av(rows * heads * dim);
    std::vector<int32_t> pos(rows * heads);
    for (size_t i = 0; i < full.size(); ++i) full[i] = sample(i);
    for (size_t i = 0; i < av.size(); ++i) av[i] = sample(i, 0.5f);
    for (int z = 0; z < rows; ++z) for (int h = 0; h < heads; ++h) pos[z * heads + h] = z * 17;
    Device<float> df(full.size()), dn(dim), da(av.size()), qr(av.size()), gr(av.size());
    Device<int32_t> dp(pos.size());
    df.put(full); dn.put(norm); da.put(av); dp.put(pos);
    // Original layer separates the first query half, then native norm + rope.
    ck(cudaMemcpy2DAsync(qr.p, dim * sizeof(float), df.p, dim * 2 * sizeof(float),
                         dim * sizeof(float), rows * heads, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
    k::native_qsa_rms_norm_weighted(qr.p, dn.p, qr.p, dim, rows * heads, k::qsa_rms_eps(), stream);
    k::native_rope_apply(qr.p, qr.p, rows * heads, dim, 64, k::RopeScaling{}, dp.p, stream);
    k::native_qsa_gate_apply(da.p, df.p, gr.p, rows * heads, dim, stream);
    const auto refq = qr.get(), refg = gr.get();
    for (int r = 0; r < 3; ++r) {
        const auto sh = shard(s, r * 8);
        std::vector<float> local(rows * 8 * dim * 2), la(rows * 8 * dim);
        std::vector<int32_t> lp(rows * 8);
        for (int z = 0; z < rows; ++z) for (int h = 0; h < 8; ++h) {
            const size_t src = size_t(z * heads + r * 8 + h), dst = size_t(z * 8 + h);
            std::copy_n(full.data() + src * 512, 512, local.data() + dst * 512);
            std::copy_n(av.data() + src * 256, 256, la.data() + dst * 256);
            lp[dst] = pos[src];
        }
        Device<float> lf(local.size()), aa(la.size()), lq(la.size()), lg(la.size());
        Device<int32_t> pp(lp.size());
        lf.put(local); aa.put(la); pp.put(lp);
        tp::qsa_shard_prepare_query(lf.p, dn.p, lq.p, pp.p, sh, rows, k::RopeScaling{}, stream);
        tp::qsa_shard_gate(aa.p, lf.p, lg.p, sh, rows, stream);
        const auto gotq = lq.get(), gotg = lg.get();
        for (int z = 0; z < rows; ++z) for (int h = 0; h < 8; ++h) {
            const size_t src = size_t(z * heads + r * 8 + h) * dim, dst = size_t(z * 8 + h) * dim;
            need(std::memcmp(gotq.data() + dst, refq.data() + src, dim * sizeof(float)) == 0, "native norm/RoPE shard mismatch");
            need(std::memcmp(gotg.data() + dst, refg.data() + src, dim * sizeof(float)) == 0, "native gate shard mismatch");
        }
    }
    std::puts("native query norm/RoPE/gating: decode and three-row prefill exact");
}

void indexer(k::QsaShapes s, void* stream) {
    constexpr int cells = 32, max_blocks = cells / 4 + 2;
    const int cap = int(k::qsa_selection_width(k::kTopkMaxCells, s));
    Device<float> tail(3 * 128), dead(128), pooled(max_blocks * 128), raw(128), norm(128), qi(4 * 128);
    Device<float> cos(32 * 32), sin(32 * 32), scores(cells), ref_scores(cells);
    Device<int32_t> pos(1), block_pos(1), step(4), ids(cap), ref_ids(cap);
    tail.zero(); dead.zero(); pooled.zero(); block_pos.zero(); scores.zero(); ref_scores.zero();
    std::vector<float> hc(32 * 32), hs(hc.size()), nn(128, 1.0f), query(4 * 128);
    k::build_rope_table(64, k::qsa_freq_base(), cells, hc.data(), hs.data());
    cos.put(hc); sin.put(hs); norm.put(nn);
    for (size_t i = 0; i < query.size(); ++i) query[i] = sample(i, 1.0f);
    qi.put(query);
    k::QsaIndexerBuffers b{tail.p, dead.p, pooled.p, block_pos.p};
    for (int t = 0; t < cells; ++t) {
        std::vector<float> rr(128);
        for (size_t i = 0; i < rr.size(); ++i) rr[i] = sample(t * 128 + i, 0.2f);
        raw.put(rr); pos.put({t});
        k::indexer_key_append(raw.p, pos.p, 0, norm.p, k::qsa_rms_eps(), b, s, cos.p, sin.p, stream);
        if (t == 8 || t == 11 || t == 18 || t == 31) {
            int32_t st[4]; k::qsa_step_fill(st, t, s);
            step.put(std::vector<int32_t>(st, st + 4)); ids.zero(); ref_ids.zero();
            const auto ts = tail.get(), ds = dead.get(), ps = pooled.get();
            const auto bs = block_pos.get();
            k::qsa_index_step(pooled.p, qi.p, nullptr, s, step.p, max_blocks, ref_scores.p, stream);
            k::topk_512_step(ref_scores.p, s, cap, step.p, ref_ids.p, stream);
            for (int rank = 0; rank < 3; ++rank) {
                tp::qsa_shard_select(pooled.p, qi.p, nullptr, step.p, max_blocks, cap,
                                      shard(s, rank * 8), scores.p, ids.p, stream);
                exact(scores.get(), ref_scores.get(), "replicated indexer scores differ");
                exact(ids.get(), ref_ids.get(), "replicated selected ids differ");
            }
            exact(tail.get(), ts, "selection modified indexer tail");
            exact(dead.get(), ds, "selection modified indexer dead key");
            exact(pooled.get(), ps, "selection modified pooled keys");
            tail.zero(); dead.zero(); pooled.zero(); block_pos.zero();
            tail.put(ts); dead.put(ds); pooled.put(ps); block_pos.put(bs);
            tp::qsa_shard_select(pooled.p, qi.p, nullptr, step.p, max_blocks, cap,
                                  shard(s, 8), scores.p, ids.p, stream);
            exact(ids.get(), ref_ids.get(), "indexer snapshot reset differs");
        }
    }
    std::puts("replicated indexer: scores/ids exact, tail/block completion and snapshot/reset covered");
}

void attention(int mode, k::QsaShapes s, const Options& o, void* stream) {
    constexpr int cells = 32, rows = 3, heads = 24, dim = 256, cap = 11;
    Cache cache(mode, s, cells);
    Device<int32_t> append_step(4);
    Device<float> key(512), value(512);
    for (int t = 0; t < cells; ++t) {
        std::vector<float> kk(512), vv(512);
        for (size_t i = 0; i < kk.size(); ++i) {
            kk[i] = sample(size_t(t) * 512 + i, 0.25f);
            vv[i] = sample(size_t(t) * 512 + i, 1.25f) + float(i / 256) * 0.4f;
        }
        key.put(kk); value.put(vv);
        int32_t st[4]; k::qsa_step_fill(st, t, s);
        append_step.put(std::vector<int32_t>(st, st + 4));
        cache.append(append_step.p, key.p, value.p, stream);
    }
    const auto snapshot = cache.snapshot();
    std::vector<float> qq(rows * heads * dim);
    for (size_t i = 0; i < qq.size(); ++i) qq[i] = sample(i, 0.75f);
    const int widths[3] = {0, 7, cap}, positions[3] = {0, 18, 31};
    std::vector<int32_t> ids(rows * cap, -1), steps(rows * 4);
    for (int z = 0; z < rows; ++z) {
        k::qsa_step_fill(steps.data() + z * 4, positions[z], s);
        steps[z * 4 + k::kStepWidth] = widths[z];
        for (int j = 0; j < widths[z]; ++j) ids[z * cap + j] = j * 2;
    }
    Device<float> q(qq.size()), ref(qq.size()), fast(qq.size());
    Device<int32_t> di(ids.size()), ds(steps.size());
    Device<uint16_t> kg(cap * 512), vg(cap * 512);
    Device<float> rw(rows * heads * cap), scratch(k::qsa_decode_attn_scratch_floats(cap, s));
    q.put(qq); di.put(ids); ds.put(steps); rw.zero();
    for (int z = 0; z < rows; ++z) {
        cache.gather(di.p + z * cap, ds.p + z * 4, cap, kg.p, vg.p, stream);
        k::qsa_attend_step(q.p + z * heads * dim, kg.p, vg.p, ds.p + z * 4, cap, s,
                           ref.p + z * heads * dim, rw.p + z * heads * cap, stream);
        k::qsa_decode_attn_step(q.p + z * heads * dim, cache.pools(), di.p + z * cap, ds.p + z * 4,
                                cap, s, scratch.p, fast.p + z * heads * dim, stream);
    }
    const auto reference = ref.get(), fast_ref = fast.get(), ref_weights = rw.get();
    std::vector<tp::QsaShard> shards{shard(s, 0), shard(s, 8), shard(s, 16), shard(s, 10)};
    auto explicit_map = shard(s, 0);
    explicit_map.explicit_heads = true;
    for (int h = 0; h < 8; ++h) explicit_map.query_heads[h] = h % 2 ? h : 23 - h;
    shards.push_back(explicit_map);
    double max_abs = 0, max_rel = 0;
    for (const auto& sh : shards) {
        std::vector<float> local(rows * sh.local_heads * dim);
        for (int z = 0; z < rows; ++z) for (int h = 0; h < sh.local_heads; ++h)
            std::copy_n(qq.data() + (z * heads + global_head(sh, h)) * dim, dim,
                        local.data() + (z * sh.local_heads + h) * dim);
        Device<float> lq(local.size()), out(local.size()), weights(rows * sh.local_heads * cap);
        lq.put(local); weights.zero();
        tp::qsa_shard_attend(lq.p, cache.pools(), di.p, ds.p, cap, sh, out.p, weights.p, rows, stream);
        const auto result = out.get(), got_weights = weights.get();
        for (int z = 0; z < rows; ++z) for (int h = 0; h < sh.local_heads; ++h) {
            const size_t src = size_t(z * heads + global_head(sh, h)) * dim;
            const size_t dst = size_t(z * sh.local_heads + h) * dim;
            need(std::memcmp(result.data() + dst, reference.data() + src, dim * sizeof(float)) == 0,
                 "attention differs from original gathered QSA");
            for (int j = 0; j < widths[z]; ++j) {
                const float w = ref_weights[size_t(z) * heads * cap + global_head(sh, h) * widths[z] + j];
                need(got_weights[(z * sh.local_heads + h) * cap + j] == w, "attention weight mismatch");
            }
            for (int d = 0; d < dim; ++d) {
                const double error = std::abs(double(result[dst + d]) - fast_ref[src + d]);
                need(std::isfinite(error), "nonfinite attention output");
                max_abs = std::max(max_abs, error);
                max_rel = std::max(max_rel, error / std::max(1e-30, std::abs(double(fast_ref[src + d]))));
                if (!o.measure) need(error <= o.atol + o.rtol * std::abs(double(fast_ref[src + d])),
                                     "fast attention tolerance exceeded");
            }
        }
        // A single decode row and the row in the prefill batch must agree.
        tp::qsa_shard_attend(lq.p + 2 * sh.local_heads * dim, cache.pools(), di.p + 2 * cap, ds.p + 8,
                              cap, sh, out.p + 2 * sh.local_heads * dim, nullptr, 1, stream);
        exact(out.get(), result, "decode/prefill per-query reference mismatch");
        exact(cache.snapshot(), snapshot, "attention modified replicated cache state");
        cache.zero(); cache.restore(snapshot);
        tp::qsa_shard_attend(lq.p, cache.pools(), di.p, ds.p, cap, sh, out.p, nullptr, rows, stream);
        exact(out.get(), result, "cache snapshot reset mismatch");
    }
    // Future selections are explicitly masked. Compare with original attention
    // on the filtered list, not an oracle that uses the new mask implementation.
    auto sh = shard(s, 8);
    steps[4 + k::kStepPos] = 5; // ids 6,8,10,12 now name future tokens.
    ds.put(steps);
    std::vector<float> local(8 * dim);
    for (int h = 0; h < 8; ++h) std::copy_n(qq.data() + (heads + 8 + h) * dim, dim, local.data() + h * dim);
    Device<float> lq(local.size()), out(local.size()), weights(8 * cap);
    lq.put(local);
    tp::qsa_shard_attend(lq.p, cache.pools(), di.p + cap, ds.p + 4, cap, sh, out.p, weights.p, 1, stream);
    const auto causal = out.get(), cw = weights.get();
    auto filtered = steps;
    filtered[4 + k::kStepWidth] = 3;
    ds.put(filtered);
    cache.gather(di.p + cap, ds.p + 4, cap, kg.p, vg.p, stream);
    k::qsa_attend_step(q.p + heads * dim, kg.p, vg.p, ds.p + 4, cap, s, ref.p, nullptr, stream);
    const auto cr = ref.get();
    for (int h = 0; h < 8; ++h) {
        need(std::memcmp(causal.data() + h * dim, cr.data() + (8 + h) * dim, dim * sizeof(float)) == 0,
             "causal filtering differs from original QSA filtered selection");
        for (int j = 3; j < 7; ++j) need(cw[h * cap + j] == 0, "future cell has nonzero weight");
    }
    std::printf("mode=%d original gathered QSA exact; fast max_abs=%.9g max_rel=%.9g; decode/prefill, causal, cache reset covered\n",
                mode, max_abs, max_rel);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = options(argc, argv);
        contract();
        if (o.inspect) return 0;
        Stream stream;
        auto s = k::qsa_real_shapes();
        s.idx_top_k = 8;
        norm_gate(s, stream.p);
        indexer(s, stream.p);
        for (int mode = 0; mode < 4; ++mode) attention(mode, s, o, stream.p);
        ck(cudaStreamSynchronize(stream.p));
        std::puts(o.measure ? "MEASURED ONLY: fast-kernel tolerances require hardware qualification" : "PASS");
        return o.measure ? 77 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
