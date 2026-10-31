// GPU component qualification, not a distributed runtime or a prefill GEMM test.
// --measure-only reports errors and returns 77: tolerances are not frozen here.
// After inspecting hardware measurements, --atol X --rtol Y gates every sample.
// --gguf FILE [--gguf FILE] --layer N --expert E samples original GGUF rows.
#include "strata/kernels/tp_cuda.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace tp = strata::kernels::tp::cuda;
namespace d = strata::distributed;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
struct Stream {
    cudaStream_t s{};
    Stream() { ck(cudaStreamCreate(&s)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
};
template<class T> struct Device {
    T* p{};
    std::size_t size{};
    explicit Device(std::size_t n) : size(n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& v) {
        require(v.size() <= size, "upload overflow");
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> v(size);
        ck(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
int block(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }
const char* codec(int t) {
    switch (t) {
        case 16: return "IQ2_XXS"; case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS"; case 21: return "IQ3_S";
        case 22: return "IQ2_S"; case 23: return "IQ4_XS";
        case 20: return "IQ4_NL"; case 42: return "Q2_0";
        default: throw std::runtime_error("unsupported sample codec");
    }
}
d::ActivationContract contract(int ff) {
    d::ActivationContract c;
    c.consumer = "expert_down"; c.gather = "all_gather_padded";
    c.quantizer = "Q8_1"; c.block_size = 32; c.reference = "global_reference";
    c.backend_policy = "cuda_native_expert_grouped"; c.layout = "contiguous";
    c.valid_counts = ff == 640 ? std::vector<int64_t>{256, 192, 192} : std::vector<int64_t>{64, 64, 64};
    c.wire_slots = *std::max_element(c.valid_counts.begin(), c.valid_counts.end());
    return c;
}
tp::MatrixShard view(int ty, int input, int output, int axis, int begin, int end,
                    const void* bytes, std::size_t stride, std::size_t capacity) {
    d::TensorSlice s;
    s.global_shape = {input, output}; s.local_shape = s.global_shape;
    s.axis = axis; s.range_start = begin; s.range_end = end; s.local_shape[axis] = end - begin;
    s.rank = 0; s.valid_lanes = {end - begin};
    s.codec = codec(ty); s.block_size = block(ty); s.scale_kind = "embedded-original";
    s.byte_row_stride = k::iq_row_bytes(ty, input);
    return {s, ty, bytes, stride, capacity};
}
std::vector<uint8_t> random_plane(int ty, int rows, int width, std::mt19937& rng) {
    std::vector<uint8_t> result(std::size_t(rows) * k::iq_row_bytes(ty, width));
    for (auto& v : result) v = uint8_t(rng());
    const auto bs = k::iq_row_bytes(ty, block(ty));
    for (std::size_t p = 0; p < result.size(); p += bs) {
        const uint16_t scale = uint16_t((5 << 10) | (rng() & 1023));
        std::memcpy(result.data() + p, &scale, 2);
    }
    return result;
}

struct Options {
    bool measure = true;
    bool inspect = false;
    bool roundoff_bound = false;
    bool roundoff_selftest = false;
    bool exact_lane = false;
    bool exact_lane_selftest = false;
    double atol = 0, rtol = 0;
    int layer = 0, expert = 0;
    std::vector<std::string> gguf;
};
Options options(int argc, char** argv) {
    Options o;
    bool atol = false, rtol = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--measure-only") { o.measure = true; continue; }
        if (arg == "--inspect-sample") { o.inspect = true; continue; }
        if (arg == "--roundoff-bound") { o.roundoff_bound = true; continue; }
        if (arg == "--roundoff-selftest") { o.roundoff_selftest = true; continue; }
        if (arg == "--exact-lane-reference") { o.exact_lane = true; continue; }
        if (arg == "--exact-lane-selftest") { o.exact_lane_selftest = true; continue; }
        require(i + 1 < argc, "missing argument value");
        const std::string v = argv[++i];
        if (arg == "--atol") { o.atol = std::stod(v); atol = true; }
        else if (arg == "--rtol") { o.rtol = std::stod(v); rtol = true; }
        else if (arg == "--gguf") o.gguf.push_back(v);
        else if (arg == "--layer") o.layer = std::stoi(v);
        else if (arg == "--expert") o.expert = std::stoi(v);
        else throw std::runtime_error("unknown argument: " + arg);
    }
    require(atol == rtol, "supply both --atol and --rtol, or neither");
    require(std::isfinite(o.atol) && std::isfinite(o.rtol) && o.atol >= 0 && o.rtol >= 0,
            "invalid tolerance");
    require(o.layer >= 0 && o.expert >= 0, "negative sample index");
    if (atol) o.measure = false;
    return o;
}
struct Metrics {
    int failures = 0;
    const Options& o;
    void compare(const char* label, const std::vector<float>& ref, const std::vector<float>& got, bool exact = false,
                 const std::vector<double>* bounds = nullptr, const std::vector<double>* norm = nullptr,
                 bool diagnostic_only = false) {
        require(ref.size() == got.size(), "comparison size mismatch");
        double max_abs = 0, err2 = 0, ref2 = 0;
        std::size_t bad = 0;
        bool all_finite = true;
        std::size_t worst = 0, bound_bad = 0;
        double max_ratio = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double e = std::abs(double(got[i]) - ref[i]);
            const bool finite = std::isfinite(ref[i]) && std::isfinite(got[i]);
            all_finite = all_finite && finite;
            max_abs = std::max(max_abs, finite ? e : std::numeric_limits<double>::infinity());
            const double limit = bounds ? (*bounds)[i] : o.atol + o.rtol * std::abs(double(ref[i]));
            const double ratio = limit > 0 ? e / limit : (e ? std::numeric_limits<double>::infinity() : 0);
            if (ratio > max_ratio) { max_ratio = ratio; worst = i; }
            if (bounds && (!finite || e > limit)) ++bound_bad;
            err2 += e * e; ref2 += double(ref[i]) * ref[i];
            if (!finite || (exact ? std::memcmp(&got[i], &ref[i], sizeof(float)) != 0 :
                              e > o.atol + o.rtol * std::abs(double(ref[i])))) ++bad;
        }
        std::printf("%s max_abs=%.9g rms=%.9g relative_l2=%.9g outside_limits=%zu exact=%d\n",
                    label, max_abs, std::sqrt(err2 / ref.size()), std::sqrt(err2 / std::max(ref2, 1e-300)), bad, exact);
        if (diagnostic_only) std::printf("  diagnostic_only=1 gate=not-selected-arithmetic\n");
        if (bad || bounds) std::printf("  worst_lane=%zu expected=%.9g got=%.9g magnitude=%.9g limit=%.9g error_over_limit=%.9g derived_bound=%d outside_bound=%zu\n",
            worst, double(ref[worst]), double(got[worst]), std::max(std::abs(double(ref[worst])), std::abs(double(got[worst]))),
            bounds ? (*bounds)[worst] : o.atol + o.rtol * std::abs(double(ref[worst])), max_ratio, bounds != nullptr, bound_bad);
        if (norm) std::printf("  worst_lane_sum_abs_terms=%.17g cancellation_condition=%.9g\n", (*norm)[worst],
            (*norm)[worst] / std::max(std::abs(double(ref[worst])), 1e-300));
        const auto rejected = bounds && o.roundoff_bound && !exact ? bound_bad : bad;
        if (!all_finite || (!diagnostic_only && rejected && (exact || !o.measure))) ++failures;
    }
};

// Minimal read-only GGUF v2/v3 metadata reader. Offsets come from the actual
// file, not an assumed rank-pack manifest. Only the requested expert is read.
template<class T> T read(std::ifstream& f) {
    T v{}; f.read(reinterpret_cast<char*>(&v), sizeof(v));
    require(bool(f), "truncated GGUF"); return v;
}
std::string string(std::ifstream& f) {
    const auto n = read<uint64_t>(f);
    require(n < (1u << 24), "GGUF string too large");
    std::string v(std::size_t(n), '\0'); f.read(v.data(), std::streamsize(n));
    require(bool(f), "truncated GGUF string"); return v;
}
void skip_value(std::ifstream& f, uint32_t type, int depth = 0) {
    require(depth < 8, "GGUF nested metadata too deep");
    static constexpr int sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    require(type < std::size(sizes), "unknown GGUF metadata type");
    if (type == 8) { (void)string(f); return; }
    if (type == 9) {
        const auto element = read<uint32_t>(f); const auto count = read<uint64_t>(f);
        require(count < (1ull << 32), "GGUF array too large");
        for (uint64_t i = 0; i < count; ++i) skip_value(f, element, depth + 1);
        return;
    }
    f.seekg(sizes[type], std::ios::cur); require(bool(f), "truncated GGUF metadata");
}
struct Tensor {
    std::string path, name;
    std::vector<uint64_t> shape;
    int type = -1;
    uint64_t offset = 0;
};
std::vector<Tensor> tensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary); require(bool(f), "cannot open GGUF");
    require(read<uint32_t>(f) == 0x46554747, "not little-endian GGUF");
    const auto version = read<uint32_t>(f); require(version == 2 || version == 3, "unsupported GGUF version");
    const auto nt = read<uint64_t>(f), nk = read<uint64_t>(f);
    require(nt < (1u << 20) && nk < (1u << 20), "GGUF header unreasonable");
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < nk; ++i) {
        const auto key = string(f); const auto type = read<uint32_t>(f);
        if (key == "general.alignment") {
            require(type == 4, "GGUF alignment must be uint32"); alignment = read<uint32_t>(f);
        } else skip_value(f, type);
    }
    require(alignment && alignment <= (1u << 20) && !(alignment & (alignment - 1)), "bad GGUF alignment");
    std::vector<Tensor> result;
    for (uint64_t i = 0; i < nt; ++i) {
        Tensor t; t.path = path; t.name = string(f);
        const auto nd = read<uint32_t>(f); require(nd > 0 && nd <= 4, "bad GGUF dimensions");
        for (uint32_t j = 0; j < nd; ++j) t.shape.push_back(read<uint64_t>(f));
        t.type = int(read<uint32_t>(f)); t.offset = read<uint64_t>(f); result.push_back(t);
    }
    const auto pos = uint64_t(f.tellg()); const auto data = (pos + alignment - 1) & ~(alignment - 1);
    for (auto& t : result) t.offset += data;
    return result;
}
std::vector<uint8_t> expert_rows(const Tensor& t, int expert) {
    require(t.shape.size() == 3 && uint64_t(expert) < t.shape[2] && t.shape[0] <= 16384 && t.shape[1] <= 16384,
            "unsupported sampled expert shape/index");
    require(t.shape[0] % block(t.type) == 0, "sample splits codec block");
    const auto bytes = k::iq_row_bytes(t.type, int64_t(t.shape[0])) * t.shape[1];
    std::vector<uint8_t> v(bytes);
    std::ifstream f(t.path, std::ios::binary); f.seekg(std::streamoff(t.offset + uint64_t(expert) * bytes));
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes)); require(bool(f), "truncated expert plane");
    std::printf("sample tensor=%s file=%s expert=%d offset=%llu bytes=%zu type=%d\n", t.name.c_str(), t.path.c_str(),
                expert, static_cast<unsigned long long>(t.offset + uint64_t(expert) * bytes), v.size(), t.type);
    return v;
}

struct Sample { int gu, down, input, ff; std::vector<uint8_t> blob; };

double half_value(const uint8_t* bytes) {
    const unsigned h = unsigned(bytes[0]) | unsigned(bytes[1]) << 8;
    const unsigned exponent = (h >> 10) & 31, mantissa = h & 1023;
    require(exponent != 31, "nonfinite scale in independent roundoff analysis");
    const double magnitude = exponent ? std::ldexp(double(1024 + mantissa), int(exponent) - 25)
                                      : std::ldexp(double(mantissa), -24);
    return h & 0x8000 ? -magnitude : magnitude;
}
double gamma(int depth) {
    const double nu = depth * std::ldexp(1.0, -24);
    require(nu < 1, "invalid floating point error depth");
    return nu / (1 - nu);
}
struct DownRoundoff { std::vector<double> norm, error; std::vector<float> scalar; };
DownRoundoff down_roundoff(const Sample& s, const k::NativeExpertLayout& layout,
                          const std::vector<uint8_t>& activation, int rows) {
    // Independent scalar interpretation of ORIGINAL GGUF codes and Q8_1 bytes.
    // Each exact dot term has an integer sum (<= 16/32*127*127), two half
    // scales whose product is exact in float, then one rounded multiply/FMA.
    // The error of each summation DAG is gamma_depth * sum(abs(exact terms)).
    // Monolithic: ceil(calls/32) lane additions + 5 butterfly adds + product.
    // TP: same maximum depth, plus two rank additions. Comparing both DAGs
    // gives (gamma_d + gamma_(d+2))*A. No observed output/error sets this bound.
    static constexpr int table[] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const int qk = block(s.down), calls = s.ff / qk * 2;
    const int depth = (calls + 31) / 32 + 5 + 1;
    DownRoundoff result{std::vector<double>(std::size_t(rows) * s.input),
                        std::vector<double>(std::size_t(rows) * s.input),
                        std::vector<float>(std::size_t(rows) * s.input)};
    for (int token = 0; token < rows; ++token) for (int out = 0; out < s.input; ++out) {
        double norm = 0;
        double exact_sum = 0;
        const auto* wrow = s.blob.data() + layout.down_off + std::size_t(out) * layout.d_row;
        for (int call = 0; call < calls; ++call) {
            const int b = call / 2, part = call % 2;
            const auto* w = wrow + std::size_t(b) * k::iq_row_bytes(s.down, qk);
            const auto* a = activation.data() + std::size_t(token) * (s.ff / 32) * 36;
            int integer_dot = 0;
            if (s.down == 20) {
                a += b * 36;
                for (int j = 0; j < 16; ++j) {
                    const int lane = part * 8 + j % 8 + (j / 8) * 16;
                    const int code = (w[2 + lane % 16] >> (lane / 16 * 4)) & 15;
                    integer_dot += table[code] * int(static_cast<int8_t>(a[4 + lane]));
                }
            } else {
                a += (b * 2 + part) * 36;
                for (int j = 0; j < 32; ++j) {
                    const int lane = part * 32 + j;
                    const int code = (w[2 + lane / 4] >> ((lane % 4) * 2)) & 3;
                    integer_dot += (code - 1) * int(static_cast<int8_t>(a[4 + j]));
                }
            }
            const double term = half_value(w) * half_value(a) * integer_dot;
            norm += std::abs(term); exact_sum += term;
        }
        const auto i = std::size_t(token) * s.input + out;
        result.norm[i] = norm;
        result.error[i] = (gamma(depth) + gamma(depth + 2)) * norm;
        result.scalar[i] = float(exact_sum);
    }
    std::printf("roundoff-policy calls=%d lane_plus_tree_product_depth=%d rank_additions=2 u=2^-24 formula=(gamma(d)+gamma(d+2))*sum_abs_original_terms\n",
                calls, depth);
    return result;
}
void roundoff_selftest() {
    for (int type : {20, 42}) {
        const auto l = k::native_expert_layout(21, type, 256, 192);
        Sample s{21, type, 256, 192, std::vector<uint8_t>(l.bytes, 0)};
        for (std::size_t o = l.down_off; o < l.bytes; o += k::iq_row_bytes(type, block(type))) {
            s.blob[o] = 0; s.blob[o + 1] = 0x3c;
        }
        std::vector<uint8_t> a(6 * 36, 0);
        for (int b = 0; b < 6; ++b) {
            a[b * 36 + 1] = 0x3c;
            std::fill(a.begin() + b * 36 + 4, a.begin() + (b + 1) * 36, 1);
        }
        const auto result = down_roundoff(s, l, a, 1);
        const double expected = 192 * (type == 20 ? -127.0 : -1.0);
        for (std::size_t i = 0; i < result.scalar.size(); ++i) {
            require(result.scalar[i] == expected && result.norm[i] == -expected && result.error[i] > 0,
                    "independent original-code scalar/bound selftest failed");
        }
    }
    std::puts("independent-roundoff-host-selftest passed GPU_NOT_RUN");
}
void exact_lane_selftest() {
    // A rounded dot product cannot reproduce the native FMA across ranks.
    const float x = 1.f + std::ldexp(1.f, -23), y = 1.f - std::ldexp(1.f, -23);
    const volatile float rounded_product = x * y;
    require(std::fma(x, y, -1.f) != rounded_product - 1.f, "FMA counterexample disappeared");
    const float finish_input = std::nextafter(1.f, 2.f), finish_bias = -.75f;
    const volatile float finish_product = finish_input * .75f;
    const float finish_separate = finish_product + finish_bias;
    require(finish_separate != std::fma(finish_input, .75f, finish_bias),
            "finish two-rounding counterexample disappeared");
    for (int type : {20, 42}) for (int width : {192, 640, 1024}) {
        const auto counts = width == 192 ? std::array<int,3>{64,64,64} :
                            width == 640 ? std::array<int,3>{256,192,192} : std::array<int,3>{384,320,320};
        const int calls = width / block(type) * 2;
        require(calls <= 64, "exact lane host geometry exceeds two slots");
        for (int lane = 0; lane < 32; ++lane) {
            int previous_rank = -1;
            for (int call = lane; call < calls; call += 32) {
                const int input = (call / 2) * block(type);
                const int owner = input < counts[0] ? 0 : input < counts[0] + counts[1] ? 1 : 2;
                int matches = 0, begin = 0;
                for (int rank = 0; rank < 3; ++rank) {
                    if (input >= begin && input < begin + counts[rank]) ++matches;
                    begin += counts[rank];
                }
                require(matches == 1 && owner >= previous_rank && call / 32 < 2,
                        "exact lane call ownership/order is ambiguous");
                previous_rank = owner;
            }
        }
    }
    std::puts("exact-lane-host-ownership-and-FMA-selftest passed GPU_NOT_RUN");
    std::puts("finish-host-separate-rounding-vs-FMA-selftest passed GPU_NOT_RUN");
}
Sample sample_real(const Options& o) {
    std::vector<Tensor> all;
    for (const auto& p : o.gguf) { const auto ts = tensors(p); all.insert(all.end(), ts.begin(), ts.end()); }
    const std::string base = "blk." + std::to_string(o.layer) + ".ffn_";
    auto find = [&](const std::string& name) -> const Tensor& {
        const Tensor* found = nullptr;
        for (const auto& t : all) if (t.name == name) { require(!found, "duplicate GGUF tensor"); found = &t; }
        require(found, "sample tensor missing from supplied GGUF splits"); return *found;
    };
    const auto& g = find(base + "gate_exps.weight"); const auto& u = find(base + "up_exps.weight");
    const auto& down = find(base + "down_exps.weight");
    require(g.shape.size() == 3 && down.shape.size() == 3 && g.shape == u.shape && g.type == u.type &&
            g.shape[0] == down.shape[1] && g.shape[1] == down.shape[0] && g.shape[2] == down.shape[2],
            "sample expert planes disagree");
    Sample s{g.type, down.type, int(g.shape[0]), int(g.shape[1]), expert_rows(g, o.expert)};
    const auto up = expert_rows(u, o.expert), dp = expert_rows(down, o.expert);
    s.blob.insert(s.blob.end(), up.begin(), up.end()); s.blob.insert(s.blob.end(), dp.begin(), dp.end());
    return s;
}

__global__ void rank_sum(const float* a, const float* b, const float* c, float* out, int n) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    if (i < n) out[i] = (a[i] + b[i]) + c[i];
}
__global__ void fill_parts(const float* source, float* parts, int n, int width) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    if (i < n) {
        const int token = i / (10 * width), expert = (i / width) % 10, lane = i % width;
        parts[i] = expert == 8 ? 0.f : source[token * width + lane];
    }
}

// Independent scalar policy control. Native finish applies optional scale and
// bias as separate branches/roundings (FMUL then FADD in the compiled kernel).
// Explicit intrinsics prevent the test compiler from contracting this oracle.
__global__ void finish_reference(const float* input, float* output, const float* bias, int n, int width) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    if (i < n) output[i] = __fadd_rn(__fmul_rn(input[i], .75f), bias[i % width]);
}

void tiny_scalar_probe() {
    // Independent hand-coded bytes: activation q=1,d=1; IQ4_NL code 0
    // means -127, Q2_0 code 0 means -1. Every dot is an exact integer.
    Stream stream;
    auto c = contract(192);
    std::vector<uint8_t> q(6 * 36, 0);
    const uint16_t one = 0x3c00;
    for (int b = 0; b < 6; ++b) {
        std::memcpy(q.data() + b * 36, &one, 2);
        std::fill(q.begin() + b * 36 + 4, q.begin() + (b + 1) * 36, uint8_t(1));
    }
    Device<uint8_t> dq(q.size()); dq.upload(q);
    tp::GlobalQ81 a{dq.p, 192, 1, q.size(), q.size()};
    for (int ty : {20, 42}) {
        const auto rb = k::iq_row_bytes(ty, 192), bs = k::iq_row_bytes(ty, block(ty));
        std::vector<uint8_t> w(3 * rb, 0);
        for (std::size_t offset = 0; offset < w.size(); offset += bs) std::memcpy(w.data() + offset, &one, 2);
        Device<uint8_t> dw(w.size()); dw.upload(w);
        Device<float> out(3);
        std::vector<float> sum(3, 0.f);
        for (int rank = 0; rank < 3; ++rank) {
            const auto offset = k::iq_row_bytes(ty, rank * 64);
            tp::down(view(ty, 192, 3, 0, rank * 64, (rank + 1) * 64, dw.p + offset,
                          rb, dw.size - offset), c, a, out.p, stream.s);
            ck(cudaStreamSynchronize(stream.s)); const auto values = out.read();
            for (int row = 0; row < 3; ++row) {
                require(values[row] == (ty == 20 ? -127.f : -1.f) * 64.f, "tiny scalar owned dot mismatch");
                sum[row] += values[row];
            }
        }
        for (float value : sum) require(value == (ty == 20 ? -127.f : -1.f) * 192.f, "tiny scalar rank sum mismatch");
        auto invalid = c; invalid.quantizer = "Q8_K"; invalid.block_size = 256;
        bool rejected = false;
        try { tp::down(view(ty, 192, 3, 0, 0, 192, dw.p, rb, dw.size), invalid, a, out.p, stream.s); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "CPU quantizer was silently accepted as CUDA Q8_1");
        auto misaligned = a;
        misaligned.data = dq.p + 1;
        rejected = false;
        try { tp::down(view(ty, 192, 3, 0, 0, 192, dw.p, rb, dw.size), c, misaligned, out.p, stream.s); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "misaligned Q8_1 base accepted");
        misaligned = a; misaligned.rows = 2; misaligned.row_stride = q.size() + 1;
        misaligned.bytes = 2 * misaligned.row_stride;
        rejected = false;
        try { tp::down(view(ty, 192, 3, 0, 0, 192, dw.p, rb, dw.size), c, misaligned, out.p, stream.s); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "misaligned Q8_1 token stride accepted");
    }
    std::puts("tiny-scalar-independent-bytes passed");
}

void run_expert_batch(const Sample& sample, int seed, Metrics& metrics) {
    constexpr int experts = 10;
    const int width = sample.input, ff = sample.ff;
    const auto layout = k::native_expert_layout(sample.gu, sample.down, width, ff);
    const auto gather = contract(ff);
    auto input_contract = gather; input_contract.gather = "none";
    Stream stream;
    std::mt19937 rng(0x4ba7 + seed);
    std::vector<float> x(width);
    for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 503.f;
    Device<float> input(width); input.upload(x);
    Device<uint8_t> input_q(tp::global_q81_bytes(1, width));
    const auto q = tp::quantize_global(input_contract, input.p, 1, width, input_q.p, input_q.size, stream.s);
    Device<float> wire(3 * experts * gather.wire_slots), hidden(experts * ff), reference_hidden(hidden.size);
    Device<uint8_t> hidden_q(experts * tp::global_q81_bytes(1, ff)), reference_q(hidden_q.size);
    const auto lane_count = std::size_t(width) * tp::down_lane_payload_width;
    Device<float> lanes(3 * experts * lane_count), output(experts * width), sequential(output.size), oracle(output.size);
    Device<float> sequential_wire(3 * lane_count), one_wire(3 * gather.wire_slots);
    wire.upload(std::vector<float>(wire.size, std::numeric_limits<float>::quiet_NaN()));
    std::vector<std::unique_ptr<Device<uint8_t>>> weights;
    // Row permutations select distinct expert planes without altering any quantized block bytes.
    for (int expert = 0; expert < experts; ++expert) {
        auto bytes = sample.blob;
        for (int row = 0; row < ff; ++row) {
            const auto source = std::size_t((row + expert) % ff) * layout.gu_row;
            std::memcpy(bytes.data() + std::size_t(row) * layout.gu_row, sample.blob.data() + source, layout.gu_row);
            std::memcpy(bytes.data() + layout.up_off + std::size_t(row) * layout.gu_row,
                        sample.blob.data() + layout.up_off + source, layout.gu_row);
        }
        for (int row = 0; row < width; ++row)
            std::memcpy(bytes.data() + layout.down_off + std::size_t(row) * layout.d_row,
                        sample.blob.data() + layout.down_off + std::size_t((row + expert) % width) * layout.d_row,
                        layout.d_row);
        auto blob = std::make_unique<Device<uint8_t>>(bytes.size()); blob->upload(bytes);
        Device<unsigned long long> ptr(1); ptr.upload({reinterpret_cast<unsigned long long>(blob->p)});
        Device<int32_t> starts(2), groups(1), dest(1), token(1);
        starts.upload({0, 1}); groups.upload({1}); dest.upload({0}); token.upload({0});
        Device<uint8_t> scratch(k::native_expert_scratch_bytes(1, ff));
        k::native_expert_grouped(layout, ptr.p, starts.p, groups.p, dest.p, token.p, 1, 1,
                                 input_q.p, scratch.p, oracle.p + expert * width, stream.s);
        int begin = 0;
        for (int rank = 0; rank < 3; ++rank) {
            const int count = int(gather.valid_counts[rank]), end = begin + count;
            Device<float> gate(count), up(count), h(count);
            const auto offset = std::size_t(begin) * layout.gu_row;
            tp::gate_up(view(sample.gu, width, ff, 1, begin, end, blob->p + offset,
                             layout.gu_row, layout.up_off - offset),
                        view(sample.gu, width, ff, 1, begin, end, blob->p + layout.up_off + offset,
                             layout.gu_row, layout.up_off - offset), input_contract, q, gate.p, up.p, h.p, stream.s);
            ck(cudaMemcpyAsync(wire.p + (rank * experts + expert) * gather.wire_slots, h.p,
                               count * sizeof(float), cudaMemcpyDeviceToDevice, stream.s));
            ck(cudaMemcpyAsync(one_wire.p + rank * gather.wire_slots, h.p,
                               count * sizeof(float), cudaMemcpyDeviceToDevice, stream.s));
            begin = end;
        }
        tp::unpack_gathered(gather, one_wire.p, 1, reference_hidden.p + expert * ff, stream.s);
        ck(cudaStreamSynchronize(stream.s));
        weights.push_back(std::move(blob));
    }
    tp::unpack_expert_batch(gather, wire.p, experts, hidden.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    metrics.compare("batch10-hidden-vs-sequential-poison-padding", reference_hidden.read(), hidden.read(), true);
    const auto qbytes = tp::global_q81_bytes(1, ff);
    for (int expert = 0; expert < experts; ++expert) {
        const auto hq = tp::quantize_global(gather, hidden.p + expert * ff, 1, ff,
                                           hidden_q.p + expert * qbytes, qbytes, stream.s);
        const auto sq = tp::quantize_global(gather, reference_hidden.p + expert * ff, 1, ff,
                                           reference_q.p + expert * qbytes, qbytes, stream.s);
        int begin = 0;
        for (int rank = 0; rank < 3; ++rank) {
            const int end = begin + int(gather.valid_counts[rank]);
            const auto offset = k::iq_row_bytes(sample.down, begin);
            const auto down = view(sample.down, ff, width, 0, begin, end,
                weights[expert]->p + layout.down_off + offset, layout.d_row,
                layout.bytes - layout.down_off - offset);
            tp::down_lanes(down, gather, hq, lanes.p + (rank * experts + expert) * lane_count, stream.s);
            tp::down_lanes(down, gather, sq, sequential_wire.p + rank * lane_count, stream.s);
            begin = end;
        }
        tp::finish_down_lanes(sequential_wire.p, sample.down, ff, width, 1,
            {uint64_t(gather.valid_counts[0]), uint64_t(gather.valid_counts[1]), uint64_t(gather.valid_counts[2])},
            sequential.p + expert * width, stream.s);
    }
    tp::finish_down_lanes(lanes.p, sample.down, ff, width, experts,
        {uint64_t(gather.valid_counts[0]), uint64_t(gather.valid_counts[1]), uint64_t(gather.valid_counts[2])},
        output.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    require(hidden_q.read() == reference_q.read(), "batch10 global hidden Q8_1 bytes differ");
    metrics.compare("batch10-exact-operands-vs-sequential", sequential.read(), output.read(), true);
    metrics.compare("batch10-exact-operands-vs-native-experts", oracle.read(), output.read(), true);
    bool rejected = false;
    try { tp::unpack_expert_batch(gather, wire.p, experts, wire.p, stream.s); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "batch hidden unpack accepted aliased buffers");
}

void run(const Sample& sample, int rows, Metrics& metrics) {
    const int width = sample.input, ff = sample.ff;
    require(ff == 192 || ff == 640, "test qualification supports FF192 tiny or FF640 original geometry");
    const auto layout = k::native_expert_layout(sample.gu, sample.down, width, ff);
    require(layout.bytes == sample.blob.size(), "native blob size mismatch");
    Stream stream;
    Device<uint8_t> blob(layout.bytes); blob.upload(sample.blob);
    std::mt19937 rng(0x5a17);
    std::vector<float> x(std::size_t(rows) * width);
    for (auto& v : x) v = float(int(rng() % 2001) - 1000) / 503.f;
    Device<float> dx(x.size()); dx.upload(x);
    auto c = contract(ff); c.gather = "none"; c.consumer = "expert_gate_up";
    Device<uint8_t> qx(tp::global_q81_bytes(rows, width));
    const auto input = tp::quantize_global(c, dx.p, rows, width, qx.p, qx.size, stream.s);
    Device<uint8_t> qx_native(qx.size);
    k::quantize_q8_1_rows(dx.p, rows, width, qx_native.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    require(qx.read() == qx_native.read(), "global input Q8_1 byte mismatch");

    // Independent existing grouped path consumes the unmodified monolithic blob.
    std::vector<int32_t> indexes(rows); for (int i = 0; i < rows; ++i) indexes[i] = i;
    Device<unsigned long long> ptr(1); ptr.upload({reinterpret_cast<unsigned long long>(blob.p)});
    Device<int32_t> starts(2), groups(1), dest(rows), token(rows);
    starts.upload({0, rows}); groups.upload({1}); dest.upload(indexes); token.upload(indexes);
    Device<uint8_t> scratch(k::native_expert_scratch_bytes(rows, ff));
    Device<float> oracle(std::size_t(rows) * width);
    k::native_expert_grouped(layout, ptr.p, starts.p, groups.p, dest.p, token.p, 1, rows,
                             qx_native.p, scratch.p, oracle.p, stream.s);

    auto gather = contract(ff);
    Device<float> wire(std::size_t(rows) * 3 * gather.wire_slots), hidden(std::size_t(rows) * ff);
    std::vector<float> poison(wire.size, std::numeric_limits<float>::quiet_NaN()); wire.upload(poison);
    std::array<std::unique_ptr<Device<float>>, 3> partials;
    Device<float> gathered_lanes(3 * std::size_t(rows) * width * tp::down_lane_payload_width);
    int begin = 0;
    for (int rank = 0; rank < 3; ++rank) {
        const int count = int(gather.valid_counts[rank]), end = begin + count;
        Device<float> gate(std::size_t(rows) * count), up(gate.size), h(gate.size);
        const auto offset = std::size_t(begin) * layout.gu_row;
        const auto g = view(sample.gu, width, ff, 1, begin, end, blob.p + offset, layout.gu_row, layout.up_off - offset);
        const auto u = view(sample.gu, width, ff, 1, begin, end, blob.p + layout.up_off + offset,
                            layout.gu_row, layout.down_off - layout.up_off - offset);
        tp::gate_up(g, u, c, input, gate.p, up.p, h.p, stream.s);
        ck(cudaMemcpy2DAsync(wire.p + rank * gather.wire_slots, 3 * gather.wire_slots * sizeof(float), h.p,
                            count * sizeof(float), count * sizeof(float), rows, cudaMemcpyDeviceToDevice, stream.s));
        ck(cudaStreamSynchronize(stream.s));
        begin = end;
    }
    tp::unpack_gathered(gather, wire.p, rows, hidden.p, stream.s);
    Device<uint8_t> hq(tp::global_q81_bytes(rows, ff));
    const auto h_image = tp::quantize_global(gather, hidden.p, rows, ff, hq.p, hq.size, stream.s);

    // Full-width single-rank arithmetic control plus exact hidden Q8_1 bytes.
    Device<float> full_g(std::size_t(rows) * ff), full_u(full_g.size), full_h(full_g.size), control(oracle.size);
    const auto g = view(sample.gu, width, ff, 1, 0, ff, blob.p, layout.gu_row, layout.up_off);
    const auto u = view(sample.gu, width, ff, 1, 0, ff, blob.p + layout.up_off, layout.gu_row, layout.up_off);
    tp::gate_up(g, u, c, input, full_g.p, full_u.p, full_h.p, stream.s);
    Device<float> gs(1), us(1), scaled_h(full_h.size);
    gs.upload({.5f}); us.upload({2.f});
    tp::gate_up(g, u, c, input, full_g.p, full_u.p, scaled_h.p, stream.s, gs.p, us.p);
    Device<uint8_t> full_hq(hq.size);
    k::quantize_q8_1_rows(full_h.p, rows, ff, full_hq.p, stream.s);
    k::iq_mmvq(sample.down, blob.p + layout.down_off, full_hq.p, control.p, ff, width, rows, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    metrics.compare("gathered-hidden", full_h.read(), hidden.read(), true);
    require(hq.read() == full_hq.read(), "hidden Q8_1 bytes differ after gather");
    metrics.compare("single-rank-native-control", oracle.read(), control.read(), true);
    Device<float> full_down(oracle.size);
    tp::down(view(sample.down, ff, width, 0, 0, ff, blob.p + layout.down_off,
                  layout.d_row, layout.bytes - layout.down_off), gather, h_image, full_down.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    metrics.compare("new-down-vs-independent-native-dot", control.read(), full_down.read(), true);
    const auto gate_host = full_g.read(), up_host = full_u.read(), scaled_host = scaled_h.read();
    std::vector<float> scaled_ref(scaled_host.size());
    for (std::size_t i = 0; i < scaled_ref.size(); ++i) {
        const float gv = gate_host[i] * .5f, uv = up_host[i] * 2.f;
        scaled_ref[i] = (gv / (1.f + std::exp(-gv))) * uv;
    }
    metrics.compare("gate-up-device-scales-vs-scalar", scaled_ref, scaled_host);

    begin = 0;
    for (int rank = 0; rank < 3; ++rank) {
        const int end = begin + int(gather.valid_counts[rank]);
        const auto offset = k::iq_row_bytes(sample.down, begin);
        const auto row_bytes = k::iq_row_bytes(sample.down, end - begin);
        // Exercise compact rank-pack rows, retaining every original block byte.
        Device<uint8_t> compact(std::size_t(width) * row_bytes);
        ck(cudaMemcpy2DAsync(compact.p, row_bytes, blob.p + layout.down_off + offset, layout.d_row,
                            row_bytes, width, cudaMemcpyDeviceToDevice, stream.s));
        partials[rank] = std::make_unique<Device<float>>(oracle.size);
        const bool original_stride = rank == 1;
        const auto w = original_stride ?
            view(sample.down, ff, width, 0, begin, end, blob.p + layout.down_off + offset,
                 layout.d_row, layout.bytes - layout.down_off - offset) :
            view(sample.down, ff, width, 0, begin, end, compact.p, row_bytes, compact.size);
        tp::down(w,
                 gather, h_image, partials[rank]->p, stream.s);
        tp::down_lanes(w, gather, h_image,
            gathered_lanes.p + rank * std::size_t(rows) * width * tp::down_lane_payload_width, stream.s);
        ck(cudaStreamSynchronize(stream.s));
        begin = end;
    }
    Device<float> reduced(oracle.size);
    rank_sum<<<unsigned((oracle.size + 255) / 256), 256, 0, stream.s>>>(partials[0]->p, partials[1]->p,
        partials[2]->p, reduced.p, int(oracle.size));
    ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream.s));
    std::printf("case gu=%s down=%s width=%d ff=%d rows=%d path=%s\n", codec(sample.gu), codec(sample.down),
                width, ff, rows, rows == 1 ? "decode" : "prefill-mmvq-reference");
    auto roundoff = down_roundoff(sample, layout, hq.read(), rows);
    metrics.compare("independent-scalar-original-block-dot", roundoff.scalar, oracle.read(), false, &roundoff.error,
                    &roundoff.norm, metrics.o.exact_lane);
    metrics.compare("rank-sum-vs-native-expert", oracle.read(), reduced.read(), false, &roundoff.error, &roundoff.norm,
                    metrics.o.exact_lane);
    Device<float> exact_output(oracle.size);
    const std::array<uint64_t, 3> valid{uint64_t(gather.valid_counts[0]), uint64_t(gather.valid_counts[1]),
                                       uint64_t(gather.valid_counts[2])};
    tp::finish_down_lanes(gathered_lanes.p, sample.down, ff, width, rows, valid, exact_output.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    metrics.compare("exact-lane-vs-native-expert", oracle.read(), exact_output.read(), true);
    Device<uint8_t> oracle_q(tp::global_q81_bytes(rows, width)), exact_q(oracle_q.size);
    k::quantize_q8_1_rows(oracle.p, rows, width, oracle_q.p, stream.s);
    k::quantize_q8_1_rows(exact_output.p, rows, width, exact_q.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    require(oracle_q.read() == exact_q.read(), "exact lane output Q8_1 codes mismatch");
    if (metrics.o.exact_lane) {
        ck(cudaMemcpyAsync(reduced.p, exact_output.p, oracle.size * sizeof(float), cudaMemcpyDeviceToDevice, stream.s));
    }

    // Ordered top-k10, per-expert scales, a nonzero shared contribution, a zero
    // part and an unselected route (weight zero), independent native combine.
    const int n = rows * 10 * width;
    Device<float> parts(n), reference_parts(n), scaled(n), route(rows * 10), scales(10), shared(oracle.size);
    Device<float> combined(oracle.size), ref_combined(oracle.size), scale(1), bias(width);
    const std::vector<float> sv{1.f, .5f, 2.f, 1.5f, .25f, 1.f, 1.f, 2.f, 1.f, .75f}; scales.upload(sv);
    std::vector<float> rw(rows * 10), sh(oracle.size), bs(width);
    for (int t = 0; t < rows; ++t) for (int e = 0; e < 10; ++e) rw[t * 10 + e] = e == 9 ? 0.f : float(e + 1) / 55.f;
    for (std::size_t i = 0; i < sh.size(); ++i) sh[i] = float(int(i % 13) - 6) / 97.f;
    for (int i = 0; i < width; ++i) bs[i] = float(i % 7) / 103.f;
    route.upload(rw); shared.upload(sh); bias.upload(bs); scale.upload({.75f});
    auto projected_ref = oracle.read();
    for (std::size_t i = 0; i < projected_ref.size(); ++i)
    {
        const volatile float rounded_product = projected_ref[i] * .75f;
        projected_ref[i] = rounded_product + bs[i % width];
    }
    Device<float> finish_control(oracle.size);
    finish_reference<<<unsigned((oracle.size + 255) / 256), 256, 0, stream.s>>>(
        oracle.p, finish_control.p, bias.p, int(oracle.size), width);
    ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream.s));
    metrics.compare("finish-independent-device-vs-cpu-rounding-control", projected_ref, finish_control.read(), true);
    for (std::size_t i = 0; i < roundoff.error.size(); ++i) {
        roundoff.norm[i] = .75 * roundoff.norm[i] + std::abs(double(bs[i % width]));
        roundoff.error[i] = .75 * roundoff.error[i] + 2 * gamma(2) * roundoff.norm[i];
    }
    tp::finish_projection(reduced.p, rows, width, scale.p, bias.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    metrics.compare("expert-scale-bias-after-rank-sum", projected_ref, reduced.read(), metrics.o.exact_lane,
                    metrics.o.exact_lane ? nullptr : &roundoff.error, metrics.o.exact_lane ? nullptr : &roundoff.norm);
    ck(cudaMemcpyAsync(oracle.p, finish_control.p, oracle.size * sizeof(float), cudaMemcpyDeviceToDevice, stream.s));
    fill_parts<<<unsigned((n + 255) / 256), 256, 0, stream.s>>>(reduced.p, parts.p, n, width);
    fill_parts<<<unsigned((n + 255) / 256), 256, 0, stream.s>>>(oracle.p, reference_parts.p, n, width);
    bool alias_rejected = false;
    try { tp::combine(parts.p, route.p, scales.p, shared.p, combined.p, rows, width, 10, shared.p, stream.s); }
    catch (const std::invalid_argument&) { alias_rejected = true; }
    require(alias_rejected, "combine accepted scale scratch aliasing shared");
    tp::combine(parts.p, route.p, scales.p, shared.p, combined.p, rows, width, 10, scaled.p, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    auto rp = reference_parts.read();
    for (std::size_t i = 0; i < rp.size(); ++i) rp[i] *= sv[(i / width) % 10];
    reference_parts.upload(rp);
    k::native_moe_combine_multi(reference_parts.p, route.p, shared.p, ref_combined.p, width, 10, rows, stream.s);
    ck(cudaStreamSynchronize(stream.s));
    auto ref = ref_combined.read();
    for (std::size_t i = 0; i < roundoff.error.size(); ++i) {
        double coefficient = 0;
        for (int expert = 0; expert < 10; ++expert) if (expert != 8)
            coefficient += std::abs(double(rw[(i / width) * 10 + expert]) * sv[expert]);
        const double norm = coefficient * roundoff.norm[i] + std::abs(double(sh[i]));
        roundoff.error[i] = coefficient * roundoff.error[i] + 2 * gamma(12) * norm;
    }
    metrics.compare("moe-topk10-scales-shared-zero-route", ref, combined.read(), metrics.o.exact_lane,
                    metrics.o.exact_lane ? nullptr : &roundoff.error);

    // Empty native groups must leave the destination untouched.
    groups.upload({0}); oracle.upload(sh);
    k::native_expert_grouped(layout, ptr.p, starts.p, groups.p, dest.p, token.p, 1, rows,
                             qx_native.p, scratch.p, oracle.p, stream.s);
    ck(cudaStreamSynchronize(stream.s)); metrics.compare("zero-selected-experts", sh, oracle.read(), true);
    run_expert_batch(sample, rows, metrics);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = options(argc, argv);
        if (o.roundoff_selftest) { roundoff_selftest(); return 0; }
        if (o.exact_lane_selftest) { exact_lane_selftest(); return 0; }
        // These environment-only experiments bypass the selected ordinary
        // grouped control before its public setters take effect.
        for (const char* name : {"STRATA_EXPERT_V2", "STRATA_EXPERT_V2K", "STRATA_DOWN_WS"}) {
            const char* value = std::getenv(name);
            require(!value || value[0] != '1', "disable experimental native expert overrides for arithmetic control");
        }
        if (o.inspect) {
            require(!o.gguf.empty(), "--inspect-sample needs --gguf");
            const auto s = sample_real(o);
            require(k::native_expert_supported(s.gu, s.down, s.input, s.ff), "sample not supported by native expert");
            const auto layout = k::native_expert_layout(s.gu, s.down, s.input, s.ff);
            require(layout.bytes == s.blob.size(), "sample native layout byte mismatch");
            std::printf("sample-inspection gu=%s down=%s width=%d ff=%d blob_bytes=%zu GPU_NOT_RUN\n",
                        codec(s.gu), codec(s.down), s.input, s.ff, s.blob.size());
            return 0;
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "SKIP: CUDA device required; no parity measurements produced\n"); return 77;
        }
        Metrics metrics{0, o};
        k::native_grouped_set_v1(true); k::iq_set_old_kernels(true); k::native_expert_set_expdec(0);
        k::native_expert_set_mode(0, 0);
        k::native_moe_combine_set_enabled(true);
        tiny_scalar_probe();
        std::mt19937 rng(0x7a531);
        for (int gu : {16, 17, 18, 21, 22, 23}) for (int down : {20, 42}) {
            for (const auto dims : {std::pair{256, 192}, std::pair{2560, 640}}) {
                Sample s{gu, down, dims.first, dims.second, random_plane(gu, 2 * dims.second, dims.first, rng)};
                const auto dp = random_plane(down, dims.first, dims.second, rng);
                s.blob.insert(s.blob.end(), dp.begin(), dp.end());
                for (int rows : {1, 4, 17}) run(s, rows, metrics);
            }
        }
        if (!o.gguf.empty()) { const auto real = sample_real(o); for (int rows : {1, 4, 17}) run(real, rows, metrics); }
        std::printf("failures=%d mode=%s atol=%.9g rtol=%.9g real_sample=%d derived_roundoff_gate=%d\n", metrics.failures,
                    o.measure ? "measurement-only-not-qualified" : "frozen-limits", o.atol, o.rtol,
                    !o.gguf.empty(), o.roundoff_bound);
        if (metrics.failures) return 1;
        return o.measure ? 77 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tp_expert_parity: %s\n", e.what()); return 2;
    }
}
