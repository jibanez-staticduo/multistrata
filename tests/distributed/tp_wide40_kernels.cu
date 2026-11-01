#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace gpu = k::tp::cuda;

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    template<class T> T* as() { return static_cast<T*>(data); }
    template<class T> void upload(const std::vector<T>& values) {
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    template<class T> std::vector<T> read(std::size_t count) const {
        std::vector<T> values(count);
        check(cudaMemcpy(values.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};

template<class T>
void check_equal(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a != b) throw std::runtime_error(label);
}

float ordered_sum(float a, float b, float c, int part) {
    volatile float partial;
    switch (part) {
        case 0: partial = b + c; return partial + a;
        case 1: partial = a + c; return partial + b;
        default: partial = a + b; return partial + c;
    }
}

void test_ring_sum_simple8(cudaStream_t stream) {
    constexpr std::size_t rows = 2047;
    constexpr std::size_t width = 2560;
    constexpr std::size_t total = rows * width;
    constexpr std::size_t max_chunk = (2u << 20) / sizeof(float);
    constexpr std::size_t channel_span = (((total + 4095) / 4096 + 7) / 8) * 4096;
    constexpr std::size_t tail_span = total - 7 * channel_span;
    static_assert(total == 5240320);
    static_assert(total * sizeof(float) == (20u << 20) - 10240);
    static_assert(channel_span == 655360);
    static_assert(tail_span == 652800);

    const std::array<std::array<float, 3>, 6> permutations{{
        {{1.e20f, -1.e20f, 1.f}},
        {{1.e20f, 1.f, -1.e20f}},
        {{-1.e20f, 1.e20f, 1.f}},
        {{-1.e20f, 1.f, 1.e20f}},
        {{1.f, 1.e20f, -1.e20f}},
        {{1.f, -1.e20f, 1.e20f}},
    }};
    const std::array<float, 3> control{{1.f, 2.f, 3.f}};
    const std::array<std::size_t, 3> full_channel_chunks{{218456, 218456, 218448}};
    const std::array<std::size_t, 3> tail_channel_chunks{{217600, 217600, 217600}};

    std::vector<std::size_t> chunk_offsets;
    for (std::size_t first = 0; first < total;) {
        chunk_offsets.push_back(first);
        first += std::min(max_chunk, total - first);
    }
    DeviceBuffer gathered(3 * total * sizeof(float));
    DeviceBuffer output(total * sizeof(float));
    for (std::size_t test = 0; test <= permutations.size(); ++test) {
        const auto& values = test == permutations.size() ? control : permutations[test];
        std::vector<float> host_gathered(3 * total);
        std::size_t gathered_first = 0;
        for (std::size_t first : chunk_offsets) {
            const std::size_t count = std::min(max_chunk, total - first);
            for (std::size_t rank = 0; rank < 3; ++rank) {
                std::fill(host_gathered.begin() + gathered_first + rank * count,
                          host_gathered.begin() + gathered_first + (rank + 1) * count,
                          values[rank]);
            }
            gathered_first += 3 * count;
        }
        gathered.upload(host_gathered);

        std::vector<float> expected(total);
        std::size_t first = 0;
        for (std::size_t channel = 0; channel < 8; ++channel) {
            const auto& chunks = channel == 7 ? tail_channel_chunks : full_channel_chunks;
            for (std::size_t part = 0; part < chunks.size(); ++part) {
                const float value = test == permutations.size()
                    ? (values[0] + values[1]) + values[2]
                    : ordered_sum(values[0], values[1], values[2], int(part));
                std::fill(expected.begin() + first, expected.begin() + first + chunks[part], value);
                first += chunks[part];
            }
        }
        if (first != total) throw std::runtime_error("SIMPLE8 CPU channel intervals do not cover the reduction");

        gathered_first = 0;
        for (first = 0; first < total;) {
            const std::size_t count = std::min(max_chunk, total - first);
            gpu::transport_ring_sum(gathered.as<float>() + gathered_first, output.as<float>() + first,
                                    count, first, total, stream);
            first += count;
            gathered_first += 3 * count;
        }
        check(cudaStreamSynchronize(stream));
        if (output.read<float>(total) != expected)
            throw std::runtime_error("SIMPLE8 RoCE ring sum differs from explicit 16KiB-cell channel intervals");
    }
}

void test_q8_chunks(cudaStream_t stream) {
    constexpr int n_in = 256;
    constexpr int rows = 40;
    std::vector<float> x(rows * n_in);
    for (int i = 0; i < rows * n_in; ++i) x[i] = std::sin(float(i) * 0.013f);
    DeviceBuffer dx(x.size() * sizeof(float));
    DeviceBuffer dq(k::native_q8_1_bytes(n_in, rows));
    dx.upload(x);
    k::native_quantize_q8_1(dx.as<float>(), dq.data, n_in, rows, stream);
    check(cudaStreamSynchronize(stream));
    const auto batched = dq.read<uint8_t>(k::native_q8_1_bytes(n_in, rows));

    const std::size_t row_bytes = k::native_q8_1_bytes(n_in, 1);
    DeviceBuffer one(row_bytes);
    for (int row : {0, 15, 16, 31, 32, 39}) {
        k::native_quantize_q8_1(dx.as<float>() + std::size_t(row) * n_in, one.data, n_in, 1, stream);
        check(cudaStreamSynchronize(stream));
        const auto expected = one.read<uint8_t>(row_bytes);
        if (!std::equal(expected.begin(), expected.end(), batched.begin() + std::size_t(row) * row_bytes))
            throw std::runtime_error("wide Q8_1 quantization differs from one-row calls");
    }

    constexpr int n_out = 32;
    constexpr int q4_block_bytes = 18;
    std::vector<uint8_t> weights(std::size_t(n_out) * n_in / 32 * q4_block_bytes);
    const uint16_t scale = 0x3000;  // finite nonzero fp16 scale
    for (int row = 0; row < n_out; ++row) {
        for (int block = 0; block < n_in / 32; ++block) {
            auto* data = weights.data() + (std::size_t(row) * (n_in / 32) + block) * q4_block_bytes;
            std::memcpy(data, &scale, sizeof(scale));
            for (int i = 2; i < q4_block_bytes; ++i) data[i] = uint8_t(0x99 + ((row + block + i) & 1) * 0x11);
        }
    }
    DeviceBuffer dw(weights.size());
    dw.upload(weights);
    DeviceBuffer multi_out(std::size_t(rows) * n_out * sizeof(float));
    DeviceBuffer single_out(std::size_t(n_out) * sizeof(float));
    k::native_mmvq_set_multi_exact(true);
    k::native_mmvq(2, dw.data, dq.data, multi_out.as<float>(), n_in, n_out, rows, stream);
    check(cudaStreamSynchronize(stream));
    const auto all = multi_out.read<float>(std::size_t(rows) * n_out);
    for (int row = 0; row < rows; ++row) {
        k::native_mmvq(2, dw.data, static_cast<uint8_t*>(dq.data) + std::size_t(row) * row_bytes,
                       single_out.as<float>(), n_in, n_out, 1, stream);
        check(cudaStreamSynchronize(stream));
        const auto expected = single_out.read<float>(n_out);
        if (std::memcmp(expected.data(), all.data() + std::size_t(row) * n_out, n_out * sizeof(float)) != 0)
            throw std::runtime_error("40-column native MMVQ differs from one-row calls");
    }
}

void test_mixer_steps_batch(cudaStream_t stream) {
    constexpr int context = 3000;
    constexpr int heads = 8;
    constexpr int position_stride = 2;
    for (const int rows : {1, 9, 24, 40}) {
        std::vector<int> capture_slots(rows);
        std::vector<int32_t> device_slots(rows);
        std::vector<int32_t> device_positions(rows * position_stride, -99);
        auto make_positions = [&](int generation) {
            std::vector<int32_t> positions(rows * position_stride, -99);
            for (int row = 0; row < rows; ++row) {
                int position = (row * 97 + generation * 211) % 2000;
                if (generation == 0 && row < 4) {
                    const int boundary_positions[] = {0, 2049, 2050, context - 1};
                    position = boundary_positions[row];
                }
                positions[row * position_stride] = position;
            }
            return positions;
        };
        for (int row = 0; row < rows; ++row) {
            capture_slots[row] = (row / 3) % 7;
            device_slots[row] = capture_slots[row];
        }
        device_positions = make_positions(0);

        DeviceBuffer slots(rows * sizeof(int32_t));
        DeviceBuffer positions(device_positions.size() * sizeof(int32_t));
        DeviceBuffer steps(rows * 4 * sizeof(int));
        DeviceBuffer row_positions(rows * heads * sizeof(int));
        DeviceBuffer reference_steps(rows * 4 * sizeof(int));
        DeviceBuffer reference_positions(rows * heads * sizeof(int));
        DeviceBuffer status(sizeof(int));
        DeviceBuffer reference_status(sizeof(int));
        slots.upload(device_slots);
        positions.upload(device_positions);

        strata::core::TensorParallelRows spec{};
        spec.count = rows;
        spec.positions = positions.as<int32_t>();
        spec.slots = slots.as<int32_t>();
        spec.independent = true;
        spec.position_stride = position_stride;

        check(cudaMemsetAsync(status.data, 0, sizeof(int), stream));
        check(cudaMemsetAsync(reference_status.data, 0, sizeof(int), stream));
        for (int row = 0; row < rows; ++row) {
            auto one = spec;
            one.count = 1;
            one.positions = positions.as<int32_t>() + row * position_stride;
            one.slots = slots.as<int32_t>() + row;
            gpu::mixer_steps(one, 1, capture_slots[row], context,
                             reference_steps.as<int>() + row * 4,
                             reference_positions.as<int>() + row * heads, heads,
                             reference_status.as<int>(), stream);
        }
        gpu::mixer_steps_batch(spec, capture_slots.data(), rows, context,
                               steps.as<int>(), row_positions.as<int>(), status.as<int>(), stream);
        check(cudaStreamSynchronize(stream));
        const auto expected_steps = reference_steps.read<int>(rows * 4);
        const auto expected_positions = reference_positions.read<int>(rows * heads);
        if (steps.read<int>(rows * 4) != expected_steps ||
            row_positions.read<int>(rows * heads) != expected_positions ||
            status.read<int>(1)[0] != 0 || reference_status.read<int>(1)[0] != 0)
            throw std::runtime_error("batched mixer steps differ from scalar rows");

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        gpu::mixer_steps_batch(spec, capture_slots.data(), rows, context,
                               steps.as<int>(), row_positions.as<int>(), status.as<int>(), stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        check(cudaGraphDestroy(graph));

        const auto replay_positions = make_positions(1);
        positions.upload(replay_positions);
        for (int& slot : capture_slots) slot += 19;
        check(cudaMemsetAsync(status.data, 0, sizeof(int), stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        const auto replay_steps = steps.read<int>(rows * 4);
        const auto replay_row_positions = row_positions.read<int>(rows * heads);
        for (int row = 0; row < rows; ++row) {
            const int p = replay_positions[row * position_stride];
            const int expected[] = {p, p + 1, (p + 1) / 4, std::min(p + 1, 2051)};
            for (int i = 0; i < 4; ++i)
                if (replay_steps[row * 4 + i] != expected[i])
                    throw std::runtime_error("captured mixer steps ignored replay positions");
            for (int head = 0; head < heads; ++head)
                if (replay_row_positions[row * heads + head] != p)
                    throw std::runtime_error("captured mixer positions ignored replay positions");
        }
        if (status.read<int>(1)[0] != 0)
            throw std::runtime_error("captured host slot metadata changed after capture");

        device_slots[rows / 2] += 1;
        slots.upload(device_slots);
        check(cudaMemsetAsync(status.data, 0, sizeof(int), stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        if (status.read<int>(1)[0] != 1)
            throw std::runtime_error("batched mixer did not flag a captured slot mismatch");

        device_slots[rows / 2] = capture_slots[rows / 2] - 19;
        device_positions[std::size_t(rows - 1) * position_stride] = context;
        slots.upload(device_slots);
        positions.upload(device_positions);
        check(cudaMemsetAsync(status.data, 0, sizeof(int), stream));
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        if (status.read<int>(1)[0] != 1)
            throw std::runtime_error("batched mixer did not flag an out-of-context position");
        check(cudaGraphExecDestroy(executable));
    }
}

void test_bf16_projection(cudaStream_t stream,int rows) {
    constexpr int K=256,N=32;
    std::vector<float> x(rows*K);std::vector<uint16_t> w(N*K);
    for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.07f)*.1f;
    for(size_t i=0;i<w.size();++i)w[i]=k::bf16_from_f32(std::cos(float(i)*.03f)*.1f);
    DeviceBuffer dx(x.size()*4),dw(w.size()*2),dy(rows*N*4);
    dx.upload(x);dw.upload(w);
    k::bf16_gemv_fp32_mmvf_multi(dx.as<float>(),K,dw.as<uint16_t>(),dy.as<float>(),N,K,N,rows,stream);
    check(cudaStreamSynchronize(stream));const auto y=dy.read<float>(rows*N);
    double err=0,den=0;
    for(int r=0;r<rows;++r)for(int n=0;n<N;++n) {
        double expected=0;
        for(int i=0;i<K;++i) {
            uint32_t bits=uint32_t(w[n*K+i])<<16;float weight;std::memcpy(&weight,&bits,4);
            expected+=double(x[r*K+i])*weight;
        }
        const double d=y[r*N+n]-expected;err+=d*d;den+=expected*expected;
    }
    if(!std::isfinite(err)||std::sqrt(err/den)>1.e-5)throw std::runtime_error("wide BF16 projection differs from CPU products");
}

void test_compact_head(cudaStream_t stream) {
    constexpr int rows=40,width=257,blocks=4;
    std::vector<float> combined(rows*12,0.f);std::vector<int32_t> expected(rows,0);
    for(int rank=0;rank<3;++rank) {
        std::vector<float> local(rows*width);
        for(int row=0;row<rows;++row)for(int col=0;col<width;++col)
            local[row*width+col]=-100.f-float(col);
        for(int row=0;row<rows;++row) {
            const int winner=row%width;
            local[row*width+winner]=row==0?-1.f:float(rank);
            if(rank==2)expected[row]=rank*width+winner;
        }
        DeviceBuffer x(local.size()*4),c(combined.size()*4);x.upload(local);
        check(cudaMemset(c.data,0,combined.size()*4));
        gpu::head_local_candidate(x.as<float>(),c.as<float>(),rows,width,rank*width,rank,stream);
        check(cudaStreamSynchronize(stream));const auto partial=c.read<float>(combined.size());
        for(size_t i=0;i<combined.size();++i)combined[i]+=partial[i];
    }
    expected[0]=0; // Equal maxima across ranks choose the lowest global ID.
    combined[39*12+2]=1.f;
    DeviceBuffer c(combined.size()*4),p(rows*4),f(rows*blocks*4);c.upload(combined);
    gpu::head_global_candidate(c.as<float>(),p.as<int32_t>(),f.as<uint32_t>(),rows,blocks,stream);
    check(cudaStreamSynchronize(stream));
    if(p.read<int32_t>(rows)!=expected)throw std::runtime_error("compact head argmax/tie mismatch");
    const auto flags=f.read<uint32_t>(rows*blocks);
    for(int row=0;row<rows;++row)for(int b=0;b<blocks;++b)
        if(flags[row*blocks+b]!=(row==39?1u:0u))throw std::runtime_error("compact head finite flags mismatch");
}

void test_hc_chunks(cudaStream_t stream, int rows) {
    constexpr int width = 10240;
    constexpr std::size_t row_values = width;
    std::vector<k::FusedGrArgs> host_args(rows);
    std::vector<DeviceBuffer*> buffers;
    std::vector<float> input(std::size_t(rows) * row_values), norm(input.size());
    std::vector<float> inject(std::size_t(rows) * 4);
    for (int row = 0; row < rows; ++row) {
        std::vector<float> r(width), w(width);
        for (int i = 0; i < width; ++i) {
            r[i] = std::sin(float(row * width + i) * 0.001f);
            w[i] = 0.75f + float(i % 13) * 0.01f;
        }
        auto* rd = new DeviceBuffer(width * sizeof(float));
        auto* wd = new DeviceBuffer(width * sizeof(float));
        auto* rsd = new DeviceBuffer(4 * sizeof(float));
        rd->upload(r);
        wd->upload(w);
        buffers.insert(buffers.end(), {rd, wd, rsd});
        host_args[row].R = rd->as<float>();
        host_args[row].w_norm = wd->as<float>();
        host_args[row].rs = rsd->as<float>();
        for (int i = 0; i < width; ++i) input[std::size_t(row) * width + i] = r[i];
        for (int i = 0; i < 4; ++i) inject[std::size_t(row) * 4 + i] = float(row * 4 + i) * 0.25f;
    }
    DeviceBuffer normalized(input.size() * sizeof(float));
    DeviceBuffer packed(input.size() * sizeof(uint16_t));
    gpu::hc_pack(host_args.data(), rows, normalized.as<float>(), packed.as<uint16_t>(), stream);
    check(cudaStreamSynchronize(stream));
    const auto got_norm = normalized.read<float>(input.size());
    const auto got_packed = packed.read<uint16_t>(input.size());

    for (int row = 0; row < rows; ++row) {
        DeviceBuffer row_norm(row_values * sizeof(float));
        DeviceBuffer row_packed(row_values * sizeof(uint16_t));
        gpu::hc_pack(host_args.data() + row, 1, row_norm.as<float>(), row_packed.as<uint16_t>(), stream);
        check(cudaStreamSynchronize(stream));
        const auto expected_norm = row_norm.read<float>(row_values);
        const auto expected_packed = row_packed.read<uint16_t>(row_values);
        if (!std::equal(expected_norm.begin(), expected_norm.end(), got_norm.begin() + std::size_t(row) * width) ||
            !std::equal(expected_packed.begin(), expected_packed.end(), got_packed.begin() + std::size_t(row) * width))
            throw std::runtime_error("wide HC pack differs from one-row calls");
    }

    std::vector<DeviceBuffer*> inject_outputs;
    for (int row = 0; row < rows; ++row) {
        auto* output = new DeviceBuffer(4 * sizeof(float));
        inject_outputs.push_back(output);
        host_args[row].inject_out = output->as<float>();
    }
    DeviceBuffer dinject(inject.size() * sizeof(float));
    dinject.upload(inject);
    gpu::hc_publish_inject(host_args.data(), dinject.as<float>(), rows, stream);
    check(cudaStreamSynchronize(stream));
    for (int row = 0; row < rows; ++row)
        check_equal(inject_outputs[row]->read<float>(4),
                    std::vector<float>(inject.begin() + row * 4, inject.begin() + row * 4 + 4),
                    "wide HC inject offsets are wrong");

    for (auto* buffer : buffers) delete buffer;
    for (auto* buffer : inject_outputs) delete buffer;
}

void test_hc_packed_silu_publish(cudaStream_t stream, int rows) {
    constexpr int packed_width = 336;
    constexpr int hc_width = 320;
    std::vector<k::FusedGrArgs> host_args(rows);
    std::vector<DeviceBuffer*> inject_outputs;
    inject_outputs.reserve(rows);
    for (int row = 0; row < rows; ++row) {
        auto* output = new DeviceBuffer(4 * sizeof(float));
        inject_outputs.push_back(output);
        host_args[row].inject_out = output->as<float>();
    }

    std::vector<float> combined(std::size_t(rows) * packed_width);
    std::vector<float> lo(std::size_t(rows) * hc_width);
    std::vector<float> inject(std::size_t(rows) * 4);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < hc_width; ++col) {
            const float value = float(row + 1) * 0.013f + float(col % 29 - 14) * 0.071f;
            combined[std::size_t(row) * packed_width + col] = value;
            lo[std::size_t(row) * hc_width + col] = value;
        }
        for (int col = 0; col < 4; ++col) {
            const float value = 100.f + float(row * 4 + col) * 0.25f;
            combined[std::size_t(row) * packed_width + hc_width + col] = value;
            inject[std::size_t(row) * 4 + col] = value;
        }
        for (int col = hc_width + 4; col < packed_width; ++col)
            combined[std::size_t(row) * packed_width + col] = -70.f - float(row * packed_width + col);
    }

    DeviceBuffer dcombined(combined.size() * sizeof(float));
    DeviceBuffer dlo(lo.size() * sizeof(float));
    DeviceBuffer dinject(inject.size() * sizeof(float));
    DeviceBuffer expected_packed(lo.size() * sizeof(uint16_t));
    DeviceBuffer actual_packed(lo.size() * sizeof(uint16_t));
    dcombined.upload(combined);
    dlo.upload(lo);
    dinject.upload(inject);

    gpu::hc_silu_pack(dlo.as<float>(), expected_packed.as<uint16_t>(), rows, stream);
    gpu::hc_publish_inject(host_args.data(), dinject.as<float>(), rows, stream);
    gpu::hc_packed_silu_publish(host_args.data(), dcombined.as<float>(),
                                 actual_packed.as<uint16_t>(), rows, stream);
    check(cudaStreamSynchronize(stream));

    check_equal(expected_packed.read<uint16_t>(lo.size()),
                actual_packed.read<uint16_t>(lo.size()),
                "packed HC SiLU output differs from separate SiLU and publish");
    for (int row = 0; row < rows; ++row)
        check_equal(inject_outputs[row]->read<float>(4),
                    std::vector<float>(inject.begin() + row * 4, inject.begin() + row * 4 + 4),
                    "packed HC inject output differs from separate publish");

    for (auto* buffer : inject_outputs) delete buffer;
}
}  // namespace

int main() try {
#if !defined(STRATA_TP_WIDE_WINDOW)
    return 77;
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    test_ring_sum_simple8(stream);
    test_q8_chunks(stream);
    test_mixer_steps_batch(stream);
    test_compact_head(stream);
    for (int rows : {17, 32, 40}) {test_hc_chunks(stream, rows);test_bf16_projection(stream,rows);}
    for (int rows : {3, 17, 24, 40}) test_hc_packed_silu_publish(stream, rows);
    check(cudaStreamDestroy(stream));
    std::puts("tp_wide40_kernels: SIMPLE8 ring sum, Q8_1 and HC 17/32/40 checks passed");
    return 0;
#endif
} catch (const std::exception& error) {
    std::fprintf(stderr, "tp_wide40_kernels: %s\n", error.what());
    return 1;
}
