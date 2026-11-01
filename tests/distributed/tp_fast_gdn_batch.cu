#include "strata/kernels/fast_gdn_batch.hpp"
#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <memory>
#include <vector>

namespace k = strata::kernels;
namespace gpu = k::tp::cuda;
namespace {
void ck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
    float* f(size_t offset = 0) { return static_cast<float*>(p) + offset; }
};

bool equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void upload(Buffer& dst, const std::vector<float>& src, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst.p, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}
}

int main() try {
    if (cudaSetDevice(0) != cudaSuccess) return 77;
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int key_heads : {5, 6}) {
        constexpr int group_count = k::kWindowMaxT >= 40 ? 8 : 2;
        const int value_heads = key_heads * 3;
        const int channels = key_heads * 5 * 128;
        const int width = value_heads * 128;
        int counts[group_count], firsts[group_count];
        for(int i=0;i<group_count;++i) {
            counts[i]=group_count==8?5:(i==0?5:3);
            firsts[i]=i==0?0:firsts[i-1]+counts[i-1];
        }
        constexpr int total_rows = group_count==8?40:8;
        constexpr int timing_iterations = 10;
        const size_t state_elements = size_t(128) * value_heads * 128;
        const size_t history_elements = size_t(channels) * 3;

        Buffer qkv(size_t(total_rows) * channels * sizeof(float));
        Buffer weights(size_t(channels) * 4 * sizeof(float));
        Buffer h_batch(size_t(total_rows) * channels * sizeof(float));
        Buffer h_ref(size_t(total_rows) * channels * sizeof(float));
        Buffer gate(size_t(total_rows) * value_heads * sizeof(float));
        Buffer beta(size_t(total_rows) * value_heads * sizeof(float));
        Buffer z(size_t(total_rows) * width * sizeof(float));
        Buffer norm(128 * sizeof(float));
        Buffer out_batch(size_t(total_rows) * width * sizeof(float));
        Buffer out_ref(size_t(total_rows) * width * sizeof(float));
        std::vector<std::unique_ptr<Buffer>> states, states_ref, histories, histories_ref, state_prefix, history_prefix, state_prefix_ref, history_prefix_ref;
        std::mt19937 rng(430 + key_heads);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<float> qkv_host(size_t(total_rows) * channels);
        std::vector<float> weights_host(size_t(channels) * 4);
        std::vector<float> gate_host(size_t(total_rows) * value_heads);
        std::vector<float> beta_host(size_t(total_rows) * value_heads);
        std::vector<float> z_host(size_t(total_rows) * width);
        std::vector<float> norm_host(128);
        std::vector<float> state_host(state_elements);
        std::vector<float> history_host(history_elements);
        for (auto& x : qkv_host) x = dist(rng);
        for (auto& x : weights_host) x = dist(rng);
        for (auto& x : gate_host) x = -0.01f - std::abs(dist(rng));
        for (auto& x : beta_host) x = 0.4f + 0.1f * dist(rng);
        for (auto& x : z_host) x = dist(rng);
        for (auto& x : norm_host) x = 0.5f + dist(rng);

        gpu::FastGdnBatchGroups groups{};
        groups.n = group_count;
        for (int group = 0; group < group_count; ++group) {
            states.push_back(std::make_unique<Buffer>(state_elements * sizeof(float)));
            states_ref.push_back(std::make_unique<Buffer>(state_elements * sizeof(float)));
            histories.push_back(std::make_unique<Buffer>(history_elements * sizeof(float)));
            histories_ref.push_back(std::make_unique<Buffer>(history_elements * sizeof(float)));
            state_prefix.push_back(std::make_unique<Buffer>(size_t(counts[group] - 1) * state_elements * sizeof(float)));
            history_prefix.push_back(std::make_unique<Buffer>(size_t(counts[group] - 1) * history_elements * sizeof(float)));
            state_prefix_ref.push_back(std::make_unique<Buffer>(size_t(counts[group]-1)*state_elements*sizeof(float)));
            history_prefix_ref.push_back(std::make_unique<Buffer>(size_t(counts[group]-1)*history_elements*sizeof(float)));
            for (auto& x : state_host) x = dist(rng);
            for (auto& x : history_host) x = dist(rng);
            groups.first[group] = firsts[group];
            groups.count[group] = counts[group];
            groups.state[group] = states[group]->f();
            groups.history[group] = histories[group]->f();
            groups.state_prefix[group] = state_prefix[group]->f();
            groups.history_prefix[group] = history_prefix[group]->f();
            groups.snapshot[group] = true;
            upload(*states[group], state_host, stream);
            upload(*states_ref[group], state_host, stream);
            upload(*histories[group], history_host, stream);
            upload(*histories_ref[group], history_host, stream);
        }
        upload(qkv, qkv_host, stream);
        upload(weights, weights_host, stream);
        upload(gate, gate_host, stream);
        upload(beta, beta_host, stream);
        upload(z, z_host, stream);
        upload(norm, norm_host, stream);
        Buffer keep_device(sizeof(int32_t));
        const int32_t keep = 1;
        ck(cudaMemcpyAsync(keep_device.p, &keep, sizeof(keep), cudaMemcpyHostToDevice, stream));

        cudaEvent_t begin, end;
        ck(cudaEventCreate(&begin));
        ck(cudaEventCreate(&end));
        ck(cudaEventRecord(begin, stream));
        for (int iteration = 0; iteration < timing_iterations; ++iteration) {
            for (int group = 0; group < group_count; ++group) {
                for (int local = 0; local < counts[group]; ++local) {
                    const int row = firsts[group] + local;
                    gpu::mixer_conv(histories_ref[group]->f(), qkv.f(size_t(row) * channels),
                                    weights.f(), h_ref.f(size_t(row) * channels), 1, channels, key_heads, stream);
                    k::gdn_step_norm_multi(states_ref[group]->f(), h_ref.f(size_t(row) * channels), channels,
                        gate.f(size_t(row) * value_heads), beta.f(size_t(row) * value_heads),
                        z.f(size_t(row) * width), norm.f(), 1.0e-6f, out_ref.f(size_t(row) * width),
                        key_heads, value_heads, 1, static_cast<const int32_t*>(keep_device.p), stream);
                    gpu::mixer_conv_commit(histories_ref[group]->f(), qkv.f(size_t(row) * channels),
                                           1, channels, stream);
                    if (local + 1 < counts[group]) {
                        ck(cudaMemcpyAsync(state_prefix_ref[group]->f(size_t(local) * state_elements),
                            states_ref[group]->p, state_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                        ck(cudaMemcpyAsync(history_prefix_ref[group]->f(size_t(local) * history_elements),
                            histories_ref[group]->p, history_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                }
            }
        }
        ck(cudaEventRecord(end, stream));
        ck(cudaEventSynchronize(end));
        float old_ms = 0.0f;
        ck(cudaEventElapsedTime(&old_ms, begin, end));
        old_ms /= timing_iterations;

        ck(cudaEventRecord(begin, stream));
        for (int iteration = 0; iteration < timing_iterations; ++iteration) {
            gpu::fast_gdn_batch(groups, qkv.f(), weights.f(), h_batch.f(), gate.f(), beta.f(), z.f(),
                                norm.f(), out_batch.f(), key_heads, 1.0e-6f, stream);
        }
        ck(cudaEventRecord(end, stream));
        ck(cudaEventSynchronize(end));
        float batch_ms = 0.0f;
        ck(cudaEventElapsedTime(&batch_ms, begin, end));
        batch_ms /= timing_iterations;

        std::vector<float> out_a(size_t(total_rows) * width), out_b(out_a.size());
        std::vector<float> h_a(size_t(total_rows) * channels), h_b(h_a.size());
        ck(cudaMemcpy(out_a.data(), out_ref.p, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(out_b.data(), out_batch.p, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(h_a.data(), h_ref.p, h_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(h_b.data(), h_batch.p, h_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        auto delta=[](const auto& a,const auto& b){double n=0,d=0,m=0;size_t count=0;for(size_t i=0;i<a.size();++i){double v=double(a[i])-b[i];n+=v*v;d+=double(a[i])*a[i];m=std::max(m,std::abs(v));count+=a[i]!=b[i];}std::printf("L2 %.9g max %.9g mismatch %zu\n",std::sqrt(n/std::max(d,1e-30)),m,count);};
        std::printf("h ");delta(h_a,h_b);std::printf("out ");delta(out_a,out_b);
        double error2=0,norm2=0;
        for(size_t i=0;i<out_a.size();++i){const double e=double(out_a[i])-out_b[i];error2+=e*e;norm2+=double(out_a[i])*out_a[i];}
        if(!equal(h_a,h_b)||!std::isfinite(error2)||std::sqrt(error2/std::max(norm2,1e-30))>1e-6)return 1;
        for (int group = 0; group < group_count; ++group) {
            std::vector<float> state_a(state_elements), state_b(state_elements);
            std::vector<float> history_a(history_elements), history_b(history_elements);
            const size_t prefix_state_elements = size_t(counts[group] - 1) * state_elements;
            const size_t prefix_history_elements = size_t(counts[group] - 1) * history_elements;
            std::vector<float> state_prefix_a(prefix_state_elements), state_prefix_b(prefix_state_elements);
            std::vector<float> history_prefix_a(prefix_history_elements), history_prefix_b(prefix_history_elements);
            ck(cudaMemcpy(state_a.data(), states_ref[group]->p, state_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(state_b.data(), states[group]->p, state_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(history_a.data(), histories_ref[group]->p, history_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(history_b.data(), histories[group]->p, history_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(state_prefix_a.data(), state_prefix_ref[group]->p,
                          state_prefix_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(state_prefix_b.data(), state_prefix[group]->p,
                          state_prefix_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(history_prefix_a.data(), history_prefix_ref[group]->p,
                          history_prefix_a.size() * sizeof(float), cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(history_prefix_b.data(), history_prefix[group]->p,
                          history_prefix_b.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::printf("state ");delta(state_a,state_b);std::printf("prefix ");delta(state_prefix_a,state_prefix_b);
            if (!equal(state_a, state_b) || !equal(history_a, history_b) ||
                !equal(state_prefix_a, state_prefix_b) || !equal(history_prefix_a, history_prefix_b)) {
                std::fprintf(stderr, "GDN batch state/snapshot mismatch for key heads %d group %d\n",
                             key_heads, group);
                return 1;
            }
        }
        std::printf("GDN batch key_heads=%d groups=%d rows=%d exact state/prefix, output L2<1e-6; old %.3f ms, batch %.3f ms\n",
                    key_heads, group_count,total_rows,old_ms, batch_ms);
        ck(cudaEventDestroy(begin));
        ck(cudaEventDestroy(end));
    }
    ck(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
}
