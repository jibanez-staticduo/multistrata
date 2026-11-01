#include "strata/distributed/nvfp4_experts.hpp"
#include "strata/kernels/tp_nvfp4.hpp"
#include "strata/kernels/tp_b12x_pack.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <dlfcn.h>

namespace strata::distributed {
namespace {
void check(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
struct Memory {
    void* data = nullptr;
    uint64_t bytes = 0;
    ~Memory() { if (data) cudaFree(data); }
    void allocate(uint64_t size) { check(cudaMalloc(&data, size)); bytes = size; }
};
struct File {
    int fd = -1;
    explicit File(const std::string& path) { fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC); if (fd < 0) throw std::runtime_error("cannot open NVFP4 expert pack"); }
    ~File() { if (fd >= 0) ::close(fd); }
    void read(uint64_t offset, void* data, size_t size) {
        size_t done = 0;
        while (done < size) {
            const auto count = ::pread(fd, static_cast<char*>(data) + done, size - done, off_t(offset + done));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("NVFP4 expert pack ended before its declared blob");
            done += size_t(count);
        }
    }
};
}
struct NativeNvfp4Experts::Impl {
    struct Layer {
        Memory gate, up, down, scales;
        Memory grouped;
        Memory decode_gu_codes,decode_gu_scales;
        Memory w1,w1s,w1a,a1,w2,w2s,w2a,a2,alpha,w1mx,w2mx,w1res,w2res;
        kernels::tp::cuda::Nvfp4Layer view{};
    };
    std::vector<std::unique_ptr<Layer>> layers;
    Memory scratch, status;
    int rank = -1, max_rows = 0;
    uint64_t bytes = 0;
    core::ModelGeometry geometry;
    std::string fixture_dir;
    Memory fixture_input,fixture_ids,fixture_weights,fixture_output;
    int fixture_rows=0;
    bool fixture_written=false;
    void* b12x_library=nullptr;
    using NativeRun=int(*)(void**,int,uint64_t,void*);
    NativeRun b12x_run=nullptr;
    bool b12x_a8=false,b12x_a16=false;
    Memory b12x_workspace,b12x_input,b12x_output,b12x_ids,b12x_routes,b12x_blocks,b12x_route_count;
    ~Impl(){layers.clear();if(b12x_library)dlclose(b12x_library);}
};
NativeNvfp4Experts::NativeNvfp4Experts() : impl_(std::make_unique<Impl>()) {}
NativeNvfp4Experts::~NativeNvfp4Experts() = default;
bool NativeNvfp4Experts::initialize(const std::string& pack, const core::ModelGeometry& geometry,
                                    int rank, int max_rows, void* stream, std::string& error) try {
    auto& p = *impl_;
    if (!p.layers.empty() || rank < 0 || rank >= 3 || max_rows < 1 || max_rows > 8192 || !stream ||
        geometry.n_ff != 640 || geometry.n_embd != 2560 || geometry.n_expert_used != 10) {
        error = "unsupported NVFP4 TP shard configuration"; return false;
    }
    if (!kernels::cpu::expert_layout_load(pack, geometry.n_layers, geometry.n_expert, error)) return false;
    const auto& layout = kernels::cpu::expert_layout();
    const char* balanced_env=std::getenv("STRATA_TP_BALANCED_EXPERTS");
    const bool balanced=balanced_env&&balanced_env[0]=='1';
    const char* packed_env=std::getenv("STRATA_TP_PREFILL_PACKED_GU");
    const bool packed_gu=packed_env&&packed_env[0]=='1';
    if(const char* library=std::getenv("STRATA_TP_B12X_LIBRARY");library&&*library) {
        if(packed_gu||std::getenv("STRATA_TP_NATIVE_GROUPED"))
            throw std::runtime_error("B12X requires separate original NVFP4 planes");
        p.b12x_library=dlopen(library,RTLD_NOW|RTLD_LOCAL);
        if(!p.b12x_library)throw std::runtime_error(dlerror());
        auto init=reinterpret_cast<int(*)()>(dlsym(p.b12x_library,"strata_b12x_native_init"));
        auto workspace=reinterpret_cast<size_t(*)(int)>(dlsym(p.b12x_library,"strata_b12x_native_workspace"));
        p.b12x_run=reinterpret_cast<Impl::NativeRun>(dlsym(p.b12x_library,"strata_b12x_native_run"));
        auto precision=reinterpret_cast<int(*)()>(dlsym(p.b12x_library,"strata_b12x_native_precision"));
        p.b12x_a8=precision&&precision()==8;
        p.b12x_a16=precision&&precision()==16;
        if(!init||!workspace||!p.b12x_run||init()!=0)
            throw std::runtime_error("B12X native library initialization or ABI failed");
        const auto bytes=workspace(40);
        if(!bytes||bytes>(1ull<<30))throw std::runtime_error("invalid B12X workspace size");
        p.b12x_workspace.allocate(bytes);p.b12x_input.allocate(40*2560*2);
        p.b12x_output.allocate(40*2560*2);p.b12x_ids.allocate(40*10*8);
        if(p.b12x_a16) {
            p.b12x_routes.allocate(4224*4);p.b12x_blocks.allocate(528*4);p.b12x_route_count.allocate(4);
        }
        if(p.b12x_a8) {
            p.status.allocate(sizeof(int));
            check(cudaMemsetAsync(p.status.data,0,sizeof(int),(cudaStream_t)stream));
        }
    }
    if(packed_gu)if(const char* grouped=std::getenv("STRATA_TP_NATIVE_GROUPED");grouped&&grouped[0]=='1')
        throw std::runtime_error("packed gate/up conflicts with grouped expert storage");
    const std::array<int, 3> valid=balanced?std::array<int,3>{224,208,208}:std::array<int,3>{256,192,192};
    const std::array<int, 3> first=balanced?std::array<int,3>{0,224,432}:std::array<int,3>{0,256,448};
    constexpr int padded = 256;
    const auto gu_row = kernels::iq_row_bytes(40, geometry.n_embd);
    const auto down_row = kernels::iq_row_bytes(40, padded);
    p.geometry = geometry; p.rank = rank; p.max_rows = max_rows;
    if(rank==0)if(const char* dir=std::getenv("STRATA_TP_EXPERT_FIXTURE_DIR");dir&&*dir) {
        if(packed_gu||std::getenv("STRATA_TP_NATIVE_GROUPED"))
            throw std::runtime_error("expert fixture requires separate original NVFP4 planes");
        p.fixture_dir=dir;
        p.fixture_input.allocate(size_t(40)*2560*4);p.fixture_output.allocate(size_t(40)*2560*4);
        p.fixture_ids.allocate(size_t(40)*10*4);p.fixture_weights.allocate(size_t(40)*10*4);
    }
    File source(pack + "/experts.bin");
    struct stat source_info{};
    if (::fstat(source.fd, &source_info) || source_info.st_size < 0 || uint64_t(source_info.st_size) != layout.total)
        throw std::runtime_error("NVFP4 expert file size differs from declared layer inventory");
    for (int64_t index = 0; index < geometry.n_layers; ++index) {
        const auto& format = layout.fmt.at(size_t(index));
        if (format.gu_type != 40 || format.d_type != 40 || !format.tail_off ||
            format.n_ff != geometry.n_ff || format.n_embd != geometry.n_embd || format.had_gu || format.had_d)
            throw std::runtime_error("NVFP4 TP requires original unrotated type-40 expert blobs");
        const auto expected = kernels::native_expert_layout(40, 40, geometry.n_embd, geometry.n_ff);
        if (format.bytes != expected.bytes || format.up_off != expected.up_off ||
            format.down_off != expected.down_off || format.tail_off != expected.tail_off ||
            format.gu_row != expected.gu_row || format.d_row != expected.d_row)
            throw std::runtime_error("NVFP4 expert metadata span differs from the native kernel layout");
        auto layer = std::make_unique<Impl::Layer>();
        const auto gu_bytes = gu_row * padded, down_bytes = down_row * size_t(geometry.n_embd);
        // MMQ's final K tile vector-loads beyond a short 256-column row.
        // Keep the same zero guard used by Strata's native gathered MMQ path.
        constexpr size_t mmq_guard = 4096;
        layer->gate.allocate(gu_bytes * geometry.n_expert + mmq_guard);
        layer->up.allocate(gu_bytes * geometry.n_expert + mmq_guard);
        layer->down.allocate(down_bytes * geometry.n_expert + mmq_guard);
        for (auto* buffer : {&layer->gate, &layer->up, &layer->down})
            check(cudaMemsetAsync(static_cast<uint8_t*>(buffer->data) + buffer->bytes - mmq_guard,
                                  0, mmq_guard, static_cast<cudaStream_t>(stream)));
        layer->scales.allocate(size_t(geometry.n_expert) * 4 * sizeof(float));
        std::vector<uint8_t> blob(size_t(format.bytes)), gate(gu_bytes), up(gu_bytes), down(down_bytes);
        std::vector<float> scales(size_t(geometry.n_expert) * 4);
        uint8_t maximum_gu=0,maximum_down=0;
        for (int64_t expert = 0; expert < geometry.n_expert; ++expert) {
            source.read(layout.blob_offset(index, expert), blob.data(), blob.size());
            std::fill(gate.begin(), gate.end(), uint8_t{}); std::fill(up.begin(), up.end(), uint8_t{});
            std::fill(down.begin(), down.end(), uint8_t{});
            const auto gu_offset = size_t(first[rank]) * gu_row, owned_gu = size_t(valid[rank]) * gu_row;
            std::memcpy(gate.data(), blob.data() + gu_offset, owned_gu);
            std::memcpy(up.data(), blob.data() + format.up_off + gu_offset, owned_gu);
            if(balanced) {
                // Each original 64-value block has four independent scales
                // and four 8-byte code groups. Cuts on 16 preserve every byte.
                for(int64_t row=0;row<geometry.n_embd;++row)for(int sub=0;sub<valid[rank]/16;++sub) {
                    const int source_sub=first[rank]/16+sub;
                    const auto* src=blob.data()+format.down_off+size_t(row)*format.d_row+size_t(source_sub/4)*36;
                    auto* dst=down.data()+size_t(row)*down_row+size_t(sub/4)*36;
                    dst[sub%4]=src[source_sub%4];
                    std::memcpy(dst+4+(sub%4)*8,src+4+(source_sub%4)*8,8);
                }
            }else {
                const auto down_offset = kernels::iq_row_bytes(40, first[rank]);
                const auto owned_down = kernels::iq_row_bytes(40, valid[rank]);
                for (int64_t row = 0; row < geometry.n_embd; ++row)
                    std::memcpy(down.data() + size_t(row) * down_row,
                                blob.data() + format.down_off + size_t(row) * format.d_row + down_offset, owned_down);
            }
            std::memcpy(scales.data() + size_t(expert) * 4, blob.data() + format.tail_off, 4 * sizeof(float));
            if(p.b12x_a16) {
                for(const auto* plane:{&gate,&up})for(size_t at=0;at<plane->size();at+=36)
                    for(int j=0;j<4;++j)maximum_gu=std::max(maximum_gu,(*plane)[at+j]);
                for(size_t at=0;at<down.size();at+=36)
                    for(int j=0;j<4;++j)maximum_down=std::max(maximum_down,down[at+j]);
            }
            for (int role = 0; role < 3; ++role)
                if (!std::isfinite(scales[size_t(expert) * 4 + role])) throw std::runtime_error("nonfinite NVFP4 expert scale");
            if(p.b12x_run&&scales[size_t(expert)*4]!=scales[size_t(expert)*4+1])
                throw std::runtime_error("B12X single FC1 alpha requires equal original gate/up tails");
            check(cudaMemcpyAsync(static_cast<uint8_t*>(layer->gate.data) + expert * gu_bytes, gate.data(), gu_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
            check(cudaMemcpyAsync(static_cast<uint8_t*>(layer->up.data) + expert * gu_bytes, up.data(), gu_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
            check(cudaMemcpyAsync(static_cast<uint8_t*>(layer->down.data) + expert * down_bytes, down.data(), down_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
            // Pageable staging is reused only after its copies finish. Loading
            // is outside graph capture and outside every measured request.
            check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        }
        check(cudaMemcpyAsync(layer->scales.data, scales.data(), layer->scales.bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
        check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        layer->view = {layer->gate.data, layer->up.data, layer->down.data,
                       static_cast<const float*>(layer->scales.data), int(geometry.n_embd), padded,
                       int(geometry.n_expert), gu_bytes, down_bytes};
        if(balanced)layer->view.valid_ff=valid[rank];
        if(p.b12x_run) {
            namespace gpu=kernels::tp::cuda;
            const auto sizes=gpu::b12x_nvfp4_pack_sizes(int(geometry.n_expert),int(geometry.n_embd),padded);
            layer->w1.allocate(sizes.w1_codes);layer->w1s.allocate(sizes.w1_scales);
            layer->w1a.allocate(sizes.w1_alpha);layer->a1.allocate(sizes.a1_gscale);
            layer->w2.allocate(sizes.w2_codes);layer->w2s.allocate(sizes.w2_scales);
            layer->w2a.allocate(sizes.w2_alpha);layer->a2.allocate(sizes.a2_gscale);
            layer->alpha.allocate(size_t(geometry.n_expert)*4);
            gpu::b12x_nvfp4_repack(layer->gate.data,layer->gate.bytes,layer->up.data,layer->up.bytes,
                layer->down.data,layer->down.bytes,(float*)layer->scales.data,layer->scales.bytes,sizes,
                layer->w1.data,layer->w1.bytes,layer->w1s.data,layer->w1s.bytes,
                (float*)layer->w1a.data,layer->w1a.bytes,(float*)layer->a1.data,layer->a1.bytes,
                layer->w2.data,layer->w2.bytes,layer->w2s.data,layer->w2s.bytes,
                (float*)layer->w2a.data,layer->w2a.bytes,(float*)layer->a2.data,layer->a2.bytes,
                int(geometry.n_expert),int(geometry.n_embd),padded,stream);
            gpu::b12x_nvfp4_runtime_alphas((float*)layer->scales.data,(float*)layer->alpha.data,
                int(geometry.n_expert),stream);
            if(p.b12x_a16) {
                auto factor=[](uint8_t b) {
                    const int e=(b>>3)&15;const float value=e?std::ldexp(1.f+(b&7)/8.f,e-7):(b&7)/512.f;
                    return value>0.f&&value<448.f?std::exp2(std::floor(std::log2(448.f/value))):1.f;
                };
                const float gu_factor=factor(maximum_gu),down_factor=factor(maximum_down);
                auto repack=[&](Memory& code,Memory& sf,int rows,int cols,int rotation,float f) {
                    Memory packed_code,packed_sf;packed_code.allocate(code.bytes);packed_sf.allocate(sf.bytes);
                    gpu::b12x_nvfp4_a16_repack((uint8_t*)code.data,(uint8_t*)sf.data,
                        (uint32_t*)packed_code.data,(uint8_t*)packed_sf.data,512,rows,cols,rotation,f,stream);
                    check(cudaStreamSynchronize((cudaStream_t)stream));
                    check(cudaFree(code.data));check(cudaFree(sf.data));
                    code.data=packed_code.data;packed_code.data=nullptr;
                    sf.data=packed_sf.data;packed_sf.data=nullptr;
                };
                repack(layer->w1,layer->w1s,512,2560,256,gu_factor);
                repack(layer->w2,layer->w2s,2560,256,0,down_factor);
                gpu::b12x_a16_global_scales((float*)layer->scales.data,(float*)layer->alpha.data,
                    (float*)layer->w2a.data,512,gu_factor,down_factor,stream);
            }
            if(p.b12x_a8) {
                layer->w1mx.allocate(size_t(512)*512*80);layer->w1res.allocate(size_t(512)*512*160);
                layer->w2mx.allocate(size_t(512)*2560*8);layer->w2res.allocate(size_t(512)*2560*16);
                gpu::b12x_nvfp4_a8_scales((uint8_t*)layer->w1s.data,(uint8_t*)layer->w1mx.data,
                    (uint8_t*)layer->w1res.data,512,512,2560,(int*)p.status.data,stream);
                gpu::b12x_nvfp4_a8_scales((uint8_t*)layer->w2s.data,(uint8_t*)layer->w2mx.data,
                    (uint8_t*)layer->w2res.data,512,2560,256,(int*)p.status.data,stream);
                int loss=0;check(cudaMemcpyAsync(&loss,p.status.data,4,cudaMemcpyDeviceToHost,(cudaStream_t)stream));
                check(cudaStreamSynchronize((cudaStream_t)stream));
                if(loss)throw std::runtime_error("B12X A8 scale decomposition loses original values at layer "+
                    std::to_string(index)+" pair "+std::to_string(loss-1));
                p.bytes+=layer->w1mx.bytes+layer->w1res.bytes+layer->w2mx.bytes+layer->w2res.bytes;
            }
            p.bytes+=layer->w1.bytes+layer->w1s.bytes+layer->w1a.bytes+layer->a1.bytes+
                layer->w2.bytes+layer->w2s.bytes+layer->w2a.bytes+layer->a2.bytes+layer->alpha.bytes;
        }
        if(const char* packed=std::getenv("STRATA_TP_DECODE_GU_VECTOR");packed&&packed[0]=='1') {
            const size_t subblocks=size_t(geometry.n_expert)*padded*(geometry.n_embd/16);
            layer->decode_gu_codes.allocate(subblocks*16);
            layer->decode_gu_scales.allocate(subblocks*2);
            kernels::tp::cuda::nvfp4_pack_decode_gu(layer->view,layer->decode_gu_codes.data,
                static_cast<uint16_t*>(layer->decode_gu_scales.data),stream);
            layer->view.decode_gu_codes=layer->decode_gu_codes.data;
            layer->view.decode_gu_scales=static_cast<const uint16_t*>(layer->decode_gu_scales.data);
        }
        if(packed_gu) {
            Memory combined;combined.allocate(2*gu_bytes*geometry.n_expert+mmq_guard);
            check(cudaMemsetAsync((uint8_t*)combined.data+2*gu_bytes*geometry.n_expert,0,mmq_guard,(cudaStream_t)stream));
            for(int64_t e=0;e<geometry.n_expert;++e) {
                auto* dst=(uint8_t*)combined.data+size_t(e)*2*gu_bytes;
                check(cudaMemcpyAsync(dst,(uint8_t*)layer->gate.data+size_t(e)*gu_bytes,gu_bytes,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                check(cudaMemcpyAsync(dst+gu_bytes,(uint8_t*)layer->up.data+size_t(e)*gu_bytes,gu_bytes,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
            }
            check(cudaStreamSynchronize((cudaStream_t)stream));
            check(cudaFree(layer->gate.data));check(cudaFree(layer->up.data));
            layer->gate.data=combined.data;layer->gate.bytes=combined.bytes;combined.data=nullptr;
            layer->up.data=nullptr;layer->up.bytes=0;
            layer->view.gate=layer->gate.data;layer->view.up=(uint8_t*)layer->gate.data+gu_bytes;
            layer->view.gu_expert_bytes=2*gu_bytes;layer->view.packed_gate_up=true;
        }
        if(const char* on=std::getenv("STRATA_TP_NATIVE_GROUPED");on&&on[0]=='1') {
            const size_t tail=gu_bytes*2+down_bytes,stride=tail+16;
            layer->grouped.allocate(stride*geometry.n_expert+mmq_guard);
            check(cudaMemsetAsync((uint8_t*)layer->grouped.data+stride*geometry.n_expert,0,mmq_guard,(cudaStream_t)stream));
            for(int64_t e=0;e<geometry.n_expert;++e) {
                auto* dst=(uint8_t*)layer->grouped.data+size_t(e)*stride;
                check(cudaMemcpyAsync(dst,(uint8_t*)layer->gate.data+size_t(e)*gu_bytes,gu_bytes,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                check(cudaMemcpyAsync(dst+gu_bytes,(uint8_t*)layer->up.data+size_t(e)*gu_bytes,gu_bytes,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                check(cudaMemcpyAsync(dst+gu_bytes*2,(uint8_t*)layer->down.data+size_t(e)*down_bytes,down_bytes,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                check(cudaMemcpyAsync(dst+tail,(float*)layer->scales.data+size_t(e)*4,16,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
            }
            check(cudaStreamSynchronize((cudaStream_t)stream));
            // One copy of each rank's shard serves both native decode and MMQ.
            for(auto* buffer:{&layer->gate,&layer->up,&layer->down}) {
                check(cudaFree(buffer->data));buffer->data=nullptr;buffer->bytes=0;
            }
            layer->view.gate=layer->grouped.data;
            layer->view.up=(uint8_t*)layer->grouped.data+gu_bytes;
            layer->view.down=(uint8_t*)layer->grouped.data+gu_bytes*2;
            layer->view.gu_expert_bytes=layer->view.down_expert_bytes=stride;
            layer->view.grouped_blobs=layer->grouped.data;layer->view.grouped_blob_bytes=stride;
            layer->view.grouped_up_offset=gu_bytes;layer->view.grouped_down_offset=gu_bytes*2;
            layer->view.grouped_tail_offset=tail;
        }
        p.bytes += layer->gate.bytes + layer->up.bytes + layer->down.bytes + layer->scales.bytes+layer->grouped.bytes+
                   layer->decode_gu_codes.bytes+layer->decode_gu_scales.bytes;
        p.layers.push_back(std::move(layer));
    }
    p.scratch.allocate(kernels::tp::cuda::nvfp4_workspace_bytes(max_rows, int(geometry.n_expert_used), int(geometry.n_embd), padded));
    std::fprintf(stderr, "TP NVFP4 rank %d: padded FF256, %llu weight bytes, prefill %s, decode FP4/Q8 DP4A\n",
                 rank, (unsigned long long)p.bytes, kernels::tp::cuda::nvfp4_prefill_dispatch());
    if(p.b12x_run)std::fprintf(stderr,"TP NVFP4 rank %d: B12X dynamic NVFP4 enabled for independent rows 9..40\n",rank);
    if(!p.status.data)p.status.allocate(sizeof(int));
    check(cudaMemsetAsync(p.status.data, 0, sizeof(int), static_cast<cudaStream_t>(stream)));
    check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    return true;
} catch (const std::exception& e) { error = e.what(); return false; }
bool NativeNvfp4Experts::local_sum(int64_t layer, const core::TensorParallelRows& rows,
                                   const float* input, const int32_t* ids, const float* weights, int k,
                                   float* output, void* stream, std::string& error) try {
    auto& p = *impl_;
    if (layer < 0 || size_t(layer) >= p.layers.size() || rows.count < 1 || rows.count > p.max_rows ||
        k != p.geometry.n_expert_used) { error = "NVFP4 routed batch exceeds initialized geometry"; return false; }
    if(p.b12x_run&&rows.independent&&rows.count>8&&rows.count<=40) {
        auto& l=*p.layers[size_t(layer)];
        kernels::f32_to_bf16_bulk(input,(uint16_t*)p.b12x_input.data,rows.count*2560,stream);
        if(p.b12x_a16)kernels::tp::cuda::b12x_a16_pack_routes(ids,int(rows.count)*k,
            (int*)p.b12x_routes.data,(int*)p.b12x_blocks.data,(int*)p.b12x_route_count.data,(int*)p.status.data,stream);
        else kernels::tp::cuda::b12x_nvfp4_prepare_ids(ids,(int64_t*)p.b12x_ids.data,
            int(rows.count)*k,int(p.geometry.n_expert),static_cast<int*>(p.status.data),stream);
        void* buffers[]={p.b12x_input.data,p.b12x_ids.data,const_cast<float*>(weights),
            l.w1.data,l.w1s.data,l.alpha.data,l.a1.data,l.w2.data,l.w2s.data,l.w2a.data,l.a2.data,
            p.b12x_output.data,p.b12x_workspace.data,l.w1mx.data,l.w2mx.data,l.w1res.data,l.w2res.data};
        if(p.b12x_a16) {
            buffers[1]=const_cast<int32_t*>(ids);buffers[13]=p.b12x_routes.data;
            buffers[14]=p.b12x_blocks.data;buffers[15]=p.b12x_route_count.data;buffers[16]=p.status.data;
        }
        if(p.b12x_run(buffers,int(rows.count),p.b12x_workspace.bytes,stream)!=0)
            throw std::runtime_error("B12X native launch failed");
        kernels::tp::cuda::transport_bf16_unpack((uint16_t*)p.b12x_output.data,output,rows.count*2560,stream);
        return true;
    }
    const bool capture=!p.fixture_dir.empty()&&!p.fixture_written&&layer==0&&rows.independent&&
                       rows.count>=16&&rows.count<=40;
    if(capture) {
        p.fixture_rows=int(rows.count);
        check(cudaMemcpyAsync(p.fixture_input.data,input,size_t(rows.count)*2560*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
        check(cudaMemcpyAsync(p.fixture_ids.data,ids,size_t(rows.count)*10*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
        check(cudaMemcpyAsync(p.fixture_weights.data,weights,size_t(rows.count)*10*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
    }
    kernels::tp::cuda::nvfp4_local_sum(p.layers[size_t(layer)]->view, input, ids, weights,
        int(rows.count), k, output, p.scratch.data, size_t(p.scratch.bytes), static_cast<int*>(p.status.data),
        stream, !rows.independent);
    if(capture)check(cudaMemcpyAsync(p.fixture_output.data,output,size_t(rows.count)*2560*4,
                                     cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
    return true;
} catch (const std::exception& e) { error = e.what(); return false; }
bool NativeNvfp4Experts::validate_execution_status(void* stream, std::string& error) {
    int status = 0;
    auto result = cudaMemcpyAsync(&status, impl_->status.data, sizeof(status), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
    if (result == cudaSuccess) result = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    if (result != cudaSuccess || status) { error = result != cudaSuccess ? cudaGetErrorString(result) : "invalid NVFP4 routed device execution"; return false; }
    auto& p=*impl_;
    if(p.fixture_rows&&!p.fixture_written)try {
        const auto dir=std::filesystem::path(p.fixture_dir);std::filesystem::create_directories(dir);
        auto write=[&](const char* name,const void* data,size_t bytes) {
            std::vector<uint8_t> host(bytes);check(cudaMemcpy(host.data(),data,bytes,cudaMemcpyDeviceToHost));
            std::ofstream file(dir/name,std::ios::binary);file.write((const char*)host.data(),std::streamsize(bytes));
            if(!file)throw std::runtime_error("cannot write real expert fixture");
        };
        const auto& layer=*p.layers.front();const auto& v=layer.view;
        write("gate.bin",v.gate,layer.gate.bytes);write("up.bin",v.up,layer.up.bytes);
        write("down.bin",v.down,layer.down.bytes);write("scales.bin",v.scales,layer.scales.bytes);
        write("input.bin",p.fixture_input.data,size_t(p.fixture_rows)*2560*4);
        write("ids.bin",p.fixture_ids.data,size_t(p.fixture_rows)*10*4);
        write("weights.bin",p.fixture_weights.data,size_t(p.fixture_rows)*10*4);
        write("output.bin",p.fixture_output.data,size_t(p.fixture_rows)*2560*4);
        std::ofstream meta(dir/"meta.txt");
        meta<<"rows "<<p.fixture_rows<<"\nhidden 2560\nff 256\nexperts 512\nvalid_ff "<<v.valid_ff
            <<"\ngu_bytes "<<v.gu_expert_bytes<<"\ndown_bytes "<<v.down_expert_bytes<<'\n';
        if(!meta)throw std::runtime_error("cannot write real expert metadata");
        p.fixture_written=true;
    }catch(const std::exception& e){error=e.what();return false;}
    return true;
}
uint64_t NativeNvfp4Experts::weight_bytes() const { return impl_->bytes; }
}
