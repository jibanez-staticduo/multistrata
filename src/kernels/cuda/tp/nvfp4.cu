#include "strata/kernels/tp_nvfp4.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace strata::kernels::tp::cuda {
namespace {
namespace mmq = strata::prefill::mmq;
struct Q8 { __half2 ds; int8_t q[32]; };
struct F4 { uint8_t d[4], q[32]; };
__device__ __forceinline__ float half_scale(uint8_t x) {
    if (!x || x == 127) return 0.f;
    unsigned e = (x >> 3) & 15, m = x & 7;
    return e ? __uint_as_float(((e + 119) << 23) | (m << 20)) : float(m) / 1024.f;
}
__device__ __forceinline__ int unpack_codes(unsigned packed) {
    const unsigned selector=(packed&7)|((packed>>4)&0x70)|((packed>>8)&0x700)|((packed>>12)&0x7000);
    const unsigned values=__byte_perm(0x03020100,0x0c080604,selector);
    const unsigned sign=(packed>>3)&0x01010101;
    return __vadd4(values^__vsub4(0,sign),sign);
}
__device__ __forceinline__ float dot_sub(const F4& b,const Q8* x,int sub) {
    const int s=sub%4;const Q8& a=x[sub/2];
    const auto* codes=reinterpret_cast<const unsigned*>(b.q+s*8);
    const auto* act=reinterpret_cast<const int*>(a.q+(sub%2)*16);
    int sum=__dp4a(unpack_codes(codes[0]&0x0f0f0f0f),act[0],0);
    sum=__dp4a(unpack_codes(codes[1]&0x0f0f0f0f),act[1],sum);
    sum=__dp4a(unpack_codes((codes[0]>>4)&0x0f0f0f0f),act[2],sum);
    sum=__dp4a(unpack_codes((codes[1]>>4)&0x0f0f0f0f),act[3],sum);
    return half_scale(b.d[s])*__low2float(a.ds)*float(sum);
}
template<int Width,int Lanes>
__device__ __forceinline__ float dot(const F4* w,const Q8* x) {
    float value=0;const int lane=threadIdx.x%Lanes;
#pragma unroll
    for(int sub=lane;sub<Width/16;sub+=Lanes)value+=dot_sub(w[sub/4],x,sub);
#pragma unroll
    for(int offset=Lanes/2;offset;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset,Lanes);
    return value;
}
template<int Lanes,bool Prefetch=false>
__device__ __forceinline__ void dot_pair(const F4* gate,const F4* up,const Q8* x,float& g,float& u) {
    g=0;u=0;const int lane=threadIdx.x%Lanes;
#pragma unroll
    for(int sub=lane;sub<160;sub+=Lanes) {
        if constexpr(Prefetch)if(sub+Lanes<160) {
            asm volatile("prefetch.global.L1 [%0];"::"l"(gate+(sub+Lanes)/4));
            asm volatile("prefetch.global.L1 [%0];"::"l"(up+(sub+Lanes)/4));
        }
        g+=dot_sub(gate[sub/4],x,sub);
        u+=dot_sub(up[sub/4],x,sub);
    }
#pragma unroll
    for(int offset=Lanes/2;offset;offset/=2){g+=__shfl_down_sync(0xffffffff,g,offset,Lanes);u+=__shfl_down_sync(0xffffffff,u,offset,Lanes);}
}
__global__ void decode_route_order(const int* ids,int routes,int* order) {
    const int route=blockIdx.x*blockDim.x+threadIdx.x;
    if(route>=routes)return;
    const int expert=ids[route];int index=0;
    for(int i=0;i<routes;++i)index+=ids[i]<expert||(ids[i]==expert&&i<route);
    order[index]=route;
}
template<int Lanes=16,bool Prefetch=false,bool Ordered=false>
__global__ void gu_decode(Nvfp4Layer layer, const Q8* x, const int* ids, int k,
                           float* h, int* status,const int* order=nullptr) {
    constexpr int per_block=128/Lanes;
    const int route = Ordered?order[blockIdx.y]:int(blockIdx.y);
    const int f = blockIdx.x * per_block + threadIdx.y * (32/Lanes) + threadIdx.x / Lanes;
    const int expert = ids[route];
    if (f >= layer.local_ff) return;
    float g = 0, u = 0;
    if (expert >= 0 && expert < layer.experts && (!layer.valid_ff || f<layer.valid_ff)) {
        const auto* gate = reinterpret_cast<const F4*>(static_cast<const uint8_t*>(layer.gate) +
            size_t(expert) * layer.gu_expert_bytes) + size_t(f) * (layer.hidden / 64);
        const auto* up = reinterpret_cast<const F4*>(static_cast<const uint8_t*>(layer.up) +
            size_t(expert) * layer.gu_expert_bytes) + size_t(f) * (layer.hidden / 64);
        const Q8* a = x + size_t(route / k) * (layer.hidden / 32);
        dot_pair<Lanes,Prefetch>(gate,up,a,g,u);
        g *= layer.scales[4 * expert];
        u *= layer.scales[4 * expert + 1];
    } else if ((expert<0||expert>=layer.experts)&&threadIdx.x == 0) atomicExch(status, 1);
    if (threadIdx.x % Lanes == 0) h[size_t(route) * layer.local_ff + f] = g / (1.f + expf(-g)) * u;
}
__global__ void repack_gu(Nvfp4Layer layer,uint4* codes,unsigned short* scales) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(512)*256*160)return;
    const int e=i/(256*160),f=(i/160)%256,sub=i%160;
    const auto* g=reinterpret_cast<const F4*>((const uint8_t*)layer.gate+size_t(e)*layer.gu_expert_bytes)+size_t(f)*40+sub/4;
    const auto* u=reinterpret_cast<const F4*>((const uint8_t*)layer.up+size_t(e)*layer.gu_expert_bytes)+size_t(f)*40+sub/4;
    const auto* gc=reinterpret_cast<const unsigned*>(g->q+(sub%4)*8);
    const auto* uc=reinterpret_cast<const unsigned*>(u->q+(sub%4)*8);
    codes[i]=make_uint4(gc[0],gc[1],uc[0],uc[1]);
    scales[i]=unsigned(g->d[sub%4])|(unsigned(u->d[sub%4])<<8);
}
__device__ __forceinline__ int packed_dot(unsigned c0,unsigned c1,int4 act) {
    int sum=__dp4a(unpack_codes(c0&0x0f0f0f0f),act.x,0);
    sum=__dp4a(unpack_codes(c1&0x0f0f0f0f),act.y,sum);
    sum=__dp4a(unpack_codes((c0>>4)&0x0f0f0f0f),act.z,sum);
    return __dp4a(unpack_codes((c1>>4)&0x0f0f0f0f),act.w,sum);
}
template<int Lanes>
__device__ __forceinline__ void dot_pair_vector(const uint4* codes,const unsigned short* scales,const Q8* x,float& g,float& u) {
    g=0;u=0;const int lane=threadIdx.x%Lanes;
#pragma unroll
    for(int sub=lane;sub<160;sub+=Lanes) {
        const uint4 c=codes[sub];const unsigned short sc=scales[sub];
        const Q8& a=x[sub/2];
        const int* aq=reinterpret_cast<const int*>(a.q+(sub%2)*16);
        const int4 act=make_int4(aq[0],aq[1],aq[2],aq[3]);
        const float ad=__low2float(a.ds);
        g+=half_scale(uint8_t(sc))*ad*float(packed_dot(c.x,c.y,act));
        u+=half_scale(uint8_t(sc>>8))*ad*float(packed_dot(c.z,c.w,act));
    }
#pragma unroll
    for(int offset=Lanes/2;offset;offset/=2){g+=__shfl_down_sync(0xffffffff,g,offset,Lanes);u+=__shfl_down_sync(0xffffffff,u,offset,Lanes);}
}
template<int Lanes=16>
__global__ void gu_decode_vector(Nvfp4Layer layer, const Q8* x, const int* ids, int k,
                           float* h, int* status) {
    constexpr int per_block=128/Lanes;
    const int route = blockIdx.y, f = blockIdx.x * per_block + threadIdx.y * (32/Lanes) + threadIdx.x / Lanes;
    const int expert = ids[route];
    if (f >= layer.local_ff) return;
    float g = 0, u = 0;
    if (expert >= 0 && expert < layer.experts && (!layer.valid_ff || f<layer.valid_ff)) {
        const auto* gate=static_cast<const uint4*>(layer.decode_gu_codes)+(size_t(expert)*256+f)*160;
        const auto* up=layer.decode_gu_scales+(size_t(expert)*256+f)*160;
        const Q8* a=x+size_t(route/k)*80;
        dot_pair_vector<Lanes>(gate,up,a,g,u);
        g *= layer.scales[4 * expert];
        u *= layer.scales[4 * expert + 1];
    } else if ((expert<0||expert>=layer.experts)&&threadIdx.x == 0) atomicExch(status, 1);
    if (threadIdx.x % Lanes == 0) h[size_t(route) * layer.local_ff + f] = g / (1.f + expf(-g)) * u;
}
template<bool Ordered=false>
__global__ void down_decode(Nvfp4Layer layer, const Q8* h, const int* ids,
                             float* parts,const int* order=nullptr) {
    const int route = Ordered?order[blockIdx.y]:int(blockIdx.y);
    const int f = blockIdx.x * 16 + threadIdx.y * 4 + threadIdx.x / 8;
    if (f >= layer.hidden) return;
    const int expert = ids[route];
    float value = 0;
    if (expert >= 0 && expert < layer.experts) {
        const auto* w = reinterpret_cast<const F4*>(static_cast<const uint8_t*>(layer.down) +
            size_t(expert) * layer.down_expert_bytes) + size_t(f) * (layer.local_ff / 64);
        const auto* input=h+size_t(route)*(layer.local_ff/32);
        if(layer.valid_ff==224)value=dot<224,8>(w,input);
        else if(layer.valid_ff==208)value=dot<208,8>(w,input);
        else value=dot<256,8>(w,input);
        value*=layer.scales[4 * expert + 2];
    }
    if (threadIdx.x % 8 == 0) parts[size_t(route) * layer.hidden + f] = value;
}
struct ReuseTask { int expert,first,count; };
template<int Batch>
__device__ __forceinline__ void gu_grouped_task(
    Nvfp4Layer layer,const Q8* x,const ReuseTask& task,const int* source,float* h,int tile,int valid_ff) {
    const int e=task.expert,first=task.first,count=task.count;
    const int f=tile*8+threadIdx.y*2+threadIdx.x/16,lane=threadIdx.x%16;
    if(f>=layer.local_ff)return;
    const F4* gate=(const F4*)((const uint8_t*)layer.gate+size_t(e)*layer.gu_expert_bytes)+size_t(f)*40;
    const F4* up=(const F4*)((const uint8_t*)layer.up+size_t(e)*layer.gu_expert_bytes)+size_t(f)*40;
    float g[Batch]={},u[Batch]={};
    if(f<valid_ff) for(int sub=lane;sub<160;sub+=16) {
        const int block=sub/4,s=sub%4;
        const auto* gc=(const unsigned*)(gate[block].q+s*8);
        const auto* uc=(const unsigned*)(up[block].q+s*8);
        const unsigned gv0=gc[0],gv1=gc[1],uv0=uc[0],uv1=uc[1];
        const int gi[4]={unpack_codes(gv0&0x0f0f0f0f),unpack_codes(gv1&0x0f0f0f0f),unpack_codes((gv0>>4)&0x0f0f0f0f),unpack_codes((gv1>>4)&0x0f0f0f0f)};
        const int ui[4]={unpack_codes(uv0&0x0f0f0f0f),unpack_codes(uv1&0x0f0f0f0f),unpack_codes((uv0>>4)&0x0f0f0f0f),unpack_codes((uv1>>4)&0x0f0f0f0f)};
        const float gs=half_scale(gate[block].d[s]),us=half_scale(up[block].d[s]);
#pragma unroll
        for(int r=0;r<Batch;++r)if(r<count) {
            const Q8& a=x[size_t(source[first+r])*80+sub/2];
            const int* q=(const int*)(a.q+(sub%2)*16);int gd=0,ud=0;
#pragma unroll
            for(int j=0;j<4;++j){gd=__dp4a(gi[j],q[j],gd);ud=__dp4a(ui[j],q[j],ud);}
            const float scale=__low2float(a.ds);g[r]+=gs*scale*float(gd);u[r]+=us*scale*float(ud);
        }
    }
#pragma unroll
    for(int r=0;r<Batch;++r) {
        for(int o=8;o;o/=2){g[r]+=__shfl_down_sync(0xffffffff,g[r],o,16);u[r]+=__shfl_down_sync(0xffffffff,u[r],o,16);}
        if(!lane&&r<count){
            if(f<valid_ff){const float v=g[r]*layer.scales[4*e];const float up=u[r]*layer.scales[4*e+1];h[size_t(first+r)*256+f]=v/(1.f+expf(-v))*up;}
            else h[size_t(first+r)*256+f]=0.f;
        }
    }
}
template<int Batch>
__device__ __forceinline__ void down_grouped_task(
    Nvfp4Layer layer,const Q8* h,const ReuseTask& task,float* parts,int tile,int valid_ff) {
    const int e=task.expert,first=task.first,count=task.count;
    const int f=tile*16+threadIdx.y*4+threadIdx.x/8,lane=threadIdx.x%8;
    if(f>=layer.hidden)return;
    const F4* w=(const F4*)((const uint8_t*)layer.down+size_t(e)*layer.down_expert_bytes)+size_t(f)*4;
    float sum[Batch]={};
    for(int sub=lane;sub<16;sub+=8) {
        if(sub>=valid_ff/16) continue;
        const F4& b=w[sub/4];const int s=sub%4;const unsigned* codes=(const unsigned*)(b.q+s*8);
        const unsigned v0=codes[0],v1=codes[1];
        const int vals[4]={unpack_codes(v0&0x0f0f0f0f),unpack_codes(v1&0x0f0f0f0f),unpack_codes((v0>>4)&0x0f0f0f0f),unpack_codes((v1>>4)&0x0f0f0f0f)};
        const float scale=half_scale(b.d[s]);
#pragma unroll
        for(int r=0;r<Batch;++r)if(r<count) {
            const Q8& a=h[size_t(first+r)*8+sub/2];const int* q=(const int*)(a.q+(sub%2)*16);int dot=0;
#pragma unroll
            for(int j=0;j<4;++j)dot=__dp4a(vals[j],q[j],dot);
            sum[r]+=scale*__low2float(a.ds)*float(dot);
        }
    }
#pragma unroll
    for(int r=0;r<Batch;++r) {
        for(int o=4;o;o/=2)sum[r]+=__shfl_down_sync(0xffffffff,sum[r],o,8);
        if(!lane&&r<count)parts[size_t(first+r)*layer.hidden+f]=sum[r]*layer.scales[4*e+2];
    }
}
__global__ void sort_routes_stable(const int* ids,int routes,int k,int experts,
                                   int* cursor,int* source,int* inverse,
                                   int* sorted_expert,int* identity) {
    if(blockIdx.x||threadIdx.x)return;
    for(int i=0;i<routes;++i) {
        int e=ids[i];
        if(e<0||e>=experts)e=0;
        const int at=cursor[e]++;
        source[at]=i/k;
        inverse[i]=at;
        sorted_expert[at]=e;
        identity[at]=at;
    }
}
__global__ void build_group_tasks(const int* bounds,int experts,int batch,
                                  ReuseTask* tasks,int* task_count) {
    if(blockIdx.x||threadIdx.x)return;
    int n=0;
    for(int e=0;e<experts;++e) {
        for(int first=bounds[e];first<bounds[e+1];first+=batch)
            tasks[n++]={e,first,min(batch,bounds[e+1]-first)};
    }
    *task_count=n;
}
__global__ void gu_grouped_static(
    Nvfp4Layer layer,const Q8* x,const int* source,const ReuseTask* tasks,
    const int* task_count,float* h,int valid_ff) {
    const int task_id=blockIdx.y;
    if(task_id>=*task_count)return;
    const ReuseTask task=tasks[task_id];
    if(task.count==1)gu_grouped_task<1>(layer,x,task,source,h,blockIdx.x,valid_ff);
    else gu_grouped_task<4>(layer,x,task,source,h,blockIdx.x,valid_ff);
}
__global__ void down_grouped_static(
    Nvfp4Layer layer,const Q8* h,const ReuseTask* tasks,const int* task_count,
    float* parts,int valid_ff) {
    const int task_id=blockIdx.y;
    if(task_id>=*task_count)return;
    const ReuseTask task=tasks[task_id];
    if(task.count==1)down_grouped_task<1>(layer,h,task,parts,blockIdx.x,valid_ff);
    else down_grouped_task<4>(layer,h,task,parts,blockIdx.x,valid_ff);
}
__global__ void count_routes(const int* ids, int routes, int k, int experts, int* counts, int* status) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routes) return;
    int e = ids[i];
    if (e < 0 || e >= experts) { atomicExch(status, 1); e = 0; }
    for (int j = 0; j < i % k; ++j)
        if (ids[i - i % k + j] == ids[i]) atomicExch(status, 1);
    atomicAdd(counts + e, 1);
}
__global__ void prefix(int* counts, int* bounds, int* cursor, int experts) {
    if (threadIdx.x) return;
    int sum = 0;
    for (int e = 0; e < experts; ++e) { bounds[e] = cursor[e] = sum; sum += counts[e]; }
    bounds[experts] = sum;
}
__global__ void sort_routes(const int* ids, int routes, int k, int experts, int* cursor,
                            int* source, int* inverse, int* sorted_expert, int* identity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routes) return;
    int e = ids[i];
    if (e < 0 || e >= experts) e = 0;
    const int at = atomicAdd(cursor + e, 1);
    source[at] = i / k; inverse[i] = at; sorted_expert[at] = e; identity[at] = at;
}
__global__ void native_group_plan(const int* bounds,const uint8_t* blobs,size_t bytes,
                                  unsigned long long* pointers,int* starts,int* count) {
    if(threadIdx.x)return;int n=0;
    for(int e=0;e<512;++e)if(bounds[e+1]>bounds[e]) {
        pointers[n]=(unsigned long long)(blobs+size_t(e)*bytes);starts[n++]=bounds[e];
    }
    starts[n]=bounds[512];*count=n;
}
__global__ void activate(const float* g, const float* u, const int* expert, const float* scales,
                          int routes, int ff, float* h) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(routes) * ff) return;
    const int e = expert[i / ff];
    const float gate = g[i] * scales[4 * e], up = u[i] * scales[4 * e + 1];
    h[i] = gate / (1.f + expf(-gate)) * up;
}
__global__ void activate_packed(const float* gu,const int* expert,const float* scales,int routes,int ff,float* h) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(routes)*ff)return;
    const size_t row=i/ff,col=i%ff;const int e=expert[row];
    const float gate=gu[row*2*ff+col]*scales[4*e],up=gu[row*2*ff+ff+col]*scales[4*e+1];
    h[i]=gate/(1.f+expf(-gate))*up;
}
__global__ void combine(const float* parts, const int* inverse, const int* ids,
                         const float* scales, const float* weights, int rows, int k,
                         int hidden, int experts, bool scaled, float* output) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * hidden) return;
    const int row = int(i / hidden), col = int(i % hidden);
    float sum = 0;
    for (int j = 0; j < k; ++j) {
        const int route = row * k + j, e = ids[route];
        if (e < 0 || e >= experts) continue;
        const int at = inverse ? inverse[route] : route;
        const float value = parts[size_t(at) * hidden + col] * (scaled ? scales[4 * e + 2] : 1.f);
        sum += weights[route] * value;
    }
    output[i] = sum;
}
struct Workspace {
    uint8_t* base; size_t used = 0;
    void* take(size_t bytes) { auto* p = base ? base + used : nullptr; used += (bytes + 255) & ~size_t(255); return p; }
};
struct Buffers {
    void *xq, *hq;
    float *g, *u, *h, *parts, *xs, *hs;
    int *counts, *bounds, *cursor, *source, *inverse, *expert, *identity, *tiles;
    unsigned long long* pointers;void* native_scratch;
    ReuseTask* reuse_tasks;int* reuse_count;
    Buffers(Workspace& w, int rows, int k, int hidden, int ff) {
        const size_t routes = size_t(rows) * k;
        xq = w.take(std::max(mmq::q8_bytes(routes, hidden), size_t(rows) * (hidden / 32) * 36));
        hq = w.take(std::max(mmq::q8_bytes(routes, ff), routes * (ff / 32) * 36));
        g = (float*)w.take(routes * ff * 4); u = (float*)w.take(routes * ff * 4);
        h = (float*)w.take(routes * ff * 4); parts = (float*)w.take(routes * hidden * 4);
        xs = (float*)w.take(routes * 4); hs = (float*)w.take(routes * 4);
        counts = (int*)w.take(512 * 4); bounds = (int*)w.take(513 * 4); cursor = (int*)w.take(512 * 4);
        source = (int*)w.take(routes * 4); inverse = (int*)w.take(routes * 4);
        expert = (int*)w.take(routes * 4); identity = (int*)w.take(routes * 4);
        tiles = (int*)w.take(513 * 4);
        pointers=(unsigned long long*)w.take(512*8);
        reuse_tasks=(ReuseTask*)w.take(routes*sizeof(ReuseTask));reuse_count=(int*)w.take(4);
        native_scratch=w.take(strata::kernels::native_expert_scratch_bytes(routes,ff));
    }
};
thread_local std::unique_ptr<mmq::Context> context;
}
void nvfp4_pack_decode_gu(const Nvfp4Layer& layer,void* codes,uint16_t* scales,void* stream) {
    if(!layer.gate||!layer.up||!codes||!scales||!stream||layer.hidden!=2560||layer.local_ff!=256||layer.experts!=512)
        throw std::invalid_argument("unsupported vector GU pack geometry");
    repack_gu<<<(size_t(512)*256*160+255)/256,256,0,(cudaStream_t)stream>>>(layer,(uint4*)codes,scales);
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
size_t nvfp4_workspace_bytes(int rows, int k, int hidden, int ff) {
    if (rows < 1 || rows > 8192 || k != 10 || hidden != 2560 || ff != 256)
        throw std::invalid_argument("unsupported NVFP4 workspace geometry");
    // Construct the host MMQ context and probe W4A4 outside every graph capture.
    if (!context) context = std::make_unique<mmq::Context>();
    (void)mmq::fp4_activations(40);
    if (std::getenv("STRATA_TP_PREFILL_TILES")) (void)mmq::nvfp4_prefill_tiles_available();
    if (std::getenv("STRATA_TP_DECODE_FP4")) (void)mmq::nvfp4_decode_tiles_available();
    Workspace workspace{nullptr}; Buffers buffers(workspace, rows, k, hidden, ff);
    return workspace.used;
}
const char* nvfp4_prefill_dispatch() {
    return mmq::fp4_activations(40) ? "NVFP4 W4A4 FP4 MMA" : "NVFP4 W4A8 integer MMA";
}
void nvfp4_local_sum(const Nvfp4Layer& layer, const float* input, const int32_t* ids,
                     const float* weights, int rows, int k, float* output, void* scratch,
                     size_t bytes, int* status, void* stream, bool prefill) {
    if (!context || !layer.gate || !layer.up || !layer.down || !layer.scales || layer.experts != 512 ||
        rows < 1 || rows > 8192 || k != 10 || layer.hidden != 2560 || layer.local_ff != 256 ||
        !input || !ids || !weights || !output || !scratch || !status || !stream)
        throw std::invalid_argument("invalid NVFP4 local expert contract");
    Workspace workspace{static_cast<uint8_t*>(scratch)};
    Buffers b(workspace, rows, k, layer.hidden, layer.local_ff);
    if (bytes < workspace.used) throw std::invalid_argument("NVFP4 workspace too small");
    const auto cs = static_cast<cudaStream_t>(stream);
    const int routes = rows * k, hidden = layer.hidden, ff = layer.local_ff;
    const bool native_grouped=layer.grouped_blobs&&!prefill;
    static const bool gu_warp32=[] { const char* e=std::getenv("STRATA_TP_GU_WARP32");return e&&e[0]=='1'; }();
    static const bool gu_prefetch=[] { const char* e=std::getenv("STRATA_TP_GU_PREFETCH");return e&&e[0]=='1'; }();
    static const bool reuse_on=[] { const char* e=std::getenv("STRATA_TP_GROUP_REUSE");return e&&e[0]=='4'; }();
    static const bool order_on=[] { const char* e=std::getenv("STRATA_TP_ROUTE_ORDER");return e&&e[0]=='1'; }();
    static const bool decode_fp4_on=[] { const char* e=std::getenv("STRATA_TP_DECODE_FP4");return e&&e[0]=='1'; }();
    const bool decode_fp4=decode_fp4_on&&!prefill&&rows>8&&mmq::fp4_activations(40)&&
                          mmq::nvfp4_decode_tiles_available();
    const bool reuse=reuse_on&&!prefill&&rows>8;
    if(reuse) {
        cudaMemsetAsync(b.counts,0,512*4,cs);
        count_routes<<<(routes+255)/256,256,0,cs>>>(ids,routes,k,512,b.counts,status);
        prefix<<<1,32,0,cs>>>(b.counts,b.bounds,b.cursor,512);
        sort_routes_stable<<<1,1,0,cs>>>(ids,routes,k,512,b.cursor,b.source,b.inverse,b.expert,b.identity);
        build_group_tasks<<<1,1,0,cs>>>(b.bounds,512,4,b.reuse_tasks,b.reuse_count);
        quantize_q8_1_rows(input,rows,hidden,b.xq,stream);
        const int valid=layer.valid_ff?layer.valid_ff:ff;
        gu_grouped_static<<<dim3(32,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.xq,b.source,b.reuse_tasks,b.reuse_count,b.h,valid);
        quantize_q8_1_rows(b.h,routes,ff,b.hq,stream);
        down_grouped_static<<<dim3(160,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.hq,b.reuse_tasks,b.reuse_count,b.parts,valid);
    } else if(native_grouped) {
        cudaMemsetAsync(b.counts,0,512*4,cs);
        count_routes<<<(routes+255)/256,256,0,cs>>>(ids,routes,k,512,b.counts,status);
        prefix<<<1,32,0,cs>>>(b.counts,b.bounds,b.cursor,512);
        sort_routes<<<(routes+255)/256,256,0,cs>>>(ids,routes,k,512,b.cursor,b.source,b.inverse,b.expert,b.identity);
        native_group_plan<<<1,32,0,cs>>>(b.bounds,(const uint8_t*)layer.grouped_blobs,
            layer.grouped_blob_bytes,b.pointers,b.tiles,b.cursor);
        quantize_q8_1_rows(input,rows,hidden,b.xq,stream);
        strata::kernels::NativeExpertLayout layout;
        layout.gu_type=layout.d_type=40;layout.n_embd=hidden;layout.n_ff=ff;
        layout.gu_row=hidden/64*36;layout.d_row=ff/64*36;
        layout.up_off=layer.grouped_up_offset;layout.down_off=layer.grouped_down_offset;
        layout.tail_off=layer.grouped_tail_offset;layout.bytes=layer.grouped_blob_bytes;
        strata::kernels::native_expert_grouped(layout,b.pointers,b.tiles,b.cursor,b.identity,b.source,
            routes,routes,b.xq,b.native_scratch,b.parts,stream,routes);
    } else if ((!prefill || rows <= 8) && !decode_fp4) {
        quantize_q8_1_rows(input, rows, hidden, b.xq, stream);
        const bool ordered=order_on&&!prefill&&rows>8;
        if(ordered) {
            decode_route_order<<<(routes+255)/256,256,0,cs>>>(ids,routes,b.identity);
            gu_decode<16,false,true><<<dim3((ff+7)/8,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.xq,ids,k,b.h,status,b.identity);
        } else if(layer.decode_gu_codes&&layer.decode_gu_scales&&rows>8)
            gu_decode_vector<16><<<dim3((ff+7)/8,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.xq,ids,k,b.h,status);
        else if(gu_prefetch&&rows>8)gu_decode<16,true><<<dim3((ff+7)/8,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.xq,ids,k,b.h,status);
        else if(gu_warp32)gu_decode<32><<<dim3((ff+3)/4,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.xq,ids,k,b.h,status);
        else gu_decode<16><<<dim3((ff + 7) / 8, routes), dim3(32,4), 0, cs>>>(layer, (Q8*)b.xq, ids, k, b.h, status);
        quantize_q8_1_rows(b.h, routes, ff, b.hq, stream);
        if(ordered)down_decode<true><<<dim3((hidden+15)/16,routes),dim3(32,4),0,cs>>>(layer,(Q8*)b.hq,ids,b.parts,b.identity);
        else down_decode<><<<dim3((hidden + 15) / 16, routes), dim3(32,4), 0, cs>>>(layer, (Q8*)b.hq, ids, b.parts);
    } else {
        cudaMemsetAsync(b.counts, 0, 512 * sizeof(int), cs);
        count_routes<<<(routes + 255) / 256, 256, 0, cs>>>(ids, routes, k, 512, b.counts, status);
        prefix<<<1,32,0,cs>>>(b.counts, b.bounds, b.cursor, 512);
        sort_routes<<<(routes + 255) / 256,256,0,cs>>>(ids, routes, k, 512, b.cursor, b.source, b.inverse, b.expert, b.identity);
        static const bool scatter_on=[] { const char* e=std::getenv("STRATA_TP_PREFILL_QUANT_SCATTER");return e&&e[0]=='1'; }();
        if(scatter_on&&prefill&&mmq::fp4_activations(40))
            mmq::quantize_nvfp4_scatter(input,b.inverse,b.xq,b.xs,hidden,hidden,rows,k,stream);
        else mmq::quantize(input, b.source, b.xq, 40, hidden, hidden, routes, stream, b.xs);
        mmq::Product p;
        p.type = 40; p.w_rows = ff; p.w_cols = hidden; p.expert_bytes = layer.gu_expert_bytes;
        p.n = 512; p.xq = b.xq; p.bounds = b.bounds; p.ids = b.identity;
        p.total_rows = routes; p.max_rows = rows; p.ld_dst = ff;
        p.y_scale = mmq::fp4_activations(40) ? b.xs : nullptr;
        static const bool tiles_on = [] { const char* e=std::getenv("STRATA_TP_PREFILL_TILES"); return e&&e[0]=='1'; }();
        const bool compact=tiles_on&&mmq::fp4_activations(40)&&mmq::nvfp4_prefill_tiles_available();
        auto product=[&] { if(decode_fp4)mmq::nvfp4_decode_tiles(p,b.tiles,stream);
                          else if(compact)mmq::nvfp4_prefill_tiles(p,b.tiles,stream);else context->run(p,stream); };
        if(layer.packed_gate_up) {
            // g and u are adjacent equally sized workspace allocations.
            p.w=layer.gate;p.dst=b.g;p.w_rows=2*ff;p.ld_dst=2*ff;product();
            activate_packed<<<(size_t(routes)*ff+255)/256,256,0,cs>>>(b.g,b.expert,layer.scales,routes,ff,b.h);
        } else {
            p.w = layer.gate; p.dst = b.g; product();
            p.w = layer.up; p.dst = b.u; product();
            activate<<<(size_t(routes) * ff + 255) / 256,256,0,cs>>>(b.g,b.u,b.expert,layer.scales,routes,ff,b.h);
        }
        mmq::quantize(b.h, nullptr, b.hq, 40, ff, ff, routes, stream, b.hs);
        p.w = layer.down; p.w_rows = hidden; p.w_cols = ff; p.expert_bytes = layer.down_expert_bytes;
        p.xq = b.hq; p.dst = b.parts; p.ld_dst = hidden;
        p.y_scale = mmq::fp4_activations(40) ? b.hs : nullptr;
        product();
    }
    combine<<<(size_t(rows) * hidden + 255) / 256,256,0,cs>>>(b.parts,
        reuse||native_grouped||decode_fp4||(prefill && rows > 8) ? b.inverse : nullptr,ids,layer.scales,weights,rows,k,hidden,512,
        decode_fp4||(prefill && rows > 8),output);
    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
}
