#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/window_limits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <climits>
#include <stdexcept>
namespace strata::kernels::tp::cuda {
namespace {
void check() { const auto e=cudaGetLastError(); if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
constexpr int kHcMaxRows =
#if defined(STRATA_TP_WIDE_WINDOW)
    40;
#else
    16;
#endif
__global__ void steps_kernel(core::TensorParallelRows spec,int rows,int expected,int context,
                               int* steps,int* positions,int heads,int* status) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows)return;
    const int p=spec.positions?spec.positions[r*spec.position_stride]:int(spec.position+r);
    if(p<0||p>=context||(spec.slots&&spec.slots[r]!=expected))atomicExch(status,1);
    steps[4*r]=p;steps[4*r+1]=p+1;steps[4*r+2]=(p+1)/4;steps[4*r+3]=min(p+1,2051);
    for(int h=0;h<heads;++h)positions[r*heads+h]=p;
}
__global__ void gates_kernel(const float* alpha,const float* raw,const float* dt,const float* a,
                               float* gate,float* beta,int count,int heads) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const float x=alpha[i]+dt[i%heads];
    gate[i]=(x>20.f?x:log1pf(expf(x)))*a[i%heads];beta[i]=1.f/(1.f+expf(-raw[i]));
}
struct MixerSlots { int ids[kHcMaxRows]; };
__global__ void steps_batch_kernel(core::TensorParallelRows spec,MixerSlots slots,int rows,int context,
                                    int* steps,int* positions,int* status) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows)return;
    const int p=spec.positions?spec.positions[r*spec.position_stride]:int(spec.position+r);
    if(p<0||p>=context||(spec.slots&&spec.slots[r]!=slots.ids[r]))atomicExch(status,1);
    steps[4*r]=p;steps[4*r+1]=p+1;steps[4*r+2]=(p+1)/4;steps[4*r+3]=min(p+1,2051);
    for(int h=0;h<8;++h)positions[r*8+h]=p;
}
__global__ void rope_positions_kernel(const int* steps,int rows,int* keys,int* index) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*2)keys[i]=steps[(i/2)*4];
    if(i<rows*4)index[i]=steps[(i/4)*4];
}
__global__ void conv_kernel(const float* history,const float* x,const float* weights,float* out,
                              int rows,int channels,int key_heads) {
    const int head=blockIdx.x,t=blockIdx.y,c=head*128+threadIdx.x;
    if(t>=rows||c>=channels)return;
    float v=0;
    for(int tap=0;tap<4;++tap) {
        const int p=t+tap-3;
        v+=weights[c*4+tap]*(p<0?history[c*3+p+3]:x[size_t(p)*channels+c]);
    }
    v=v/(1.f+expf(-v));
    if(head<2*key_heads) {
        __shared__ float sums[4];
        float sum=v*v;
        for(int o=16;o;o/=2)sum+=__shfl_xor_sync(0xffffffff,sum,o);
        if((threadIdx.x&31)==0)sums[threadIdx.x/32]=sum;
        __syncthreads();
        const float total=sums[0]+sums[1]+sums[2]+sums[3];
        v*=rsqrtf(total+1.e-6f);
    }
    out[size_t(t)*channels+c]=v;
}
__global__ void commit_conv_kernel(float* history,const float* x,int rows,int channels) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=channels)return;
    float tail[3];
    for(int j=0;j<3;++j) { const int p=rows-3+j;tail[j]=p<0?history[c*3+p+3]:x[size_t(p)*channels+c]; }
    for(int j=0;j<3;++j)history[c*3+j]=tail[j];
}
__global__ void recurrence_kernel(float* state,const float* h,const float* gate,const float* beta,
                                   const float* z,const float* norm,float* y,int rows,int kh,int vh) {
    const int head=blockIdx.x,col=threadIdx.x,key=head%kh,channels=(2*kh+vh)*128;
    float s[128];
    for(int r=0;r<128;++r)s[r]=state[(size_t(r)*vh+head)*128+col];
    __shared__ float q[128],k[128],sums[4];
    for(int t=0;t<rows;++t) {
        const float* input=h+size_t(t)*channels;
        q[col]=input[key*128+col];k[col]=input[(kh+key)*128+col];
        __syncthreads();
        const float decay=expf(gate[t*vh+head]);
        float sk=0;
        for(int r=0;r<128;++r){s[r]*=decay;sk+=s[r]*k[r];}
        const float delta=(input[(2*kh+head)*128+col]-sk)*beta[t*vh+head];
        float out=0;
        for(int r=0;r<128;++r){s[r]+=k[r]*delta;out+=s[r]*q[r];}
        out*=0.08838834764831845f;
        float sum=out*out;
        for(int o=16;o;o/=2)sum+=__shfl_xor_sync(0xffffffff,sum,o);
        if((col&31)==0)sums[col/32]=sum;
        __syncthreads();
        const float scale=rsqrtf((sums[0]+sums[1]+sums[2]+sums[3])/128.f+1.e-6f);
        y[(size_t(t)*vh+head)*128+col]=out*scale*norm[col]/(1.f+expf(-z[(size_t(t)*vh+head)*128+col]));
        __syncthreads();
    }
    for(int r=0;r<128;++r)state[(size_t(r)*vh+head)*128+col]=s[r];
}
__global__ void shared_swiglu_kernel(const float* gate,const float* up,uint16_t* hidden,int64_t elements) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=elements)return;
    const float g=gate[i];
    const float value=__fdividef(g,1.0f+__expf(-g))*up[i];
    const float saturated=isnan(value)?value:fminf(fmaxf(value,-65504.0f),65504.0f);
    hidden[i]=f16_from_f32(saturated);
}
__global__ void shared_add_kernel(float* local,const float* shared,const float* gate,int count,
                                  int width,bool apply_sigmoid) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int total=count*width;
    if(i>=total)return;
    const float scale=apply_sigmoid?1.0f/(1.0f+__expf(-gate[i/width])):1.0f;
    local[i]+=shared[i]*scale;
}
struct HcRows { FusedGrArgs args[16]; int count; };
__global__ void hc_pack_kernel(HcRows rows,float* normalized,uint16_t* packed) {
    const int row=blockIdx.x,c=blockIdx.y,tid=threadIdx.x;
    const auto a=rows.args[row];
    const float scale=a.apply?2.f/(1.f+expf(-a.inj_prev[c]/4.f)):0.f;
    float values[10],sum=0;int at=0;
    for(int d=tid;d<2560;d+=256,++at) {
        float x=a.R[c*2560+d];
        if(a.apply)x=fmaf(a.bo_prev[d],scale,x);
        values[at]=x;sum+=x*x;
    }
    __shared__ float sums[8];
    for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if((tid&31)==0)sums[tid/32]=sum;
    __syncthreads();
    if(tid<32) {
        sum=tid<8?sums[tid]:0;
        for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(!tid)sums[0]=rsqrtf(sum/2560.f+a.eps);
    }
    __syncthreads();const float rs=sums[0];
    if(!tid)a.rs[c]=rs;
    at=0;
    for(int d=tid;d<2560;d+=256,++at) {
        const float x=values[at];
        if(a.apply)a.R_out[c*2560+d]=x;
        const float v=x*rs*a.w_norm[c*2560+d];
        const size_t i=size_t(row)*10240+c*2560+d;
        normalized[i]=v;
        packed[i]=bf16_from_f32(v);
    }
}
__global__ void hc_silu_pack_kernel(float* lo,uint16_t* packed,int elements) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=elements)return;
    const float x=lo[i]*.25f,v=x/(1.f+expf(-x));lo[i]=v;
    packed[i]=bf16_from_f32(v);
}
__global__ void hc_inject_kernel(HcRows rows,const float* inject) {
    const int i=threadIdx.x;if(i<rows.count*4)rows.args[i/4].inject_out[i%4]=inject[i];
}
__global__ void hc_packed_silu_kernel(HcRows rows,const float* combined,uint16_t* packed) {
    const int row=blockIdx.x,i=threadIdx.x;
    const float* source=combined+size_t(row)*336;
    for(int col=i;col<320;col+=blockDim.x) {
        const float x=source[col]*.25f,v=x/(1.f+expf(-x));
        packed[size_t(row)*320+col]=bf16_from_f32(v);
    }
    if(i<4)rows.args[row].inject_out[i]=source[320+i];
}
__global__ void transport_pack_kernel(const float* input,uint16_t* output,int64_t count) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    output[i]=bf16_from_f32(input[i]);
}
__global__ void transport_unpack_kernel(const uint16_t* input,float* output,int64_t count) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)output[i]=__uint_as_float(unsigned(input[i])<<16);
}
__global__ void head_local_kernel(const float* logits,float* candidates,int width,int first,int rank) {
    const int row=blockIdx.x,tid=threadIdx.x;
    float best=__int_as_float(0xff800000);int id=INT_MAX,bad=0;
    for(int i=tid;i<width;i+=blockDim.x) {
        const float value=logits[size_t(row)*width+i];bad|=!isfinite(value);
        if(value>best||(value==best&&first+i<id)){best=value;id=first+i;}
    }
    __shared__ float values[256];__shared__ int ids[256],errors[256];
    values[tid]=best;ids[tid]=id;errors[tid]=bad;__syncthreads();
    for(int offset=128;offset;offset/=2) {
        if(tid<offset) {
            const float value=values[tid+offset];const int other=ids[tid+offset];
            if(value>values[tid]||(value==values[tid]&&other<ids[tid])){values[tid]=value;ids[tid]=other;}
            errors[tid]|=errors[tid+offset];
        }
        __syncthreads();
    }
    if(!tid){float* out=candidates+size_t(row)*12+rank*4;out[0]=values[0];out[1]=float(ids[0]);out[2]=float(errors[0]);out[3]=0;}
}
__global__ void head_global_kernel(const float* candidates,int32_t* picks,uint32_t* flags,int rows,int blocks) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
    const float* x=candidates+size_t(row)*12;float best=__int_as_float(0xff800000);int id=INT_MAX;bool bad=false;
    for(int rank=0;rank<3;++rank){const float value=x[rank*4];const int other=int(x[rank*4+1]);
        bad|=x[rank*4+2]!=0;if(value>best||(value==best&&other<id)){best=value;id=other;}}
    picks[row]=id;if(flags)for(int b=0;b<blocks;++b)flags[row*blocks+b]=bad?1u:0u;
}
}
void mixer_steps(const core::TensorParallelRows& spec,int rows,int expected,int context,
                  int* steps,int* positions,int heads,int* status,void* stream) {
    auto device=spec;device.capture_slots=nullptr;
    steps_kernel<<<(rows+127)/128,128,0,(cudaStream_t)stream>>>(device,rows,expected,context,steps,positions,heads,status);check();
}
void mixer_gates(const float* alpha,const float* raw,const float* dt,const float* a,
                  float* gate,float* beta,int rows,int heads,void* stream) {
    gates_kernel<<<(rows*heads+127)/128,128,0,(cudaStream_t)stream>>>(alpha,raw,dt,a,gate,beta,rows*heads,heads);check();
}
void mixer_rope_positions(const int* steps,int rows,int* keys,int* index,void* stream) {
    rope_positions_kernel<<<(rows*4+127)/128,128,0,(cudaStream_t)stream>>>(steps,rows,keys,index);check();
}
void mixer_steps_batch(const core::TensorParallelRows& spec,const int* slots,int rows,int context,
                       int* steps,int* positions,int* status,void* stream) {
    if(!slots||rows<1||rows>kHcMaxRows||context<1||!steps||!positions||!status||!stream)
        throw std::invalid_argument("invalid batched mixer metadata");
    MixerSlots capture{};
    for(int i=0;i<rows;++i)capture.ids[i]=slots[i];
    auto device=spec;device.capture_slots=nullptr;device.segments=nullptr;device.segment_count=0;
    steps_batch_kernel<<<1,128,0,(cudaStream_t)stream>>>(device,capture,rows,context,steps,positions,status);check();
}
void mixer_conv(float* history,const float* qkv,const float* weights,float* h,
                 int rows,int channels,int key_heads,void* stream) {
    conv_kernel<<<dim3(channels/128,rows),128,0,(cudaStream_t)stream>>>(history,qkv,weights,h,rows,channels,key_heads);check();
}
void mixer_conv_commit(float* history,const float* qkv,int rows,int channels,void* stream) {
    commit_conv_kernel<<<(channels+127)/128,128,0,(cudaStream_t)stream>>>(history,qkv,rows,channels);check();
}
void mixer_recurrence(float* state,const float* h,const float* gate,const float* beta,
                       const float* z,const float* norm,float* y,int rows,int kh,int vh,void* stream) {
    recurrence_kernel<<<vh,128,0,(cudaStream_t)stream>>>(state,h,gate,beta,z,norm,y,rows,kh,vh);check();
}
void shared_swiglu_f16(const float* gate,const float* up,uint16_t* hidden,int64_t elements,void* stream) {
    if(!gate||!up||!hidden||elements<=0||!stream)throw std::invalid_argument("invalid TP shared SwiGLU");
    shared_swiglu_kernel<<<unsigned((elements+255)/256),256,0,(cudaStream_t)stream>>>(gate,up,hidden,elements);check();
}
void shared_add(float* local_sum,const float* shared,const float* gate_logits,int rows,int width,
                bool apply_sigmoid,void* stream) {
    if(!local_sum||!shared||rows<=0||width<=0||(apply_sigmoid&&!gate_logits)||!stream)
        throw std::invalid_argument("invalid TP shared add");
    shared_add_kernel<<<unsigned((int64_t(rows)*width+255)/256),256,0,(cudaStream_t)stream>>>(
        local_sum,shared,gate_logits,rows,width,apply_sigmoid);check();
}
void hc_pack(const FusedGrArgs* args,int rows,float* normalized,uint16_t* packed,void* stream) {
    if(rows<1||rows>kHcMaxRows)throw std::invalid_argument("invalid fused HC rows");
    for(int first=0;first<rows;first+=16) {
        HcRows r{};r.count=std::min(16,rows-first);
        for(int i=0;i<r.count;++i)r.args[i]=args[first+i];
        hc_pack_kernel<<<dim3(r.count,4),256,0,(cudaStream_t)stream>>>(
            r,normalized+size_t(first)*10240,packed+size_t(first)*10240);check();
    }
}
void hc_silu_pack(float* lo,uint16_t* packed,int rows,void* stream) {
    hc_silu_pack_kernel<<<(rows*320+255)/256,256,0,(cudaStream_t)stream>>>(lo,packed,rows*320);check();
}
void hc_publish_inject(const FusedGrArgs* args,const float* inject,int rows,void* stream) {
    if(rows<1||rows>kHcMaxRows)throw std::invalid_argument("invalid fused HC rows");
    for(int first=0;first<rows;first+=16) {
        HcRows r{};r.count=std::min(16,rows-first);
        for(int i=0;i<r.count;++i)r.args[i]=args[first+i];
        hc_inject_kernel<<<1,64,0,(cudaStream_t)stream>>>(r,inject+size_t(first)*4);check();
    }
}
void hc_packed_silu_publish(const FusedGrArgs* args,const float* combined,uint16_t* packed,int rows,void* stream) {
    if(!args||!combined||!packed||rows<1||rows>kHcMaxRows||!stream)
        throw std::invalid_argument("invalid packed HC rows");
    for(int first=0;first<rows;first+=16) {
        HcRows r{};r.count=std::min(16,rows-first);
        for(int i=0;i<r.count;++i)r.args[i]=args[first+i];
        hc_packed_silu_kernel<<<r.count,256,0,(cudaStream_t)stream>>>(
            r,combined+size_t(first)*336,packed+size_t(first)*320);check();
    }
}
void transport_bf16_pack(const float* input,uint16_t* output,int64_t count,void* stream) {
    transport_pack_kernel<<<unsigned((count+255)/256),256,0,(cudaStream_t)stream>>>(input,output,count);check();
}
void transport_bf16_unpack(const uint16_t* input,float* output,int64_t count,void* stream) {
    transport_unpack_kernel<<<unsigned((count+255)/256),256,0,(cudaStream_t)stream>>>(input,output,count);check();
}
__global__ void ring_sum_kernel(const float* values,float* output,uint64_t count,
                                uint64_t first,uint64_t total) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    // NCCL SIMPLE assigns whole 16 KiB cells; only the final channel is short.
    const uint64_t channel_span=((total+4095)/4096+7)/8*4096;
    const uint64_t channel=(first+i)/channel_span;
    const uint64_t channel_count=min(channel_span,total-channel*channel_span);
    const uint64_t position=first+i-channel*channel_span;
    const uint64_t chunk=min(uint64_t(524288),((channel_count+2)/3+3)/4*4);
    const uint64_t loop_first=position/(3*chunk)*(3*chunk);
    const uint64_t remaining=channel_count-loop_first;
    const uint64_t live_chunk=remaining<3*chunk?((remaining+2)/3+3)/4*4:chunk;
    const uint64_t part=(position-loop_first)/live_chunk;
    const float a=values[i],b=values[count+i],c=values[2*count+i];
    output[i]=part==0?__fadd_rn(__fadd_rn(b,c),a):
              part==1?__fadd_rn(__fadd_rn(a,c),b):__fadd_rn(__fadd_rn(a,b),c);
}
void transport_ring_sum(const float* gathered,float* output,uint64_t count,
                         uint64_t first,uint64_t total,void* stream) {
    if(!gathered||!output||!count||total%8||!total||first>total||count>total-first)
        throw std::invalid_argument("invalid ring-ordered RoCE sum");
    ring_sum_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>(gathered,output,count,first,total);
    check();
}
void head_local_candidate(const float* logits,float* candidates,int rows,int width,int first,int rank,void* stream) {
    head_local_kernel<<<rows,256,0,(cudaStream_t)stream>>>(logits,candidates,width,first,rank);check();
}
void head_global_candidate(const float* candidates,int32_t* picks,uint32_t* flags,int rows,int blocks,void* stream) {
    head_global_kernel<<<(rows+127)/128,128,0,(cudaStream_t)stream>>>(candidates,picks,flags,rows,blocks);check();
}
}
