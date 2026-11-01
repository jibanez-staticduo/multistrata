#include "strata/kernels/fast_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

// Instantiate the existing FP16x3 WY scan with compact head geometry. One
// staging buffer fits GB10's 99 KiB opt-in limit; the equations stay the same.
#define STRATA_GDN_TC_STAGES 1
namespace strata::kernels::tp::cuda {
namespace {
template<int Keys>struct Instance {
    static size_t bytes();
    static void run(float*,const float*,const float*,const float*,float*,int,void*,size_t,cudaStream_t);
};
}
}

namespace strata::kernels::tp::cuda::gdn5 {
constexpr int S=128,HK=5,HV=HK*3,C=HK*5*128,VPK=3,GCH=32,GSB=2048,GTP=33;
__device__ __forceinline__ void gdn_cp4(float* dst,const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));
}
__device__ __forceinline__ void gdn_cp16(float* dst,const float* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));
}
__device__ __forceinline__ void gdn_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void gdn_cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
#include "../../../prefill/gdn_tc.cuh"
size_t bytes() {return kTcTiBytes+kTcTeBytes+kTcScBytes+kTcKqBytes+kTcPiBytes;}
void run(float* state,const float* h,const float* gate,const float* beta,float* out,int rows,
         void* scratch,size_t size,cudaStream_t stream) {
    if(size<bytes())throw std::runtime_error("compact GDN TC workspace too small");
    auto* b=static_cast<unsigned char*>(scratch);
    GtcScratch sc{(uint16_t*)b,(float*)(b+kTcTiBytes),(float4*)(b+kTcTiBytes+kTcTeBytes),
        (uint16_t*)(b+kTcTiBytes+kTcTeBytes+kTcScBytes),(uint16_t*)(b+kTcTiBytes+kTcTeBytes+kTcScBytes+kTcKqBytes)};
    static bool configured=false;
    if(!configured) {
        if(cudaFuncSetAttribute(gdn_tc_scan_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,
             int(kTcScanSmem<false>))!=cudaSuccess)throw std::runtime_error("GB10 compact GDN TC shared memory unavailable");
        configured=true;
    }
    for(int first=0;first<rows;first+=GSB) {
        const int chunks=(std::min(GSB,rows-first)+GCH-1)/GCH;
        gdn_tc_prep_kernel<<<dim3(chunks,HK),256,kTcPrepSmem,stream>>>(h,gate,beta,sc.ti,sc.te,sc.sc,sc.kq,sc.pi,first,rows);
        gdn_tc_scan_kernel<false><<<HV,256,kTcScanSmem<false>,stream>>>(state,h,sc.ti,sc.te,sc.sc,sc.kq,sc.pi,out,first,chunks,rows);
    }
}
}

namespace strata::kernels::tp::cuda::gdn6 {
constexpr int S=128,HK=6,HV=HK*3,C=HK*5*128,VPK=3,GCH=32,GSB=2048,GTP=33;
__device__ __forceinline__ void gdn_cp4(float* dst,const float* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));
}
__device__ __forceinline__ void gdn_cp16(float* dst,const float* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));
}
__device__ __forceinline__ void gdn_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void gdn_cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
#include "../../../prefill/gdn_tc.cuh"
size_t bytes() {return kTcTiBytes+kTcTeBytes+kTcScBytes+kTcKqBytes+kTcPiBytes;}
void run(float* state,const float* h,const float* gate,const float* beta,float* out,int rows,
         void* scratch,size_t size,cudaStream_t stream) {
    if(size<bytes())throw std::runtime_error("compact GDN TC workspace too small");
    auto* b=static_cast<unsigned char*>(scratch);
    GtcScratch sc{(uint16_t*)b,(float*)(b+kTcTiBytes),(float4*)(b+kTcTiBytes+kTcTeBytes),
        (uint16_t*)(b+kTcTiBytes+kTcTeBytes+kTcScBytes),(uint16_t*)(b+kTcTiBytes+kTcTeBytes+kTcScBytes+kTcKqBytes)};
    static bool configured=false;
    if(!configured) {
        if(cudaFuncSetAttribute(gdn_tc_scan_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,
             int(kTcScanSmem<false>))!=cudaSuccess)throw std::runtime_error("GB10 compact GDN TC shared memory unavailable");
        configured=true;
    }
    for(int first=0;first<rows;first+=GSB) {
        const int chunks=(std::min(GSB,rows-first)+GCH-1)/GCH;
        gdn_tc_prep_kernel<<<dim3(chunks,HK),256,kTcPrepSmem,stream>>>(h,gate,beta,sc.ti,sc.te,sc.sc,sc.kq,sc.pi,first,rows);
        gdn_tc_scan_kernel<false><<<HV,256,kTcScanSmem<false>,stream>>>(state,h,sc.ti,sc.te,sc.sc,sc.kq,sc.pi,out,first,chunks,rows);
    }
}
}

namespace strata::kernels::tp::cuda {
size_t fast_gdn_workspace_bytes(int keys) {
    if(keys==5)return gdn5::bytes();
    if(keys==6)return gdn6::bytes();
    throw std::invalid_argument("compact GDN supports five or six key heads");
}
void fast_gdn_prompt(float* state,const float* h,const float* gate,const float* beta,
                       const float* z,const float* norm,float* output,int rows,int keys,
                       void* scratch,size_t bytes,void* stream) {
    const size_t scan_bytes=fast_gdn_workspace_bytes(keys);
    auto* raw=reinterpret_cast<float*>(static_cast<unsigned char*>(scratch)+scan_bytes);
    if(keys==5)gdn5::run(state,h,gate,beta,raw,rows,scratch,bytes,(cudaStream_t)stream);
    else if(keys==6)gdn6::run(state,h,gate,beta,raw,rows,scratch,bytes,(cudaStream_t)stream);
    else throw std::invalid_argument("compact GDN supports five or six key heads");
    const int heads=rows*keys*3;
    // The native norm accepts at most 65535 independent head rows per call.
    for(int first=0;first<heads;first+=65535) {
        const size_t offset=size_t(first)*128;
        native_gdn_out_norm(raw+offset,z+offset,norm,output+offset,
                            std::min(65535,heads-first),128,1.e-6f,stream);
    }
    auto result=cudaGetLastError();if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
}
}
