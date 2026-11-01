#include "strata/kernels/fast_gdn.hpp"
#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
namespace k=strata::kernels;
namespace gpu=k::tp::cuda;
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct B{void* p=nullptr;explicit B(size_t bytes){ck(cudaMalloc(&p,bytes));}~B(){cudaFree(p);}float* f(size_t off=0){return (float*)p+off;}};
}
int main()try {
    if(cudaSetDevice(0)!=cudaSuccess)return 77;
    cudaStream_t cs;ck(cudaStreamCreateWithFlags(&cs,cudaStreamNonBlocking));
    for(int keys:{5,6}) for(int rows:{97,8191}) {
        const int heads=keys*3,channels=keys*640,width=heads*128;
        const size_t state_n=size_t(128)*heads*128;
        B state(state_n*4),oracle(state_n*4),h(rows*channels*4),gate(rows*heads*4),beta(rows*heads*4),z(rows*width*4),norm(128*4),out(rows*width*4),ref(rows*width*4),keep(4);
        B workspace(gpu::fast_gdn_workspace_bytes(keys)+rows*width*4);
        std::mt19937 rng(83+keys);std::normal_distribution<float>d(0,.1);
        std::vector<float> hh(rows*channels),gg(rows*heads),bb(rows*heads),zz(rows*width),nn(128,1),ss(state_n);
        for(float& v:hh)v=d(rng);for(float& v:zz)v=d(rng);for(float& v:ss)v=d(rng);
        for(int t=0;t<rows;++t)for(int head=0;head<2*keys;++head){double sum=0;for(int j=0;j<128;++j)sum+=hh[t*channels+head*128+j]*hh[t*channels+head*128+j];for(int j=0;j<128;++j)hh[t*channels+head*128+j]/=std::sqrt(sum+1.e-6);}
        for(size_t i=0;i<gg.size();++i){gg[i]=-.02f-.01f*std::abs(d(rng));bb[i]=.5f+.1f*d(rng);}
        auto upload=[&](B& b,const std::vector<float>& v){ck(cudaMemcpyAsync(b.p,v.data(),v.size()*4,cudaMemcpyHostToDevice,cs));};
        upload(h,hh);upload(gate,gg);upload(beta,bb);upload(z,zz);upload(norm,nn);upload(state,ss);upload(oracle,ss);
        int one=1;ck(cudaMemcpyAsync(keep.p,&one,4,cudaMemcpyHostToDevice,cs));
        for(int t=0;t<rows;++t)k::gdn_step_norm_multi(oracle.f(),h.f(t*channels),channels,gate.f(t*heads),beta.f(t*heads),z.f(t*width),norm.f(),1.e-6f,ref.f(t*width),keys,heads,1,(int*)keep.p,cs);
        gpu::fast_gdn_prompt(state.f(),h.f(),gate.f(),beta.f(),z.f(),norm.f(),out.f(),rows,keys,workspace.p,gpu::fast_gdn_workspace_bytes(keys)+rows*width*4,cs);
        std::vector<float>a(rows*width),b(rows*width),sa(state_n),sb(state_n);
        ck(cudaMemcpyAsync(a.data(),out.p,a.size()*4,cudaMemcpyDeviceToHost,cs));ck(cudaMemcpyAsync(b.data(),ref.p,b.size()*4,cudaMemcpyDeviceToHost,cs));
        ck(cudaMemcpyAsync(sa.data(),state.p,sa.size()*4,cudaMemcpyDeviceToHost,cs));ck(cudaMemcpyAsync(sb.data(),oracle.p,sb.size()*4,cudaMemcpyDeviceToHost,cs));ck(cudaStreamSynchronize(cs));
        auto relative=[](const auto& a,const auto& b){double n=0,d=0;for(size_t i=0;i<a.size();++i){double x=a[i]-b[i];n+=x*x;d+=double(b[i])*b[i];}return std::sqrt(n/d);};
        const double e=relative(a,b),s=relative(sa,sb);std::printf("compact GDN keys%d rows%d output L2 %.8g state %.8g\n",keys,rows,e,s);
        if(!std::isfinite(e)||!std::isfinite(s)||e>.0001||s>.0001)return 1;
    }
    ck(cudaStreamDestroy(cs));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
