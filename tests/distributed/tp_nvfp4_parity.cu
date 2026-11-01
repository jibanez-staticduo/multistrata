#include "strata/kernels/tp_nvfp4.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <memory>

namespace k = strata::kernels::tp::cuda;
namespace mmq = strata::prefill::mmq;
namespace {
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
};
__global__ void initialize(unsigned char* data, size_t bytes, int seed) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= bytes) return;
    const int in = i % 36;
    // Unit block scales. Codes depend on the matrix, expert, row and block.
    data[i] = in < 4 ? 56 : (unsigned char)((i * 13 + seed * 7 + i / 36) & 255);
}
float weight(int expert, int row, int col, int width, int seed) {
    const size_t offset = (size_t(expert) * 256 * 2560 / width + row) * (width / 64) * 36;
    const int sub = (col % 64) / 16, lane = col % 16;
    const size_t i = offset + (col / 64) * 36 + 4 + sub * 8 + lane % 8;
    const int byte = (i * 13 + seed * 7 + i / 36) & 255;
    const int code = lane < 8 ? byte & 15 : byte >> 4;
    const float values[8] = {0,.5,1,1.5,2,3,4,6};
    return (code & 8 ? -1.f : 1.f) * values[code & 7];
}
struct Fp4Mmq { uint8_t scales[16], codes[128]; };
__global__ void unpack_fp4(const Fp4Mmq* blocks, const float* scales, float* out, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r=i/cols,c=i%cols,sub=(c%256)/16,lane=c%16;
    const auto& b=blocks[(c/256)*rows+r];
    const int value=b.codes[sub*8+lane%8],q=lane<8?value&15:value>>4;
    const int values[8]={0,1,2,3,4,6,8,12};
    const int s=b.scales[sub],exp=(s>>3)&15,man=s&7;
    const float scale=(!s||s==127)?0.f:exp?ldexpf(1.f+float(man)/8,exp-8):float(man)/1024;
    out[i]=float(q&8?-values[q&7]:values[q&7])*scale*scales[r];
}
std::vector<float> fp4_round(const std::vector<float>& input,int rows,int cols,cudaStream_t stream) {
    Buffer in(input.size()*4),q(mmq::q8_bytes(rows,cols)),scales(rows*4),out(input.size()*4);
    check(cudaMemcpyAsync(in.p,input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));
    mmq::quantize((float*)in.p,nullptr,q.p,40,cols,cols,rows,stream,(float*)scales.p);
    unpack_fp4<<<(input.size()+255)/256,256,0,stream>>>((Fp4Mmq*)q.p,(float*)scales.p,(float*)out.p,rows,cols);
    std::vector<float> result(input.size());
    check(cudaMemcpyAsync(result.data(),out.p,result.size()*4,cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    return result;
}
}
int main(int argc, char**) try {
    if (cudaSetDevice(0) != cudaSuccess) return 77;
    constexpr int N = 2560, F = 256, E = 512, K = 10, R = 16;
    const size_t gb = size_t(F) * N / 64 * 36, db = size_t(N) * F / 64 * 36;
    Buffer gate(gb*E+4096), up(gb*E+4096), down(db*E+4096), scales(E*16), input(R*N*4), ids(R*K*4),
           weights(R*K*4), output(R*N*4), status(4);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for (const auto item : {std::pair{gate.p,1},std::pair{up.p,2},std::pair{down.p,3}})
        initialize<<<(gb*E+255)/256,256,0,stream>>>((unsigned char*)item.first,gb*E,item.second);
    for(auto* p:{gate.p,up.p,down.p})check(cudaMemsetAsync((char*)p+gb*E,0,4096,stream));
    std::vector<float> tails(E*4), x(R*N), route_weights(R*K);
    std::vector<int> routes(R*K);
    for (int e=0;e<E;++e) { tails[4*e]=.004f*(1+e%3); tails[4*e+1]=.003f; tails[4*e+2]=.002f; }
    for (int i=0;i<R*N;++i) x[i]=.2f*std::sin(float(i)*.17f);
    for (int i=0;i<R*K;++i) { routes[i]=i%K==0?511:(i*31+i/K)%500; route_weights[i]=float(1+i%K)/55.f; }
    check(cudaMemcpyAsync(scales.p,tails.data(),tails.size()*4,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(input.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(ids.p,routes.data(),routes.size()*4,cudaMemcpyHostToDevice,stream));
    check(cudaMemcpyAsync(weights.p,route_weights.data(),route_weights.size()*4,cudaMemcpyHostToDevice,stream));
    check(cudaMemsetAsync(status.p,0,4,stream));
    k::Nvfp4Layer layer{gate.p,up.p,down.p,(float*)scales.p,N,F,E,gb,db};
    std::unique_ptr<Buffer> packed_gu;
    std::unique_ptr<Buffer> decode_codes,decode_scales;
    if(std::getenv("STRATA_TP_DECODE_GU_VECTOR")) {
        const size_t subblocks=size_t(E)*F*(N/16);
        decode_codes=std::make_unique<Buffer>(subblocks*16);
        decode_scales=std::make_unique<Buffer>(subblocks*2);
        k::nvfp4_pack_decode_gu(layer,decode_codes->p,(uint16_t*)decode_scales->p,stream);
        layer.decode_gu_codes=decode_codes->p;layer.decode_gu_scales=(uint16_t*)decode_scales->p;
    }
    if(std::getenv("STRATA_TP_PREFILL_PACKED_GU")) {
        packed_gu=std::make_unique<Buffer>(2*gb*E+4096);
        check(cudaMemsetAsync((char*)packed_gu->p+2*gb*E,0,4096,stream));
        for(int e=0;e<E;++e) {
            check(cudaMemcpyAsync((char*)packed_gu->p+2*gb*e,(char*)gate.p+gb*e,gb,cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync((char*)packed_gu->p+2*gb*e+gb,(char*)up.p+gb*e,gb,cudaMemcpyDeviceToDevice,stream));
        }
        layer.gate=packed_gu->p;layer.up=(char*)packed_gu->p+gb;
        layer.gu_expert_bytes=2*gb;layer.packed_gate_up=true;
    }
    std::unique_ptr<Buffer> grouped;
    if(std::getenv("STRATA_TP_NATIVE_GROUPED")) {
        const size_t tail=gb*2+db,stride=tail+16;
        grouped=std::make_unique<Buffer>(stride*E+4096);
        check(cudaMemsetAsync((char*)grouped->p+stride*E,0,4096,stream));
        for(int e=0;e<E;++e) {
            auto* dst=(char*)grouped->p+size_t(e)*stride;
            check(cudaMemcpyAsync(dst,(char*)gate.p+size_t(e)*gb,gb,cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(dst+gb,(char*)up.p+size_t(e)*gb,gb,cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(dst+gb*2,(char*)down.p+size_t(e)*db,db,cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(dst+tail,(float*)scales.p+size_t(e)*4,16,cudaMemcpyDeviceToDevice,stream));
        }
        layer.gate=grouped->p;layer.up=(char*)grouped->p+gb;layer.down=(char*)grouped->p+gb*2;
        layer.gu_expert_bytes=layer.down_expert_bytes=stride;
        layer.grouped_blobs=grouped->p;layer.grouped_blob_bytes=stride;
        layer.grouped_up_offset=gb;layer.grouped_down_offset=gb*2;layer.grouped_tail_offset=tail;
    }
    Buffer workspace(k::nvfp4_workspace_bytes(R,K,N,F));
    const size_t bytes=k::nvfp4_workspace_bytes(R,K,N,F);
    if (mmq::fp4_activations(40)) {
        if (!mmq::nvfp4_prefill_tiles_available()) throw std::runtime_error("missing FP4 prefill tiles");
        for(int cols:{256,2560}) for(int tokens:{17,2048}) {
            const int total=tokens*K;
            std::vector<float> values(size_t(tokens)*cols);
            for(size_t i=0;i<values.size();++i)values[i]=.2f*std::sin(float(i)*.031f);
            std::vector<int> source(total),inverse(total);
            for(int visit=0;visit<total;++visit) {
                const int packed=total-1-visit;
                source[packed]=visit/K;inverse[visit]=packed;
            }
            const size_t size=mmq::q8_bytes(total,cols);
            Buffer a(values.size()*4),src(total*4),inv(total*4),ref(size),got(size),rs(total*4),gs(total*4);
            check(cudaMemcpyAsync(a.p,values.data(),values.size()*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(src.p,source.data(),total*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(inv.p,inverse.data(),total*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemsetAsync(ref.p,0,size,stream));check(cudaMemsetAsync(got.p,0,size,stream));
            auto verify=[&] {
                mmq::quantize((float*)a.p,(int*)src.p,ref.p,40,cols,cols,total,stream,(float*)rs.p);
                std::vector<unsigned char> expected(size),actual(size),scales_ref(total*4),scales_got(total*4);
                check(cudaMemcpyAsync(expected.data(),ref.p,size,cudaMemcpyDeviceToHost,stream));
                check(cudaMemcpyAsync(actual.data(),got.p,size,cudaMemcpyDeviceToHost,stream));
                check(cudaMemcpyAsync(scales_ref.data(),rs.p,total*4,cudaMemcpyDeviceToHost,stream));
                check(cudaMemcpyAsync(scales_got.data(),gs.p,total*4,cudaMemcpyDeviceToHost,stream));
                check(cudaStreamSynchronize(stream));
                if(expected!=actual||scales_ref!=scales_got)throw std::runtime_error("FP4 scatter differs from gather");
            };
            mmq::quantize_nvfp4_scatter((float*)a.p,(int*)inv.p,got.p,(float*)gs.p,cols,cols,tokens,K,stream);
            verify();
            cudaGraph_t graph;cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            mmq::quantize_nvfp4_scatter((float*)a.p,(int*)inv.p,got.p,(float*)gs.p,cols,cols,tokens,K,stream);
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            std::reverse(source.begin(),source.end());
            for(int visit=0;visit<total;++visit)inverse[visit]=visit;
            check(cudaMemcpyAsync(src.p,source.data(),total*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(inv.p,inverse.data(),total*4,cudaMemcpyHostToDevice,stream));
            check(cudaGraphLaunch(exec,stream));verify();
            check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
            std::printf("NVFP4 scatter K%d tokens%d bytes/scales/replay bitexact PASS\n",cols,tokens);
            if(cols==2560&&tokens==2048) {
                cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
                for(bool scatter:{false,true}) {
                    check(cudaEventRecord(begin,stream));
                    for(int repeat=0;repeat<20;++repeat) {
                        if(scatter)mmq::quantize_nvfp4_scatter((float*)a.p,(int*)inv.p,got.p,(float*)gs.p,cols,cols,tokens,K,stream);
                        else mmq::quantize((float*)a.p,(int*)src.p,ref.p,40,cols,cols,total,stream,(float*)rs.p);
                    }
                    check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                    float ms;check(cudaEventElapsedTime(&ms,begin,end));
                    std::printf("NVFP4 quantization tokens%d scatter%d mean_ms %.6f\n",tokens,int(scatter),ms/20);
                }
                check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            }
        }
        // Empty experts, full expert 511, multiple column tiles and a partial
        // last tile exercise the compact prefix lookup against native MMQ.
        for (int columns : {256,2560}) for (bool skew : {false,true}) {
            const int tokens=packed_gu?2048:257;const int total=tokens*K;
            const int outputs=columns==256?N:F;
            std::vector<int> bounds(E+1), identity(total);
            for(int i=0;i<total;++i)++bounds[1+(skew?(i%K==0?511:i%K-1):i%E)];
            for(int e=0;e<E;++e)bounds[e+1]+=bounds[e];
            for(int i=0;i<total;++i)identity[i]=i;
            std::vector<float> activations(size_t(total)*columns);
            for(size_t i=0;i<activations.size();++i)activations[i]=.2f*std::sin(float(i)*.031f);
            Buffer a(activations.size()*4),q(mmq::q8_bytes(total,columns)),ys(total*4),
                bd((E+1)*4),ix(total*4),tiles((E+1)*4),ref(size_t(total)*outputs*4),got(size_t(total)*outputs*4);
            check(cudaMemcpyAsync(a.p,activations.data(),activations.size()*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(bd.p,bounds.data(),bounds.size()*4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(ix.p,identity.data(),identity.size()*4,cudaMemcpyHostToDevice,stream));
            mmq::quantize((float*)a.p,nullptr,q.p,40,columns,columns,total,stream,(float*)ys.p);
            mmq::Product p;p.type=40;p.w=columns==256?down.p:gate.p;p.w_rows=outputs;p.w_cols=columns;
            p.expert_bytes=gb;p.n=E;p.xq=q.p;p.bounds=(int*)bd.p;p.ids=(int*)ix.p;
            p.total_rows=total;p.max_rows=tokens;p.ld_dst=outputs;p.y_scale=(float*)ys.p;
            mmq::Context context;p.dst=(float*)ref.p;context.run(p,stream);
            p.dst=(float*)got.p;mmq::nvfp4_prefill_tiles(p,(int*)tiles.p,stream);
            std::vector<float> expected(size_t(total)*outputs),actual(expected.size());
            check(cudaMemcpyAsync(expected.data(),ref.p,expected.size()*4,cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(actual.data(),got.p,actual.size()*4,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            if(std::memcmp(expected.data(),actual.data(),expected.size()*4))
                throw std::runtime_error("compact prefill differs from native MMQ");
            mmq::nvfp4_decode_tiles(p,(int*)tiles.p,stream);
            check(cudaMemcpyAsync(actual.data(),got.p,actual.size()*4,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            if(std::memcmp(expected.data(),actual.data(),expected.size()*4))
                throw std::runtime_error("J8 decode differs from native MMQ");
            std::printf("NVFP4 tiles K%d skew%d bitexact PASS\n",columns,int(skew));
            if(columns==2560&&packed_gu) {
                Buffer gu(size_t(total)*2*F*4),up_ref(size_t(total)*F*4);
                p.w=up.p;p.w_rows=F;p.ld_dst=F;p.expert_bytes=gb;p.dst=(float*)up_ref.p;
                mmq::nvfp4_prefill_tiles(p,(int*)tiles.p,stream);
                auto pair=[&] {
                    p.w=gate.p;p.w_rows=F;p.ld_dst=F;p.expert_bytes=gb;p.dst=(float*)ref.p;
                    mmq::nvfp4_prefill_tiles(p,(int*)tiles.p,stream);
                    p.w=up.p;p.dst=(float*)up_ref.p;mmq::nvfp4_prefill_tiles(p,(int*)tiles.p,stream);
                };
                auto fused=[&] {
                    p.w=packed_gu->p;p.w_rows=2*F;p.ld_dst=2*F;p.expert_bytes=2*gb;p.dst=(float*)gu.p;
                    mmq::nvfp4_prefill_tiles(p,(int*)tiles.p,stream);
                };
                pair();fused();
                std::vector<float> gate_ref(size_t(total)*F),up_values(gate_ref.size()),combined(2*gate_ref.size());
                check(cudaMemcpyAsync(gate_ref.data(),ref.p,gate_ref.size()*4,cudaMemcpyDeviceToHost,stream));
                check(cudaMemcpyAsync(up_values.data(),up_ref.p,up_values.size()*4,cudaMemcpyDeviceToHost,stream));
                check(cudaMemcpyAsync(combined.data(),gu.p,combined.size()*4,cudaMemcpyDeviceToHost,stream));
                check(cudaStreamSynchronize(stream));
                for(int row=0;row<total;++row)for(int col=0;col<F;++col)
                    if(combined[size_t(row)*2*F+col]!=gate_ref[size_t(row)*F+col]||
                       combined[size_t(row)*2*F+F+col]!=up_values[size_t(row)*F+col])
                        throw std::runtime_error("packed gate/up differs from separate products");
                cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
                for(bool packed:{false,true}) {
                    check(cudaEventRecord(begin,stream));
                    for(int repeat=0;repeat<20;++repeat){if(packed)fused();else pair();}
                    check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                    float ms;check(cudaEventElapsedTime(&ms,begin,end));
                    std::printf("NVFP4 gate/up tokens%d skew%d packed%d mean_ms %.6f\n",tokens,int(skew),int(packed),ms/20);
                }
                check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            }
        }
    }
    if (argc > 1) {
        for (int rows : {16}) for (bool prefill : {false,true}) {
            k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,rows,K,(float*)output.p,
                              workspace.p,bytes,(int*)status.p,stream,prefill);
            check(cudaStreamSynchronize(stream));
            cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,rows,K,(float*)output.p,
                              workspace.p,bytes,(int*)status.p,stream,prefill);
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
            for(int repeat=0;repeat<5;++repeat)check(cudaGraphLaunch(exec,stream));
            check(cudaStreamSynchronize(stream));
            cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
            check(cudaEventRecord(begin,stream));
            for(int repeat=0;repeat<40;++repeat)
                check(cudaGraphLaunch(exec,stream));
            check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            float ms=0;check(cudaEventElapsedTime(&ms,begin,end));
            std::printf("NVFP4 rows%d MMQ%d time_ms %.6f\n",rows,int(prefill),ms/40);
            check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
            check(cudaGraphExecDestroy(exec));
        }
    }
    double worst=0;
    for (int mode : {0,1,2}) {
        const bool prefill=mode==2;
        const int active=mode==1?8:R;
        k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,active,K,(float*)output.p,
                            workspace.p,bytes,(int*)status.p,stream,prefill);
        std::vector<float> got(R*N);
        check(cudaMemcpyAsync(got.data(),output.p,got.size()*4,cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        double numerator=0,denominator=0;
        const bool fp4=(prefill || (active>8 && std::getenv("STRATA_TP_DECODE_FP4"))) && mmq::fp4_activations(40);
        const auto rounded=fp4?fp4_round(x,R,N,stream):x;
        // Two independent rows exercise per-row routing, tails and SwiGLU.
        for (int r : {0,active-1}) {
            std::vector<double> ref(N,0), h(F);
            for(int j=0;j<K;++j) {
                const int e=routes[r*K+j];
                for(int f=0;f<F;++f) {
                    double g=0,u=0;
                    for(int c=0;c<N;++c) { g+=weight(e,f,c,N,1)*rounded[r*N+c]; u+=weight(e,f,c,N,2)*rounded[r*N+c]; }
                    g*=tails[4*e]; u*=tails[4*e+1]; h[f]=g/(1+std::exp(-g))*u;
                }
                if(fp4) {
                    std::vector<float> hf(h.begin(),h.end());
                    hf=fp4_round(hf,1,F,stream);
                    h.assign(hf.begin(),hf.end());
                }
                for(int c=0;c<N;++c) {
                    double v=0;
                    for(int f=0;f<F;++f) v+=weight(e,c,f,F,3)*h[f];
                    ref[c]+=v*tails[4*e+2]*route_weights[r*K+j];
                }
            }
            for(int c=0;c<N;++c) { const double d=got[r*N+c]-ref[c]; numerator+=d*d; denominator+=ref[c]*ref[c]; }
        }
        const double relative=std::sqrt(numerator/std::max(denominator,1e-30));
        worst=std::max(worst,relative);
        std::printf("NVFP4 %s rows%d relative L2 %.8f\n",prefill?"prefill":"decode",active,relative);
        if (!std::isfinite(relative) || relative>.02) return 1;
    }
    if(std::getenv("STRATA_TP_ROUTE_ORDER")||std::getenv("STRATA_TP_GROUP_REUSE")) {
        k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,R,K,(float*)output.p,
            workspace.p,bytes,(int*)status.p,stream,false);
        check(cudaStreamSynchronize(stream));
        cudaGraph_t graph;cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,R,K,(float*)output.p,
            workspace.p,bytes,(int*)status.p,stream,false);
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        auto changed=routes;
        for(auto& id:changed)id=(id+137)%512;
        check(cudaMemcpyAsync(ids.p,changed.data(),changed.size()*4,cudaMemcpyHostToDevice,stream));
        k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,R,K,(float*)output.p,
            workspace.p,bytes,(int*)status.p,stream,false);
        std::vector<float> expected(size_t(R)*N),actual(expected.size());
        check(cudaMemcpyAsync(expected.data(),output.p,expected.size()*4,cudaMemcpyDeviceToHost,stream));
        check(cudaGraphLaunch(exec,stream));
        check(cudaMemcpyAsync(actual.data(),output.p,actual.size()*4,cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        if(expected!=actual)throw std::runtime_error("ordered route graph retained stale IDs");
        for(int invalid:{-1,512}) {
            check(cudaMemsetAsync(status.p,0,4,stream));
            check(cudaMemcpyAsync(ids.p,&invalid,4,cudaMemcpyHostToDevice,stream));
            check(cudaGraphLaunch(exec,stream));int error=0;
            check(cudaMemcpyAsync(&error,status.p,4,cudaMemcpyDeviceToHost,stream));
            check(cudaStreamSynchronize(stream));
            if(error!=1)throw std::runtime_error("ordered route graph missed invalid ID");
        }
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
        std::printf("NVFP4 route order changed-ID replay and invalid IDs PASS\n");
    }
    int bad=-1;
    check(cudaMemcpyAsync(ids.p,&bad,4,cudaMemcpyHostToDevice,stream));
    k::nvfp4_local_sum(layer,(float*)input.p,(int*)ids.p,(float*)weights.p,R,K,(float*)output.p,
                        workspace.p,bytes,(int*)status.p,stream,false);
    check(cudaMemcpyAsync(&bad,status.p,4,cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    if(bad!=1)return 2;
    check(cudaStreamDestroy(stream));
    std::printf("NVFP4 parity passed, worst %.8f; invalid route detected\n",worst);
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 2; }
