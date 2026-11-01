#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void* p=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n){ck(cudaMalloc(&p,n));ck(cudaMemset(p,0,n));}
    ~Buffer(){cudaFree(p);}
    template<class T>T* as(){return static_cast<T*>(p);}
    template<class T>void upload(const std::vector<T>& v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
void equal(Buffer& a,Buffer& b,const char* message) {
    std::vector<unsigned char>x(a.bytes),y(b.bytes);
    ck(cudaMemcpy(x.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(y.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));
    if(x!=y)throw std::runtime_error(message);
}
constexpr int cells=64,heads=2,width=256,idx=128,snapshot=513;
struct State {
    Buffer key{cells*heads*width},value{cells*heads*width};
    Buffer ks{cells*heads*4*2},vs{cells*heads*4*2},table{16*4};
    Buffer tail{3*idx*4},dead{idx*4},pooled{(cells/4+1)*idx*4},pos{4};
    Buffer history{5*snapshot*4},raw{5*idx*4};
    State(){std::vector<int> t(16);for(int i=0;i<16;++i)t[i]=(i*5+3)%16;table.upload(t);}
    k::QsaIndexerBuffers indexer(){return {tail.as<float>(),dead.as<float>(),pooled.as<float>(),pos.as<int>()};}
};
void run(cudaStream_t stream,int count) {
    constexpr int groups=8;
    const int rows=groups*count;
    auto shapes=k::qsa_real_shapes();
    std::vector<std::unique_ptr<State>> a,b;
    k::KvQ8AppendGroups kv{};k::QsaIndexerGroups ix{};
    kv.n=ix.n=groups;
    for(int i=0;i<groups;++i) {
        a.push_back(std::make_unique<State>());b.push_back(std::make_unique<State>());
        auto& s=*a.back();kv.k[i]=s.key.as<int8_t>();kv.v[i]=s.value.as<int8_t>();
        kv.k_scale[i]=s.ks.as<uint16_t>();kv.v_scale[i]=s.vs.as<uint16_t>();kv.page_table[i]=s.table.as<int>();
        kv.first[i]=ix.first[i]=i*count;kv.count[i]=ix.count[i]=count;
        ix.buffers[i]=s.indexer();ix.history_prefix[i]=s.history.as<float>();
        ix.raw_prefix[i]=s.raw.as<float>();ix.snapshot[i]=count>1;
    }
    std::vector<float> keys(rows*512),values(keys.size()),raw(rows*idx),gamma(idx);
    std::vector<int> steps(rows*4);
    for(size_t i=0;i<keys.size();++i){keys[i]=std::sin(float(i)*.03f);values[i]=std::cos(float(i)*.07f);}
    for(size_t i=0;i<raw.size();++i)raw[i]=std::sin(float(i)*.19f)*.1f;
    for(int i=0;i<idx;++i)gamma[i]=1.f+.1f*std::cos(float(i));
    auto fill_steps=[&](int generation) {
        for(int g=0;g<groups;++g)for(int t=0;t<count;++t) {
            const int row=g*count+t,p=(g%4)+generation*8+t;
            steps[row*4]=p;steps[row*4+1]=p+1;steps[row*4+2]=(p+1)/4;steps[row*4+3]=p+1;
        }
    };
    fill_steps(0);
    Buffer dk(keys.size()*4),dv(values.size()*4),dr(raw.size()*4),dg(idx*4),ds(steps.size()*4);
    Buffer dead(rows*idx*4),ref_dead(rows*idx*4);
    dk.upload(keys);dv.upload(values);dr.upload(raw);dg.upload(gamma);ds.upload(steps);
    auto grouped=[&] {
        k::kv_append_q8_groups(kv,ds.as<int>(),4,dk.as<float>(),dv.as<float>(),512,rows,shapes,stream);
        k::native_qsa_indexer_append_groups(dr.as<float>(),ds.as<int>(),4,dead.as<float>(),dg.as<float>(),
            1.e-6f,ix,shapes,cells,k::rope_scaling(),stream);
    };
    auto scalar=[&] {
        for(int g=0;g<groups;++g)for(int t=0;t<count;++t) {
            auto& s=*b[g];const int row=g*count+t;
            k::kv_append_q8_steps(s.key.as<int8_t>(),s.value.as<int8_t>(),s.ks.as<uint16_t>(),s.vs.as<uint16_t>(),
                s.table.as<int>(),ds.as<int>()+row*4,4,dk.as<float>()+row*512,dv.as<float>()+row*512,512,1,shapes,stream,nullptr);
            if(count>1) {
                auto* dst=s.history.as<char>()+size_t(t)*snapshot*4;
                ck(cudaMemcpyAsync(dst,s.tail.p,3*idx*4,cudaMemcpyDeviceToDevice,stream));
                ck(cudaMemcpyAsync(dst+3*idx*4,s.dead.p,idx*4,cudaMemcpyDeviceToDevice,stream));
                ck(cudaMemcpyAsync(dst+4*idx*4,s.pos.p,4,cudaMemcpyDeviceToDevice,stream));
            }
            k::native_qsa_indexer_append_steps(dr.as<float>()+row*idx,ds.as<int>()+row*4,4,1,0,dg.as<float>(),
                1.e-6f,s.indexer(),shapes,cells,k::rope_scaling(),stream);
            if(count>1)ck(cudaMemcpyAsync(s.raw.as<float>()+t*idx,dr.as<float>()+row*idx,idx*4,cudaMemcpyDeviceToDevice,stream));
            ck(cudaMemcpyAsync(ref_dead.as<float>()+row*idx,s.dead.p,idx*4,cudaMemcpyDeviceToDevice,stream));
        }
    };
    cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));grouped();
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));ck(cudaGraphDestroy(graph));
    for(int generation=0;generation<2;++generation) {
        fill_steps(generation);ds.upload(steps);scalar();ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
        equal(dead,ref_dead,"grouped indexer per-row dead differs");
        for(int g=0;g<groups;++g) {
            equal(a[g]->key,b[g]->key,"grouped KV keys differ");equal(a[g]->value,b[g]->value,"grouped KV values differ");
            equal(a[g]->ks,b[g]->ks,"grouped KV key scales differ");equal(a[g]->vs,b[g]->vs,"grouped KV value scales differ");
            equal(a[g]->tail,b[g]->tail,"grouped indexer tail differs");equal(a[g]->dead,b[g]->dead,"grouped indexer dead differs");
            equal(a[g]->pooled,b[g]->pooled,"grouped indexer pooled differs");equal(a[g]->pos,b[g]->pos,"grouped indexer position differs");
            equal(a[g]->history,b[g]->history,"grouped indexer history snapshot differs");equal(a[g]->raw,b[g]->raw,"grouped indexer raw snapshot differs");
        }
    }
    ck(cudaGraphExecDestroy(exec));
    std::printf("QSA append groups8 count%d state/KV/snapshots/replay exact PASS\n",count);
}
}
int main()try {
    if(cudaSetDevice(0)!=cudaSuccess)return 77;
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int count:{1,3,5})run(stream,count);
    ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
