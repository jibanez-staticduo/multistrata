#include "strata/distributed/fast_execution.hpp"
#include "distributed/nccl_api.hpp"

#include <cstdio>
#include <vector>

using namespace strata::distributed;
class Api final : public nccl::NcclApi {
public:
    bool live = false;
    unsigned waits = 0;
    std::vector<uint64_t> counts;
    bool inspect_shared = false;
    Failure identity(std::string&, std::string&) const override { return {}; }
    Failure comm_init(const nccl::NcclConfig&) override { return {}; }
    Failure enqueue_all_reduce_sum(const void* in, void* out, uint64_t count, ScalarType, cudaStream_t) override {
        if (!live || in != out) return Failure::fail(FailureCode::bad_operation, "invalid reduction");
        if (inspect_shared && *static_cast<float*>(out) != 7.f)
            return Failure::fail(FailureCode::bad_operation, "shared partial was not added before reduction");
        counts.push_back(count); return {};
    }
    Failure enqueue_all_gather(const void*, void*, uint64_t, ScalarType, cudaStream_t) override {
        return Failure::fail(FailureCode::bad_operation, "hidden gathers are forbidden");
    }
    Failure poll_completion(Deadline) override { return Failure::fail(FailureCode::bad_operation, "per-layer waits are forbidden"); }
    Failure begin_stream_step(cudaStream_t) override { live = true; return {}; }
    Failure finish_stream_step(Deadline) override { ++waits; live = false; return {}; }
    bool async_failure(Failure&) const override { return false; }
    void comm_abort() noexcept override { live = false; }
};
class SharedState final : public strata::core::TensorParallelHooks {
public:
    int calls=0;
    bool fail=false;
    bool has_sharded_shared()const override{return true;}
    bool shared_expert_local_add(int64_t,const strata::core::TensorParallelRows&,const float*,
                                float* output,void*,std::string& error)override {
        ++calls;if(fail){error="shared provider refused";return false;}*output+=5.f;return true;
    }
    bool mixer(int64_t,const strata::core::TensorParallelRows&,const float*,float*,void*,std::string&)override{return false;}
    bool experts(int64_t,const strata::core::TensorParallelRows&,const float*,const int32_t*,const float*,int,float*,void*,std::string&)override{return false;}
};
int main() {
    Api api;
    unsigned calls = 0;
    auto mixer = [&](int64_t, const strata::core::TensorParallelRows&, const float*, float*, void*, std::string&) {
        ++calls; return true;
    };
    auto experts = [&](int64_t, const strata::core::TensorParallelRows&, const float*, const int32_t*,
                       const float*, int, float*, void*, std::string&) { ++calls; return true; };
    FastTensorParallel tp(api, 2560, mixer, experts);
    std::string error;
    auto stream = reinterpret_cast<void*>(uintptr_t{1});
    float input = 1, output = 0;
    int32_t id = 0;
    if (!tp.begin(stream, error)) return 1;
    for (int layer = 0; layer < 48; ++layer) {
        if (!tp.mixer(layer, {8}, &input, &output, stream, error) ||
            !tp.experts(layer, {8}, &input, &id, &input, 10, &output, stream, error)) return 2;
    }
    if (!tp.finish({}, error) || api.waits != 1 || calls != 96 || api.counts.size() != 96) return 3;
    for (auto count : api.counts) if (count != 8 * 2560) return 4;
    if (tp.finish({}, error)) return 5;
    Api shared_api;shared_api.inspect_shared=true;
    SharedState state;
    auto local=[](int64_t,const strata::core::TensorParallelRows&,const float*,const int32_t*,
                  const float*,int,float* out,void*,std::string&){*out=2.f;return true;};
    FastTensorParallel shared_tp(shared_api,2560,mixer,local);shared_tp.state_provider(&state);
    if(!shared_tp.has_sharded_shared()||!shared_tp.begin(stream,error)||
       !shared_tp.experts(0,{8},&input,&id,&input,10,&output,stream,error)||
       state.calls!=1||shared_api.counts.size()!=1||!shared_tp.finish({},error))return 6;
    state.fail=true;
    if(!shared_tp.begin(stream,error)||shared_tp.experts(0,{8},&input,&id,&input,10,&output,stream,error)||
       shared_api.counts.size()!=1||error!="shared provider refused")return 7;
    std::puts("fast TP: 48 layers, 96 output reductions, one wait, no hidden gathers");
}
