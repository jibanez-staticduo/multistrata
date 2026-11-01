#include "fast_roce_api.hpp"
#include "strata/distributed/segmented_reduce.hpp"
#include <cstdlib>
#include <stdexcept>
#include "strata/kernels/fast_mixer.hpp"
namespace strata::distributed {
namespace {
class RoceOnlyApi final:public nccl::NcclApi {
    roce::RoceBackend& backend_;
    uint64_t maximum_;
    void* gather_scratch_=nullptr;
public:
    RoceOnlyApi(roce::RoceBackend& b,uint64_t maximum):backend_(b),maximum_(maximum){
        const auto e=cudaMalloc(&gather_scratch_,maximum*3);
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    }
    ~RoceOnlyApi()override {
        if(backend_.requires_process_exit())std::_Exit(2);
        if(gather_scratch_)cudaFree(gather_scratch_);
    }
    Failure identity(std::string& version,std::string& provenance)const override {
        version="native-roce-v1";provenance=backend_.provenance();return {};
    }
    Failure comm_init(const nccl::NcclConfig&)override{return {};}
    Failure begin_stream_step(cudaStream_t s)override{return backend_.begin_stream_step(s);}
    Failure finish_stream_step(Deadline d)override{return backend_.finish_stream_step(d);}
    Failure enqueue_all_reduce_sum(const void* in,void* out,uint64_t n,ScalarType type,cudaStream_t s)override {
        static const bool ring_order=[] {
            const char* value=std::getenv("STRATA_TP_ROCE_RING_ORDER");return value&&value[0]=='1';
        }();
        if(ring_order&&type==ScalarType::fp32&&n>((2u<<20)/4)&&n%8==0) {
            const uint64_t limit=maximum_/4;
            for(uint64_t first=0;first<n;first+=limit) {
                const uint64_t count=std::min(limit,n-first);
                if(auto f=backend_.raw_all_reduce_ring_sum(
                    {const_cast<float*>(static_cast<const float*>(in)+first),count,type},
                    {static_cast<float*>(out)+first,count,type},count,first,n,s))return f;
            }
            return {};
        }
        return enqueue_segmented_reduce(in,out,n,type,maximum_,[&](const void* x,void* y,uint64_t count,ScalarType dtype) {
            return backend_.raw_all_reduce_sum({const_cast<void*>(x),count,dtype},{y,count,dtype},count,s);
        });
    }
    Failure enqueue_all_gather(const void* in,void* out,uint64_t n,ScalarType type,cudaStream_t s)override {
        const auto bytes=scalar_type_bytes(type);
        uint64_t total=0;
        if(!in||!out||!bytes||!n||!element_bytes(n,type,total)||total%16||total>SIZE_MAX/3)
            return Failure::fail(FailureCode::bad_extent,"invalid RoCE-only all-gather span");
        if(total<=maximum_)return backend_.raw_all_gather({const_cast<void*>(in),n,type},{out,n*3,type},n,s);
        const auto* source=static_cast<const uint8_t*>(in);auto* destination=static_cast<uint8_t*>(out);
        for(uint64_t first=0;first<total;first+=maximum_) {
            const uint64_t count=std::min(maximum_,total-first),elements=count/bytes;
            if(auto f=backend_.raw_all_gather({const_cast<uint8_t*>(source+first),elements,type},
                {gather_scratch_,elements*3,type},elements,s))return f;
            const auto e=cudaMemcpy2DAsync(destination+first,total,gather_scratch_,count,count,3,
                cudaMemcpyDeviceToDevice,s);
            if(e!=cudaSuccess)return Failure::fail(FailureCode::comm_failed,cudaGetErrorString(e));
        }
        return {};
    }
    Failure poll_completion(Deadline d)override{return backend_.raw_poll(d);}
    bool async_failure(Failure& f)const override{return backend_.raw_async_failure(f);}
    bool requires_process_exit()const override{return backend_.requires_process_exit();}
    void comm_abort()noexcept override{backend_.raw_abort();}
};
class FastRoceApi final:public nccl::NcclApi {
    std::unique_ptr<nccl::NcclApi> nccl_;
    roce::RoceBackend& roce_;
    uint64_t maximum_;
    bool segment_large_;
public:
    FastRoceApi(std::unique_ptr<nccl::NcclApi> n,roce::RoceBackend& r,uint64_t maximum,bool segment_large)
        :nccl_(std::move(n)),roce_(r),maximum_(maximum),segment_large_(segment_large){}
    Failure identity(std::string& v,std::string& p)const override{return nccl_->identity(v,p);}
    Failure comm_init(const nccl::NcclConfig& c)override{return nccl_->comm_init(c);}
    Failure begin_stream_step(cudaStream_t s)override {
        if(auto f=nccl_->begin_stream_step(s))return f;
        if(auto f=roce_.begin_stream_step(s)){nccl_->comm_abort();return f;}
        return {};
    }
    Failure finish_stream_step(Deadline d)override {
        if(auto f=nccl_->finish_stream_step(d)){roce_.raw_abort();return f;}
        if(auto f=roce_.finish_stream_step(d)){nccl_->comm_abort();return f;}
        return {};
    }
    Failure enqueue_all_reduce_sum(const void* in,void* out,uint64_t n,ScalarType type,cudaStream_t s)override {
        const auto bytes=scalar_type_bytes(type);
        if(segment_large_&&type==ScalarType::fp32&&bytes&&n&&n>maximum_/bytes&&n<=UINT64_MAX/bytes&&(n*bytes)%16==0&&
           reinterpret_cast<uintptr_t>(in)%16==0&&reinterpret_cast<uintptr_t>(out)%16==0)
            return enqueue_segmented_reduce(in,out,n,type,maximum_,[&](const void* x,void* y,uint64_t count,ScalarType dtype) {
                return roce_.raw_all_reduce_sum({const_cast<void*>(x),count,dtype},{y,count,dtype},count,s);
            });
        if(bytes&&n&&n<=maximum_/bytes&&(n*bytes)%16==0&&
           reinterpret_cast<uintptr_t>(in)%16==0&&reinterpret_cast<uintptr_t>(out)%16==0)
            return roce_.raw_all_reduce_sum({const_cast<void*>(in),n,type},{out,n,type},n,s);
        return nccl_->enqueue_all_reduce_sum(in,out,n,type,s);
    }
    Failure enqueue_all_gather(const void* in,void* out,uint64_t n,ScalarType type,cudaStream_t s)override {
        return nccl_->enqueue_all_gather(in,out,n,type,s);
    }
    Failure poll_completion(Deadline d)override{return nccl_->poll_completion(d);}
    bool async_failure(Failure& f)const override{return roce_.raw_async_failure(f)||nccl_->async_failure(f);}
    bool requires_process_exit()const override{return roce_.requires_process_exit()||nccl_->requires_process_exit();}
    void comm_abort()noexcept override{roce_.raw_abort();nccl_->comm_abort();}
};
}
std::unique_ptr<nccl::NcclApi> make_roce_only_api(roce::RoceBackend& r,uint64_t maximum) {
    return std::make_unique<RoceOnlyApi>(r,maximum);
}
std::unique_ptr<nccl::NcclApi> make_fast_roce_api(std::unique_ptr<nccl::NcclApi> n,roce::RoceBackend& r,uint64_t maximum,bool segment_large) {
    return std::make_unique<FastRoceApi>(std::move(n),r,maximum,segment_large);
}
}
