#pragma once
#include "strata/distributed/roce_backend.hpp"
#include "strata/distributed/session.hpp"
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace strata::distributed {

// Bytes travel through the bit-preserving gather, never a floating-point sum.
class RoceControlExchange {
public:
    explicit RoceControlExchange(roce::RoceBackend& backend):backend_(backend) {
        try {
            check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking));
            check(cudaMallocHost(&send_,kSlotBytes));check(cudaMallocHost(&receive_,3*kSlotBytes));
            check(cudaMalloc(&device_send_,kSlotBytes));check(cudaMalloc(&device_receive_,3*kSlotBytes));
        }catch(...){release();throw;}
    }
    ~RoceControlExchange() {
        // A pending NIC/GPU reference must be handled by the outer exit guard.
        if(backend_.requires_process_exit())std::_Exit(2);
        if(stream_)cudaStreamSynchronize(stream_);
        release();
    }
    Failure operator()(std::string_view local,std::vector<std::string>& output,Deadline at) {
        if(local.size()>kMaximum)return Failure::fail(FailureCode::oversized_frame,"RDMA control envelope exceeds cap");
        std::vector<std::string> staged;
        // Exchange lengths first: every rank must enqueue the same byte count.
        std::memset(send_,0,16);const uint32_t size=uint32_t(local.size());
        std::memcpy(send_,&size,4);
        if(auto f=gather(16,at))return f;
        uint32_t sizes[3]{};size_t maximum=0;
        for(int rank=0;rank<3;++rank) {
            std::memcpy(&sizes[rank],static_cast<char*>(receive_)+rank*16,4);
            if(sizes[rank]>kMaximum)return Failure::fail(FailureCode::oversized_frame,"peer RDMA control envelope exceeds cap");
            maximum=std::max(maximum,size_t(sizes[rank]));
        }
        const size_t bytes=std::max<size_t>(16,(maximum+15)&~size_t(15));
        std::memset(send_,0,bytes);std::memcpy(send_,local.data(),local.size());
        if(auto f=gather(bytes,at))return f;
        for(int rank=0;rank<3;++rank)
            staged.emplace_back(static_cast<char*>(receive_)+rank*bytes,sizes[rank]);
        output.swap(staged);return {};
    }
private:
    void release()noexcept {
        if(device_send_)cudaFree(device_send_);
        if(device_receive_)cudaFree(device_receive_);
        if(send_)cudaFreeHost(send_);
        if(receive_)cudaFreeHost(receive_);
        if(stream_)cudaStreamDestroy(stream_);
    }
    static constexpr size_t kMaximum=256u*1024u,kSlotBytes=kMaximum;
    static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
    Failure gather(size_t bytes,Deadline at) {
        auto error=[](cudaError_t e){return Failure::fail(FailureCode::comm_failed,cudaGetErrorString(e));};
        if(auto e=cudaMemcpyAsync(device_send_,send_,bytes,cudaMemcpyHostToDevice,stream_);e!=cudaSuccess)return error(e);
        if(auto f=backend_.begin_stream_step(stream_))return f;
        if(auto f=backend_.raw_all_gather({device_send_,bytes/4,ScalarType::fp32},
            {device_receive_,3*bytes/4,ScalarType::fp32},bytes/4,stream_))return f;
        if(auto e=cudaMemcpyAsync(receive_,device_receive_,3*bytes,cudaMemcpyDeviceToHost,stream_);e!=cudaSuccess)return error(e);
        // Control collectives finish in tens of microseconds. Polling the stream
        // here avoids the generic backend's millisecond sleep between polls.
        for(;;) {
            const auto e=cudaStreamQuery(stream_);
            if(e==cudaSuccess)break;
            if(e!=cudaErrorNotReady)return error(e);
            if(std::chrono::steady_clock::now()>=at.end) {
                backend_.raw_abort();
                return Failure::fail(FailureCode::deadline_expired,"RDMA control stream deadline expired");
            }
            Failure failure;if(backend_.raw_async_failure(failure))return failure;
        }
        if(auto f=backend_.finish_stream_step(at))return f;
        return {};
    }
    roce::RoceBackend& backend_;
    cudaStream_t stream_=nullptr;
    void *send_=nullptr,*receive_=nullptr,*device_send_=nullptr,*device_receive_=nullptr;
};
}
