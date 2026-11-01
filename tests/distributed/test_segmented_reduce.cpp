#include "strata/distributed/segmented_reduce.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace strata::distributed;
int main()try {
    alignas(16)std::array<uint8_t,4096> input{},output{};
    for(auto type:{ScalarType::fp32,ScalarType::bf16})for(uint64_t bytes:{16,1024,1040,4096}) {
        uint64_t next=0;unsigned calls=0;
        auto f=enqueue_segmented_reduce(input.data(),output.data(),bytes/scalar_type_bytes(type),type,1024,
            [&](const void* x,void* y,uint64_t count,ScalarType got) {
                if(got!=type||x!=input.data()+next||y!=output.data()+next||count*scalar_type_bytes(type)>1024)
                    throw std::runtime_error("chunk offset/type/size mismatch");
                next+=count*scalar_type_bytes(type);++calls;return Failure{};
            });
        if(f||next!=bytes||calls!=(bytes+1023)/1024)throw std::runtime_error("chunk coverage mismatch");
    }
    unsigned calls=0;
    uint64_t inplace=0;
    auto same=enqueue_segmented_reduce(output.data(),output.data(),1024,ScalarType::fp32,1024,
        [&](const void* x,void* y,uint64_t count,ScalarType) {
            if(x!=y)throw std::runtime_error("in-place chunk alias changed");inplace+=count;return Failure{};
        });
    if(same||inplace!=1024)throw std::runtime_error("in-place chunk coverage mismatch");
    auto failure=enqueue_segmented_reduce(input.data(),output.data(),1024,ScalarType::fp32,1024,
        [&](auto,auto,auto,auto){++calls;return Failure::fail(FailureCode::enqueue_failed,"injected");});
    if(failure.code!=FailureCode::enqueue_failed||calls!=1)throw std::runtime_error("partial failure was hidden");
    for(uint64_t count:std::array<uint64_t,3>{0,1,UINT64_MAX}) {
        auto f=enqueue_segmented_reduce(input.data(),output.data(),count,ScalarType::fp32,1024,
            [&](auto,auto,auto,auto){throw std::runtime_error("invalid span enqueued");return Failure{};});
        if(!f)throw std::runtime_error("invalid extent accepted");
    }
    std::puts("segmented reduction coverage/type/failure/overflow checks passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
