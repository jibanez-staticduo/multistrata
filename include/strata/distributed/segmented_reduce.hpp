#pragma once
#include "strata/distributed/transport.hpp"
#include <algorithm>
#include <cstdint>

namespace strata::distributed {
// The caller agrees one logical reduction; chunks execute on its ordered stream.
template<class Enqueue>
Failure enqueue_segmented_reduce(const void* input,void* output,uint64_t elements,
                                  ScalarType type,uint64_t maximum,Enqueue&& enqueue) {
    uint64_t bytes=0;
    const auto width=scalar_type_bytes(type);
    if(!input||!output||!elements||!width||!element_bytes(elements,type,bytes)||
       maximum<16||maximum%16||bytes%16||reinterpret_cast<uintptr_t>(input)%16||
       reinterpret_cast<uintptr_t>(output)%16)
        return Failure::fail(FailureCode::bad_extent,"invalid segmented reduction span");
    const uint64_t limit=maximum/width;
    const auto* in=static_cast<const uint8_t*>(input);auto* out=static_cast<uint8_t*>(output);
    for(uint64_t first=0;first<elements;) {
        const uint64_t count=std::min(limit,elements-first);
        if(auto f=enqueue(in+first*width,out+first*width,count,type))return f;
        first+=count;
    }
    return {};
}
}
