#pragma once
#include "nccl_api.hpp"
#include "strata/distributed/roce_backend.hpp"
namespace strata::distributed {
std::unique_ptr<nccl::NcclApi> make_fast_roce_api(std::unique_ptr<nccl::NcclApi>,roce::RoceBackend&,uint64_t maximum,
                                               bool segment_large=false);
std::unique_ptr<nccl::NcclApi> make_roce_only_api(roce::RoceBackend&,uint64_t maximum);
}
