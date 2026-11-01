#pragma once
#include "strata/core/geometry.hpp"
#include "strata/core/tensor_parallel.hpp"
#include <memory>
#include <string>

namespace strata::distributed {
class NativeNvfp4Experts {
public:
    NativeNvfp4Experts();
    ~NativeNvfp4Experts();
    bool initialize(const std::string& pack, const core::ModelGeometry&, int rank,
                    int max_rows, void* stream, std::string& error);
    bool local_sum(int64_t layer, const core::TensorParallelRows&, const float* input,
                   const int32_t* ids, const float* weights, int top_k, float* output,
                   void* stream, std::string& error);
    bool validate_execution_status(void* stream, std::string& error);
    uint64_t weight_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
