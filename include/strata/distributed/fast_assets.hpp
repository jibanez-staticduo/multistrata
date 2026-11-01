#pragma once

#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/ngram.hpp"

#include <memory>
#include <string>

namespace strata::distributed {
// Dense/embedding/PLE assets for the production NVFP4 executor. Routed weights
// are loaded separately by the shard provider; no full expert GPU cache here.
class FastAssets {
public:
    FastAssets();
    ~FastAssets();
    bool load(const std::string& pack, const std::string& dense_gguf,
              const std::string& embedding_gguf, const std::string& ple_gguf,
              std::string& error);
    const core::WeightTable& weights() const;
    const core::ModelGeometry& geometry() const;
    const core::NativeHead& head() const;
    kernels::PleTable& ple();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
