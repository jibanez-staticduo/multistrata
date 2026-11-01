#pragma once

#include "strata/core/tensor_parallel.hpp"
#include "strata/distributed/transport.hpp"

#include <functional>
#include <memory>

namespace strata::distributed {
namespace nccl { class NcclApi; }

// Providers enqueue only local GPU work; the shared adapter owns the two
// mathematical output reductions and one completion boundary per compute step.
class FastTensorParallel final : public core::TensorParallelHooks {
public:
    using Mixer = std::function<bool(int64_t, const core::TensorParallelRows&,
                                    const float*, float*, void*, std::string&)>;
    using Experts = std::function<bool(int64_t, const core::TensorParallelRows&,
                                      const float*, const int32_t*, const float*, int,
                                      float*, void*, std::string&)>;
    FastTensorParallel(nccl::NcclApi&, int64_t hidden, Mixer, Experts);
    ~FastTensorParallel();
    bool begin(void* stream, std::string& error);
    bool finish(Deadline, std::string& error);
    void abort() noexcept;
    bool requires_process_exit() const noexcept;
    void state_provider(core::TensorParallelHooks* provider) { state_ = provider; }
    void execution_validator(std::function<bool(void*, std::string&)> validator) { validate_ = std::move(validator); }
    bool bind_sessions(core::SessionState* const*, int, std::string&) override;
    bool reset_slots(const int32_t*, int, void*, std::string&) override;
    bool begin_slots(const int32_t*, const int32_t*, int, bool, void*, std::string&) override;
    bool commit_slots(const int32_t*, const int32_t*, int, void*, std::string&) override;
    bool rollback_slots(const int32_t*, const int64_t*, int, void*, std::string&) override;
    bool validate_execution_status(void*, std::string&) override;
    bool has_fast_hc() const override { return state_ && state_->has_fast_hc(); }
    bool use_fast_hc(int rows) const override { return state_ && state_->use_fast_hc(rows); }
    bool hyper_connection(const kernels::FusedGrArgs*, int, void*, std::string&) override;
    bool has_sharded_shared() const override { return state_ && state_->has_sharded_shared(); }
    void output_head(const void* weights, int type, size_t row_bytes, int vocab, int rank);
    bool has_output_head() const override { return head_weights_ != nullptr; }
    bool output_head(const void*, float*, int, void*, std::string&) override;
    void configure_compact_head(bool eligible) override { compact_head_active_ = compact_head_enabled_ && eligible; }
    bool compact_greedy_head() const override { return compact_head_active_; }
    bool output_head_greedy(const void*, int32_t*, uint32_t*, int, int, void*, std::string&) override;
    bool prepare_dense_output_head(void*, std::string&);
    bool has_dense_output_head(int) const override;
    bool dense_output_head(const float*, float*, int32_t*, uint32_t*, int, int, void*, std::string&) override;
    bool mixer(int64_t, const core::TensorParallelRows&, const float*, float*,
               void*, std::string&) override;
    bool experts(int64_t, const core::TensorParallelRows&, const float*,
                 const int32_t*, const float*, int, float*, void*, std::string&) override;
private:
    bool reduce(const core::TensorParallelRows&, float*, void*, std::string&);
    nccl::NcclApi& api_;
    int64_t hidden_;
    Mixer mixer_;
    Experts experts_;
    void* stream_ = nullptr;
    bool failed_ = false;
    core::TensorParallelHooks* state_ = nullptr;
    std::function<bool(void*, std::string&)> validate_;
    const void* head_weights_ = nullptr;
    int head_type_ = -1, head_vocab_ = 0, head_rank_ = 0;
    size_t head_row_bytes_ = 0;
    void* head_scratch_ = nullptr;
    void* head_candidates_ = nullptr;
    bool compact_head_enabled_ = false, compact_head_active_ = false;
    void* reduction_scratch_ = nullptr;
    bool reduction_prefill_only_ = false;
    struct DenseHead;
    std::unique_ptr<DenseHead> dense_head_;
};
} // namespace strata::distributed
