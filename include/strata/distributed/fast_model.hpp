#pragma once

#include "strata/core/session.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/tensor_parallel.hpp"
#include "strata/distributed/multislot.hpp"

#include <memory>
#include <string>

namespace strata::distributed {

// Shared weights, prompt workspace and batched verifier; mutable session/KV/PLE
// state belongs to each slot. The caller wraps every forward in one agreed NCCL
// stream step and does not publish a sampled ID until it completes.
class FastModel {
public:
    FastModel(const core::WeightTable&, const core::ModelGeometry&, const core::NativeHead&,
              kernels::PleTable&, core::TensorParallelHooks&, int slots, int context,
              int prompt_chunk, void* stream);
    ~FastModel();
    bool initialize(std::string& error);
    bool configure_mtp(const std::string& path,int drafts,int rank,std::string& error);
    bool propose(const MultiSlotProposal&,std::vector<int32_t>&,int draft_limit,std::string&);
    bool propose_batch(const std::vector<MultiSlotProposal>&,std::vector<std::vector<int32_t>>&,
                       int draft_limit,std::string&);
    bool verify_proposals(const std::vector<MultiSlotProposal>&,const std::vector<std::vector<int32_t>>&,
                            std::string& error);
    bool select_prefixes(const std::vector<MultiSlotProposal>&,std::vector<MultiSlotAccepted>&,std::string&);
    bool commit_prefixes(const std::vector<MultiSlotProposal>&,const std::vector<MultiSlotAccepted>&,std::string&);
    void observe_mtp(double round_ms);
    bool retire(std::string& error);
    bool execute(const std::vector<MultiSlotBatchRow>&, std::string& error);
    bool reset(int slot, std::string& error);
    bool sampling(int slot, const ServeRequest&, std::string& error);
    int32_t sampled(int slot) const;
    bool sample(int slot, const ServeRequest&, const std::vector<int32_t>& history,
                uint64_t draw, int32_t& token, std::string& error);
    const std::vector<core::SessionState*>& sessions() const;
    bool requires_process_exit() const noexcept;
    bool mtp_enabled() const noexcept;
    bool copy_logits(int slot,std::vector<float>& output,std::string& error);
private:
    bool propose_local_batch(const std::vector<MultiSlotProposal>&,std::vector<std::vector<int32_t>>&,
                             int draft_limit,std::string&);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::distributed
