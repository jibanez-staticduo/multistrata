#include "strata/distributed/fast_model.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/draft_policy.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <thread>

namespace strata::distributed {
namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Memory {
    void* data = nullptr;
    ~Memory() { if (data) cudaFree(data); }
    void allocate(size_t bytes) { check(cudaMalloc(&data, bytes)); }
};
struct Slot {
    core::SessionState state;
    Memory arena, ple_row, ple_scratch;
    std::vector<float> ple_host;
    uint64_t position = 0;
    int32_t sampled = -1;
    int last_row = -1;
    Memory draft_rows;
    std::unique_ptr<core::MtpDrafter> draft;
    std::vector<int32_t> pending_picks;
    uint64_t pending_position=0;
    bool draft_caught_up = false;
    spec::DraftPolicy policy{5};
    spec::DraftPolicy::ChainState draft_chain;
    std::vector<float> draft_probs;
    int proposed_rows=1,accepted_rows=1;
    int draft_cap=5;
    bool adaptive_drafts=true;
    bool compact_head_eligible=false;
    ~Slot() { draft.reset(); core::session_release(state); }
};
void wire_ple(const core::WeightTable& weights, Slot& slot, kernels::PleTable& table) {
    auto weight = [&](const char* name) -> const core::WeightRef& {
        auto* value = weights.find(std::string("blk.1.") + name);
        if (!value || (!value->data && !value->native_data)) throw std::runtime_error("missing NVFP4 PLE weight");
        return *value;
    };
    auto& state = slot.state;
    const auto& key = weight("ple_key.weight");
    const auto& convolution = weight("ple_conv1d.weight");
    if (!key.quantized()) state.ple.w.key_bf16 = static_cast<const uint16_t*>(key.data);
    else if (key.native_data) {
        state.ple.w.key_native_data = key.native_data;
        state.ple.w.key_native_type = key.native_type;
        state.ple.w.key_native_q8_1 = key.native_q8_1;
    } else throw std::runtime_error("unsupported NVFP4 PLE key storage");
    state.ple.w.value_bf16 = static_cast<const uint16_t*>(weight("ple_value.weight").data);
    state.ple.w.norm_key = static_cast<const float*>(weight("ple_norm_key.weight").data);
    state.ple.w.norm_query = static_cast<const float*>(weight("ple_norm_query.weight").data);
    state.ple.w.norm_conv = static_cast<const float*>(weight("ple_norm_conv.weight").data);
    if (convolution.bytes != uint64_t(convolution.elements) * 2)
        throw std::runtime_error("NVFP4 PLE convolution must retain FP16 storage");
    state.ple.w.conv1d_f16 = static_cast<const uint16_t*>(convolution.data);
    slot.ple_row.allocate(kernels::NG_N_EMBD * sizeof(float));
    slot.ple_scratch.allocate(core::ple_run_scratch_bytes());
    slot.ple_host.resize(kernels::NG_N_EMBD);
    state.ple.table = &table;
    state.ple.hist = state.ple_hist;
    state.ple.token = &state.ple_token;
    state.ple.prev = state.ple_prev;
    state.ple.emb_dev = static_cast<float*>(slot.ple_row.data);
    state.ple.emb_host = slot.ple_host.data();
    state.ple.scratch = static_cast<float*>(slot.ple_scratch.data);
    state.ple.consts = kernels::ple_artifact_consts();
    if (!state.ple.ready()) throw std::runtime_error("NVFP4 PLE is not ready");
}
}
struct FastModel::Impl {
    const core::WeightTable& weights;
    const core::ModelGeometry& geometry;
    const core::NativeHead& head;
    kernels::PleTable& ple;
    core::TensorParallelHooks& hooks;
    int count, context, chunk;
    cudaStream_t stream;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<core::SessionState*> sessions;
    // Declared last: destroy every captured graph before freeing slot memory.
    std::unique_ptr<prefill::Prefill> prompt;
    std::unique_ptr<core::Verifier> verify;
    Memory selected_slots;
    Memory sample_history, sample_bias, sample_output;
    std::vector<float> host_bias;
    std::vector<int32_t> staged_slots;
    std::vector<int32_t> greedy_picks;
    bool failed = false;
    int drafts=0,rank=0;
    bool async_drafts=false,distributed_drafts=false;
    uint64_t mtp_windows=0,mtp_verified=0,mtp_accepted=0;
    std::vector<int> proposal_slots,proposal_first,proposal_count;
    std::vector<int32_t> proposal_tokens;
    std::vector<int64_t> proposal_positions;
    Impl(const core::WeightTable& w, const core::ModelGeometry& g, const core::NativeHead& h,
         kernels::PleTable& p, core::TensorParallelHooks& t, int n, int c, int pc, void* s)
        : weights(w), geometry(g), head(h), ple(p), hooks(t), count(n), context(c), chunk(pc),
          stream(static_cast<cudaStream_t>(s)) {}
    bool capacity(Slot& slot, uint64_t cells, std::string& error) {
        if (cells > uint64_t(context)) { error = "request exceeds NVFP4 context"; return false; }
        for (int64_t q = 0; q < slot.state.qsa_alloc; ++q)
            if (!core::qsa_kv_elastic_grow_state(slot.state.qsa_states[q + slot.state.qsa_ord0], int64_t(cells))) {
                error = "shared KV memory budget exhausted"; return false;
            }
        if(slot.draft && !core::qsa_kv_elastic_grow_state(slot.draft->kv_state_rw(),int64_t(cells))) {
            error="draft KV memory budget exhausted";return false;
        }
        return true;
    }
};
FastModel::FastModel(const core::WeightTable& weights, const core::ModelGeometry& geometry,
                     const core::NativeHead& head, kernels::PleTable& ple,
                     core::TensorParallelHooks& hooks, int slots, int context, int prompt_chunk, void* stream)
    : impl_(std::make_unique<Impl>(weights, geometry, head, ple, hooks, slots, context, prompt_chunk, stream)) {}
FastModel::~FastModel() {
    // Graph executables borrow row tables and all slot buffers. Retire them
    // before member destruction starts releasing those addresses.
    impl_->verify.reset();
    impl_->prompt.reset();
}
bool FastModel::retire(std::string& error) {
    auto& p=*impl_;
    if(cudaStreamSynchronize(p.stream)!=cudaSuccess){error="cannot retire unfinished TP graphs";return false;}
    if (p.rank == 0 && p.mtp_windows)
        std::fprintf(stderr,"TP MTP windows=%llu verified=%llu committed=%llu\n",
            (unsigned long long)p.mtp_windows,(unsigned long long)p.mtp_verified,
            (unsigned long long)p.mtp_accepted);
    if(p.distributed_drafts)for(size_t i=0;i<p.slots.size();++i)if(p.slots[i]->draft)
        std::fprintf(stderr,"TP draft owner rank=%d slot=%zu\n",p.rank,i);
    p.verify.reset();p.prompt.reset();return true;
}
bool FastModel::initialize(std::string& error) try {
    auto& p = *impl_;
    if (p.verify || p.count < 1 || p.count > 8 || p.context < 4 || p.context > 262144 ||
        p.chunk < 64 || p.chunk > 8192 || !p.stream) { error = "invalid final NVFP4 executor configuration"; return false; }
    core::qsa_set_kv_int8(true);
    core::qsa_set_kv_elastic(true, std::min(p.context, 2048));
    if (!core::qsa_kv_elastic()) { error = "final NVFP4 executor requires CUDA VMM"; return false; }
    for (int i = 0; i < p.count; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->arena.allocate(core::session_bytes(p.geometry, p.context, p.geometry.n_expert_used));
        if (!core::session_init(p.geometry, p.context, p.geometry.n_expert_used, slot->arena.data, slot->state)) {
            error = "cannot allocate NVFP4 slot state"; return false;
        }
        wire_ple(p.weights, *slot, p.ple);
        core::session_zero(slot->state, p.geometry, nullptr, p.stream);
        p.sessions.push_back(&slot->state);
        p.slots.push_back(std::move(slot));
    }
    check(cudaStreamSynchronize(p.stream));
    p.verify = std::make_unique<core::Verifier>();
    p.verify->set_stream(p.stream);
    p.verify->set_tensor_parallel(&p.hooks);
    p.selected_slots.allocate(kernels::kWindowMaxT * sizeof(int32_t));
    p.sample_history.allocate(size_t(p.context) * sizeof(int32_t));
    p.sample_bias.allocate(size_t(p.geometry.n_vocab) * sizeof(float));
    p.sample_output.allocate(sizeof(int32_t));
    p.host_bias.resize(size_t(p.geometry.n_vocab));
    p.staged_slots.resize(kernels::kWindowMaxT);
    p.verify->set_tensor_parallel_slot_table(static_cast<const int32_t*>(p.selected_slots.data));
    p.verify->set_batch_graph_limit(16);
    if (!p.verify->init(p.weights, p.geometry, *p.sessions.front(), {}, &p.head, kernels::kWindowMaxT, error) ||
        !p.verify->init_slots(p.sessions, error)) return false;
    p.prompt = std::make_unique<prefill::Prefill>();
    p.prompt->set_tensor_parallel(&p.hooks);
    if (!p.prompt->init(p.weights, p.geometry, *p.sessions.front(), nullptr, nullptr, nullptr,
                         p.chunk, p.stream, error)) return false;
    return p.hooks.bind_sessions(p.sessions.data(), int(p.sessions.size()), error);
} catch (const std::exception& e) { impl_->failed = true; error = e.what(); return false; }
bool FastModel::execute(const std::vector<MultiSlotBatchRow>& rows, std::string& error) try {
    auto& p = *impl_;
    if (p.failed || !p.verify || rows.empty() || rows.size() > size_t(p.count)) { error = "invalid NVFP4 batch"; return false; }
    std::vector<int> selected;
    std::vector<int32_t> tokens, output(rows.size());
    std::vector<int64_t> positions;
    const bool packed=rows.size()>1&&rows.front().phase==StepPhase::prefill;
    p.hooks.configure_compact_head(std::all_of(rows.begin(),rows.end(),[&](const auto& row) {
        return row.slot<p.slots.size()&&p.slots[row.slot]->compact_head_eligible;
    }));
    std::vector<std::vector<int64_t>> packed_tokens(rows.size());
    std::vector<prefill::PrefillPackedPrompt> packed_prompts;
    for (const auto& row : rows) {
        if(packed&&row.phase!=StepPhase::prefill){error="mixed packed prompt/decode batch";return false;}
        if (row.slot >= p.slots.size() || row.tokens.empty() ||
            row.position != p.slots[row.slot]->position ||
            std::find(selected.begin(), selected.end(), int(row.slot)) != selected.end()) {
            error = "NVFP4 batch identity/position mismatch"; return false;
        }
        auto& slot = *p.slots[row.slot];
        if (!p.capacity(slot, row.position + row.tokens.size(), error)) return false;
        if (row.phase == StepPhase::prefill) {
            if(packed) {
                const size_t index=selected.size();
                packed_tokens[index].assign(row.tokens.begin(),row.tokens.end()-1);
                if(!packed_tokens[index].empty())packed_prompts.push_back({&slot.state,int32_t(row.slot),
                    packed_tokens[index].data(),int64_t(packed_tokens[index].size()),int64_t(row.position)});
                if(slot.draft)slot.draft->set_prompt_len(int64_t(row.position+row.tokens.size()));
            }else {
            if (rows.size() != 1 || !p.prompt->rebind_session(slot.state, error)) return false;
            p.prompt->set_tensor_parallel_slot(int32_t(row.slot));
            if(slot.draft) {
                slot.draft->set_prompt_len(int64_t(row.position+row.tokens.size()));
                p.prompt->on_chunk=[&](const float* residual,int64_t count,int64_t position,std::string& e) {
                    std::vector<int32_t> next(size_t(count),0);
                    for(int64_t i=0;i<count;++i)next[size_t(i)]=row.tokens.at(size_t(position-row.position+i+1));
                    return p.prompt->draft_kv(*slot.draft,residual,next.data(),count,position,e) ||
                        (e.empty()&&slot.draft->prefill(residual,next.data(),count,position,e));
                };
            }
            if (row.tokens.size() > 1) {
                std::vector<int64_t> prompt(row.tokens.begin(), row.tokens.end() - 1);
                if (!p.prompt->run(prompt.data(), int64_t(prompt.size()), int64_t(row.position), error)) return false;
            }
            p.prompt->on_chunk = {};
            }
        } else if (row.phase != StepPhase::decode || row.tokens.size() != 1) {
            error = "invalid NVFP4 execution phase"; return false;
        }
        selected.push_back(int(row.slot));
        tokens.push_back(row.tokens.back());
        positions.push_back(int64_t(row.position + row.tokens.size() - 1));
    }
    if(packed&&!packed_prompts.empty()) {
        p.prompt->on_packed_chunk=[&](int32_t id,const float* residual,int64_t count,int64_t position,std::string& e) {
            auto& slot=*p.slots.at(size_t(id));if(!slot.draft)return true;
            const auto found=std::find_if(rows.begin(),rows.end(),[&](const auto& row){return row.slot==uint32_t(id);});
            if(found==rows.end()){e="packed draft slot is not in the request batch";return false;}
            std::vector<int32_t> next(size_t(count),0);
            for(int64_t i=0;i<count;++i)next[size_t(i)]=found->tokens.at(size_t(position-found->position+i+1));
            return p.prompt->draft_kv(*slot.draft,residual,next.data(),count,position,e)||
                (e.empty()&&slot.draft->prefill(residual,next.data(),count,position,e));
        };
        const bool ok=p.prompt->run_packed(packed_prompts,error);
        p.prompt->on_packed_chunk={};
        if(!ok)return false;
    }
    for (size_t i = 0; i < selected.size(); ++i) p.staged_slots[i] = selected[i];
    std::vector<int32_t> begin_ids(selected.begin(),selected.end()),begin_rows(selected.size(),1);
    if(!p.hooks.begin_slots(begin_ids.data(),begin_rows.data(),int(begin_ids.size()),false,p.stream,error))return false;
    check(cudaMemcpyAsync(p.selected_slots.data, p.staged_slots.data(), selected.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice, p.stream));
    if (!p.verify->run_slot_rows(selected.data(), int(rows.size()), tokens.data(), positions.data(),
                                  nullptr, nullptr, output.data(), error) ||
        !p.verify->commit_slots(error)) { p.failed = true; return false; }
    p.greedy_picks=output;
    std::vector<int32_t> committed(selected.begin(), selected.end()), keep(selected.size(), 1);
    if (!p.hooks.commit_slots(committed.data(), keep.data(), int(keep.size()), p.stream, error)) {
        p.failed = true; return false;
    }
    check(cudaStreamSynchronize(p.stream));
    if (p.verify->nonfinite_row() >= 0) { p.failed = true; error = "nonfinite NVFP4 logits"; return false; }
    for (size_t i = 0; i < rows.size(); ++i) {
        auto& slot = *p.slots[rows[i].slot];
        slot.position += rows[i].tokens.size();
        slot.sampled = output[i];
        slot.last_row = int(i);
        if(slot.draft) {
            const size_t bytes=size_t(p.geometry.hc_dim())*4;
            check(cudaMemcpyAsync(slot.draft_rows.data,p.verify->final_R(int(i)),bytes,cudaMemcpyDeviceToDevice,p.stream));
            slot.pending_picks.clear();
            slot.draft_caught_up = false;
        }
    }
    return true;
} catch (const std::exception& e) { impl_->failed = true; error = e.what(); return false; }
bool FastModel::reset(int index, std::string& error) try {
    auto& p = *impl_;
    if (p.failed || index < 0 || size_t(index) >= p.slots.size()) { error = "invalid NVFP4 reset slot"; return false; }
    if (!p.verify->wait_commit(error)) return false;
    auto& slot = *p.slots[size_t(index)];
    const int32_t id = index;
    if (!p.hooks.reset_slots(&id, 1, p.stream, error)) return false;
    core::session_zero(slot.state, p.geometry, nullptr, p.stream);
    check(cudaStreamSynchronize(p.stream));
    for (int64_t q = 0; q < slot.state.qsa_alloc; ++q)
        if (!core::qsa_kv_elastic_shrink_state(slot.state.qsa_states[q + slot.state.qsa_ord0], std::min(p.context, 2048))) {
            error = "cannot release retired NVFP4 KV pages"; return false;
        }
    slot.position = 0; slot.sampled = -1; slot.last_row = -1;
    slot.pending_picks.clear();
    slot.draft_caught_up = false;
    if(slot.draft){
        core::qsa_state_zero(slot.draft->kv_state_rw(),p.geometry,slot.draft->stream());
        if(!slot.draft->idle(error)||!core::qsa_kv_elastic_shrink_state(slot.draft->kv_state_rw(),std::min(p.context,2048)))return false;
    }
    return true;
} catch (const std::exception& e) { impl_->failed = true; error = e.what(); return false; }
bool FastModel::sampling(int slot, const ServeRequest& request, std::string& error) {
    auto& p = *impl_;
    if (!p.verify || slot < 0 || size_t(slot) >= p.slots.size()) { error = "invalid sampling slot"; return false; }
    // Only root samples after the whole rank group has completed. The graph's
    // plain greedy pick can be reused only without request transformations.
    kernels::SamplerParams greedy;
    greedy.greedy = true;
    greedy.temperature = 0;
    p.verify->set_slot_sampling(slot, greedy);
    p.slots[size_t(slot)]->compact_head_eligible=
        (request.sampling.greedy||request.sampling.temperature<=0.f)&&
        request.sampling.penalty_last_n==0&&request.logit_bias.empty();
    const char* cost=std::getenv("STRATA_SPEC_COST");
    p.slots[size_t(slot)]->adaptive_drafts=cost&&cost[0]=='1'&&
        (request.sampling.greedy || request.sampling.temperature<=0.f);
    if(auto& owner=p.slots[size_t(slot)];owner->draft) {
        if(owner->adaptive_drafts) {
            Slot* state=owner.get();
            owner->draft->set_chain_rule([state](int index,float probability) {
                return state->policy.keep_draft(state->draft_chain,index,probability,state->draft_cap);
            });
        } else owner->draft->set_chain_rule({});
    }
    return true;
}
bool FastModel::configure_mtp(const std::string& path,int drafts,int rank,std::string& error)try {
    auto& p=*impl_;
    if(!p.verify||drafts<1||drafts>4||rank<0||rank>2)throw std::runtime_error("invalid fast MTP configuration");
    p.drafts=drafts;p.rank=rank;
    const char* distributed=std::getenv("STRATA_TP_DISTRIBUTED_DRAFTS");
    p.distributed_drafts=distributed&&distributed[0]=='1';
    if(rank!=0&&!p.distributed_drafts)return true;
    const char* async=std::getenv("STRATA_TP_ASYNC_DRAFTS");
    p.async_drafts=async&&async[0]=='1';
    std::string draft_vocab;
    if(const char* file=std::getenv("STRATA_TP_DRAFT_VOCAB")) {
        draft_vocab=file;std::ifstream input(draft_vocab,std::ios::binary|std::ios::ate);
        const auto bytes=input.tellg();
        if(!input||bytes<4||bytes%4||bytes>p.geometry.n_vocab*4)
            throw std::runtime_error("invalid fast draft vocabulary file");
        input.seekg(0);std::set<int32_t> unique;
        for(std::streamoff at=0;at<bytes;at+=4) {
            int32_t token=-1;input.read(reinterpret_cast<char*>(&token),4);
            if(!input||token<0||token>=p.geometry.n_vocab||!unique.insert(token).second)
                throw std::runtime_error("invalid or duplicate fast draft vocabulary token");
        }
    }
    core::MtpDrafter* shared=nullptr;
    for(size_t index=0;index<p.slots.size();++index) {
        if(p.distributed_drafts&&int(index%3)!=rank)continue;
        auto& slot=p.slots[index];
        slot->draft=std::make_unique<core::MtpDrafter>();
        if(p.async_drafts)slot->draft->set_force_capture(true);
        if(!draft_vocab.empty())slot->draft->set_draft_vocab(draft_vocab);
        if(const char* norm=std::getenv("STRATA_TP_MTP_HNORM");norm&&std::string_view(norm)=="stream")
            slot->draft->set_hnorm_per_stream(true);
        if(!slot->draft->load(path,p.geometry,slot->state,5,error,32768,shared))return false;
        slot->draft_rows.allocate(size_t(5)*p.geometry.hc_dim()*4);
        if(!slot->draft->bind(p.weights,&p.head,(float*)slot->draft_rows.data,error,shared))return false;
        slot->draft->set_max_drafts(drafts);slot->draft->set_ple_session(&slot->state);
        if(p.async_drafts&&!slot->draft->prepare_chain(error))return false;
        slot->policy.set_robust(true);
        if(!shared)shared=slot->draft.get();
    }
    return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastModel::propose_batch(const std::vector<MultiSlotProposal>& requests,
    std::vector<std::vector<int32_t>>& output,int draft_limit,std::string& error)try {
    auto& p=*impl_;
    if(!p.distributed_drafts)return propose_local_batch(requests,output,draft_limit,error);
    output.assign(requests.size(),{});
    std::vector<MultiSlotProposal> local;
    std::vector<size_t> indices;
    for(size_t i=0;i<requests.size();++i)if(int(requests[i].slot%3)==p.rank) {
        indices.push_back(i);local.push_back(requests[i]);
    }
    if(local.empty())return true;
    std::vector<std::vector<int32_t>> drafts;
    if(!propose_local_batch(local,drafts,draft_limit,error))return false;
    for(size_t i=0;i<indices.size();++i)output[indices[i]]=std::move(drafts[i]);
    return true;
}catch(const std::exception& e){impl_->failed=true;error=e.what();return false;}
bool FastModel::propose_local_batch(const std::vector<MultiSlotProposal>& requests,
    std::vector<std::vector<int32_t>>& output,int draft_limit,std::string& error)try {
    auto& p=*impl_;output.assign(requests.size(),{});
    const bool asynchronous=p.async_drafts&&requests.size()>1&&
        std::none_of(requests.begin(),requests.end(),[&](const auto& r){return p.slots.at(r.slot)->adaptive_drafts;});
    if(!asynchronous) {
        for(size_t i=0;i<requests.size();++i) {
            const auto& r=requests[i];
            if(draft_limit>0&&r.request.max_new-r.draw>1&&r.history.size()+1<uint64_t(p.context)&&
               !propose(r,output[i],draft_limit,error))return false;
        }
        return true;
    }
    std::vector<int> limits(requests.size());
    for(size_t i=0;i<requests.size();++i) {
        const auto& r=requests[i];auto& slot=*p.slots.at(r.slot);
        if(!slot.draft||slot.position!=r.history.size())throw std::runtime_error("invalid async draft state");
        const auto room=std::min<uint64_t>(r.request.max_new-r.draw,uint64_t(p.context)-slot.position);
        limits[i]=std::min({p.drafts,draft_limit,int(room?room-1:0)});
        if(limits[i]>0&&!p.capacity(slot,slot.position+limits[i]+1,error))return false;
    }
    // All accepted residual copies precede the independent drafter streams.
    // Launch each chain before waiting for any one slot's result.
    check(cudaStreamSynchronize(p.stream));
    for(size_t i=0;i<requests.size();++i) {
        if(limits[i]<1)continue;
        const auto& r=requests[i];auto& slot=*p.slots[r.slot];
        const bool first=slot.pending_picks.empty();
        const int32_t* tokens=first?&r.feedback:slot.pending_picks.data();
        const int count=first?1:int(slot.pending_picks.size());
        const auto position=first?int64_t(slot.position-1):int64_t(slot.pending_position);
        slot.draft->set_window_R((float*)slot.draft_rows.data);
        if(!slot.draft->chain_launch(count,tokens,position,count-1,nullptr,0,limits[i],0,error)) {
            p.failed=true;return false;
        }
    }
    for(size_t i=0;i<requests.size();++i) {
        if(limits[i]<1)continue;
        auto& slot=*p.slots[requests[i].slot];int ready=0;
        while(!(ready=slot.draft->chain_poll(error)))std::this_thread::yield();
        if(ready<0){p.failed=true;return false;}
        output[i].assign(slot.draft->chain_tok(),slot.draft->chain_tok()+limits[i]);
        slot.draft_probs.assign(slot.draft->chain_prob(),slot.draft->chain_prob()+limits[i]);
        slot.draft_caught_up=true;
    }
    return true;
}catch(const std::exception& e){impl_->failed=true;error=e.what();return false;}
bool FastModel::propose(const MultiSlotProposal& request,std::vector<int32_t>& output,int draft_limit,std::string& error)try {
    auto& p=*impl_;auto& slot=*p.slots.at(request.slot);output.clear();
    if(!slot.draft||slot.position!=request.history.size())throw std::runtime_error("invalid fast MTP proposal state");
    const auto room=std::min<uint64_t>(request.request.max_new-request.draw,uint64_t(p.context)-slot.position);
    const int limit=std::min({p.drafts,draft_limit,int(room>0?room-1:0)});
    if(limit<1)throw std::runtime_error("no room for a fast MTP proposal");
    if(!p.capacity(slot,slot.position+uint64_t(limit)+1,error))return false;
    slot.draft->set_max_drafts(limit);
    slot.draft_cap=limit+1;
    check(cudaStreamSynchronize(p.stream));
    int32_t result[4]{};float probabilities[4]{};int produced=0;
    slot.policy.set_context(int64_t(slot.position));
    slot.draft_chain={};
    if(slot.pending_picks.empty()) {
        if(!slot.draft->draft_first(limit+1,(float*)slot.draft_rows.data,request.feedback,
            int64_t(slot.position-1),result,error,probabilities,0,&produced))return false;
    }else {
        slot.draft->set_window_R((float*)slot.draft_rows.data);
        if(!slot.draft->draft(int(slot.pending_picks.size()),slot.pending_picks.data(),int64_t(slot.pending_position),
            int(slot.pending_picks.size()-1),result,error,probabilities,0,&produced))return false;
    }
    slot.draft_caught_up = true;
    const int kept=slot.adaptive_drafts?std::min(produced,std::max(1,slot.draft_chain.kept)):produced;
    output.assign(result,result+kept);slot.draft_probs.assign(probabilities,probabilities+kept);return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastModel::verify_proposals(const std::vector<MultiSlotProposal>& requests,
    const std::vector<std::vector<int32_t>>& drafts,std::string& error)try {
    auto& p=*impl_;
    p.proposal_slots.clear();p.proposal_first.clear();p.proposal_count.clear();p.proposal_tokens.clear();p.proposal_positions.clear();
    if(requests.empty()||requests.size()!=drafts.size())throw std::runtime_error("invalid fast proposal batch");
    p.hooks.configure_compact_head(std::all_of(requests.begin(),requests.end(),[&](const auto& r) {
        return r.slot<p.slots.size()&&p.slots[r.slot]->compact_head_eligible;
    }));
    const int per_slot=std::min(p.drafts+1,kernels::kWindowMaxT/int(requests.size()));
    std::vector<int32_t> ids,counts;
    for(size_t i=0;i<requests.size();++i) {
        const auto& r=requests[i];auto& slot=*p.slots.at(r.slot);
        const int count=int(std::min({uint64_t(per_slot),uint64_t(drafts[i].size()+1),
            uint64_t(r.request.max_new-r.draw),uint64_t(p.context)-slot.position}));
        if(count<1||slot.position!=r.history.size())throw std::runtime_error("invalid proposal position/budget");
        if(!p.capacity(slot,slot.position+count,error))return false;
        p.proposal_first.push_back(int(p.proposal_slots.size()));p.proposal_count.push_back(count);ids.push_back(int32_t(r.slot));counts.push_back(count);
        slot.proposed_rows=count;
        for(int row=0;row<count;++row) {
            p.proposal_slots.push_back(int(r.slot));p.proposal_tokens.push_back(row?drafts[i][row-1]:r.feedback);
            p.proposal_positions.push_back(int64_t(slot.position+row));
        }
    }
    const int count=int(p.proposal_slots.size());std::vector<int32_t> output(size_t(count),0);
    for(int i=0;i<count;++i)p.staged_slots[i]=p.proposal_slots[i];
    check(cudaMemcpyAsync(p.selected_slots.data,p.staged_slots.data(),size_t(count)*4,cudaMemcpyHostToDevice,p.stream));
    if(!p.hooks.begin_slots(ids.data(),counts.data(),int(ids.size()),true,p.stream,error))return false;
    p.verify->set_tensor_parallel_speculative(true);
    if(!p.verify->run_slot_rows(p.proposal_slots.data(),count,p.proposal_tokens.data(),p.proposal_positions.data(),
        nullptr,nullptr,output.data(),error))return false;
    p.greedy_picks=std::move(output);
    return p.verify->nonfinite_row()<0;
}catch(const std::exception& e){impl_->failed=true;error=e.what();return false;}
bool FastModel::select_prefixes(const std::vector<MultiSlotProposal>& requests,
    std::vector<MultiSlotAccepted>& accepted,std::string& error)try {
    auto& p=*impl_;accepted.clear();
    for(size_t i=0;i<requests.size();++i) {
        const auto& r=requests[i];auto history=r.history;MultiSlotAccepted result;
        const int first=p.proposal_first.at(i),count=p.proposal_count.at(i);
        for(int j=0;j<count;++j) {
            const int token=p.proposal_tokens[first+j];history.push_back(token);
            p.slots[r.slot]->last_row=first+j;int32_t picked=-1;
            if(!sample(int(r.slot),r.request,history,r.draw+j,picked,error))return false;
            result.consumed.push_back(token);result.tokens.push_back(picked);
            if(j+1<count&&picked!=p.proposal_tokens[first+j+1])break;
        }
        accepted.push_back(std::move(result));
    }return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastModel::commit_prefixes(const std::vector<MultiSlotProposal>& requests,
    const std::vector<MultiSlotAccepted>& accepted,std::string& error)try {
    auto& p=*impl_;std::vector<int> keep(size_t(p.count),0);std::vector<int32_t> ids,counts;
    for(size_t i=0;i<requests.size();++i) {
        const int slot=int(requests[i].slot),n=int(accepted.at(i).tokens.size());
        if(n<1||n>p.proposal_count.at(i))throw std::runtime_error("invalid accepted prefix");
        for(int j=0;j<n;++j)if(accepted[i].consumed[j]!=p.proposal_tokens[p.proposal_first[i]+j])throw std::runtime_error("accepted input differs from proposal");
        keep[slot]=n;ids.push_back(slot);counts.push_back(n);
    }
    if(!p.verify->commit_slot_prefixes(keep.data(),error)||!p.hooks.commit_slots(ids.data(),counts.data(),int(ids.size()),p.stream,error))return false;
    for(size_t i=0;i<requests.size();++i) {
        auto& slot=*p.slots[requests[i].slot];const size_t n=accepted[i].tokens.size();
        if(slot.draft) {
            // Full batches skip proposal generation. Preserve every target pair
            // before replacing the previous residual window with this one.
            if (!slot.draft_caught_up) {
                check(cudaStreamSynchronize(p.stream));
                if (slot.pending_picks.empty()) {
                    const int32_t feedback = requests[i].feedback;
                    if (!slot.draft->prefill((float*)slot.draft_rows.data, &feedback, 1,
                                             int64_t(slot.position - 1), error)) return false;
                } else if (!slot.draft->prefill((float*)slot.draft_rows.data, slot.pending_picks.data(),
                                                int64_t(slot.pending_picks.size()),
                                                int64_t(slot.pending_position), error)) return false;
            }
            check(cudaMemcpyAsync(slot.draft_rows.data,p.verify->final_R(p.proposal_first[i]),
                n*size_t(p.geometry.hc_dim())*4,cudaMemcpyDeviceToDevice,p.stream));
            slot.pending_position=slot.position;slot.pending_picks=accepted[i].tokens;
            slot.draft_caught_up = false;
        }
        slot.position+=n;
        slot.accepted_rows=int(n);
        p.mtp_verified += uint64_t(p.proposal_count[i]);
        p.mtp_accepted += n;
    }
    ++p.mtp_windows;
    p.verify->set_tensor_parallel_speculative(false);return true;
}catch(const std::exception& e){impl_->failed=true;error=e.what();return false;}
void FastModel::observe_mtp(double round_ms) {
    auto& p=*impl_;
    if(p.rank!=0&&!p.distributed_drafts)return;
    for(size_t i=0;i<p.proposal_first.size();++i) {
        auto& slot=*p.slots[size_t(p.proposal_slots[p.proposal_first[i]])];
        if(p.distributed_drafts&&!slot.draft)continue;
        const int drafts=std::min(int(slot.draft_probs.size()),slot.proposed_rows-1);
        slot.policy.observe_probs(slot.draft_probs.data(),drafts,slot.accepted_rows-1);
        slot.policy.observe(false,slot.proposed_rows,slot.accepted_rows-1,0,round_ms);
    }
}
bool FastModel::sample(int index, const ServeRequest& request, const std::vector<int32_t>& history,
                        uint64_t draw, int32_t& token, std::string& error) try {
    auto& p = *impl_;
    if (p.failed || index < 0 || size_t(index) >= p.slots.size() || history.empty() || history.size() > size_t(p.context)) {
        error = "invalid committed NVFP4 sample state"; return false;
    }
    auto& slot = *p.slots[size_t(index)];
    const auto* logits = p.verify->logits_device(slot.last_row);
    if (!logits) { error = "NVFP4 logits row is unavailable"; return false; }
    if(const char* dir=std::getenv("STRATA_TP_LOGITS_DIR")) {
        std::vector<float> values(size_t(p.geometry.n_vocab));
        if(!p.verify->copy_logits(slot.last_row,values.data())){error="cannot capture sampled TP logits";return false;}
        std::filesystem::create_directories(dir);
        std::ofstream out(std::filesystem::path(dir)/(request.request_id+"-"+std::to_string(history.size()-1)+".f32"),std::ios::binary);
        out.write((const char*)values.data(),std::streamsize(values.size()*4));
        if(!out){error="cannot write sampled TP logits";return false;}
    }
    auto params = request.sampling;
    // The verifier has already returned the plain greedy picks for every row.
    // Reuse them only when the request adds no history or logit transformation.
    if((params.greedy||params.temperature<=0.f)&&params.penalty_last_n==0&&request.logit_bias.empty()) {
        token=p.greedy_picks.at(size_t(slot.last_row));
        return token>=0&&token<p.geometry.n_vocab;
    }
    params.counter = draw;
    params.penalty_last_n = std::min(params.penalty_last_n, p.context);
    check(cudaMemcpyAsync(p.sample_history.data, history.data(), history.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice, p.stream));
    if (!request.logit_bias.empty()) {
        std::fill(p.host_bias.begin(), p.host_bias.end(), 0.f);
        for (const auto& [id, value] : request.logit_bias) p.host_bias.at(size_t(id)) = value;
        check(cudaMemcpyAsync(p.sample_bias.data, p.host_bias.data(), p.host_bias.size() * sizeof(float),
                              cudaMemcpyHostToDevice, p.stream));
        params.logit_bias = static_cast<const float*>(p.sample_bias.data);
    } else params.logit_bias = nullptr;
    kernels::sample_tokens(logits, 1, int(p.geometry.n_vocab), static_cast<const int*>(p.sample_history.data),
                            int(history.size()), params, static_cast<int*>(p.sample_output.data), p.stream);
    check(cudaMemcpyAsync(&token, p.sample_output.data, sizeof(token), cudaMemcpyDeviceToHost, p.stream));
    check(cudaStreamSynchronize(p.stream));
    return token >= 0 && token < p.geometry.n_vocab;
} catch (const std::exception& e) { impl_->failed = true; error = e.what(); return false; }
int32_t FastModel::sampled(int slot) const { return impl_->slots.at(size_t(slot))->sampled; }
const std::vector<core::SessionState*>& FastModel::sessions() const { return impl_->sessions; }
bool FastModel::requires_process_exit() const noexcept { return impl_->failed; }
bool FastModel::mtp_enabled() const noexcept { return impl_->drafts>0; }
bool FastModel::copy_logits(int slot,std::vector<float>& output,std::string& error) {
    auto& p=*impl_;
    if(slot<0||size_t(slot)>=p.slots.size()||!p.verify){error="invalid logits capture slot";return false;}
    output.resize(size_t(p.geometry.n_vocab));
    if(!p.verify->copy_logits(p.slots[slot]->last_row,output.data())){error="cannot read final TP logits";return false;}
    return true;
}
} // namespace strata::distributed
