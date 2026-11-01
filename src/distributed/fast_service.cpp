#include "strata/distributed/fast_service.hpp"
#include "strata/kernels/window_limits.hpp"
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <charconv>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <iomanip>
#include <algorithm>

namespace strata::distributed {
namespace {
class StepWatchdog {
public:
    explicit StepWatchdog(Deadline deadline) : worker_([this, deadline] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!changed_.wait_until(lock, deadline.end, [this] { return done_; })) std::_Exit(2);
    }) {}
    ~StepWatchdog() {
        { std::lock_guard<std::mutex> lock(mutex_); done_ = true; }
        changed_.notify_one();
        worker_.join();
    }
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool done_ = false;
    std::thread worker_;
};
}
Failure run_fast_service(MultiSlotConfig config, Session& session, FastModel& model,
                         FastTensorParallel& tp, const MultiSlotServiceCallbacks& io,
                         void* stream, uint32_t timeout_ms) {
    auto callbacks = io;
    const char* compact=std::getenv("STRATA_TP_CONTROL_COMPACT");
    const bool compact_control=compact&&compact[0]=='1';
    const char* distributed=std::getenv("STRATA_TP_DISTRIBUTED_DRAFTS");
    const bool distributed_drafts=distributed&&distributed[0]=='1';
    std::string speculative_publication;
    auto failed = [](const std::string& text) { return Failure::fail(FailureCode::comm_failed, text); };
    callbacks.scheduler.broadcast = [&](std::string_view label, std::string& payload) {
        Deadline at;
        if (auto f = session.operation_deadline(at)) return f;
        return session.broadcast_control_payload(label, payload, at);
    };
    callbacks.scheduler.agree_payload = [&](std::string_view label, std::string_view payload) {
        // The proposal broadcast validates these identities before GPU work;
        // the publication agreement below also seals the completed commit.
        if(compact_control&&label=="spec-batch")return Failure{};
        Deadline at;
        if (auto f = session.operation_deadline(at)) return f;
        if(compact_control&&label=="spec-publish-batch") {
            if(speculative_publication.empty())return failed("missing completed speculative commit");
            auto result=session.agree_control_payload(label,speculative_publication+std::string(payload),at);
            speculative_publication.clear();return result;
        }
        return session.agree_control_payload(label, payload, at);
    };
    callbacks.scheduler.admitted = [&](uint32_t slot, const ServeRequest& request) {
        std::string error;
        return model.sampling(int(slot), request, error) ? Failure{} : failed(error);
    };
    callbacks.scheduler.execute = [&](uint32_t slot, StepPhase phase, std::string_view,
                                       int32_t, uint64_t) {
        if (phase != StepPhase::reset && phase != StepPhase::cancel)
            return failed("scalar forward is unavailable in the production TP executor");
        std::string error;
        return model.reset(int(slot), error) ? Failure{} : failed(error);
    };
    callbacks.scheduler.execute_batch = [&](const std::vector<MultiSlotBatchRow>& rows) {
        std::string error;
        // Verifier and Prefill share this ordered stream. Providers and NCCL
        // remain inside the GPU work; only the complete step is a host boundary.
        Deadline deadline;
        if (auto f = session.operation_deadline(deadline)) return f;
        StepWatchdog watchdog(deadline);
        if (!tp.begin(stream, error)) return failed(error);
        if (!model.execute(rows, error) || !tp.finish(deadline, error) || !tp.validate_execution_status(stream, error)) {
            tp.abort();
            return failed(error);
        }
        // A peer failure cannot publish another rank's apparently valid token.
        return session.agree_control_payload("fast-step-committed-v1", "ok", deadline);
    };
    callbacks.scheduler.sample = [&](uint32_t slot, const ServeRequest& request, const std::vector<int32_t>& history,
                                      uint64_t draw, int32_t& token) {
        std::string error;
        if(const char* dir=std::getenv("STRATA_TP_LOGITS_DIR")) {
            std::vector<float> logits;
            if(!model.copy_logits(int(slot),logits,error))return failed(error);
            std::filesystem::create_directories(dir);
            std::ofstream out(std::filesystem::path(dir)/(request.request_id+"-"+std::to_string(history.size()-1)+".f32"),std::ios::binary);
            out.write((const char*)logits.data(),std::streamsize(logits.size()*4));
            if(!out)return failed("cannot write diagnostic TP logits");
        }
        return model.sample(int(slot), request, history, draw, token, error) ? Failure{} : failed(error);
    };
    if(model.mtp_enabled())callbacks.scheduler.speculate = [&](const std::vector<MultiSlotProposal>& requests,
                                         std::vector<MultiSlotAccepted>& accepted) {
        const auto started=std::chrono::steady_clock::now();
        const bool phase_timing=std::getenv("STRATA_TP_PHASE_TIMING")!=nullptr;
        auto phase_started=started;
        std::ostringstream phase_times;
        auto phase=[&](const char* name) {
            if(!phase_timing)return;
            const auto now=std::chrono::steady_clock::now();
            phase_times<<' '<<name<<'='<<std::chrono::duration<double,std::milli>(now-phase_started).count();
            phase_started=now;
        };
        std::string error,wire;
        std::vector<std::vector<int32_t>> drafts(requests.size());
        if(distributed_drafts) {
            const int limit=std::min(4,kernels::kWindowMaxT/int(requests.size())-1);
            if(!model.propose_batch(requests,drafts,limit,error))return failed(error);
            std::ostringstream out;
            for(size_t i=0;i<requests.size();++i)if(requests[i].slot%3==config.rank) {
                const auto& r=requests[i];out<<i<<' '<<r.slot<<' '<<std::quoted(r.request.request_id)<<' '
                    <<r.history.size()<<' '<<r.draw<<' '<<r.feedback<<' '<<drafts[i].size()<<' ';
                for(auto token:drafts[i])out<<token<<' ';
            }
            wire=out.str();
        } else if(config.rank==0) {
            std::ostringstream out;
            const int limit=std::min(4,kernels::kWindowMaxT/int(requests.size())-1);
            if(!model.propose_batch(requests,drafts,limit,error))return failed(error);
            for(size_t i=0;i<requests.size();++i) {
                // A full row budget leaves no room for draft verification.
                // Skip producer work whose output would all be discarded.
                if(compact_control) {
                    const auto& r=requests[i];out<<r.slot<<' '<<std::quoted(r.request.request_id)<<' '
                        <<r.history.size()<<' '<<r.draw<<' '<<r.feedback<<' ';
                }
                out<<drafts[i].size()<<' ';for(auto token:drafts[i])out<<token<<' ';
            }
            wire=out.str();
        }
        phase("propose");
        Deadline deadline;if(auto f=session.operation_deadline(deadline))return f;
        if(distributed_drafts) {
            std::vector<std::string> parts;
            if(auto f=session.exchange_control_payloads("fast-mtp-owned-proposals-v1",wire,parts,deadline))return f;
            if(parts.size()!=3)return failed("MTP proposal owner count mismatch");
            std::vector<bool> seen(requests.size(),false);
            for(size_t owner=0;owner<parts.size();++owner) {
                std::istringstream in(parts[owner]);
                for(;;) {
                    in>>std::ws;if(in.eof())break;
                    size_t index=0;uint32_t slot=0;std::string id;uint64_t position=0,draw=0;int32_t feedback=-1;int count=-1;
                    if(!(in>>index>>slot>>std::quoted(id)>>position>>draw>>feedback>>count)||index>=requests.size()||
                       seen[index]||slot%3!=owner||count<0||count>4)return failed("invalid owned MTP proposal");
                    const auto& r=requests[index];
                    if(slot!=r.slot||id!=r.request.request_id||position!=r.history.size()||draw!=r.draw||feedback!=r.feedback)
                        return failed("owned MTP proposal identity mismatch");
                    auto& proposal=drafts[index];proposal.resize(size_t(count));
                    for(auto& token:proposal)if(!(in>>token)||token<0||token>=config.n_vocab)
                        return failed("invalid owned MTP draft token");
                    seen[index]=true;
                }
            }
            if(std::find(seen.begin(),seen.end(),false)!=seen.end())return failed("missing owned MTP proposal");
        } else if(auto f=session.broadcast_control_payload("fast-mtp-proposals-v1",wire,deadline))return f;
        phase("proposals_broadcast");
        if(!distributed_drafts) {
            std::istringstream in(wire);
            for(size_t i=0;i<drafts.size();++i) {
                auto& proposal=drafts[i];
                if(compact_control) {
                    uint32_t slot=0;std::string id;uint64_t position=0,draw=0;int32_t feedback=-1;
                    const auto& r=requests[i];
                    if(!(in>>slot>>std::quoted(id)>>position>>draw>>feedback)||slot!=r.slot||
                       id!=r.request.request_id||position!=r.history.size()||draw!=r.draw||feedback!=r.feedback)
                        return failed("MTP proposal identity mismatch");
                }
                int count=-1;if(!(in>>count)||count<0||count>4)return failed("invalid MTP proposal count");
                proposal.resize(size_t(count));
                for(auto& token:proposal)if(!(in>>token)||token<0||token>=config.n_vocab)return failed("invalid MTP draft token");
            }
            in>>std::ws;if(!in.eof())return failed("trailing MTP proposal bytes");
        }
        StepWatchdog watchdog(deadline);
        if(!tp.begin(stream,error)||!model.verify_proposals(requests,drafts,error)){tp.abort();return failed(error);}
        phase("verify");
        if(!tp.finish(deadline,error)||!tp.validate_execution_status(stream,error)){tp.abort();return failed(error);}
        phase("finish");
        wire.clear();
        if(config.rank==0) {
            if(!model.select_prefixes(requests,accepted,error))return failed(error);
            std::ostringstream out;
            for(const auto& prefix:accepted) {
                out<<prefix.tokens.size()<<' ';
                for(size_t i=0;i<prefix.tokens.size();++i)out<<prefix.consumed[i]<<' '<<prefix.tokens[i]<<' ';
            }
            wire=out.str();
        }
        phase("select");
        if(auto f=session.broadcast_control_payload("fast-mtp-verdict-v1",wire,deadline))return f;
        phase("verdict_broadcast");
        {
            accepted.clear();std::istringstream in(wire);
            for(size_t i=0;i<requests.size();++i) {
                int count=0;if(!(in>>count)||count<1||count>5)return failed("invalid MTP verdict count");
                MultiSlotAccepted prefix;prefix.consumed.resize(size_t(count));prefix.tokens.resize(size_t(count));
                for(int j=0;j<count;++j)if(!(in>>prefix.consumed[j]>>prefix.tokens[j])||prefix.tokens[j]<0||
                    prefix.tokens[j]>=config.n_vocab)return failed("invalid MTP verdict token");
                accepted.push_back(std::move(prefix));
            }
            in>>std::ws;if(!in.eof())return failed("trailing MTP verdict bytes");
        }
        if(!model.commit_prefixes(requests,accepted,error)){tp.abort();return failed(error);}
        phase("commit");
        model.observe_mtp(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
        Failure result;
        if(compact_control)speculative_publication="committed:"+std::to_string(wire.size())+":"+wire;
        else result=session.agree_control_payload("fast-mtp-committed-v1","ok",deadline);
        phase("agreement");
        if(phase_timing)std::fprintf(stderr,"TP phases rank=%u slots=%zu%s\n",config.rank,requests.size(),phase_times.str().c_str());
        return result;
    };
    callbacks.scheduler.abort_all = [&] {
        // NCCL graph references must be released before communicator teardown.
        // Failed in-flight steps retain their process-exit guard instead.
        std::string error;
        if(!tp.requires_process_exit())model.retire(error);
        tp.abort();
    };
    return run_multislot_service(config, callbacks, timeout_ms);
}
}
