#include "strata/distributed/limits.hpp"
#include "strata/distributed/multislot.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::distributed {
namespace {
constexpr size_t kChunk = 24u << 10;
Failure invalid(std::string_view reason) { return Failure::fail(FailureCode::bad_frame, reason); }
}

MultiSlotScheduler::MultiSlotScheduler(MultiSlotConfig config, MultiSlotCallbacks callbacks)
    : config_(config), callbacks_(std::move(callbacks)) {
    if (config.rank > 2 || !config.slots || config.slots > kMultiSlotMaximum ||
        config.n_vocab <= 0 || !config.context || config.context > kDistributedMaxContext ||
        !config.request_limit || !callbacks_.broadcast || !callbacks_.execute ||
        !callbacks_.abort_all || (config.rank == 0 && !callbacks_.sample))
        throw std::invalid_argument("invalid multislot configuration or callbacks");
    slots_.resize(config.slots);
}

size_t MultiSlotScheduler::active() const {
    return size_t(std::count_if(slots_.begin(), slots_.end(), [](const Slot& s) { return bool(s.request); }));
}
std::optional<std::string> MultiSlotScheduler::request(uint32_t slot) const {
    if (slot >= slots_.size() || !slots_[slot].request) return {};
    return slots_[slot].request->request_id;
}
MultiSlotResult MultiSlotScheduler::terminal() const {
    return {MultiSlotOutcome::failed, kMultiSlotMaximum, failure_, {}};
}
MultiSlotResult MultiSlotScheduler::reject(std::string reason) const {
    return {MultiSlotOutcome::rejected, kMultiSlotMaximum, {}, std::move(reason)};
}
MultiSlotResult MultiSlotScheduler::abort(Failure failure) {
    if (!failure_) {
        failure_ = failure ? std::move(failure) : Failure::fail(FailureCode::aborted, "multislot group aborted");
        try { callbacks_.abort_all(); } catch (...) { /* Preserve terminal state even for a broken adapter. */ }
        for (auto& s : slots_) s = Slot{};
    }
    return terminal();
}

Failure MultiSlotScheduler::agree(std::string_view kind, const std::string& expected) {
    if (action_ == std::numeric_limits<uint64_t>::max()) return invalid("multislot action sequence exhausted");
    const std::string label = "serve-multislot-v1:" + std::to_string(action_) + ":" + std::string(kind);
    // Configuration participates even in malformed/idle actions, before parser decisions or slot state.
    const std::string length = "slots=" + std::to_string(config_.slots) + ":vocab=" +
        std::to_string(config_.n_vocab) + ":context=" + std::to_string(config_.context) +
        ":limit=" + std::to_string(config_.request_limit) + ":bytes=" + std::to_string(expected.size());
    if (callbacks_.agree_payload && length.size() + 1 + expected.size() <= kChunk) {
        std::string payload = length;
        payload.push_back('\n');
        payload += expected;
        if (auto f = callbacks_.agree_payload(label, payload)) return f;
        ++action_;
        return {};
    }
    std::string bytes = config_.rank == 0 ? length : std::string{};
    if (auto f = callbacks_.broadcast(label + ":length", bytes)) return f;
    if (bytes != length) return invalid("multislot action length disagrees");
    for (size_t at = 0; at < expected.size(); at += kChunk) {
        const std::string chunk = expected.substr(at, kChunk);
        bytes = config_.rank == 0 ? chunk : std::string{};
        if (auto f = callbacks_.broadcast(label + ":chunk=" + std::to_string(at / kChunk), bytes)) return f;
        if (bytes != chunk) return invalid("multislot action disagrees with local state");
    }
    // Peers enter this barrier only after checking every byte locally. Root must not mutate or emit
    // merely because the last chunk was delivered to a peer that subsequently rejected it.
    bytes.clear();
    if (auto f = callbacks_.broadcast(label + ":checked", bytes)) return f;
    if (!bytes.empty()) return invalid("invalid multislot acknowledgement");
    ++action_;
    return {};
}

std::string MultiSlotScheduler::identity(uint32_t slot) const {
    const auto& s = slots_[slot];
    return "tick=" + std::to_string(tick_) + ":slot=" + std::to_string(slot) +
           ":request=" + s.request->request_id + ":position=" + std::to_string(s.history.size()) +
           ":prompt=" + std::to_string(s.prompt) + ":draw=" + std::to_string(s.generated);
}

Failure MultiSlotScheduler::execute(uint32_t slot, StepPhase phase, int32_t token, uint64_t position) {
    const auto envelope = identity(slot) + ":phase=" + std::to_string(uint32_t(phase)) +
                          ":token=" + std::to_string(token) + ":execute_position=" + std::to_string(position);
    if (auto f = agree("execute", envelope)) return f;
    return callbacks_.execute(slot, phase, slots_[slot].request->request_id, token, position);
}

MultiSlotResult MultiSlotScheduler::submit(std::string_view json) try {
    if (failed()) return terminal();
    // Admission is bounded before copying; the parser enforces the same line limit.
    if (json.size() > kDistributedMaxLineBytes) {
        if (auto f = agree("oversized-admission", std::to_string(json.size()))) return abort(f);
        return reject("generate JSON exceeds 4 MiB");
    }
    if (auto f = agree("submit", std::string(json))) return abort(f);
    ServeRequest candidate;
    std::string error;
    if (!parse_serve_request(json, config_.n_vocab, config_.context, candidate, error) ||
        candidate.kind != ServeRequest::Kind::generate) return reject("invalid generate: " + error);
    auto free = std::find_if(slots_.begin(), slots_.end(), [](const Slot& s) { return !s.request; });
    std::string decision = "accepted";
    if (admitted_.count(candidate.request_id)) decision = "duplicate request identity";
    else if (admitted_.size() == config_.request_limit) decision = "session request limit reached";
    else if (free == slots_.end()) decision = "all slots busy";
    const auto index = free == slots_.end() ? kMultiSlotMaximum : uint32_t(free - slots_.begin());
    if (auto f = agree("admission", candidate.request_id + ":slot=" + std::to_string(index) + ":" + decision))
        return abort(f);
    if (decision != "accepted") return reject(decision);
    if (callbacks_.admitted) if (auto f = callbacks_.admitted(index, candidate)) return abort(f);
    admitted_.insert(candidate.request_id);
    free->request = std::move(candidate);
    return {MultiSlotOutcome::accepted, index, {}, {}};
} catch (...) {
    return abort(Failure::fail(FailureCode::comm_failed, "exception during multislot admission"));
}

MultiSlotResult MultiSlotScheduler::cancel(std::string_view id) try {
    if (failed()) return terminal();
    if (id.size() > 128) {
        if (auto f = agree("oversized-cancel", std::to_string(id.size()))) return abort(f);
        return reject("invalid cancellation identity");
    }
    auto slot = std::find_if(slots_.begin(), slots_.end(), [&](const Slot& s) {
        return s.request && s.request->request_id == id;
    });
    const uint32_t index = slot == slots_.end() ? kMultiSlotMaximum : uint32_t(slot - slots_.begin());
    if (auto f = agree("cancel", std::string(id) + ":slot=" + std::to_string(index))) return abort(f);
    if (slot == slots_.end()) return reject("cancellation has no active request");
    slot->cancel = true;
    return {MultiSlotOutcome::accepted, index, {}, {}};
} catch (...) {
    return abort(Failure::fail(FailureCode::comm_failed, "exception during multislot cancellation"));
}

Failure MultiSlotScheduler::sampled(std::string_view id, int32_t& token) {
    std::string bytes;
    if (config_.rank == 0) {
        if (token < 0 || token >= config_.n_vocab) return invalid("sample outside vocabulary");
        for (int shift : {24, 16, 8, 0}) bytes.push_back(char((uint32_t(token) >> shift) & 255));
    }
    if (auto f = callbacks_.broadcast("serve-multislot-sampled-v1:" + std::string(id), bytes)) return f;
    if (bytes.size() != 4) return invalid("invalid multislot sampled envelope");
    uint32_t value = 0;
    for (unsigned char byte : bytes) value = (value << 8) | byte;
    if (value >= uint32_t(config_.n_vocab)) return invalid("sample outside vocabulary");
    token = int32_t(value);
    return {};
}

MultiSlotResult MultiSlotScheduler::tick() try {
    if (failed()) return terminal();
    if (callbacks_.execute_batch) return tick_batched();
    uint32_t index = kMultiSlotMaximum;
    for (uint32_t n = 0; n < config_.slots; ++n) {
        const uint32_t at = (cursor_ + n) % config_.slots;
        if (slots_[at].request) { index = at; break; }
    }
    if (auto f = agree("select", "tick=" + std::to_string(tick_) + ":cursor=" + std::to_string(cursor_) +
                        ":selected=" + std::to_string(index))) return abort(f);
    if (index == kMultiSlotMaximum) return {};
    if (tick_ == std::numeric_limits<uint64_t>::max()) return abort(invalid("multislot tick exhausted"));
    auto& s = slots_[index];
    const std::string id = s.request->request_id;
    if (s.cancel) {
        if (auto f = agree("cancel-selected", identity(index))) return abort(f);
        // An untouched slot has no active Worker request to cancel.
        if (!s.history.empty()) if (auto f = execute(index, StepPhase::cancel, -1, 0)) return abort(f);
        s = Slot{};
        cursor_ = (index + 1) % config_.slots;
        ++tick_;
        if (config_.rank == 0 && callbacks_.emit) callbacks_.emit("cancelled", index, id, -1);
        return {MultiSlotOutcome::progressed, index, {}, {}};
    }
    const bool prefill = s.prompt < s.request->ids.size();
    const int32_t input = prefill ? s.request->ids[s.prompt] : s.feedback;
    if (input < 0) return abort(invalid("decode lacks committed feedback"));
    if (auto f = execute(index, prefill ? StepPhase::prefill : StepPhase::decode, input, s.history.size()))
        return abort(f);
    if (auto f = execute(index, StepPhase::commit, -1, 0)) return abort(f);
    s.history.push_back(input);
    if (prefill) ++s.prompt;
    if (s.prompt == s.request->ids.size()) {
        int32_t token = -1;
        if (config_.rank == 0) if (auto f = callbacks_.sample(index, *s.request, s.history, s.generated, token))
            return abort(f);
        if (auto f = sampled(identity(index), token)) return abort(f);
        if (auto f = agree("publish", identity(index) + ":token=" + std::to_string(token))) return abort(f);
        const bool done = s.generated + 1 == s.request->max_new;
        // Finish reset before publishing the final token, so a failing reset cannot leak output.
        if (done) if (auto f = execute(index, StepPhase::reset, -1, 0)) return abort(f);
        s.feedback = token;
        ++s.generated;
        if (done) s = Slot{};
        cursor_ = (index + 1) % config_.slots;
        ++tick_;
        if (config_.rank == 0 && callbacks_.emit) {
            callbacks_.emit("token", index, id, token);
            if (done) callbacks_.emit("done", index, id, -1);
        }
    } else {
        cursor_ = (index + 1) % config_.slots;
        ++tick_;
    }
    return {MultiSlotOutcome::progressed, index, {}, {}};
} catch (...) {
    return abort(Failure::fail(FailureCode::comm_failed, "exception during multislot tick"));
}

MultiSlotResult MultiSlotScheduler::tick_batched() {
    if (!config_.prefill_chunk || config_.prefill_chunk > 8192)
        return abort(invalid("invalid production prefill chunk"));
    if (tick_ == std::numeric_limits<uint64_t>::max()) return abort(invalid("multislot tick exhausted"));
    // Retire cancellation before selecting GPU rows. No cancelled slot can be
    // included in the next graph, and reset never changes another slot's state.
    for (uint32_t i = 0; i < config_.slots; ++i) {
        auto& s = slots_[i];
        if (!s.request || !s.cancel) continue;
        const auto id = s.request->request_id;
        if (auto f = agree("batch-cancel", identity(i))) return abort(f);
        if (!s.history.empty()) if (auto f = execute(i, StepPhase::cancel, -1, 0)) return abort(f);
        s = Slot{};
        if (config_.rank == 0 && callbacks_.emit) callbacks_.emit("cancelled", i, id, -1);
    }
    std::vector<uint32_t> prompt, decode;
    for (uint32_t n = 0; n < config_.slots; ++n) {
        const auto i = (cursor_ + n) % config_.slots;
        const auto& s = slots_[i];
        if (!s.request) continue;
        (s.prompt < s.request->ids.size() ? prompt : decode).push_back(i);
    }
    const bool prefill = !prompt.empty() && (prefer_prefill_ || decode.empty());
    if(!prefill && !decode.empty() && callbacks_.speculate) {
        std::vector<MultiSlotProposal> proposals;
        std::string envelope="tick="+std::to_string(tick_);
        for(auto slot:decode) {
            const auto& s=slots_[slot];
            proposals.push_back({slot,*s.request,s.history,s.feedback,s.generated});
            envelope+=':'+identity(slot);
        }
        if(auto f=agree("spec-batch",envelope))return abort(f);
        std::vector<MultiSlotAccepted> accepted;
        if(auto f=callbacks_.speculate(proposals,accepted))return abort(f);
        if(accepted.size()!=proposals.size())return abort(invalid("speculative batch result count mismatch"));
        for(size_t i=0;i<accepted.size();++i) {
            const auto slot=proposals[i].slot;
            const auto& a=accepted[i];const auto& s=slots_[slot];
            if(a.tokens.empty()||a.tokens.size()!=a.consumed.size()||
                a.tokens.size()>s.request->max_new-s.generated||a.consumed.front()!=s.feedback)
                return abort(invalid("invalid speculative prefix result"));
            for(size_t j=0;j<a.tokens.size();++j) {
                if(a.tokens[j]<0||a.tokens[j]>=config_.n_vocab||a.consumed[j]<0||a.consumed[j]>=config_.n_vocab||
                    (j && a.consumed[j]!=a.tokens[j-1]))return abort(invalid("invalid speculative token chain"));
            }
        }
        if (!callbacks_.agree_payload) {
            // Preserve the established per-slot agreement sequence for legacy adapters.
            for(size_t i=0;i<accepted.size();++i) {
                const auto slot=proposals[i].slot;auto& s=slots_[slot];const auto id=s.request->request_id;
                const auto& a=accepted[i];std::string decision=identity(slot);
                for(auto token:a.tokens)decision+=':'+std::to_string(token);
                if(auto f=agree("spec-publish",decision))return abort(f);
                s.history.insert(s.history.end(),a.consumed.begin(),a.consumed.end());
                s.generated+=a.tokens.size();s.feedback=a.tokens.back();const bool done=s.generated==s.request->max_new;
                if(done)if(auto f=execute(slot,StepPhase::reset,-1,0))return abort(f);
                if(done)s=Slot{};
                if(config_.rank==0 && callbacks_.emit) {
                    for(auto token:a.tokens)callbacks_.emit("token",slot,id,token);
                    if(done)callbacks_.emit("done",slot,id,-1);
                }
            }
        } else {
            // Length-prefix every value so identities and token sequences cannot collide.
            std::string decision;
            auto append_field = [&](std::string_view field) {
                decision += std::to_string(field.size());
                decision.push_back(':');
                decision.append(field);
            };
            append_field(std::to_string(accepted.size()));
            for(size_t i=0;i<accepted.size();++i) {
                append_field(identity(proposals[i].slot));
                append_field(std::to_string(accepted[i].tokens.size()));
                for(const auto token:accepted[i].tokens) append_field(std::to_string(token));
            }
            if(auto f=agree("spec-publish-batch",decision))return abort(f);
            std::vector<std::string> request_ids;
            std::vector<bool> done_slots;
            request_ids.reserve(accepted.size());
            done_slots.reserve(accepted.size());
            for(size_t i=0;i<accepted.size();++i) {
                const auto slot=proposals[i].slot;auto& s=slots_[slot];const auto id=s.request->request_id;
                const auto& a=accepted[i];
                s.history.insert(s.history.end(),a.consumed.begin(),a.consumed.end());
                s.generated+=a.tokens.size();s.feedback=a.tokens.back();const bool done=s.generated==s.request->max_new;
                request_ids.push_back(id);
                done_slots.push_back(done);
            }
            for(size_t i=0;i<accepted.size();++i) {
                const auto slot=proposals[i].slot;
                if(done_slots[i]) if(auto f=execute(slot,StepPhase::reset,-1,0))return abort(f);
            }
            for(size_t i=0;i<accepted.size();++i) {
                const auto slot=proposals[i].slot;
                const auto& id=request_ids[i];
                const bool done=done_slots[i];
                const auto& a=accepted[i];
                if(done)slots_[slot]=Slot{};
                if(config_.rank==0 && callbacks_.emit) {
                    for(auto token:a.tokens)callbacks_.emit("token",slot,id,token);
                    if(done)callbacks_.emit("done",slot,id,-1);
                }
            }
        }
        cursor_=(decode.back()+1)%config_.slots;prefer_prefill_=true;++tick_;
        return {MultiSlotOutcome::progressed,decode.front(),{}, {}};
    }
    std::vector<MultiSlotBatchRow> rows;
    if (prefill) {
        const size_t count=config_.packed_prefill?prompt.size():1;
        size_t remaining=config_.prefill_chunk;
        for(size_t at=0;at<count&&remaining;++at) {
            const auto i=prompt[at];const auto& s=slots_[i];
            const auto quota=std::max<size_t>(1,remaining/(count-at));
            const auto end=std::min(s.request->ids.size(),s.prompt+quota);
            rows.push_back({i,StepPhase::prefill,s.request->request_id,
                {s.request->ids.begin()+s.prompt,s.request->ids.begin()+end},s.history.size()});
            remaining-=end-s.prompt;
        }
    } else for (const auto i : decode) {
        const auto& s = slots_[i];
        if (s.feedback < 0) return abort(invalid("decode batch lacks committed feedback"));
        rows.push_back({i, StepPhase::decode, s.request->request_id, {s.feedback}, s.history.size()});
    }
    std::string envelope = "tick=" + std::to_string(tick_) + ":rows=" + std::to_string(rows.size());
    for (const auto& row : rows) {
        envelope += ":slot=" + std::to_string(row.slot) + ":request=" + row.request +
                    ":position=" + std::to_string(row.position) + ":phase=" + std::to_string(uint32_t(row.phase));
        for (const auto token : row.tokens) { envelope.push_back(':'); envelope += std::to_string(token); }
    }
    if (auto f = agree("batch-execute", envelope)) return abort(f);
    if (rows.empty()) return {};
    if (auto f = callbacks_.execute_batch(rows)) return abort(f);
    // Only an entirely successful, committed GPU batch advances host histories.
    for (const auto& row : rows) {
        auto& s = slots_[row.slot];
        s.history.insert(s.history.end(), row.tokens.begin(), row.tokens.end());
        if (prefill) s.prompt += row.tokens.size();
    }
    for (const auto& row : rows) {
        auto& s = slots_[row.slot];
        if (s.prompt != s.request->ids.size()) continue;
        const auto id = s.request->request_id;
        int32_t token = -1;
        if (config_.rank == 0) if (auto f = callbacks_.sample(row.slot, *s.request, s.history, s.generated, token))
            return abort(f);
        if (auto f = sampled(identity(row.slot), token)) return abort(f);
        if (auto f = agree("batch-publish", identity(row.slot) + ":token=" + std::to_string(token))) return abort(f);
        const bool done = s.generated + 1 == s.request->max_new;
        if (done) if (auto f = execute(row.slot, StepPhase::reset, -1, 0)) return abort(f);
        s.feedback = token;
        ++s.generated;
        if (done) s = Slot{};
        if (config_.rank == 0 && callbacks_.emit) {
            callbacks_.emit("token", row.slot, id, token);
            if (done) callbacks_.emit("done", row.slot, id, -1);
        }
    }
    cursor_ = (rows.back().slot + 1) % config_.slots;
    prefer_prefill_ = !prefill;
    ++tick_;
    return {MultiSlotOutcome::progressed, rows.front().slot, {}, {}};
}
}  // namespace strata::distributed
