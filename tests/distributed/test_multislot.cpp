#include "strata/distributed/multislot.hpp"

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {
using namespace strata::distributed;
int checks = 0, failures = 0;
void check(bool value, const char* what) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
Failure bad() { return Failure::fail(FailureCode::comm_failed, "injected failure"); }
std::string generate(std::string id, uint32_t prompt = 1, uint32_t output = 3) {
    std::string json = "{\"type\":\"generate\",\"request_id\":\"" + id + "\",\"ids\":[";
    for (uint32_t i = 0; i < prompt; ++i) json += (i ? "," : "") + std::to_string(i % 97 + 1);
    return json + "],\"max_new\":" + std::to_string(output) +
        ",\"sampling\":{\"seed\":18446744073709551615,\"repetition_penalty\":1.2}}";
}

// All five mutable state families influence the output. A single shared history or a partially
// reset state cannot pass the ordinary execution reference comparison.
struct ModelState {
    std::vector<int32_t> kv, qsa, indexer, ple;
    uint64_t gdn = 7;
    uint64_t positions = 0;
    bool operator==(const ModelState&) const = default;
};
void forward(ModelState& s, int32_t token) {
    s.kv.push_back(token);
    s.gdn = s.gdn * 131 + uint32_t(token);
    s.qsa.push_back(int32_t((s.gdn + s.positions) % 97));
    s.indexer.push_back(int32_t((uint32_t(token) * 17 + s.positions) % 97));
    s.ple.push_back(int32_t((s.gdn ^ (s.positions * 997)) % 97));
    ++s.positions;
}
int32_t pick(const ModelState& s, const ServeRequest& req, uint64_t draw) {
    uint64_t value = s.gdn + req.sampling.seed + draw * 719;
    for (const auto* v : {&s.kv, &s.qsa, &s.indexer, &s.ple})
        for (const auto t : *v) value = value * 131 + uint32_t(t);
    return int32_t(value % 97);
}
std::pair<std::vector<int32_t>, ModelState> ordinary(const std::string& json) {
    ServeRequest request;
    std::string error;
    check(parse_serve_request(json, 100, 65536, request, error), "ordinary reference parses public request");
    ModelState state;
    for (auto token : request.ids) forward(state, token);
    std::vector<int32_t> result;
    for (uint64_t draw = 0; draw < request.max_new; ++draw) {
        if (draw) forward(state, result.back());
        result.push_back(pick(state, request, draw));
    }
    return {result, state};
}
struct Exchange { std::string label, bytes; };
struct Call {
    uint32_t slot;
    StepPhase phase;
    std::string id;
    int32_t token;
    uint64_t position;
    bool operator==(const Call&) const = default;
};
struct FakeSlot {
    ModelState committed, pending;
    bool has_pending = false;
};
struct Rig {
    std::array<FakeSlot, 8> states;
    std::map<std::string, std::vector<int32_t>> output;
    std::map<std::string, std::vector<uint64_t>> draws;
    std::map<std::string, ModelState> completed;
    std::vector<Call> calls;
    std::vector<Exchange> wire;
    std::vector<std::string> events;
    std::vector<uint32_t> forwards;
    size_t sample_count = 0, abort_count = 0;
    size_t read = 0;
    size_t compact_calls = 0;
    std::string fail_compact_label;
    bool failed_phase = false;
    StepPhase fail_phase = StepPhase::abort;
    std::string fail_label;
    bool fail_sample = false;
    bool throw_emit = false;
    uint32_t rank;
    size_t batch_calls = 0, largest_decode_batch = 0, largest_prompt_chunk = 0;
    std::unique_ptr<MultiSlotScheduler> scheduler;
    Rig(uint32_t slots, uint32_t rank_ = 0, const std::vector<Exchange>* replay = nullptr,
        uint32_t limit = 4096, uint32_t context = 65536, bool batched = false,
        bool compact = false) : rank(rank_) {
        MultiSlotCallbacks cb;
        cb.broadcast = [&, replay](std::string_view label, std::string& bytes) {
            check(bytes.size() <= 24u * 1024u, "broadcast uses bounded chunks");
            if (!fail_label.empty() && label.find(fail_label) != std::string_view::npos) return bad();
            if (rank == 0) wire.push_back({std::string(label), bytes});
            else {
                check(bytes.empty(), "follower supplies no broadcast input");
                if (!replay || read == replay->size()) return bad();
                const auto& frame = (*replay)[read++];
                if (frame.label != label) return bad();
                bytes = frame.bytes;
            }
            return Failure{};
        };
        if (compact) cb.agree_payload = [&, replay](std::string_view label, std::string_view payload) {
            ++compact_calls;
            if (!fail_compact_label.empty() && label.find(fail_compact_label) != std::string_view::npos) return bad();
            if (rank == 0) wire.push_back({std::string(label) + ":payload", std::string(payload)});
            else {
                if (!replay || read == replay->size()) return bad();
                const auto& frame = (*replay)[read++];
                if (frame.label != std::string(label) + ":payload" || frame.bytes != payload) return bad();
            }
            return Failure{};
        };
        cb.execute = [&](uint32_t slot, StepPhase phase, std::string_view request, int32_t token, uint64_t pos) {
            if (rank == 0) check(!wire.empty() && (wire.back().label.ends_with(":checked") ||
                                                   wire.back().label.ends_with(":payload")),
                                 "every backend mutation follows an acknowledged exact action");
            calls.push_back({slot, phase, std::string(request), token, pos});
            if (phase == fail_phase) { failed_phase = true; return bad(); }
            auto& s = states.at(slot);
            if (phase == StepPhase::prefill || phase == StepPhase::decode) {
                check(!s.has_pending && pos == s.committed.positions, "slot's position starts at zero and advances independently");
                s.pending = s.committed;
                forward(s.pending, token);
                s.has_pending = true;
                forwards.push_back(slot);
            } else if (phase == StepPhase::commit) {
                check(s.has_pending, "commit has a full pending state");
                s.committed = s.pending; s.has_pending = false;
            } else if (phase == StepPhase::reset || phase == StepPhase::cancel) {
                check(!s.has_pending, "reset and cancel run between committed inputs");
                completed[std::string(request)] = s.committed;
                s = FakeSlot{};
            } else check(false, "scheduler executes only declared scalar phases");
            return Failure{};
        };
        if (rank == 0) cb.sample = [&](uint32_t slot, const ServeRequest& req,
                                       const std::vector<int32_t>& history, uint64_t draw, int32_t& out) {
            ++sample_count;
            const auto& state = states[slot];
            check(!state.has_pending && history == state.committed.kv,
                  "sampling sees committed complete state and exactly its own history");
            check(req.sampling.seed == UINT64_MAX, "admission retains all 64 seed bits");
            draws[req.request_id].push_back(draw);
            if (fail_sample) return bad();
            out = pick(state.committed, req, draw);
            return Failure{};
        };
        if (rank == 0) cb.emit = [&](std::string_view type, uint32_t slot, std::string_view req, int32_t token) {
            if (throw_emit) throw std::runtime_error("injected emit exception");
            check(!states[slot].has_pending && abort_count == 0, "events only escape successful commits");
            if (type == "token") output[std::string(req)].push_back(token);
            events.push_back(std::string(type) + ":" + std::string(req));
        };
        cb.abort_all = [&] { ++abort_count; for (auto& s : states) s = FakeSlot{}; };
        if (batched) cb.execute_batch = [&](const std::vector<MultiSlotBatchRow>& rows) {
            ++batch_calls;
            if (!rows.empty() && rows.front().phase == StepPhase::decode)
                largest_decode_batch = std::max(largest_decode_batch, rows.size());
            if (fail_phase == StepPhase::decode) return bad();
            for (const auto& row : rows) {
                auto& state = states[row.slot];
                check(!state.has_pending && state.committed.positions == row.position,
                      "batched inputs start at the independently committed position");
                if (row.phase == StepPhase::prefill) largest_prompt_chunk = std::max(largest_prompt_chunk, row.tokens.size());
                for (const auto token : row.tokens) forward(state.committed, token);
            }
            return Failure{};
        };
        scheduler = std::make_unique<MultiSlotScheduler>(MultiSlotConfig{rank, slots, 100, context, limit}, std::move(cb));
    }
    void drain() {
        for (int guard = 0; scheduler->active() && guard < 500; ++guard)
            check(scheduler->tick().outcome == MultiSlotOutcome::progressed, "runnable tick progresses");
        check(scheduler->active() == 0, "bounded test workload drains");
    }
};

void parity_and_followers() {
    for (uint32_t capacity : {1u, 4u, 8u}) {
        Rig root(capacity);
        std::vector<std::string> inputs;
        for (uint32_t i = 0; i < capacity; ++i) {
            inputs.push_back(generate("r" + std::to_string(i), i % 4 + 1, i % 3 + 2));
            const auto result = root.scheduler->submit(inputs.back());
            check(result.outcome == MultiSlotOutcome::accepted && result.slot == i, "admission takes lowest free slot");
        }
        root.drain();
        for (uint32_t i = 0; i < capacity; ++i) {
            const auto id = "r" + std::to_string(i);
            const auto ref = ordinary(inputs[i]);
            check(root.output[id] == ref.first && root.completed[id] == ref.second,
                  "C1/C4/C8 interleave equals independent ordinary execution of all five state families");
            std::vector<uint64_t> expected;
            for (size_t j = 0; j < ref.first.size(); ++j) expected.push_back(j);
            check(root.draws[id] == expected, "each request has its own fixed-seed draw counter");
            check(root.forwards[i] == i, "first round services arrival order once each");
        }
        bool mixed = false;
        for (size_t i = 1; i < root.calls.size(); ++i)
            if (root.calls[i].phase == StepPhase::prefill)
                for (size_t j = 0; j < i; ++j) mixed |= root.calls[j].phase == StepPhase::decode;
        if (capacity > 1) check(mixed, "long prefill interleaves with other requests' decode");
        for (uint32_t rank : {1u, 2u}) {
            Rig follower(capacity, rank, &root.wire);
            for (const auto& json : inputs)
                check(follower.scheduler->submit(json).outcome == MultiSlotOutcome::accepted, "follower admits exact same request");
            follower.drain();
            check(follower.calls == root.calls && follower.completed == root.completed &&
                  follower.scheduler->global_tick() == root.scheduler->global_tick(),
                  "all followers apply exact slot/phase/input/position sequence and full state");
            check(follower.read == root.wire.size() && follower.events.empty() && follower.sample_count == 0,
                  "followers consume every root message and never sample or emit");
        }
    }
}

void production_batching() {
    Rig rig(8, 0, nullptr, 4096, 65536, true);
    std::vector<std::string> inputs;
    for (int i = 0; i < 8; ++i) {
        inputs.push_back(generate("batch" + std::to_string(i), 33 + i, 32));
        check(rig.scheduler->submit(inputs.back()).outcome == MultiSlotOutcome::accepted, "batch admission succeeds");
    }
    rig.drain();
    for (const auto& input : inputs) {
        ServeRequest request; std::string error;
        check(parse_serve_request(input, 100, 65536, request, error), "batch reference request parses");
        const auto expected = ordinary(input);
        check(rig.output[request.request_id] == expected.first && rig.completed[request.request_id] == expected.second,
              "real row batches preserve isolated scalar reference states and output");
    }
    check(rig.largest_prompt_chunk >= 33 && rig.largest_decode_batch > 1,
          "production path emits whole prompt chunks and multiple decode slots in one callback");
    for (uint32_t rank : {1u, 2u}) {
        Rig follower(8, rank, &rig.wire, 4096, 65536, true);
        for (const auto& input : inputs) follower.scheduler->submit(input);
        follower.drain();
        check(follower.read == rig.wire.size() && follower.sample_count == 0 && follower.events.empty(),
              "production followers execute the identical batches without sampling or publishing");
    }
    Rig cancelled(4, 0, nullptr, 4096, 65536, true);
    cancelled.scheduler->submit(generate("cancelled", 80, 20));
    cancelled.scheduler->submit(generate("survives", 50, 6));
    cancelled.scheduler->tick();
    cancelled.scheduler->cancel("cancelled");
    cancelled.drain();
    check(cancelled.output["survives"] == ordinary(generate("survives", 50, 6)).first &&
          cancelled.output["cancelled"].size() == 1,
          "production cancellation retires one row while an independent slot continues");
    Rig failed(4, 0, nullptr, 4096, 65536, true);
    failed.scheduler->submit(generate("fail", 20, 3));
    failed.fail_phase = StepPhase::decode;
    check(failed.scheduler->tick().outcome == MultiSlotOutcome::failed && failed.output.empty() && failed.abort_count == 1,
          "a failed batch cannot publish a token and aborts all state");
}

void cancellation_and_reuse() {
    Rig rig(4);
    check(rig.scheduler->submit(generate("partial", 7)).slot == 0, "partial prompt admitted");
    check(rig.scheduler->submit(generate("decode", 1, 5)).slot == 1, "short prompt admitted");
    check(rig.scheduler->submit(generate("untouched", 3)).slot == 2, "untouched prompt admitted");
    check(rig.scheduler->tick().slot == 0 && rig.scheduler->tick().slot == 1, "first two slots committed");
    check(rig.scheduler->cancel("partial").outcome == MultiSlotOutcome::accepted &&
          rig.scheduler->cancel("untouched").outcome == MultiSlotOutcome::accepted,
          "cancel queues between committed turns");
    check(rig.scheduler->active() == 3 && rig.scheduler->tick().slot == 2 &&
          rig.scheduler->active() == 2 && rig.scheduler->tick().slot == 0,
          "cancellations consume their selected turns and preserve other active slot");
    check(rig.completed["partial"].positions == 1 && !rig.completed.count("untouched") &&
          rig.states[1].committed.positions == 1 && rig.output["partial"].empty(),
          "partial cancellation clears only selected full state and untouched worker is not cancelled");
    check(rig.scheduler->submit(generate("replacement", 2, 2)).slot == 0,
          "lowest cancelled slot is reused for a fresh identity");
    rig.drain();
    const auto ref = ordinary(generate("replacement", 2, 2));
    check(rig.completed["replacement"] == ref.second && rig.output["replacement"] == ref.first,
          "reused slot has fresh histories, positions, state and sampling counter");
    check(rig.scheduler->submit(generate("partial")).outcome == MultiSlotOutcome::rejected &&
          rig.scheduler->cancel("partial").outcome == MultiSlotOutcome::rejected && !rig.scheduler->failed(),
          "retired identity cannot be reused or affect a new occupant");
    Rig decode(4);
    decode.scheduler->submit(generate("cancel_decode", 1, 5));
    decode.scheduler->submit(generate("survivor", 2, 2));
    decode.scheduler->tick(); decode.scheduler->tick(); decode.scheduler->tick();
    decode.scheduler->cancel("cancel_decode");
    decode.drain();
    check(decode.output["cancel_decode"].size() == 2 && decode.completed["cancel_decode"].positions == 2 &&
          decode.output["survivor"] == ordinary(generate("survivor", 2, 2)).first,
          "decode cancellation preserves prior committed outputs and ready peer continues");
}

void admission_bounds() {
    for (const uint32_t slots : {1u, 4u, 8u}) {
        Rig boundary(slots, 0, nullptr, 4096, 65792);
        check(boundary.scheduler->submit(generate("overflow", 65536, 257)).outcome == MultiSlotOutcome::rejected &&
              boundary.calls.empty() && boundary.scheduler->active() == 0,
              "C1/C4/C8 output overflow rejects before slot state changes");
        check(boundary.scheduler->submit(generate("boundary", 65536, 256)).outcome == MultiSlotOutcome::accepted &&
              boundary.scheduler->active() == 1 && boundary.scheduler->request(0) == "boundary",
              "C1/C4/C8 scheduler admits exact 64K prompt plus 256 outputs without truncation");
    }
    bool bad_context = false;
    try { Rig invalid(1, 0, nullptr, 4096, kDistributedMaxContext + 1); }
    catch (const std::invalid_argument&) { bad_context = true; }
    check(bad_context, "scheduler rejects context above bounded capacity");
    Rig rig(1, 0, nullptr, 2);
    check(rig.scheduler->submit("{}").outcome == MultiSlotOutcome::rejected &&
          rig.scheduler->submit(generate("too_long", 65536, 1)).outcome == MultiSlotOutcome::rejected &&
          rig.scheduler->active() == 0 && rig.calls.empty(), "malformed or over-context admission does not mutate slot state");
    Rig line_limit(1, 0, nullptr, 4096, 65792);
    auto large_request = "{\"type\":\"generate\",\"request_id\":\"large\",\"ids\":[1],\"max_new\":3,"
                         "\"sampling\":{\"stop\":\"" + std::string((1u << 20), 'x') + "\"}}";
    check(large_request.size() > (1u << 20) &&
          line_limit.scheduler->submit(large_request).outcome == MultiSlotOutcome::accepted,
          "multislot admission accepts valid requests above the former one MiB limit");
    check(line_limit.scheduler->submit(std::string(kDistributedMaxLineBytes + 1, 'x')).outcome ==
              MultiSlotOutcome::rejected,
          "multislot admission rejects a request at 4 MiB plus one byte");
    rig.scheduler->submit(generate("a", 1, 1));
    check(rig.scheduler->submit(generate("overflow")).outcome == MultiSlotOutcome::rejected &&
          rig.scheduler->submit(generate("a")).outcome == MultiSlotOutcome::rejected &&
          !rig.scheduler->failed() && rig.scheduler->request(0) == "a", "busy and duplicate rejection keeps session alive");
    rig.drain();
    rig.scheduler->submit(generate("b", 1, 1)); rig.drain();
    check(rig.scheduler->submit(generate("overflow")).outcome == MultiSlotOutcome::rejected &&
          rig.scheduler->active() == 0, "retired identity storage has explicit bounded session budget");
    check(!rig.scheduler->request(8) && rig.scheduler->tick().outcome == MultiSlotOutcome::idle,
          "invalid slot query and idle tick do no work");
    for (uint32_t count : {0u, 9u}) {
        bool threw = false;
        try { Rig invalid(count); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "capacity outside one through eight is rejected");
    }
    Rig large(1);
    const auto json = generate("large", 20000, 1);
    check(large.scheduler->submit(json).outcome == MultiSlotOutcome::accepted && large.wire.size() > 5,
          "large exact public admission uses multiple bounded frames");
    large.scheduler->cancel("large"); large.drain();
}

void failures_withhold_output() {
    for (int mode = 0; mode < 7; ++mode) {
        Rig rig(4);
        rig.scheduler->submit(generate("victim", 1, 1));
        rig.scheduler->submit(generate("peer", 4, 2));
        if (mode == 0) rig.fail_phase = StepPhase::prefill;
        if (mode == 1) rig.fail_phase = StepPhase::commit;
        if (mode == 2) rig.fail_sample = true;
        if (mode == 3) rig.fail_label = "sampled-v1";
        if (mode == 4) rig.fail_label = ":publish:";
        if (mode == 5) rig.fail_phase = StepPhase::reset;
        if (mode == 6) rig.fail_label = ":select:";
        check(rig.scheduler->tick().outcome == MultiSlotOutcome::failed && rig.events.empty() &&
              rig.scheduler->active() == 0 && rig.abort_count == 1,
              "forward/commit/sample/broadcast/publish/reset/selection failure invalidates whole group and withholds output");
        const auto before = rig.calls.size(), frames = rig.wire.size();
        check(rig.scheduler->tick().outcome == MultiSlotOutcome::failed &&
              rig.scheduler->submit(generate("later")).outcome == MultiSlotOutcome::failed &&
              rig.scheduler->cancel("peer").outcome == MultiSlotOutcome::failed &&
              rig.calls.size() == before && rig.wire.size() == frames && rig.abort_count == 1,
              "failed group accepts no subsequent commands or output and aborts only once");
        for (const auto& s : rig.states) check(s.committed == ModelState{} && !s.has_pending, "abort discards every slot state");
    }
    Rig thrown(4);
    thrown.scheduler->submit(generate("throw", 1, 2));
    thrown.scheduler->submit(generate("peer", 4, 2));
    thrown.throw_emit = true;
    check(thrown.scheduler->tick().outcome == MultiSlotOutcome::failed && thrown.events.empty() &&
          thrown.abort_count == 1 && thrown.scheduler->active() == 0 &&
          thrown.scheduler->tick().outcome == MultiSlotOutcome::failed,
          "emit exception aborts every slot and suppresses all subsequent events");
    Rig cancel_failed(4);
    cancel_failed.scheduler->submit(generate("partial", 3, 1));
    cancel_failed.scheduler->submit(generate("peer", 3, 1));
    cancel_failed.scheduler->tick();
    cancel_failed.scheduler->cancel("partial");
    cancel_failed.scheduler->tick();
    cancel_failed.fail_phase = StepPhase::cancel;
    check(cancel_failed.scheduler->tick().outcome == MultiSlotOutcome::failed &&
          cancel_failed.events.empty() && cancel_failed.abort_count == 1 && cancel_failed.scheduler->active() == 0,
          "failed partial-prefill cancellation invalidates all slots before any output");
}

void disagreements() {
    Rig root(4);
    const auto json = generate("a", 2, 2);
    root.scheduler->submit(json); root.scheduler->tick();
    {
        Rig different(8, 1, &root.wire);
        check(different.scheduler->submit(json).outcome == MultiSlotOutcome::failed && different.calls.empty(),
              "configuration disagreement fails before admission mutation");
    }
    {
        Rig different(4, 1, &root.wire);
        check(different.scheduler->submit(generate("b", 2, 2)).outcome == MultiSlotOutcome::failed &&
              different.calls.empty() && different.abort_count == 1, "different public admission fails before any model mutation");
    }
    {
        auto wire = root.wire;
        for (auto& frame : wire) if (frame.label.find(":select:chunk=") != std::string::npos) {
            frame.bytes = "tick=0:cursor=0:selected=1"; break;
        }
        Rig different(4, 1, &wire);
        different.scheduler->submit(json);
        check(different.scheduler->tick().outcome == MultiSlotOutcome::failed && different.calls.empty(),
              "different root selection fails against exact local round robin state");
    }
    {
        auto wire = root.wire;
        for (auto& frame : wire) if (frame.label.find(":execute:chunk=") != std::string::npos) {
            frame.bytes.back() = '9'; break;
        }
        Rig different(4, 2, &wire);
        different.scheduler->submit(json);
        check(different.scheduler->tick().outcome == MultiSlotOutcome::failed && different.calls.empty(),
              "position/token/phase envelope disagreement fails before execution");
    }
    {
        Rig stale(4, 1, &root.wire);
        stale.scheduler->submit(json); stale.scheduler->tick();
        stale.read = 0;
        check(stale.scheduler->tick().outcome == MultiSlotOutcome::failed, "replayed stale action sequence is terminal");
    }
    {
        Rig divergent(4, 1, &root.wire);
        divergent.scheduler->submit(json);
        check(divergent.scheduler->cancel("a").outcome == MultiSlotOutcome::failed,
              "cancel versus tick command ordering disagreement is terminal");
    }
}
void speculative_batching() {
    std::vector<int32_t> published;
    size_t steps=0,resets=0;
    size_t compact_publish_calls=0;
    StepPhase fail_reset=StepPhase::abort;
    MultiSlotCallbacks cb;
    cb.broadcast=[](std::string_view,std::string&){return Failure{};};
    cb.agree_payload=[&](std::string_view label,std::string_view) {
        if(label.find("spec-publish-batch")!=std::string_view::npos)++compact_publish_calls;
        return Failure{};
    };
    cb.abort_all=[]{};
    cb.execute=[&](uint32_t,StepPhase phase,std::string_view,int32_t,uint64_t) {
        if(phase==StepPhase::reset)++resets;
        if(phase==fail_reset)return bad();
        return Failure{};
    };
    cb.execute_batch=[](const auto&){return Failure{};};
    cb.sample=[](uint32_t,const ServeRequest&,const auto&,uint64_t,int32_t& token){token=4;return Failure{};};
    cb.emit=[&](std::string_view kind,uint32_t,std::string_view,int32_t token){if(kind=="token")published.push_back(token);};
    cb.speculate=[&](const std::vector<MultiSlotProposal>& rows,std::vector<MultiSlotAccepted>& out) {
        ++steps;check(rows.size()==1,"speculative adapter receives active decode rows");
        const auto& r=rows.front();check(r.feedback==4&&r.draw==1&&r.history.size()==2,"proposal preserves committed request state");
        out={{{4,5,6},{5,6,7}}};return Failure{};
    };
    MultiSlotScheduler scheduler({0,1,100,32,4096,64},cb);
    check(scheduler.submit(generate("spec",2,4)).outcome==MultiSlotOutcome::accepted,"speculative request admitted");
    check(scheduler.tick().outcome==MultiSlotOutcome::progressed,"prompt completes before speculative decode");
    check(scheduler.tick().outcome==MultiSlotOutcome::progressed&&steps==1&&resets==1&&scheduler.active()==0&&
        compact_publish_calls==1&&published==std::vector<int32_t>({4,5,6,7}),
        "whole accepted prefix uses one batch agreement and publishes after successful commit and reset");
    published.clear();
    cb.speculate=[](const auto&,std::vector<MultiSlotAccepted>& out){out={{{4,99},{5,6}}};return Failure{};};
    MultiSlotScheduler invalid_chain({0,1,100,32,4096,64},cb);
    invalid_chain.submit(generate("invalid",2,4));invalid_chain.tick();
    check(invalid_chain.tick().outcome==MultiSlotOutcome::failed&&published==std::vector<int32_t>({4}),
        "invalid speculative chain publishes no prefix tokens");

    std::vector<uint32_t> batch_sizes;
    cb.speculate=[&](const std::vector<MultiSlotProposal>& rows,std::vector<MultiSlotAccepted>& out) {
        batch_sizes.push_back(rows.size());
        out.clear();
        for(const auto& row:rows) out.push_back({{row.feedback},{5}});
        return Failure{};
    };
    MultiSlotScheduler batch({0,2,100,32,4096,64},cb);
    batch.submit(generate("spec_a",2,3));
    batch.submit(generate("spec_b",2,3));
    for(int i=0;i<3;++i) batch.tick();
    const auto publish_agreements_before_batch=compact_publish_calls;
    published.clear();
    check(batch.tick().outcome==MultiSlotOutcome::progressed,
          "two-row speculative test progresses");
    check(!batch_sizes.empty()&&batch_sizes.back()==2,
          "speculative adapter receives two rows in one batch");
    check(compact_publish_calls==publish_agreements_before_batch+1,
          "two speculative slots use one batch publish agreement");
    check(published.size()==2, "two-row speculative batch publishes one token per slot");

    published.clear();
    resets=0;
    fail_reset=StepPhase::reset;
    MultiSlotScheduler reset_failure({0,2,100,32,4096,64},cb);
    reset_failure.submit(generate("reset_a",2,3));
    reset_failure.submit(generate("reset_b",2,3));
    for(int i=0;i<3;++i) reset_failure.tick();
    published.clear();
    reset_failure.tick();
    check(reset_failure.failed(), "reset failure aborts the speculative batch");
    check(published.empty(), "reset failure withholds every token in the speculative batch");
}

void compact_agreement() {
    Rig root(1, 0, nullptr, 4096, 65536, false, true);
    const auto request = generate("compact", 1, 1);
    check(root.scheduler->submit(request).outcome == MultiSlotOutcome::accepted &&
          root.compact_calls == 2 && root.wire.size() == 2,
          "small submit and admission each use one compact agreement");
    Rig follower(1, 1, &root.wire, 4096, 65536, false, true);
    check(follower.scheduler->submit(request).outcome == MultiSlotOutcome::accepted &&
          follower.read == root.wire.size() && follower.compact_calls == 2,
          "follower validates exact compact payloads in the same sequence");

    Rig mismatch(1, 1, &root.wire, 4096, 65536, false, true);
    check(mismatch.scheduler->submit(generate("different", 1, 1)).outcome == MultiSlotOutcome::failed &&
          mismatch.abort_count == 1 && mismatch.scheduler->active() == 0,
          "compact agreement mismatch aborts before admission mutates slot state");

    Rig failed(1, 0, nullptr, 4096, 65536, false, true);
    failed.fail_compact_label = ":admission";
    check(failed.scheduler->submit(request).outcome == MultiSlotOutcome::failed &&
          failed.abort_count == 1 && failed.scheduler->active() == 0,
          "compact agreement failure aborts before admission mutates slot state");

    Rig large(1, 0, nullptr, 4096, 65536, false, true);
    const std::string large_request =
        "{\"type\":\"generate\",\"request_id\":\"large\",\"ids\":[1],\"max_new\":1,"
        "\"sampling\":{\"stop\":\"" + std::string(25u * 1024u, 'x') + "\"}}";
    check(large_request.size() > 24u * 1024u &&
          large.scheduler->submit(large_request).outcome == MultiSlotOutcome::accepted &&
          large.compact_calls == 1 && large.wire.size() > 3,
          "oversized agreement falls back to the bounded legacy fragment protocol");
}
void packed_prompt_budget() {
    std::vector<MultiSlotBatchRow> captured;
    MultiSlotCallbacks cb;
    cb.broadcast=[](std::string_view,std::string&){return Failure{};};
    cb.agree_payload=[](std::string_view,std::string_view){return Failure{};};
    cb.execute=[](uint32_t,StepPhase,std::string_view,int32_t,uint64_t){return Failure{};};
    cb.execute_batch=[&](const auto& rows){captured=rows;return Failure{};};
    cb.abort_all=[]{};
    cb.sample=[](uint32_t,const ServeRequest&,const auto&,uint64_t,int32_t& out){out=1;return Failure{};};
    MultiSlotScheduler scheduler({0,4,100,65536,4096,8,true},cb);
    scheduler.submit(generate("packed-a",13,2));scheduler.submit(generate("packed-b",9,2));
    scheduler.submit(generate("packed-c",1,2));
    check(scheduler.tick().outcome==MultiSlotOutcome::progressed&&captured.size()==3,
          "packed prompt includes every active prompt within aggregate budget");
    size_t total=0;for(const auto& row:captured)total+=row.tokens.size();
    check(total<=8&&captured[0].tokens.size()==2&&captured[1].tokens.size()==3&&captured[2].tokens.size()==1,
          "packed quota is deterministic and bounded across unequal lengths");
    check(captured[0].position==0&&captured[1].position==0&&captured[2].position==0,
          "packed prompt rows preserve independent initial positions");
}
}
int main() {
    parity_and_followers();
    production_batching();
    cancellation_and_reuse();
    admission_bounds();
    failures_withhold_output();
    disagreements();
    speculative_batching();
    compact_agreement();
    packed_prompt_budget();
    std::printf("multislot: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
