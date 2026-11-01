#include "strata/distributed/multislot_service.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
using namespace strata::distributed;
int checks = 0, failures = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
Failure injected() { return Failure::fail(FailureCode::comm_failed, "injected service failure"); }
std::string generate(std::string_view id, unsigned prompt = 1, unsigned output = 3) {
    std::string json = "{\"type\":\"generate\",\"request_id\":\"" + std::string(id) + "\",\"ids\":[";
    for (unsigned i = 0; i < prompt; ++i) json += (i ? "," : "") + std::to_string(i % 91 + 1);
    return json + "],\"max_new\":" + std::to_string(output) +
        ",\"sampling\":{\"seed\":18446744073709551615,\"repetition_penalty\":1.2}}";
}
std::string cancel(std::string_view id) {
    return "{\"type\":\"cancel\",\"request_id\":\"" + std::string(id) + "\"}";
}
int32_t pick(const std::vector<int32_t>& history, uint64_t draw) {
    uint64_t hash = 7 + draw * 719;
    for (const auto token : history) hash = hash * 131 + uint32_t(token);
    return int32_t(hash % 97);
}
std::vector<int32_t> ordinary(const std::string& json) {
    ServeRequest request;
    std::string error;
    check(parse_serve_request(json, 100, 65536, request, error), "reference parses public generate JSON");
    auto history = request.ids;
    std::vector<int32_t> output;
    for (uint64_t draw = 0; draw < request.max_new; ++draw) {
        if (draw) history.push_back(output.back());
        output.push_back(pick(history, draw));
    }
    return output;
}
struct Frame { std::string label, bytes; };
struct Event { std::string type, id, state, reason; int32_t token; };
struct Call {
    uint32_t slot; StepPhase phase; std::string id; int32_t token; uint64_t position;
    bool operator==(const Call&) const = default;
};
struct Rig {
    MultiSlotConfig config{0, 4, 100, 65536, 4096};
    std::vector<std::string> input;
    std::vector<uint32_t> waits;
    std::vector<Frame> wire;
    std::vector<Event> events;
    std::vector<Call> calls;
    std::array<std::vector<int32_t>, 8> histories;
    std::array<int32_t, 8> pending{};
    std::map<std::string, std::vector<int32_t>> output;
    size_t read = 0, polled = 0, health_calls = 0, samples = 0, aborted = 0;
    size_t forwards = 0, previous_poll_forwards = 0;
    size_t fail_health = 0;
    bool fail_poll = false, fail_sample = false, throw_poll = false;
    uint64_t clock = 100;
    std::map<std::string, RequestTiming> timings;
    StepPhase fail_phase = StepPhase::abort;
    std::string fail_label;
    const std::vector<Frame>* replay = nullptr;

    size_t count(std::string_view type, std::string_view id = {}) const {
        return size_t(std::count_if(events.begin(), events.end(), [&](const Event& event) {
            return event.type == type && (id.empty() || event.id == id);
        }));
    }
    void idle(unsigned n = 100) { input.insert(input.end(), n, ""); }
    MultiSlotServiceCallbacks callbacks() {
        MultiSlotServiceCallbacks cb;
        if (config.rank == 0) {
            cb.poll = [&](uint32_t wait, std::string& line, bool& eof) {
                waits.push_back(wait);
                check(forwards - previous_poll_forwards <= 1,
                      "service performs at most one forward between input polls");
                previous_poll_forwards = forwards;
                if (throw_poll) throw std::runtime_error("poll failure");
                if (fail_poll) return injected();
                eof = polled == input.size();
                if (!eof) line = input[polled++];
                return Failure{};
            };
            cb.emit = [&](std::string_view type, std::string_view id, int32_t token,
                          std::string_view state, std::string_view reason) {
                if (type == "ready") check(health_calls == 1 && wire.size() == 1 &&
                    wire.back().label == "multislot-ready-v1" &&
                    wire.back().bytes == "slots=" + std::to_string(config.slots),
                    "ready follows successful health and full readiness agreement");
                events.push_back({std::string(type), std::string(id), std::string(state), std::string(reason), token});
                if (type == "token") output[std::string(id)].push_back(token);
            };
            cb.now_ns = [&] { return clock; };
            cb.scheduler.sample = [&](uint32_t slot, const ServeRequest& request,
                                      const std::vector<int32_t>& history, uint64_t draw, int32_t& token) {
                ++samples;
                check(history == histories[slot], "sampler receives only its slot's committed history");
                check(request.sampling.seed == UINT64_MAX && request.sampling.penalty_repeat == 1.2f,
                      "service preserves public sampling parameters");
                if (fail_sample) return injected();
                token = pick(history, draw);
                return Failure{};
            };
        }
        cb.health = [&] { return ++health_calls == fail_health ? injected() : Failure{}; };
        cb.scheduler.broadcast = [&](std::string_view label, std::string& bytes) {
            check(bytes.size() <= 24u * 1024u, "every collective payload respects 24 KiB limit");
            if (!fail_label.empty() && label.find(fail_label) != std::string_view::npos) return injected();
            if (config.rank == 0) wire.push_back({std::string(label), bytes});
            else {
                check(bytes.empty(), "follower never supplies root broadcast bytes");
                if (!replay || read == replay->size()) return injected();
                const auto& frame = (*replay)[read++];
                if (frame.label != label) return injected();
                bytes = frame.bytes;
            }
            return Failure{};
        };
        cb.scheduler.execute = [&](uint32_t slot, StepPhase phase, std::string_view id,
                                   int32_t token, uint64_t position) {
            calls.push_back({slot, phase, std::string(id), token, position});
            if (phase == fail_phase) return injected();
            if (phase == StepPhase::prefill || phase == StepPhase::decode) {
                ++forwards;
                clock += 10;
                check(position == histories[slot].size(), "each slot advances its own position");
                pending[slot] = token;
            } else if (phase == StepPhase::commit) histories[slot].push_back(pending[slot]);
            else if (phase == StepPhase::reset || phase == StepPhase::cancel) histories[slot].clear();
            else check(false, "service executes only scheduler forward, commit and retirement phases");
            if (phase == StepPhase::commit) clock += 3;
            return Failure{};
        };
        cb.scheduler.abort_all = [&] {
            ++aborted;
            for (auto& history : histories) history.clear();
        };
        return cb;
    }
    Failure run(uint32_t budget = 60000) { return run_multislot_service(config, callbacks(), budget); }
};

void concurrent_admission_and_followers() {
    Rig root;
    for (const auto id : {"a", "b", "c", "d"}) root.input.push_back(generate(id, 4, 5));
    root.input.push_back(generate("overflow"));
    root.input.push_back(generate("a"));
    root.idle();
    root.input.push_back(generate("reuse", 2, 4));
    root.idle();
    check(!root.run(), "concurrent service closes successfully on EOF");
    check(root.count("ready") == 1 && root.aborted == 1, "ready is emitted once and EOF aborts pool once");
    check(root.count("rejected", "overflow") == 1 && root.count("rejected", "a") == 1,
          "full pool and duplicate admission reject without ending service");
    for (const auto id : {"a", "b", "c", "d"}) {
        check(root.output[id] == ordinary(generate(id, 4, 5)) && root.count("done", id) == 1,
              "interleaved output matches ordinary target history for every admitted request");
    }
    check(root.output["reuse"] == ordinary(generate("reuse", 2, 4)), "retired slot starts fresh for new request");
    for (uint32_t slot = 0; slot < 4; ++slot)
        check(std::any_of(root.calls.begin(), root.calls.end(), [=](const Call& call) { return call.slot == slot; }),
              "concurrent requests fill every configured slot");
    check(root.waits.front() == 1000 && std::find(root.waits.begin(), root.waits.end(), 0) != root.waits.end(),
          "poll blocks only when idle and stays nonblocking while any slot is active");
    check(std::any_of(root.events.begin(), root.events.end(), [](const Event& e) {
        return e.type == "health" && e.state == "busy";
    }) && root.events.back().type == "health" && root.events.back().state == "ready",
          "health tracks active work and returns ready after retirement");
    for (uint32_t rank : {1u, 2u}) {
        Rig follower;
        follower.config.rank = rank;
        follower.replay = &root.wire;
        check(!follower.run() && follower.read == root.wire.size() && follower.calls == root.calls,
              "follower replays exact root wire and same backend mutations to EOF");
        check(follower.polled == 0 && follower.samples == 0 && follower.events.empty() && follower.aborted == 1,
              "follower needs no poll, sample or emit functions");
    }
}

void cancellation_and_shutdown() {
    Rig root;
    root.input = {generate("victim", 4, 4), generate("survivor", 2, 3), cancel("victim")};
    root.idle();
    root.input.push_back(cancel("victim"));
    root.input.push_back(cancel("victim"));
    root.input.push_back(generate("finished", 1, 1));
    root.input.push_back(cancel("finished"));
    root.input.push_back(cancel("finished"));
    root.input.push_back(cancel("unknown"));
    root.input.push_back("{\"type\":\"shutdown\"}");
    check(!root.run(), "shutdown closes service cleanly");
    check(root.count("cancelled", "victim") == 1 && root.count("token", "victim") == 0,
          "partial prefill cancellation publishes acknowledgement without tokens");
    check(root.output["survivor"] == ordinary(generate("survivor", 2, 3)),
          "cancellation leaves surviving slot output identical to ordinary execution");
    check(root.count("done", "finished") == 1 && root.count("cancelled", "finished") == 1,
          "late cancellation after done is acknowledged exactly once");
    check(root.count("rejected", "unknown") == 1 && root.aborted == 1,
          "unknown cancellation is nonterminal and shutdown aborts pool once");
    Rig follower;
    follower.config.rank = 1;
    follower.replay = &root.wire;
    check(!follower.run() && follower.read == root.wire.size() && follower.calls == root.calls,
          "follower agrees active and late cancellations through clean shutdown");
    Rig active;
    active.input = {generate("active", 3, 4), "{\"type\":\"shutdown\"}"};
    check(!active.run() && active.count("token") == 0 && active.aborted == 1 && active.histories[0].empty(),
          "shutdown discards active partial history without publishing output");
    Rig eof;
    eof.input = {generate("active", 3, 4)};
    check(!eof.run() && eof.count("token") == 0 && eof.aborted == 1 && eof.histories[0].empty(),
          "EOF discards active partial history without publishing output");
}

void failures_withhold_output() {
    for (unsigned mode = 0; mode < 11; ++mode) {
        Rig rig;
        rig.input = {generate("victim", 1, 1)};
        if (mode == 0) rig.fail_health = 1;
        if (mode == 1) rig.fail_health = 2;
        if (mode == 2) rig.fail_poll = true;
        if (mode == 3) rig.throw_poll = true;
        if (mode == 4) rig.fail_phase = StepPhase::prefill;
        if (mode == 5) rig.fail_phase = StepPhase::commit;
        if (mode == 6) rig.fail_sample = true;
        if (mode == 7) rig.fail_label = "sampled-v1";
        if (mode == 8) rig.fail_phase = StepPhase::reset;
        if (mode == 9) rig.fail_label = "multislot-ready-v1";
        if (mode == 10) rig.fail_label = "multislot-message-v1";
        check(bool(rig.run()) && rig.aborted == 1 && rig.count("token") == 0 && rig.count("done") == 0,
              "health/input/forward/commit/sample/wire/reset failures abort and withhold tokens");
        if (mode == 0 || mode == 9) check(rig.events.empty(), "failed startup never emits ready");
    }
    Rig malformed;
    malformed.input = {"{}"};
    check(bool(malformed.run()) && malformed.calls.empty() && malformed.aborted == 1,
          "malformed public JSON aborts before backend mutation");
}

void budgets_chunks_and_validation() {
    Rig short_budget;
    short_budget.input = {"", generate("small", 1, 1)};
    check(!short_budget.run(100) && short_budget.waits.front() == 25 && short_budget.count("done") == 1,
          "short operation budget bounds idle polling and still services requests");
    Rig large;
    large.input = {generate("large", 20000, 1), cancel("large")};
    check(!large.run(), "large valid admission and cancellation complete");
    check(std::count_if(large.wire.begin(), large.wire.end(), [](const Frame& frame) {
        return frame.label.starts_with("multislot-message-v1:bytes=") && frame.bytes.size() == 24u * 1024u;
    }) >= 2, "large service input is divided into multiple bounded wire chunks");
    Rig oversize;
    oversize.input = {std::string(kDistributedMaxLineBytes + 1, 'x')};
    check(bool(oversize.run()) && oversize.calls.empty() && oversize.aborted == 1,
          "root line above 4 MiB aborts before scheduling or broadcasting payload");
    for (const uint32_t slots : {0u, 2u, 3u, 9u}) {
        Rig invalid;
        invalid.config.slots = slots;
        check(bool(invalid.run()) && invalid.events.empty() && invalid.wire.empty() && invalid.aborted == 0,
              "unsupported service capacity rejects before callbacks");
    }
    Rig invalid_budget;
    check(bool(invalid_budget.run(99)) && invalid_budget.health_calls == 0,
          "operation budgets below minimum reject before health");
    for (const uint32_t slots : {1u, 4u, 8u}) {
        Rig boundary;
        boundary.config.slots = slots;
        boundary.config.context = kDistributedMaxContext;
        boundary.input = {generate("boundary", 65536, 256), cancel("boundary")};
        check(!boundary.run() && boundary.count("cancelled") == 1 && boundary.count("done") == 0,
              "C1/C4/C8 admit and cancel 64K prompt plus 256 outputs at expanded capacity");
    }
    Rig invalid_context;
    invalid_context.config.context = kDistributedMaxContext + 1;
    check(bool(invalid_context.run()) && invalid_context.events.empty() && invalid_context.wire.empty(),
          "context above bounded capacity rejects before service readiness");
    for (uint32_t slots : {1u, 8u}) {
        Rig supported;
        supported.config.slots = slots;
        supported.input = {generate("supported", 1, 1)};
        check(!supported.run() && supported.count("done") == 1, "one and eight slot policies are supported");
    }
    Rig corrupt;
    corrupt.config.rank = 1;
    const std::vector<Frame> wrong_ready{{"multislot-ready-v1", "slots=8"}};
    corrupt.replay = &wrong_ready;
    check(bool(corrupt.run()) && corrupt.calls.empty() && corrupt.aborted == 1,
          "readiness disagreement aborts follower before execution");
}
void request_timing_is_per_request() {
    Rig rig;
    rig.input = {generate("blocker", 2, 3), generate("b", 1, 3), generate("c", 1, 3),
                 generate("d", 1, 3), cancel("blocker"), generate("queued", 2, 1)};
    rig.idle(100);
    auto callbacks = rig.callbacks();
    callbacks.emit_timing = [&](std::string_view type, std::string_view id, const RequestTiming& timing) {
        check(type == "done" || type == "cancelled", "timed multislot event names its terminal type");
        if (id == "queued") rig.timings[std::string(id)] = timing;
    };
    check(!run_multislot_service(rig.config, callbacks), "interleaved timed requests complete");
    const auto queued = rig.timings.find("queued");
    check(queued != rig.timings.end(),
          "completed queued request emits its own timing record");
    if (queued != rig.timings.end()) {
        check(queued->second.queue_ns >= 13 && queued->second.prefill_ns == 26 &&
              queued->second.decode_ns == 0 && queued->second.generated_tokens == 1 &&
              queued->second.prompt_tokens == 2,
              "waiting slot records neighboring work as queue but excludes it from own prefill");
    }
}
void disabled_timing_never_reads_clock() {
    Rig rig;
    rig.input = {generate("cancel-a", 3, 4), cancel("cancel-a"),
                 generate("cancel-b", 2, 4), cancel("cancel-b")};
    auto callbacks = rig.callbacks();
    callbacks.now_ns = []() -> uint64_t {
        throw std::runtime_error("clock must remain unused when timing is disabled");
    };
    check(!run_multislot_service(rig.config, callbacks) &&
          rig.count("cancelled", "cancel-a") == 1 && rig.count("cancelled", "cancel-b") == 1,
          "disabled timings neither read the clock nor disrupt repeated cancellations");
}
void timing_callback_does_not_replace_required_events() {
    Rig rig;
    auto callbacks = rig.callbacks();
    callbacks.emit = {};
    callbacks.emit_timing = [](std::string_view, std::string_view, const RequestTiming&) {};
    const auto result = run_multislot_service(rig.config, callbacks);
    check(result.code == FailureCode::invalid_config && rig.health_calls == 0,
          "timing-only callback is rejected before startup work");
}
void cancelled_multislot_reports_partial_timing() {
    Rig rig;
    rig.input = {generate("partial", 3, 4), cancel("partial")};
    auto callbacks = rig.callbacks();
    std::string terminal;
    RequestTiming captured;
    callbacks.emit_timing = [&](std::string_view type, std::string_view id, const RequestTiming& timing) {
        terminal = std::string(type);
        check(id == "partial", "multislot cancel timing keeps request identity");
        captured = timing;
    };
    check(!run_multislot_service(rig.config, callbacks), "multislot partial cancellation completes");
    check(terminal == "cancelled" && captured.prompt_tokens == 3 && captured.generated_tokens == 0 &&
          captured.prefill_ns == 13,
          "multislot cancelled terminal reports only committed partial prefill");
}
} // namespace

int main() {
    concurrent_admission_and_followers();
    cancellation_and_shutdown();
    failures_withhold_output();
    budgets_chunks_and_validation();
    request_timing_is_per_request();
    disabled_timing_never_reads_clock();
    cancelled_multislot_reports_partial_timing();
    timing_callback_does_not_replace_required_events();
    std::printf("multislot_service: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
