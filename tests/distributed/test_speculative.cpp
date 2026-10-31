// CPU state oracle only. No model, drafter, transport or GPU execution.
#include "strata/distributed/speculative.hpp"
#include "strata/distributed/limits.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
using namespace strata::distributed;
int checks = 0, failures = 0;
void check(bool ok, const char* text) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL: %s\n", text); }
}
Failure error() { return Failure::fail(FailureCode::comm_failed, "injected failure"); }

struct State {
    std::array<std::array<uint8_t, 67>, 7> bytes{}; // KV/GDN/conv/indexer/PLE/R/metadata.
    uint64_t position = 0;
    bool operator==(const State&) const = default;
};
void advance(State& state, int32_t token) {
    for (size_t kind = 0; kind < state.bytes.size(); ++kind) {
        auto& bytes = state.bytes[kind];
        for (size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = uint8_t(bytes[(i + 1) % bytes.size()] * 17u + uint32_t(token) * (kind + 1) +
                               state.position * 7 + i);
    }
    ++state.position;
}
int32_t target(const std::vector<int32_t>& history, uint64_t draw, bool sampled) {
    uint64_t hash = 29;
    for (auto id : history) hash = hash * 131 + uint32_t(id);
    if (sampled) hash ^= (draw + 71) * 0x9e3779b97f4a7c15ULL;
    return int32_t(hash % 97);
}
SpeculativeWindow make_window(size_t width, size_t mismatch, bool sampled) {
    SpeculativeWindow w{"request-A", 1, 3, 8, {4}, {1, 2, 3}, 5};
    auto history = w.history;
    for (size_t row = 0; row + 1 < width; ++row) {
        history.push_back(w.inputs.back());
        const auto pick = target(history, w.draw_base + row, sampled);
        w.inputs.push_back(row == mismatch ? (pick + 1) % 97 : pick);
    }
    return w;
}

struct Frame { std::string label, bytes; };
struct Fake {
    State state, snapshot;
    bool pending = false, saved = false, sampled = false;
    uint32_t rank = 0;
    size_t calls = 0, fail_call = 0, broadcast_calls = 0, fail_broadcast = 0;
    size_t rollbacks = 0, decodes = 0, commits = 0, records = 0;
    bool throw_sample = false;
    std::vector<std::pair<std::vector<int32_t>, uint64_t>> draws;
    std::vector<State> recorded_states;
    std::vector<Frame> frames;
    size_t read_frame = 0;
    std::string corrupt_label;
    enum class Corruption { flip, trailing, length } corruption = Corruption::flip;

    Fake() { for (auto token : {1, 2, 3}) advance(state, token); }
    SpeculativeCallbacks callbacks() {
        SpeculativeCallbacks cb;
        cb.execute = [&](StepPhase phase, int32_t token, uint64_t position) -> Failure {
            ++calls;
            if (phase == StepPhase::checkpoint) {
                if (pending) return error();
                snapshot = state; saved = true;
            } else if (phase == StepPhase::decode) {
                if (pending || state.position != position) return error();
                advance(state, token); pending = true; ++decodes;
            } else if (phase == StepPhase::commit) {
                if (!pending) return error();
                pending = false; ++commits;
            } else if (phase == StepPhase::rollback) {
                if (!saved) return error();
                state = snapshot; pending = false; ++rollbacks;
            } else return error();
            return calls == fail_call ? error() : Failure{};
        };
        cb.sample = [&](const std::vector<int32_t>& history, uint64_t draw, int32_t& token) -> Failure {
            if (throw_sample) throw std::runtime_error("sample failure");
            draws.emplace_back(history, draw);
            token = target(history, draw, sampled);
            return {};
        };
        cb.record_target_row = [&](size_t row, int32_t, int32_t) -> Failure {
            if (row != records || pending) return error();
            ++records; recorded_states.push_back(state); return {};
        };
        cb.broadcast = [&](std::string_view label, std::string& bytes) -> Failure {
            ++broadcast_calls;
            if (rank == 0) frames.push_back({std::string(label), bytes});
            else {
                if (read_frame >= frames.size() || frames[read_frame].label != label) return error();
                bytes = frames[read_frame++].bytes;
            }
            if (!corrupt_label.empty() && label.starts_with(corrupt_label)) {
                if (corruption == Corruption::length) bytes = "2097153";
                else if (corruption == Corruption::trailing) bytes.push_back('x');
                else if (!bytes.empty()) bytes.back() ^= 1;
            }
            return broadcast_calls == fail_broadcast ? error() : Failure{};
        };
        return cb;
    }
};

void oracle_cases() {
    for (bool sampled : {false, true}) for (size_t width = 1; width <= 5; ++width)
        for (size_t mismatch = 0; mismatch <= width; ++mismatch)
            for (uint64_t budget : {uint64_t(1), uint64_t(3), uint64_t(5)}) {
                auto w = make_window(width, mismatch, sampled); w.output_budget = budget;
                const size_t attempted = std::min(width, size_t(budget));
                const size_t keep = std::min(attempted, mismatch + 1);
                Fake root; root.sampled = sampled;
                SpeculativeVerifier verifier(0, 97, 64);
                SpeculativeResult result;
                check(!verifier.verify(w, root.callbacks(), result), "valid speculative window resolves");
                auto history = w.history;
                std::vector<int32_t> expected;
                int32_t feedback = w.inputs[0];
                State ordinary;
                for (auto id : history) advance(ordinary, id);
                for (size_t row = 0; row < keep; ++row) {
                    history.push_back(feedback); advance(ordinary, feedback);
                    feedback = target(history, w.draw_base + row, sampled); expected.push_back(feedback);
                }
                check(result.tokens == expected && root.state == ordinary && !root.pending &&
                      result.next_position == ordinary.position && result.next_feedback == feedback,
                      "resolved tokens and every mutable state byte match ordinary autoregressive oracle");
                check(result.accepted_rows == keep && result.accepted_drafts == keep - 1 &&
                      result.consumed_inputs == std::vector<int32_t>(w.inputs.begin(), w.inputs.begin() + keep),
                      "consumed prefix excludes correction and bonus feedback");
                check(root.decodes == attempted + (keep < attempted ? keep : 0) && root.commits == root.decodes &&
                      root.rollbacks == size_t(keep < attempted), "replay executes exactly retained inputs with fresh operations");
                check(root.draws.size() == attempted && root.records == attempted, "every target row sampled and recorded before overwrite");
                auto attempted_history = w.history;
                for (size_t row = 0; row < attempted; ++row) {
                    attempted_history.push_back(w.inputs[row]);
                    check(root.draws[row].first == attempted_history && root.draws[row].second == w.draw_base + row &&
                          root.recorded_states[row].position == w.position + row + 1,
                          "target history draw index and final residual row retain logical query identity");
                }
                Fake follower; follower.rank = 2; follower.frames = root.frames;
                auto cb = follower.callbacks(); cb.sample = {}; cb.record_target_row = {};
                SpeculativeVerifier other(2, 97, 64);
                SpeculativeResult peer;
                check(!other.verify(w, cb, peer) && peer.tokens == result.tokens && follower.state == root.state &&
                      follower.draws.empty() && follower.read_frame == root.frames.size(),
                      "follower validates root decision and reconstructs exact canonical state without sampling");
            }
}

void invalid_windows() {
    const auto valid = make_window(5, 2, false);
    for (const uint32_t context : {kDistributedMaxContext, kDistributedMaxContext + 1}) {
        Fake f; SpeculativeVerifier verifier(0, 97, context); SpeculativeResult out;
        const auto failure = verifier.verify(valid, f.callbacks(), out);
        check(context == kDistributedMaxContext ? !failure && !out.tokens.empty() :
              bool(failure) && out.tokens.empty() && f.calls == 0,
              "speculative context capacity is bounded before checkpoint and output");
    }
    std::vector<SpeculativeWindow> bad;
    auto add = [&](auto change) { auto w = valid; change(w); bad.push_back(std::move(w)); };
    add([](auto& w) { w.request_id.clear(); }); add([](auto& w) { w.window_id = 0; });
    add([](auto& w) { w.inputs.clear(); }); add([](auto& w) { w.inputs.resize(6); });
    add([](auto& w) { w.output_budget = 0; }); add([](auto& w) { w.history.pop_back(); });
    add([](auto& w) { w.inputs[0] = -1; }); add([](auto& w) { w.inputs[1] = 97; });
    add([](auto& w) { w.history[0] = 97; }); add([](auto& w) { w.position = 64; });
    add([](auto& w) { w.draw_base = std::numeric_limits<uint64_t>::max(); });
    for (const auto& w : bad) {
        Fake f; SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        out.tokens = {1};
        check(bool(verifier.verify(w, f.callbacks(), out)) && out.tokens.empty() && f.calls == 0,
              "invalid bounds identity counter or token rejected without model operation or output");
    }
    {
        Fake f; SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(!verifier.verify(valid, f.callbacks(), out), "duplicate fixture succeeds once");
        const auto calls = f.calls;
        check(bool(verifier.verify(valid, f.callbacks(), out)) && out.tokens.empty() && f.calls == calls,
              "duplicate window cannot reuse checkpoint or publish twice");
    }
    {
        Fake f; SpeculativeVerifier verifier(0, 97, 5); SpeculativeResult out;
        check(!verifier.verify(valid, f.callbacks(), out) && f.records == 2 && out.next_position <= 5,
              "remaining context clips attempted window before target forward");
    }
}

void failures_withhold() {
    const auto w = make_window(5, 1, true);
    Fake successful; successful.sampled = true; SpeculativeResult result;
    SpeculativeVerifier good(0, 97, 64);
    check(!good.verify(w, successful.callbacks(), result), "failure fixture includes rollback and replay");
    for (size_t fail = 1; fail <= successful.calls; ++fail) {
        Fake f; f.sampled = true; f.fail_call = fail;
        SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, f.callbacks(), out)) && out.tokens.empty() && out.consumed_inputs.empty(),
              "failure after any checkpoint forward commit restore or replay withholds all output");
        const auto calls = f.calls;
        check(bool(verifier.verify(w, f.callbacks(), out)) && f.calls == calls && out.tokens.empty(),
              "failed verifier cannot retry mutated session");
    }
    for (size_t fail = 1; fail <= successful.broadcast_calls; ++fail) {
        Fake f; f.sampled = true; f.fail_broadcast = fail;
        SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, f.callbacks(), out)) && out.tokens.empty(),
              "proposal decision or final agreement failure withholds output");
    }
    for (std::string label : {"spec-window-v1:bytes", "spec-decision-v1:bytes", "spec-resolved-v1:bytes"}) {
        Fake f; f.corrupt_label = label;
        SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, f.callbacks(), out)) && out.tokens.empty(), "corrupt envelope cannot publish output");
    }
    {
        Fake f; f.throw_sample = true; SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, f.callbacks(), out)) && out.tokens.empty(), "callback exception withholds output");
    }
    {
        Fake f; auto cb = f.callbacks(); cb.record_target_row = [](size_t, int32_t, int32_t) { return error(); };
        SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, cb, out)) && out.tokens.empty(), "residual capture failure withholds output");
    }
    {
        Fake f; auto cb = f.callbacks(); cb.sample = [](const auto&, uint64_t, int32_t& token) { token = 97; return Failure{}; };
        SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, cb, out)) && out.tokens.empty(), "out of vocabulary target result withholds output");
    }
    for (bool changed_context : {false, true}) {
        auto altered = w;
        if (!changed_context) altered.history[0] = 4;
        Fake f; f.rank = 1; f.frames = successful.frames;
        SpeculativeVerifier verifier(1, 97, changed_context ? 65 : 64); SpeculativeResult out;
        check(bool(verifier.verify(altered, f.callbacks(), out)) && f.calls == 0 && out.tokens.empty(),
              "follower rejects differing baseline history or context before checkpoint");
    }
    for (int change = 0; change < 4; ++change) {
        Fake f; f.rank = 1; f.frames = successful.frames;
        for (auto& frame : f.frames) if (frame.label.starts_with("spec-decision-v1:bytes")) {
            if (change == 0) frame.bytes[20] ^= 1; // Bound request identity.
            else if (change == 1) frame.bytes[frame.bytes.size() - 5 * 8 - 1] = 0; // keep == 0.
            else if (change == 2) frame.bytes[frame.bytes.size() - 1] = char(127); // Invalid ID.
            else frame.bytes[frame.bytes.size() - 5 * 8 - 1] = 5; // Contradicts first mismatch.
        }
        SpeculativeVerifier verifier(1, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, f.callbacks(), out)) && out.tokens.empty(),
              "follower rejects stale identity illegal keep invalid ID or false acceptance decision");
    }
    {
        Fake f; f.corrupt_label = "spec-window-v1:length"; f.corruption = Fake::Corruption::length;
        SpeculativeVerifier verifier(0, 97, 64); SpeculativeResult out;
        check(bool(verifier.verify(w, f.callbacks(), out)) && f.calls == 0 && out.tokens.empty(),
              "oversized control frame rejected before allocation or model work");
    }
}

void consecutive_windows() {
    for (bool sampled : {false, true}) {
        Fake f; f.sampled = sampled;
        SpeculativeVerifier verifier(0, 97, 64);
        auto w = make_window(1, 0, sampled);
        std::vector<int32_t> actual, ordinary;
        auto ordinary_history = w.history;
        int32_t input = w.inputs[0];
        State reference = f.state;
        for (size_t row = 0; row < 19; ++row) {
            ordinary_history.push_back(input); advance(reference, input);
            input = target(ordinary_history, w.draw_base + row, sampled); ordinary.push_back(input);
        }
        for (uint64_t id = 1; actual.size() < ordinary.size(); ++id) {
            w.window_id = id;
            w.output_budget = ordinary.size() - actual.size();
            auto attempted_history = w.history;
            while (w.inputs.size() < 5) {
                attempted_history.push_back(w.inputs.back());
                const auto row = w.inputs.size() - 1;
                const auto pick = target(attempted_history, w.draw_base + row, sampled);
                w.inputs.push_back(row == id % 5 ? (pick + 1) % 97 : pick);
            }
            f.records = 0; f.recorded_states.clear();
            SpeculativeResult out;
            check(!verifier.verify(w, f.callbacks(), out), "consecutive speculative window resolves with increasing identity");
            actual.insert(actual.end(), out.tokens.begin(), out.tokens.end());
            w.history.insert(w.history.end(), out.consumed_inputs.begin(), out.consumed_inputs.end());
            w.inputs = {out.next_feedback};
            w.position = out.next_position; w.draw_base += out.tokens.size();
        }
        check(actual == ordinary && f.state == reference && w.history == ordinary_history,
              "multiple windows preserve ordinary seeded output history and full state across rejected draw attempts");
    }
}

void categorical_sampling_qualification() {
    // Freeze the target law and acceptance bands before observing the samples.
    constexpr size_t seeds = 10000, output_count = 12;
    std::array<size_t, 3> histogram{};
    size_t rejected = 0, accepted = 0;
    auto draw = [](uint64_t seed, uint64_t index, const std::vector<int32_t>& history) {
        uint64_t value = seed * 0x9e3779b97f4a7c15ULL + index;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        value ^= value >> 31;
        const double u = double(value >> 11) * 0x1.0p-53;
        const int shift = history.empty() ? 0 : history.back() % 3;
        return int32_t(((u < .2) ? 0 : (u < .5 ? 1 : 2)) + shift) % 3;
    };
    for (uint64_t seed = 0; seed < seeds; ++seed) {
        Fake f;
        auto callbacks = f.callbacks();
        callbacks.sample = [&](const auto& history, uint64_t index, int32_t& token) {
            token = draw(seed, index, history);
            return Failure{};
        };
        SpeculativeVerifier verifier(0, 97, 64);
        SpeculativeWindow w{"categorical", 0, 3, 0, {4}, {1, 2, 3}, output_count};
        auto reference_history = w.history;
        State reference_state = f.state;
        std::vector<int32_t> expected, actual;
        int32_t feedback = 4;
        for (size_t row = 0; row < output_count; ++row) {
            reference_history.push_back(feedback);
            advance(reference_state, feedback);
            feedback = draw(seed, row, reference_history);
            expected.push_back(feedback);
        }
        while (actual.size() < output_count) {
            ++w.window_id;
            // Deliberately biased proposals independent of the target draws.
            while (w.inputs.size() < 5) w.inputs.push_back(int32_t((seed + w.window_id) % 3));
            f.records = 0;
            f.recorded_states.clear();
            SpeculativeResult result;
            check(!verifier.verify(w, callbacks, result), "categorical MTP window completes");
            if (result.tokens.empty()) return;
            accepted += result.accepted_drafts;
            rejected += result.accepted_rows < std::min(w.inputs.size(), size_t(w.output_budget));
            actual.insert(actual.end(), result.tokens.begin(), result.tokens.end());
            w.history.insert(w.history.end(), result.consumed_inputs.begin(), result.consumed_inputs.end());
            w.inputs = {result.next_feedback};
            w.position = result.next_position;
            w.draw_base += result.tokens.size();
            w.output_budget = output_count - actual.size();
        }
        check(actual == expected && f.state == reference_state && w.history == reference_history,
              "categorical MTP preserves every seeded ordinary draw and rollback state");
        // First input is 4, so undo its history-dependent rotation.
        ++histogram[size_t((actual.front() + 2) % 3)];
    }
    check(accepted > 0 && rejected > 0, "categorical qualification exercises acceptance and rejection");
    for (size_t token = 0; token < 3; ++token) {
        const double expected = std::array{.2, .3, .5}[token];
        const double observed = double(histogram[token]) / seeds;
        check(observed > expected - .025 && observed < expected + .025,
              "categorical output law remains within frozen 2.5 percentage point band");
    }
    std::printf("categorical sampling seeds=%zu law_counts=%zu,%zu,%zu accepted_drafts=%zu rejected_windows=%zu\n",
                seeds, histogram[0], histogram[1], histogram[2], accepted, rejected);
}
}
int main() {
    oracle_cases(); invalid_windows(); failures_withhold(); consecutive_windows();
    categorical_sampling_qualification();
    std::printf("speculative CPU oracle: %d checks, %d failures (no model/GPU execution)\n", checks, failures);
    return failures ? 1 : 0;
}
