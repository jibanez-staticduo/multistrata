#include "strata/distributed/limits.hpp"
#include "strata/distributed/autoregressive.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <optional>
#include <set>

namespace strata::distributed {
namespace {
Failure bad(std::string_view text) { return Failure::fail(FailureCode::bad_operation, text); }

// Length is agreed before any chunk is read, so a missing/extra chunk cannot shift the next command.
Failure message(uint32_t rank, const AutoregressiveCallbacks& cb, std::string& bytes) {
    std::string length = rank == 0 ? std::to_string(bytes.size()) : std::string{};
    if (auto f = cb.broadcast("serve-message-length-v1", length)) return f;
    size_t count = 0;
    const auto parsed = std::from_chars(length.data(), length.data() + length.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != length.data() + length.size() ||
        count > kAutoregressiveMaxLine + 1) return bad("invalid serve message length");
    std::string received;
    received.reserve(count);
    for (size_t offset = 0; offset < count; offset += kAutoregressiveChunk) {
        const size_t size = std::min(kAutoregressiveChunk, count - offset);
        std::string chunk = rank == 0 ? bytes.substr(offset, size) : std::string{};
        const auto tag = "serve-message-v1:bytes=" + std::to_string(count) + ":offset=" + std::to_string(offset);
        if (auto f = cb.broadcast(tag, chunk)) return f;
        if (chunk.size() != size) return bad("invalid serve message chunk length");
        received += chunk;
    }
    bytes.swap(received);
    return {};
}
}

Failure run_autoregressive(uint32_t rank, int32_t n_vocab, uint32_t context,
                           const AutoregressiveCallbacks& cb, uint32_t operation_budget_ms) {
    if (rank >= kTpWorld || n_vocab <= 0 || context < 4 || context > kDistributedMaxContext ||
        operation_budget_ms < 100 ||
        !cb.broadcast || !cb.health || !cb.execute ||
        (rank == 0 && (!cb.poll || !cb.sample || !cb.emit)))
        return Failure::fail(FailureCode::invalid_config, "invalid autoregressive callbacks or geometry");
    if (cb.mtp_enabled && rank == 0 && (!cb.begin_request || !cb.prompt_row || !cb.propose ||
                                       !cb.record_target_row || !cb.accepted_window))
        return Failure::fail(FailureCode::invalid_config, "MTP service requires all root adapter hooks");
    SpeculativeVerifier verifier(rank, n_vocab, context);
    uint64_t window_id = 0;
    std::optional<ServeRequest> active;
    std::set<std::string> retired, cancelled;
    std::vector<int32_t> history;
    size_t prompt = 0;
    uint32_t generated = 0;
    int32_t feedback = -1;
    bool worker_active = false;
    RequestTiming timing;
    uint64_t admitted_ns = 0;
    bool timing_active = false, queue_recorded = false;
    const bool timing_enabled = rank == 0 && bool(cb.emit_timing);
    const auto now_ns = [&]() {
        if (cb.now_ns) return cb.now_ns();
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    auto emit = [&](std::string_view type, std::string_view request = {}, int32_t token = -1) {
        if (rank != 0) return;
        const auto state = active ? "busy" : "ready";
        if ((type == "done" || type == "cancelled") && cb.emit_timing)
            cb.emit_timing(type, request, token, state, timing);
        else if (cb.emit) cb.emit(type, request, token, state);
    };
    const auto timed = [&](auto&& operation) -> std::pair<Failure, uint64_t> {
        if (!timing_enabled) return {operation(), 0};
        const auto start = now_ns();
        auto result = operation();
        const auto end = now_ns();
        return {std::move(result), end >= start ? end - start : 0};
    };
    const auto mark_started = [&]() {
        if (timing_enabled && timing_active && !queue_recorded) {
            const auto started_ns = now_ns();
            timing.queue_ns = started_ns >= admitted_ns ? started_ns - admitted_ns : 0;
            queue_recorded = true;
        }
    };
    const auto time_draft_operation = [&](auto&& operation) -> Failure {
        if (!timing_enabled) return operation();
        const auto start = now_ns();
        auto result = operation();
        const auto end = now_ns();
        if (!result && end >= start) timing.mtp_draft_ns += end - start;
        return result;
    };
    // READY is published only after a real three-rank round trip and local transport health checks.
    if (auto f = cb.health()) return f;
    std::string initial = rank == 0 ? "ready" : std::string{};
    if (auto f = cb.broadcast("serve-ready-v1", initial)) return f;
    if (initial != "ready") return bad("invalid serve readiness marker");
    emit("ready");
    for (;;) {
        if (auto f = cb.health()) return f;
        std::string action;
        if (rank == 0) {
            bool eof = false;
            // Followers already wait inside the next broadcast while rank zero polls stdin.
            const uint32_t idle_wait = std::min(1000u, operation_budget_ms / 4);
            if (auto f = cb.poll(active ? 0 : idle_wait, action, eof)) return f;
            if (action.size() > kAutoregressiveMaxLine) return bad("serve input exceeds one MiB");
            action = eof && action.empty() ? "S" : action.empty() ? "H" : "J" + action;
        }
        if (auto f = message(rank, cb, action)) return f;
        if (action.empty()) return bad("empty serve action");
        if (action == "S") return {};
        if (action == "H") {
            emit("health");
        } else if (action.front() == 'J') {
            ServeRequest request;
            std::string error;
            if (!parse_serve_request(std::string_view(action).substr(1), n_vocab, context, request, error))
                return bad("invalid serve request: " + error);
            if (request.kind == ServeRequest::Kind::shutdown) return {};
            if (request.kind == ServeRequest::Kind::cancel) {
                if (active && request.request_id == active->request_id) {
                    if (worker_active) if (auto f = cb.execute(StepPhase::cancel, request.request_id, -1, 0)) return f;
                    retired.insert(request.request_id);
                    cancelled.insert(request.request_id);
                    active.reset(); worker_active = false; feedback = -1; history.clear(); timing_active = false;
                    emit("cancelled", request.request_id);
                } else if (retired.contains(request.request_id)) {
                    // A stream consumer can close on its final token before it reads done. Acknowledge its
                    // queued cancellation once, without resetting the idle worker or another active request.
                    if (cancelled.insert(request.request_id).second) emit("cancelled", request.request_id);
                } else return bad("cancel does not name an active or retired request");
            } else {
                if (active || retired.contains(request.request_id)) return bad("busy or reused serve request identity");
                active = std::move(request); prompt = 0; generated = 0; feedback = -1; history.clear();
                timing = {};
                timing.prompt_tokens = active->ids.size();
                admitted_ns = now_ns();
                timing_active = true; queue_recorded = false;
                if (cb.mtp_enabled && rank == 0)
                    if (auto f = time_draft_operation([&] { return cb.begin_request(*active); })) return f;
            }
        } else return bad("unknown serve action");
        if (!active) continue;
        const auto& request = *active;
        const bool prefill = prompt < request.ids.size();
        if (cb.mtp_enabled && !prefill) {
            const int32_t current_feedback = feedback;
            if (current_feedback < 0) return bad("MTP decode has no sampled feedback");
            std::vector<int32_t> drafts;
            std::string proposal;
            if (rank == 0) {
                if (auto f = time_draft_operation([&] {
                        return cb.propose(history, current_feedback, generated, drafts);
                    })) return f;
                if (drafts.size() > 4) return bad("MTP producer exceeded four drafts");
                for (auto token : drafts) {
                    if (token < 0 || token >= n_vocab) return bad("MTP producer returned invalid token");
                    for (int shift : {24, 16, 8, 0}) proposal.push_back(char((uint32_t(token) >> shift) & 255));
                }
            }
            if (auto f = message(rank, cb, proposal)) return f;
            if (proposal.size() > 16 || proposal.size() % 4) return bad("invalid MTP proposal frame");
            drafts.clear();
            for (size_t offset = 0; offset < proposal.size(); offset += 4) {
                uint32_t token = 0;
                for (size_t byte = 0; byte < 4; ++byte) token = (token << 8) | uint8_t(proposal[offset + byte]);
                if (token >= uint32_t(n_vocab)) return bad("invalid MTP proposal ID");
                drafts.push_back(int32_t(token));
            }
            SpeculativeWindow window{request.request_id, ++window_id, history.size(), generated,
                                      {current_feedback}, history, request.max_new - generated};
            window.inputs.insert(window.inputs.end(), drafts.begin(), drafts.end());
            SpeculativeCallbacks spec;
            spec.broadcast = cb.broadcast;
            spec.execute = [&](StepPhase phase, int32_t token, uint64_t position) {
                mark_started();
                const auto [result, elapsed] = timed([&] {
                    return cb.execute(phase, request.request_id, token, position);
                });
                if (!result && rank == 0) {
                    if (phase == StepPhase::decode || phase == StepPhase::commit) timing.decode_ns += elapsed;
                    else if (phase == StepPhase::checkpoint) timing.mtp_checkpoint_ns += elapsed;
                    else if (phase == StepPhase::rollback) timing.mtp_rollback_ns += elapsed;
                }
                return result;
            };
            spec.sample = [&](const auto& tail, uint64_t draw, int32_t& token) {
                const auto [result, elapsed] = timed([&] { return cb.sample(request, tail, draw, token); });
                if (!result && rank == 0) timing.sampling_ns += elapsed;
                return result;
            };
            if (rank == 0) {
                spec.record_target_row = [&](size_t row, int32_t input, int32_t next) {
                    return time_draft_operation([&] { return cb.record_target_row(row, input, next); });
                };
            }
            SpeculativeResult result;
            if (auto f = verifier.verify(window, spec, result)) return f;
            if (rank == 0)
                if (auto f = time_draft_operation([&] { return cb.accepted_window(window, result); })) return f;
            // Finish root's accepted-row bookkeeping before any output can escape.
            std::string ready = rank == 0 ? "accepted" : std::string{};
            if (auto f = cb.broadcast("serve-mtp-accepted-v1", ready)) return f;
            if (ready != "accepted") return bad("MTP adapter did not finish accepted catch-up");
            history.insert(history.end(), result.consumed_inputs.begin(), result.consumed_inputs.end());
            feedback = result.next_feedback;
            generated += uint32_t(result.tokens.size());
            if (rank == 0) timing.generated_tokens = generated;
            for (auto token : result.tokens) emit("token", request.request_id, token);
            if (generated == request.max_new) {
                const auto id = request.request_id;
                if (auto f = cb.execute(StepPhase::reset, id, -1, 0)) return f;
                retired.insert(id); active.reset(); worker_active = false; feedback = -1; history.clear(); timing_active = false;
                emit("done", id);
            }
            continue;
        }
        const int32_t input = prefill ? request.ids[prompt++] : feedback;
        if (input < 0) return bad("decode has no committed sampled token");
        mark_started();
        const auto phase = prefill ? StepPhase::prefill : StepPhase::decode;
        const auto [compute, compute_ns] = timed([&] {
            return cb.execute(phase, request.request_id, input, history.size());
        });
        if (compute) return compute;
        worker_active = true;
        const auto [commit, commit_ns] = timed([&] {
            return cb.execute(StepPhase::commit, request.request_id, -1, 0);
        });
        if (commit) return commit;
        if (rank == 0) {
            if (prefill) timing.prefill_ns += compute_ns + commit_ns;
            else timing.decode_ns += compute_ns + commit_ns;
        }
        history.push_back(input);
        if (prompt < request.ids.size()) {
            if (cb.mtp_enabled && rank == 0)
                if (auto f = time_draft_operation([&] {
                        return cb.prompt_row(request.ids[prompt], history.size() - 1);
                    })) return f;
            continue;
        }
        int32_t token = -1;
        if (rank == 0) {
            const auto [sample, elapsed] = timed([&] { return cb.sample(request, history, generated, token); });
            if (sample) return sample;
            timing.sampling_ns += elapsed;
        }
        std::string picked;
        if (rank == 0) {
            for (int shift : {24, 16, 8, 0}) picked.push_back(static_cast<char>((uint32_t(token) >> shift) & 0xff));
        }
        if (auto f = cb.broadcast("serve-sampled-id-v1:request=" + request.request_id +
                                  ":draw=" + std::to_string(generated), picked)) return f;
        if (picked.size() != 4) return bad("invalid sampled token envelope");
        uint32_t value = 0;
        for (unsigned char byte : picked) value = (value << 8) | byte;
        if (value >= uint32_t(n_vocab)) return bad("sampled token is outside vocabulary");
        feedback = static_cast<int32_t>(value);
        ++generated;
        if (rank == 0) timing.generated_tokens = generated;
        emit("token", request.request_id, feedback);
        if (generated == request.max_new) {
            const auto id = request.request_id;
            if (auto f = cb.execute(StepPhase::reset, id, -1, 0)) return f;
            retired.insert(id); active.reset(); worker_active = false; feedback = -1; history.clear(); timing_active = false;
            emit("done", id);
        }
    }
}
}  // namespace strata::distributed
