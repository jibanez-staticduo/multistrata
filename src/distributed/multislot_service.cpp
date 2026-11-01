#include "strata/distributed/limits.hpp"
#include "strata/distributed/multislot_service.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <set>
#include <unordered_map>

namespace strata::distributed {
namespace {
Failure bad(std::string_view message) { return Failure::fail(FailureCode::bad_frame, message); }
Failure message(uint32_t rank, const MultiSlotCallbacks& cb, std::string& bytes) {
    constexpr size_t chunk_size = 24u << 10, limit = kDistributedMaxLineBytes + 1;
    std::string length = rank == 0 ? std::to_string(bytes.size()) : std::string{};
    if (auto f = cb.broadcast("multislot-message-length-v1", length)) return f;
    size_t count = 0;
    const auto parsed = std::from_chars(length.data(), length.data() + length.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != length.data() + length.size() || count > limit)
        return bad("invalid multislot input length");
    std::string received;
    received.reserve(count);
    for (size_t offset = 0; offset < count; offset += chunk_size) {
        const size_t size = std::min(chunk_size, count - offset);
        std::string chunk = rank == 0 ? bytes.substr(offset, size) : std::string{};
        if (auto f = cb.broadcast("multislot-message-v1:bytes=" + std::to_string(count) +
                                  ":offset=" + std::to_string(offset), chunk)) return f;
        if (chunk.size() != size) return bad("invalid multislot input chunk");
        received += chunk;
    }
    bytes.swap(received);
    return {};
}
}
Failure run_multislot_service(MultiSlotConfig config, const MultiSlotServiceCallbacks& cb,
                              uint32_t operation_budget_ms) {
    if (config.rank >= kTpWorld || (config.slots != 1 && config.slots != 4 && config.slots != 8) ||
        config.context < 4 || config.context > kDistributedMaxContext || config.n_vocab <= 0 || operation_budget_ms < 100 ||
        !cb.health || !cb.scheduler.broadcast || !cb.scheduler.execute || !cb.scheduler.abort_all ||
        (config.rank == 0 && (!cb.poll || !cb.emit || !cb.scheduler.sample)))
        return Failure::fail(FailureCode::invalid_config, "invalid multislot service configuration");
    std::set<std::string> retired, acknowledged;
    std::unordered_map<std::string, RequestTiming> timings;
    std::unordered_map<std::string, uint64_t> admitted_ns;
    std::unordered_map<std::string, bool> queue_recorded;
    std::unordered_map<std::string, StepPhase> pending_compute;
    const bool timing_enabled = config.rank == 0 && bool(cb.emit_timing);
    const auto now_ns = [&]() {
        if (cb.now_ns) return cb.now_ns();
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    auto timed = [&](auto&& operation) {
        if (!timing_enabled) return std::pair{operation(), uint64_t{0}};
        const auto start = now_ns();
        auto result = operation();
        const auto end = now_ns();
        return std::pair{std::move(result), end >= start ? end - start : uint64_t{0}};
    };
    auto callbacks = cb.scheduler;
    callbacks.emit = [&](std::string_view type, uint32_t, std::string_view request, int32_t token) {
        if (type == "done" || type == "cancelled") retired.insert(std::string(request));
        if (type == "cancelled") acknowledged.insert(std::string(request));
        if (config.rank != 0) return;
        if (type == "token" && timing_enabled) {
            auto found = timings.find(std::string(request));
            if (found != timings.end()) ++found->second.generated_tokens;
        }
        if ((type == "done" || type == "cancelled") && timing_enabled) {
            cb.emit_timing(type, request, timings[std::string(request)]);
            timings.erase(std::string(request));
            admitted_ns.erase(std::string(request));
            queue_recorded.erase(std::string(request));
            pending_compute.erase(std::string(request));
        } else if (cb.emit) cb.emit(type, request, token, "ready", {});
        if (type == "cancelled" && !timing_enabled) {
            timings.erase(std::string(request));
            admitted_ns.erase(std::string(request));
            queue_recorded.erase(std::string(request));
            pending_compute.erase(std::string(request));
        }
    };
    const auto original_execute = callbacks.execute;
    callbacks.execute = [&](uint32_t slot, StepPhase phase, std::string_view request,
                            int32_t token, uint64_t position) {
        const std::string id(request);
        if (timing_enabled && (phase == StepPhase::prefill || phase == StepPhase::decode)) {
            auto found = timings.find(id);
            if (found != timings.end() && !queue_recorded[id]) {
                const auto at = now_ns();
                const auto start = admitted_ns[id];
                found->second.queue_ns = at >= start ? at - start : 0;
                queue_recorded[id] = true;
            }
        }
        const auto [result, elapsed] = timed([&] { return original_execute(slot, phase, request, token, position); });
        if (!result && timing_enabled) {
            auto found = timings.find(id);
            if (found != timings.end()) {
                if (phase == StepPhase::prefill) found->second.prefill_ns += elapsed;
                else if (phase == StepPhase::decode) found->second.decode_ns += elapsed;
                else if (phase == StepPhase::commit) {
                    auto previous = pending_compute.find(id);
                    if (previous != pending_compute.end()) {
                        if (previous->second == StepPhase::prefill) found->second.prefill_ns += elapsed;
                        else found->second.decode_ns += elapsed;
                        pending_compute.erase(previous);
                    }
                }
            }
        }
        if (!result && timing_enabled && (phase == StepPhase::prefill || phase == StepPhase::decode))
            pending_compute[id] = phase;
        return result;
    };
    const auto original_sample = callbacks.sample;
    const auto original_speculate = callbacks.speculate;
    if(original_speculate)callbacks.speculate = [&](const std::vector<MultiSlotProposal>& requests,
                                                     std::vector<MultiSlotAccepted>& accepted) {
        const auto [result,elapsed]=timed([&]{return original_speculate(requests,accepted);});
        if(!result && timing_enabled)for(const auto& request:requests) {
            auto found=timings.find(request.request.request_id);
            if(found!=timings.end())found->second.decode_ns+=elapsed;
        }
        return result;
    };
    const auto original_batch = callbacks.execute_batch;
    if (original_batch) callbacks.execute_batch = [&](const std::vector<MultiSlotBatchRow>& rows) {
        const auto start = timing_enabled ? now_ns() : 0;
        if (timing_enabled) for (const auto& row : rows) {
            auto found = timings.find(row.request);
            if (found != timings.end() && !queue_recorded[row.request]) {
                const auto admitted = admitted_ns[row.request];
                found->second.queue_ns = start >= admitted ? start - admitted : 0;
                queue_recorded[row.request] = true;
            }
        }
        const auto [result, elapsed] = timed([&] { return original_batch(rows); });
        if (!result && timing_enabled) for (const auto& row : rows) {
            auto found = timings.find(row.request);
            if (found == timings.end()) continue;
            // All rows overlap in the same batch. Each request records its
            // observed batch wall time; these values must not be summed as GPU time.
            if (row.phase == StepPhase::prefill) found->second.prefill_ns += elapsed;
            else found->second.decode_ns += elapsed;
        }
        return result;
    };
    callbacks.sample = [&](uint32_t slot, const ServeRequest& request, const std::vector<int32_t>& history,
                           uint64_t draw, int32_t& token) {
        const auto [result, elapsed] = timed([&] { return original_sample(slot, request, history, draw, token); });
        if (!result && timing_enabled) {
            auto found = timings.find(request.request_id);
            if (found != timings.end()) found->second.sampling_ns += elapsed;
        }
        return result;
    };
    MultiSlotScheduler scheduler(config, std::move(callbacks));
    auto fail = [&](Failure f) { return scheduler.abort(std::move(f)).failure; };
    try {
        if (auto f = cb.health()) return fail(f);
        std::string ready = config.rank == 0 ? "slots=" + std::to_string(config.slots) : std::string{};
        if (auto f = cb.scheduler.broadcast("multislot-ready-v1", ready)) return fail(f);
        if (ready != "slots=" + std::to_string(config.slots)) return fail(bad("multislot ready policy mismatch"));
        if (config.rank == 0) cb.emit("ready", {}, -1, "ready", {});
        for (;;) {
            if (auto f = cb.health()) return fail(f);
            std::string action;
            if (config.rank == 0) {
                bool eof = false;
                const auto wait = scheduler.active() ? 0 : std::min(1000u, operation_budget_ms / 4);
                if (auto f = cb.poll(wait, action, eof)) return fail(f);
                if (action.size() > kDistributedMaxLineBytes) return fail(bad("multislot input exceeds 4 MiB"));
                action = eof && action.empty() ? "S" : action.empty() ? "H" : "J" + action;
            }
            if (auto f = message(config.rank, cb.scheduler, action)) return fail(f);
            if (action == "S") { scheduler.abort(Failure::fail(FailureCode::aborted, "multislot input closed")); return {}; }
            if (action != "H") {
                if (action.empty() || action.front() != 'J') return fail(bad("invalid multislot action"));
                const auto json = std::string_view(action).substr(1);
                ServeRequest request;
                std::string error;
                if (!parse_serve_request(json, config.n_vocab, config.context, request, error))
                    return fail(bad("invalid multislot request: " + error));
                if (request.kind == ServeRequest::Kind::shutdown) {
                    scheduler.abort(Failure::fail(FailureCode::aborted, "multislot shutdown")); return {};
                }
                const auto result = request.kind == ServeRequest::Kind::generate ? scheduler.submit(json) :
                                                                                 scheduler.cancel(request.request_id);
                if (result.failure) return result.failure;
                if (timing_enabled && request.kind == ServeRequest::Kind::generate &&
                    result.outcome == MultiSlotOutcome::accepted) {
                    auto [it, inserted] = timings.try_emplace(request.request_id);
                    if (inserted) {
                        it->second.prompt_tokens = request.ids.size();
                        admitted_ns[request.request_id] = now_ns();
                        queue_recorded[request.request_id] = false;
                    }
                }
                if (result.outcome == MultiSlotOutcome::rejected && config.rank == 0) {
                    if (request.kind == ServeRequest::Kind::cancel && retired.contains(request.request_id)) {
                        if (acknowledged.insert(request.request_id).second)
                            cb.emit("cancelled", request.request_id, -1, "ready", {});
                    } else {
                        if (retired.size() >= config.request_limit && !retired.contains(request.request_id))
                            return fail(bad("multislot terminal identity limit reached"));
                        retired.insert(request.request_id);
                        cb.emit("rejected", request.request_id, -1, "ready", result.reason);
                    }
                }
            }
            // Admit the bounded burst already queued before packing prompt rows.
            if(config.packed_prefill&&action!="H"&&scheduler.active()<config.slots)continue;
            const auto result = scheduler.tick();
            if (result.failure) return result.failure;
            if (config.rank == 0) cb.emit("health", {}, -1, scheduler.active() ? "busy" : "ready", {});
        }
    } catch (...) { return fail(Failure::fail(FailureCode::comm_failed, "multislot service callback threw")); }
}
} // namespace strata::distributed
