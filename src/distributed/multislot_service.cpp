#include "strata/distributed/limits.hpp"
#include "strata/distributed/multislot_service.hpp"

#include <algorithm>
#include <charconv>
#include <set>

namespace strata::distributed {
namespace {
Failure bad(std::string_view message) { return Failure::fail(FailureCode::bad_frame, message); }
Failure message(uint32_t rank, const MultiSlotCallbacks& cb, std::string& bytes) {
    constexpr size_t chunk_size = 24u << 10, limit = (1u << 20) + 1;
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
    auto callbacks = cb.scheduler;
    callbacks.emit = [&](std::string_view type, uint32_t, std::string_view request, int32_t token) {
        if (type == "done" || type == "cancelled") retired.insert(std::string(request));
        if (type == "cancelled") acknowledged.insert(std::string(request));
        if (config.rank == 0) cb.emit(type, request, token, "ready", {});
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
                if (action.size() > (1u << 20)) return fail(bad("multislot input exceeds one MiB"));
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
            const auto result = scheduler.tick();
            if (result.failure) return result.failure;
            if (config.rank == 0) cb.emit("health", {}, -1, scheduler.active() ? "busy" : "ready", {});
        }
    } catch (...) { return fail(Failure::fail(FailureCode::comm_failed, "multislot service callback threw")); }
}
} // namespace strata::distributed
