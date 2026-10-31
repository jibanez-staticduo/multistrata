#include "strata/distributed/autoregressive.hpp"

#include <algorithm>
#include <charconv>
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
    if (rank >= kTpWorld || n_vocab <= 0 || context < 4 || context > 65536 ||
        operation_budget_ms < 100 ||
        !cb.broadcast || !cb.health || !cb.execute || (rank == 0 && (!cb.poll || !cb.sample || !cb.emit)))
        return Failure::fail(FailureCode::invalid_config, "invalid autoregressive callbacks or geometry");
    std::optional<ServeRequest> active;
    std::set<std::string> retired, cancelled;
    std::vector<int32_t> history;
    size_t prompt = 0;
    uint32_t generated = 0;
    std::optional<int32_t> feedback;
    bool worker_active = false;
    auto emit = [&](std::string_view type, std::string_view request = {}, int32_t token = -1) {
        if (rank == 0) cb.emit(type, request, token, active ? "busy" : "ready");
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
                    active.reset(); worker_active = false; feedback.reset(); history.clear();
                    emit("cancelled", request.request_id);
                } else if (retired.contains(request.request_id)) {
                    // A stream consumer can close on its final token before it reads done. Acknowledge its
                    // queued cancellation once, without resetting the idle worker or another active request.
                    if (cancelled.insert(request.request_id).second) emit("cancelled", request.request_id);
                } else return bad("cancel does not name an active or retired request");
            } else {
                if (active || retired.contains(request.request_id)) return bad("busy or reused serve request identity");
                active = std::move(request); prompt = 0; generated = 0; feedback.reset(); history.clear();
            }
        } else return bad("unknown serve action");
        if (!active) continue;
        const auto& request = *active;
        const bool prefill = prompt < request.ids.size();
        const int32_t input = prefill ? request.ids[prompt++] : feedback.value_or(-1);
        if (input < 0) return bad("decode has no committed sampled token");
        if (auto f = cb.execute(prefill ? StepPhase::prefill : StepPhase::decode,
                                request.request_id, input, history.size())) return f;
        worker_active = true;
        if (auto f = cb.execute(StepPhase::commit, request.request_id, -1, 0)) return f;
        history.push_back(input);
        if (prompt < request.ids.size()) continue;
        int32_t token = -1;
        if (rank == 0) if (auto f = cb.sample(request, history, generated, token)) return f;
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
        emit("token", request.request_id, *feedback);
        if (generated == request.max_new) {
            const auto id = request.request_id;
            if (auto f = cb.execute(StepPhase::reset, id, -1, 0)) return f;
            retired.insert(id); active.reset(); worker_active = false; feedback.reset(); history.clear();
            emit("done", id);
        }
    }
}
}  // namespace strata::distributed
