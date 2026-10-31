#include "strata/distributed/speculative.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <utility>

namespace strata::distributed {
namespace {
Failure bad(std::string_view text) { return Failure::fail(FailureCode::bad_operation, text); }
void number(std::string& out, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) out.push_back(char((value >> shift) & 255));
}
bool read(std::string_view bytes, size_t& offset, uint64_t& value) {
    if (offset > bytes.size() || bytes.size() - offset < 8) return false;
    value = 0;
    for (size_t i = 0; i < 8; ++i) value = (value << 8) | uint8_t(bytes[offset++]);
    return true;
}
std::string proposal(const SpeculativeWindow& w, uint32_t context, int32_t vocab) {
    std::string out = "strata-spec-window-v1";
    number(out, w.request_id.size()); out += w.request_id;
    for (auto value : {w.window_id, w.position, w.draw_base, w.output_budget,
                       uint64_t(context), uint64_t(vocab), uint64_t(w.inputs.size())}) number(out, value);
    for (auto token : w.inputs) number(out, uint32_t(token));
    number(out, w.history.size());
    for (auto token : w.history) number(out, uint32_t(token));
    return out;
}
// Keep exact history agreement within the control transport's bounded frame size.
Failure transfer(uint32_t rank, const SpeculativeCallbacks& cb, std::string_view label, std::string& bytes) {
    constexpr size_t limit = 2u << 20, chunk_size = 24u << 10;
    std::string length = rank == 0 ? std::to_string(bytes.size()) : std::string{};
    if (auto f = cb.broadcast(std::string(label) + ":length", length)) return f;
    size_t count = 0;
    const auto parsed = std::from_chars(length.data(), length.data() + length.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != length.data() + length.size() || count > limit)
        return bad("invalid speculative envelope length");
    std::string received;
    received.reserve(count);
    for (size_t offset = 0; offset < count; offset += chunk_size) {
        const auto size = std::min(chunk_size, count - offset);
        std::string chunk = rank == 0 ? bytes.substr(offset, size) : std::string{};
        const auto tag = std::string(label) + ":bytes=" + std::to_string(count) + ":offset=" + std::to_string(offset);
        if (auto f = cb.broadcast(tag, chunk)) return f;
        if (chunk.size() != size) return bad("invalid speculative envelope chunk");
        received += chunk;
    }
    bytes.swap(received);
    return {};
}
}

Failure SpeculativeVerifier::verify(const SpeculativeWindow& w, const SpeculativeCallbacks& cb,
                                    SpeculativeResult& out) {
    out = {};
    if (failed_) return bad("speculative verifier was invalidated");
    // Every error invalidates this coordinator, including malformed decisions and callback exceptions.
    failed_ = true;
    try {
        if (rank_ >= kTpWorld || n_vocab_ <= 0 || !context_ || context_ > 65536 || !cb.execute ||
            !cb.broadcast || (rank_ == 0 && !cb.sample)) return bad("invalid speculative configuration");
        if (w.request_id.empty() || w.request_id.size() > 1024 || !w.window_id || w.window_id <= last_window_ ||
            w.inputs.empty() || w.inputs.size() > 5 || !w.output_budget || w.position >= context_ ||
            w.history.size() != w.position) return bad("invalid speculative window identity or bounds");
        const auto valid = [&](int32_t token) { return token >= 0 && token < n_vocab_; };
        if (!std::all_of(w.inputs.begin(), w.inputs.end(), valid) ||
            !std::all_of(w.history.begin(), w.history.end(), valid)) return bad("invalid speculative token");
        const size_t rows = size_t(std::min({uint64_t(w.inputs.size()), w.output_budget,
                                           uint64_t(context_) - w.position}));
        if (w.draw_base > std::numeric_limits<uint64_t>::max() - (rows - 1))
            return bad("speculative draw counter overflow");
        const auto expected = proposal(w, context_, n_vocab_);
        std::string agreed = rank_ == 0 ? expected : std::string{};
        if (auto f = transfer(rank_, cb, "spec-window-v1", agreed)) return f;
        if (agreed != expected) return bad("speculative proposal disagrees with local request/window/history");
        last_window_ = w.window_id;
        if (auto f = cb.execute(StepPhase::checkpoint, -1, w.position)) return f;
        std::vector<int32_t> picked;
        std::vector<int32_t> history = w.history;
        for (size_t row = 0; row < rows; ++row) {
            if (auto f = cb.execute(StepPhase::decode, w.inputs[row], w.position + row)) return f;
            if (auto f = cb.execute(StepPhase::commit, -1, 0)) return f;
            history.push_back(w.inputs[row]);
            if (rank_ == 0) {
                int32_t token = -1;
                if (auto f = cb.sample(history, w.draw_base + row, token)) return f;
                if (!valid(token)) return bad("target sampler returned invalid speculative token");
                picked.push_back(token);
                if (cb.record_target_row)
                    if (auto f = cb.record_target_row(row, w.inputs[row], token)) return f;
            }
        }
        size_t keep = rows;
        if (rank_ == 0) {
            for (size_t row = 0; row + 1 < rows; ++row)
                if (picked[row] != w.inputs[row + 1]) { keep = row + 1; break; }
        }
        std::string decision;
        if (rank_ == 0) {
            decision = expected;
            number(decision, rows); number(decision, keep);
            for (auto token : picked) number(decision, uint32_t(token));
        }
        if (auto f = transfer(rank_, cb, "spec-decision-v1", decision)) return f;
        if (!decision.starts_with(expected)) return bad("speculative decision belongs to another window");
        size_t offset = expected.size();
        uint64_t count = 0, retained = 0;
        if (!read(decision, offset, count) || count != rows || !read(decision, offset, retained) ||
            retained == 0 || retained > rows) return bad("invalid speculative decision extent");
        std::vector<int32_t> accepted_picks;
        for (size_t row = 0; row < rows; ++row) {
            uint64_t token = 0;
            if (!read(decision, offset, token) || token >= uint64_t(n_vocab_)) return bad("invalid speculative decision token");
            accepted_picks.push_back(int32_t(token));
        }
        if (offset != decision.size() || (rank_ == 0 && accepted_picks != picked))
            return bad("speculative decision changed target draws");
        size_t correct_keep = rows;
        for (size_t row = 0; row + 1 < rows; ++row)
            if (accepted_picks[row] != w.inputs[row + 1]) { correct_keep = row + 1; break; }
        if (retained != correct_keep) return bad("speculative decision violates exact target equality");
        keep = size_t(retained);
        if (keep < rows) {
            if (auto f = cb.execute(StepPhase::rollback, -1, w.position)) return f;
            for (size_t row = 0; row < keep; ++row) {
                if (auto f = cb.execute(StepPhase::decode, w.inputs[row], w.position + row)) return f;
                if (auto f = cb.execute(StepPhase::commit, -1, 0)) return f;
            }
        }
        // Publication waits until every rank has completed any restore and replay.
        std::string resolved = rank_ == 0 ? decision : std::string{};
        if (auto f = transfer(rank_, cb, "spec-resolved-v1", resolved)) return f;
        if (resolved != decision) return bad("speculative final agreement changed decision");
        SpeculativeResult result;
        result.tokens.assign(accepted_picks.begin(), accepted_picks.begin() + keep);
        result.consumed_inputs.assign(w.inputs.begin(), w.inputs.begin() + keep);
        result.accepted_rows = keep;
        result.accepted_drafts = keep - 1;
        result.next_feedback = result.tokens.back();
        result.next_position = w.position + keep;
        out = std::move(result);
        failed_ = false;
        return {};
    } catch (...) { return bad("speculative callback threw; session must abort"); }
}
} // namespace strata::distributed
