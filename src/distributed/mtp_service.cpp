#include "strata/distributed/mtp_service.hpp"

#include "strata/core/mtp.hpp"
#include "strata/core/native_head.hpp"

#include <array>
#include <limits>

namespace strata::distributed {
namespace {
Failure bad(std::string_view text) { return Failure::fail(FailureCode::bad_operation, text); }
}
struct MtpService::Impl {
    struct Rows {
        float* data = nullptr;
        ~Rows() { if (data) cudaFree(data); }
    } storage;
    float* rows = nullptr; // Borrowed by drafter, whose destructor runs before storage.
    core::MtpDrafter drafter;
    const core::ModelGeometry* geometry = nullptr;
    core::SessionState* state = nullptr;
    cudaStream_t model_stream = nullptr;
    size_t row_bytes = 0;
    uint32_t max_drafts = 0;
    bool initialized = false, active = false, first = true;
    uint64_t prompt_len = 0, pending_position = 0;
    std::vector<int32_t> pending_picks;
    size_t recorded = 0;
    ~Impl() {
        // All public methods complete the drafter's work before returning.
        std::string ignored;
        drafter.idle(ignored);
    }
    Failure stage(size_t row) {
        if (!initialized || row > max_drafts || !state->R) return bad("invalid MTP residual row");
        const auto status = cudaMemcpyAsync(reinterpret_cast<char*>(rows) + row * row_bytes,
                                             state->R, row_bytes, cudaMemcpyDeviceToDevice, model_stream);
        if (status != cudaSuccess) return bad(cudaGetErrorString(status));
        // The drafter has its own stream. Complete this copy before giving it the borrowed rows.
        const auto done = cudaStreamSynchronize(model_stream);
        return done == cudaSuccess ? Failure{} : bad(cudaGetErrorString(done));
    }
};
MtpService::MtpService() : impl_(std::make_unique<Impl>()) {}
MtpService::~MtpService() = default;

bool MtpService::initialize(const std::string& runtime, uint32_t drafts, const core::ModelGeometry& geometry,
                           core::SessionState& state, const core::WeightTable& weights,
                           const core::NativeHead& head, void* model_stream, std::string& err) {
    auto& p = *impl_;
    if (p.initialized || drafts < 1 || drafts > 4 || !model_stream || state.max_cells < 4 ||
        state.max_cells > 65536 || geometry.hc_dim() <= 0) { err = "invalid MTP service configuration"; return false; }
    p.geometry = &geometry; p.state = &state; p.max_drafts = drafts;
    p.model_stream = static_cast<cudaStream_t>(model_stream);
    p.row_bytes = size_t(geometry.hc_dim()) * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&p.rows), (drafts + 1) * p.row_bytes) != cudaSuccess) {
        err = "MTP target residual rows do not fit"; return false;
    }
    p.storage.data = p.rows;
    // A full resident window avoids draft KV wraparound for this first integration.
    if (!p.drafter.load(runtime, geometry, state, int(drafts + 1), err, state.max_cells) ||
        !p.drafter.bind(weights, &head, p.rows, err)) return false;
    p.drafter.set_max_drafts(int(drafts));
    p.drafter.set_chain_rule({});
    if (!p.drafter.prepare_prefill(err) || !p.drafter.reserve_prefill(1, err)) return false;
    // Materialize ordinary round/step graphs before READY, then discard the dummy draft state.
    if (cudaMemset(p.rows, 0, (drafts + 1) * p.row_bytes) != cudaSuccess) { err = "MTP warmup rows failed"; return false; }
    std::array<int32_t, 5> tokens{};
    std::array<int32_t, 4> warm{};
    for (int rows = 1; rows <= int(drafts + 1) && rows <= state.max_cells; ++rows) {
        int produced = 0;
        if (!p.drafter.draft(rows, tokens.data(), 0, rows - 1, warm.data(), err,
                              nullptr, 0.0f, &produced)) return false;
    }
    core::qsa_state_zero(p.drafter.kv_state_rw(), geometry, p.drafter.stream());
    p.drafter.set_cell_floor(0);
    p.drafter.set_prompt_len(0);
    if (!p.drafter.idle(err)) return false;
    p.initialized = true;
    return true;
}
Failure MtpService::begin_request(const ServeRequest& request) {
    auto& p = *impl_;
    if (!p.initialized || request.ids.empty() || request.ids.size() > uint64_t(p.state->max_cells))
        return bad("invalid MTP request");
    core::qsa_state_zero(p.drafter.kv_state_rw(), *p.geometry, p.drafter.stream());
    p.drafter.set_prompt_len(int64_t(request.ids.size()));
    p.drafter.set_cell_floor(0);
    std::string err;
    if (!p.drafter.idle(err)) return bad(err);
    p.prompt_len = request.ids.size(); p.pending_picks.clear(); p.recorded = 0;
    p.active = true; p.first = true;
    return {};
}
Failure MtpService::prompt_row(int32_t next_token, uint64_t position) {
    auto& p = *impl_;
    if (!p.active || position + 1 >= p.prompt_len) return bad("invalid MTP prompt pair");
    if (auto f = p.stage(0)) return f;
    std::string err;
    if (!p.drafter.prefill(p.rows, &next_token, 1, int64_t(position), err)) return bad(err);
    return {};
}
Failure MtpService::propose(const std::vector<int32_t>& history, int32_t feedback, uint64_t draw,
                            std::vector<int32_t>& drafts) {
    drafts.clear();
    auto& p = *impl_;
    if (!p.active || history.empty() || history.size() >= uint64_t(p.state->max_cells) || !draw)
        return bad("invalid MTP proposal context");
    std::array<int32_t, 4> ids{};
    int produced = 0;
    std::string err;
    if (p.first) {
        if (history.size() != p.prompt_len) return bad("MTP first window lost prompt position");
        if (auto f = p.stage(0)) return f;
        if (!p.drafter.draft_first(int(p.max_drafts + 1), p.rows, feedback, int64_t(history.size() - 1),
                                  ids.data(), err, nullptr, 0.0f, &produced)) return bad(err);
        p.first = false;
    } else {
        if (p.pending_picks.empty() || p.pending_position + p.pending_picks.size() != history.size() ||
            p.pending_picks.back() != feedback) return bad("MTP accepted row history disagrees with next window");
        if (!p.drafter.draft(int(p.pending_picks.size()), p.pending_picks.data(), int64_t(p.pending_position),
                            int(p.pending_picks.size() - 1), ids.data(), err, nullptr, 0.0f, &produced)) return bad(err);
    }
    if (produced < 0 || produced > int(p.max_drafts)) return bad("MTP producer returned invalid draft count");
    drafts.assign(ids.begin(), ids.begin() + produced);
    p.recorded = 0;
    return {};
}
Failure MtpService::record_target_row(size_t row, int32_t, int32_t) {
    auto& p = *impl_;
    if (!p.active || row != p.recorded) return bad("MTP target row order mismatch");
    if (auto f = p.stage(row)) return f;
    ++p.recorded;
    return {};
}
Failure MtpService::accepted_window(const SpeculativeWindow& window, const SpeculativeResult& result) {
    auto& p = *impl_;
    if (!p.active || result.accepted_rows == 0 || result.accepted_rows > p.recorded ||
        result.tokens.size() != result.accepted_rows || result.next_position != window.position + result.accepted_rows)
        return bad("invalid MTP accepted window");
    p.pending_position = window.position;
    p.pending_picks = result.tokens;
    return {};
}
uint64_t MtpService::vram_bytes() const {
    return impl_->drafter.vram_bytes() + (impl_->max_drafts + 1) * impl_->row_bytes;
}
} // namespace strata::distributed
