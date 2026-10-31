#include "strata/distributed/model_worker.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace strata::distributed {
namespace {
constexpr size_t kCommandLimit = 1024 * 1024;
Failure bad(std::string_view why) { return Failure::fail(FailureCode::bad_operation, why); }

class CanonicalWriter {
 public:
    bool number(uint64_t value) {
        if (bytes.size() > kCommandLimit - 8) return false;
        for (int shift = 56; shift >= 0; shift -= 8) bytes.push_back(char((value >> shift) & 255));
        return true;
    }
    bool field(std::string_view value) {
        if (value.size() > kCommandLimit || !number(value.size()) || value.size() > kCommandLimit - bytes.size())
            return false;
        bytes.append(value);
        return true;
    }
    std::string bytes;
};
}

Failure canonical_step_command(const StepCommand& command, std::string& out) {
    out.clear();
    CanonicalWriter w;
    bool valid = w.field("strata-model-worker-command-v1") && w.field(command.session) &&
                 w.field(command.request) && w.number(command.slot) && w.number(command.step) &&
                 w.number(static_cast<uint32_t>(command.phase)) && w.field(command.manifest_hash) &&
                 w.number(command.tokens.size());
    for (const auto token : command.tokens) valid = valid && w.number(std::bit_cast<uint32_t>(token));
    valid = valid && w.number(command.positions.size());
    for (const auto position : command.positions) valid = valid && w.number(position);
    valid = valid && w.number(command.expected_collective_schedule.size());
    for (const auto& item : command.expected_collective_schedule) {
        const auto& d = item.descriptor;
        valid = valid && w.number(d.operation.sequence) && w.number(d.operation.layer) &&
                w.number(d.operation.step) && w.number(static_cast<uint32_t>(d.kind)) &&
                w.number(static_cast<uint32_t>(d.type)) && w.number(d.elements) && w.number(d.valid_counts.size());
        for (const auto count : d.valid_counts) valid = valid && w.number(count);
        valid = valid && w.field(item.route);
    }
    if (!valid) return Failure::fail(FailureCode::oversized_frame, "step agreement exceeds one MiB");
    out = std::move(w.bytes);
    return {};
}

Failure model_forward_schedule(const StepCommand& command, const core::ModelGeometry& g, uint64_t sequence,
                               std::vector<ScheduledCollective>& out) {
    out.clear();
    if (g.n_layers != 48 || g.n_embd != 2560 || g.qsa_interval != 4 || g.n_expert_used != 10 ||
        g.n_ff != 640 || g.ssm_state_size != 128 || g.ssm_k_heads != 16 || g.ssm_v_heads != 48 ||
        g.n_head != 24 || g.head_dim != 256) return bad("unsupported C1 model schedule geometry");
    if (command.phase != StepPhase::prefill && command.phase != StepPhase::decode) return {};
    if (command.tokens.empty() || command.tokens.size() > 8) return bad("model command supports one to eight tokens");
    const uint64_t count = uint64_t(command.tokens.size()) * uint64_t(g.n_layers) * uint64_t(2 + 2 * g.n_expert_used);
    if (sequence == 0 || count > std::numeric_limits<uint64_t>::max() - sequence) return bad("model schedule sequence exhausted");
    auto append = [&](int64_t layer, std::string stage, std::vector<uint64_t> valid) {
        CollectiveDescriptor d;
        d.operation = {sequence++, uint64_t(layer), command.step};
        d.elements = uint64_t(g.n_embd);
        if (!valid.empty()) {
            d.kind = CollectiveKind::all_gather_padded;
            d.elements = *std::max_element(valid.begin(), valid.end());
            d.valid_counts = std::move(valid);
        }
        out.push_back({std::move(d), std::move(stage)});
    };
    for (size_t token = 0; token < command.tokens.size(); ++token) {
        for (int64_t layer = 0; layer < g.n_layers; ++layer) {
            const bool qsa = core::is_qsa_layer(g, layer);
            append(layer, qsa ? "qsa_gated_reference" : "gdn_values",
                   qsa ? std::vector<uint64_t>{2048, 2048, 2048} : std::vector<uint64_t>{2304, 1920, 1920});
            append(layer, qsa ? "qsa_output_rank_zero_reference" : "mixer", {});
            for (int64_t slot = 0; slot < g.n_expert_used; ++slot) {
                append(layer, "expert_hidden_" + std::to_string(slot), {256, 192, 192});
                append(layer, "expert_output_" + std::to_string(slot), {});
            }
        }
    }
    return {};
}

}  // namespace strata::distributed

#if defined(STRATA_DISTRIBUTED_CUDA)
#include "strata/distributed/model.hpp"

#include <cuda_runtime.h>
#include <thread>

namespace strata::distributed {
namespace {
Failure cuda_failure(cudaError_t result) {
    return result == cudaSuccess ? Failure{} : Failure::fail(FailureCode::async_error, cudaGetErrorString(result));
}
bool identical(const ScheduledCollective& a, const ScheduledCollective& b) {
    const auto& x = a.descriptor; const auto& y = b.descriptor;
    return x.operation.sequence == y.operation.sequence && x.operation.layer == y.operation.layer &&
           x.operation.step == y.operation.step && x.kind == y.kind && x.type == y.type &&
           x.elements == y.elements && x.valid_counts == y.valid_counts && a.route == b.route;
}
}

struct ModelWorkerBackend::Impl {
    ModelWorkerConfig config;
    ModelForward& model;
    SessionCollectives& collectives;
    Session& session;
    GuardedTransport& transport;
    float* device_logits = nullptr;
    float* host_logits = nullptr;
    cudaEvent_t completion = nullptr;
    bool initialized = false;
    bool dead = false;
    bool candidate = false;
    bool published = false;
    std::string agreed_command;
    std::string execution_receipt;

    Impl(ModelWorkerConfig c, ModelForward& m, SessionCollectives& l, Session& s, GuardedTransport& t)
        : config(std::move(c)), model(m), collectives(l), session(s), transport(t) {}
    ~Impl() {
        if (completion) cudaEventDestroy(completion);
        if (device_logits) cudaFree(device_logits);
        if (host_logits) cudaFreeHost(host_logits);
    }
    Failure wait() {
        Deadline deadline;
        if (auto f = session.operation_deadline(deadline)) return f;
        if (auto f = cuda_failure(cudaEventRecord(completion, static_cast<cudaStream_t>(config.stream)))) return f;
        for (;;) {
            const auto result = cudaEventQuery(completion);
            if (result == cudaSuccess) return transport.check_health();
            if (result != cudaErrorNotReady) return cuda_failure(result);
            if (auto f = transport.check_health()) return f;
            if (std::chrono::steady_clock::now() >= deadline.end)
                return Failure::fail(FailureCode::deadline_expired, "model completion deadline expired");
            std::this_thread::yield();
        }
    }
    Failure agree_bytes(std::string_view label, std::string_view bytes) {
        Deadline deadline;
        if (auto f = session.operation_deadline(deadline)) return f;
        constexpr size_t part_bytes = 24 * 1024;
        const size_t parts = std::max(size_t(1), (bytes.size() + part_bytes - 1) / part_bytes);
        for (size_t i = 0; i < parts; ++i) {
            const auto part = bytes.substr(std::min(i * part_bytes, bytes.size()), part_bytes);
            const auto tag = std::string(label) + ":bytes=" + std::to_string(bytes.size()) + ":part=" +
                             std::to_string(i) + "/" + std::to_string(parts);
            if (auto f = session.agree_control_payload(tag, part, deadline)) return f;
        }
        return {};
    }
    Failure finite_logits() {
        const size_t bytes = size_t(model.geometry().n_vocab) * sizeof(float);
        if (auto f = cuda_failure(cudaMemcpyAsync(host_logits, device_logits, bytes, cudaMemcpyDeviceToHost,
                                                  static_cast<cudaStream_t>(config.stream)))) return f;
        if (auto f = wait()) return f;
        if (!std::all_of(host_logits, host_logits + model.geometry().n_vocab,
                         [](float value) { return std::isfinite(value); })) return bad("nonfinite CUDA logits");
        return {};
    }
};

ModelWorkerBackend::ModelWorkerBackend(ModelWorkerConfig c, ModelForward& m, SessionCollectives& l,
                                       Session& s, GuardedTransport& t)
    : impl_(std::make_unique<Impl>(std::move(c), m, l, s, t)) {}
ModelWorkerBackend::~ModelWorkerBackend() = default;

Failure ModelWorkerBackend::plan(const StepCommand& command, std::vector<ScheduledCollective>& out) const {
    const auto& p = *impl_;
    if (p.dead) return Failure::fail(FailureCode::aborted, "model worker was invalidated");
    if (!p.config.stream || p.config.position_base != 0 || command.session != p.config.worker.session ||
        command.manifest_hash != p.config.worker.manifest_hash || p.config.worker.rank >= kTpWorld)
        return bad("model worker identity/stream/position base is invalid");
    if (command.tokens.size() != command.positions.size()) return bad("model token/position mismatch");
    for (size_t i = 0; i < command.tokens.size(); ++i) {
        if (command.tokens[i] < 0 || command.tokens[i] >= p.model.geometry().n_vocab ||
            command.positions[i] > uint64_t(std::numeric_limits<int32_t>::max()))
            return bad("model token or position outside supported range");
    }
    return model_forward_schedule(command, p.model.geometry(), p.collectives.sequence(), out);
}

Failure ModelWorkerBackend::agree(const StepCommand& command) {
    auto& p = *impl_;
    std::string bytes;
    if (auto f = canonical_step_command(command, bytes)) return f;
    if (auto f = p.agree_bytes("model-command-v1:base=0", bytes)) return f;
    p.agreed_command = std::move(bytes);
    return {};
}

Failure ModelWorkerBackend::begin(const StepCommand& command) {
    auto& p = *impl_;
    std::string bytes;
    if (auto f = canonical_step_command(command, bytes)) return f;
    if (p.dead || bytes != p.agreed_command) return bad("begin requires exact complete command agreement");
    p.published = false;
    p.collectives.step(command.step);
    p.collectives.reset_observations();
    p.collectives.expect_schedule(command.expected_collective_schedule);
    std::string err;
    if (command.phase == StepPhase::init) {
        if (p.initialized) return bad("model worker already initialized");
        const auto vocab = p.model.geometry().n_vocab;
        if (vocab <= 0 || uint64_t(vocab) > std::numeric_limits<size_t>::max() / sizeof(float)) return bad("invalid vocabulary size");
        const size_t count = size_t(vocab) * sizeof(float);
        if (auto f = cuda_failure(cudaMalloc(reinterpret_cast<void**>(&p.device_logits), count))) return f;
        if (auto f = cuda_failure(cudaHostAlloc(reinterpret_cast<void**>(&p.host_logits), count, cudaHostAllocDefault))) return f;
        if (auto f = cuda_failure(cudaEventCreateWithFlags(&p.completion, cudaEventDisableTiming))) return f;
        if (!p.model.initialize(err) || !p.model.reset(p.config.stream, err)) return bad(err);
        p.initialized = true;
    } else if (command.phase == StepPhase::prefill || command.phase == StepPhase::decode) {
        if (!p.initialized || p.candidate) return bad("forward requires initialized model without pending candidate");
        for (size_t i = 0; i < command.tokens.size(); ++i) {
            if (!p.model.token(command.tokens[i], int64_t(command.positions[i]), p.config.position_base,
                               p.device_logits, p.config.stream, err)) return bad(err);
            if (auto f = p.finite_logits()) return f;
        }
        p.candidate = true;
    } else if (command.phase == StepPhase::reset || command.phase == StepPhase::cancel) {
        if (!p.model.reset(p.config.stream, err)) return bad(err);
        p.candidate = false;
    } else if (command.phase == StepPhase::commit) {
        if (!p.candidate) return bad("model commit requires a finite candidate");
        if (auto f = p.finite_logits()) return f;
    } else return bad("unsupported model phase");
    return p.wait();
}

Failure ModelWorkerBackend::collective(const ScheduledCollective&) {
    return bad("model-driven collectives cannot be replayed by the worker");
}

Failure ModelWorkerBackend::finish(const StepCommand& command, BackendCompletion& out) {
    auto& p = *impl_;
    out = {};
    const auto& observed = p.collectives.observed();
    if (observed.size() != command.expected_collective_schedule.size() ||
        !std::equal(observed.begin(), observed.end(), command.expected_collective_schedule.begin(), identical))
        return bad("actual model execution did not complete the declared schedule");
    if (auto f = p.wait()) return f;
    // Receipt confirms completed command and logical position. It does not claim to checksum model
    // KV/GDN/QSA/PLE bytes; numerical state parity still requires the GPU qualification harness.
    const std::string receipt = "noncrypto-execution-receipt-v1:position=" + std::to_string(p.model.next_position());
    if (auto f = p.agree_bytes("model-stage-complete-v1", p.agreed_command + receipt)) return f;
    p.execution_receipt = receipt;
    if (command.phase == StepPhase::commit) {
        p.candidate = false;
        p.published = p.config.worker.rank == 0;
    }
    out = {true, true, true, receipt};
    return {};
}

void ModelWorkerBackend::abort() noexcept {
    auto& p = *impl_;
    if (p.dead) return;
    p.dead = true;
    p.published = false;
    p.candidate = false;
    p.model.invalidate();
    p.transport.abort("model worker aborted");
    p.session.abort("model worker aborted");
}
void ModelWorkerBackend::discard_session() noexcept {
    auto& p = *impl_;
    p.dead = true;
    p.published = p.candidate = false;
    p.agreed_command.clear();
    p.execution_receipt.clear();
    p.model.invalidate();
}
const float* ModelWorkerBackend::committed_logits() const {
    return impl_->published && !impl_->dead ? impl_->host_logits : nullptr;
}
size_t ModelWorkerBackend::logits_size() const { return size_t(impl_->model.geometry().n_vocab); }

}  // namespace strata::distributed
#endif
