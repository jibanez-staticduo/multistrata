#include "strata/distributed/model_worker.hpp"
#include "strata/distributed/expert_operands.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>

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
    if (static_cast<uint32_t>(command.phase) > static_cast<uint32_t>(StepPhase::rollback))
        return bad("unknown step phase");
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
                               std::vector<ScheduledCollective>& out, bool batched_expert_collectives,
                               const ExpertOperandSchedule& expert_operands,
                               bool layer_major_prefill_batch,
                               bool prefill_expert_batch) {
    out.clear();
    if (static_cast<uint32_t>(command.phase) > static_cast<uint32_t>(StepPhase::rollback))
        return bad("unknown model step phase");
    if (g.n_layers != 48 || g.n_embd != 2560 || g.qsa_interval != 4 || g.n_expert_used != 10 ||
        g.n_ff != 640 || g.ssm_state_size != 128 || g.ssm_k_heads != 16 || g.ssm_v_heads != 48 ||
        g.n_head != 24 || g.head_dim != 256) return bad("unsupported C1 model schedule geometry");
    if (command.phase != StepPhase::prefill && command.phase != StepPhase::decode) return {};
    if (command.tokens.empty() || command.tokens.size() > 8) return bad("model command supports one to eight tokens");
    if (expert_operands.compact && expert_operands.down_codecs.size() != size_t(g.n_layers))
        return bad("compact expert schedule requires per-rank ownership and one down codec per layer");
    if (expert_operands.compact) {
        for (const int type : expert_operands.down_codecs) {
            try {
                (void)compact_down_payload_width(type, int(g.n_ff), expert_operands.valid_counts);
            } catch (...) {
                return bad("invalid codec or ownership in compact expert schedule");
            }
        }
    }
    const bool expert_rows = prefill_expert_batch && command.phase == StepPhase::prefill &&
                             command.tokens.size() > 1;
    if (expert_rows && (!layer_major_prefill_batch || !batched_expert_collectives || !expert_operands.compact))
        return bad("batched routed prefill requires layer-major mode, batched collectives and compact exact operands");
    const uint64_t row_count = uint64_t(command.tokens.size());
    const uint64_t count = expert_rows ?
        uint64_t(g.n_layers) * (2 * row_count + 2) :
        row_count * uint64_t(g.n_layers) *
        uint64_t(batched_expert_collectives ? 4 : 2 + 2 * g.n_expert_used);
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
    Failure append_failure;
    auto append_layer = [&](int64_t layer, bool batch_experts, bool append_mixer = true) {
        const bool qsa = core::is_qsa_layer(g, layer);
        if (append_mixer) {
            append(layer, qsa ? "qsa_gated_reference" : "gdn_values",
                   qsa ? std::vector<uint64_t>{2048, 2048, 2048} : std::vector<uint64_t>{2304, 1920, 1920});
            append(layer, qsa ? "qsa_output_rank_zero_reference" : "mixer", {});
        }
        if (batch_experts) {
            const uint64_t entries = row_count * uint64_t(g.n_expert_used);
            const uint64_t hidden = entries * 256;
            uint64_t lane_width = 128;
            if (expert_operands.compact) {
                try {
                    lane_width = uint64_t(compact_down_payload_width(
                        expert_operands.down_codecs[size_t(layer)], int(g.n_ff), expert_operands.valid_counts));
                } catch (...) { append_failure = bad("invalid compact expert schedule"); return; }
            }
            const uint64_t lanes = entries * uint64_t(g.n_embd) * lane_width;
            append(layer, "expert_hidden_rows", {hidden, hidden, hidden});
            append(layer, "expert_lane_terms_rows", {lanes, lanes, lanes});
        } else if (batched_expert_collectives) {
            // Transport includes each row's padding; logical valid counts remain 256/192/192.
            const uint64_t hidden = uint64_t(g.n_expert_used) * 256;
            uint64_t lane_width = 128;
            if (expert_operands.compact) {
                try {
                    lane_width = uint64_t(compact_down_payload_width(expert_operands.down_codecs[size_t(layer)],
                                                                    int(g.n_ff), expert_operands.valid_counts));
                } catch (...) { append_failure = bad("invalid compact expert schedule"); return; }
            }
            const uint64_t lanes = uint64_t(g.n_expert_used) * uint64_t(g.n_embd) * lane_width;
            append(layer, "expert_hidden_batch", {hidden, hidden, hidden});
            append(layer, "expert_lane_terms_batch", {lanes, lanes, lanes});
        } else for (int64_t slot = 0; slot < g.n_expert_used; ++slot) {
            append(layer, "expert_hidden_" + std::to_string(slot),
                   expert_operands.compact ?
                       std::vector<uint64_t>(expert_operands.valid_counts.begin(), expert_operands.valid_counts.end()) :
                       std::vector<uint64_t>{256, 192, 192});
            uint64_t lane_width = 128;
            if (expert_operands.compact) {
                try {
                    lane_width = uint64_t(compact_down_payload_width(expert_operands.down_codecs[size_t(layer)],
                                                                    int(g.n_ff), expert_operands.valid_counts));
                } catch (...) { append_failure = bad("invalid compact expert schedule"); return; }
            }
            const uint64_t lanes = uint64_t(g.n_embd) * lane_width;
            append(layer, "expert_lane_terms_" + std::to_string(slot), {lanes, lanes, lanes});
        }
    };
    const bool layer_major = layer_major_prefill_batch && command.phase == StepPhase::prefill &&
                             command.tokens.size() > 1;
    if (layer_major) {
        for (int64_t layer = 0; layer < g.n_layers; ++layer) {
            if (expert_rows) {
                for (size_t token = 0; token < command.tokens.size(); ++token) {
                    append(layer, core::is_qsa_layer(g, layer) ? "qsa_gated_reference" : "gdn_values",
                           core::is_qsa_layer(g, layer) ? std::vector<uint64_t>{2048, 2048, 2048} :
                                                          std::vector<uint64_t>{2304, 1920, 1920});
                    append(layer, core::is_qsa_layer(g, layer) ? "qsa_output_rank_zero_reference" : "mixer", {});
                    (void)token;
                }
                append_layer(layer, true, false);
                if (append_failure) return append_failure;
            } else {
                for (size_t token = 0; token < command.tokens.size(); ++token)
                    { (void)token; append_layer(layer, false); if (append_failure) return append_failure; }
            }
        }
    } else {
        for (size_t token = 0; token < command.tokens.size(); ++token) {
            for (int64_t layer = 0; layer < g.n_layers; ++layer)
                { (void)token; append_layer(layer, false); if (append_failure) return append_failure; }
        }
    }
    return {};
}

}  // namespace strata::distributed

#if defined(STRATA_DISTRIBUTED_CUDA)
#include "strata/distributed/model.hpp"
#include "nccl_api.hpp"
#include "strata/distributed/process_exit_guard.hpp"
#include "strata/distributed/graph_watchdog.hpp"

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
    cudaGraph_t decode_graph = nullptr;
    cudaGraphExec_t decode_exec = nullptr;
    int* host_route_status = nullptr;
    bool graph_warmed = false;
    bool graph_pending = false;
    bool logits_readback_pending = false;
    bool graph_unsafe = false;
    uint64_t graph_replays = 0;
    bool initialized = false;
    bool dead = false;
    bool candidate = false;
    bool published = false;
    std::string agreed_command;
    std::string execution_receipt;
    ModelCheckpoint checkpoint;
    ModelCheckpointIdentity checkpoint_identity;
    bool checkpoint_committed = false;
    std::string request;

    Impl(ModelWorkerConfig c, ModelForward& m, SessionCollectives& l, Session& s, GuardedTransport& t)
        : config(std::move(c)), model(m), collectives(l), session(s), transport(t) {}
    bool release_graph() {
        if (!decode_graph && !decode_exec) return true;
        if (graph_unsafe || graph_pending || logits_readback_pending || collectives.requires_process_exit()) return false;
        if (decode_exec && cudaGraphExecDestroy(decode_exec) != cudaSuccess) return false;
        decode_exec = nullptr;
        if (decode_graph && cudaGraphDestroy(decode_graph) != cudaSuccess) return false;
        decode_graph = nullptr;
        std::string err;
        if (!collectives.leave_capture(err)) return false;
        std::fprintf(stderr, "MODEL_GRAPH completed replays=%llu\n", static_cast<unsigned long long>(graph_replays));
        return true;
    }
    ~Impl() {
        // Unknown DMA completion must retain every borrowed model allocation.
        if (graph_unsafe || graph_pending || logits_readback_pending || collectives.requires_process_exit()) std::_Exit(2);
        if (!release_graph()) std::_Exit(2);
        if (host_route_status) cudaFreeHost(host_route_status);
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
        if (auto f = bounded_readback(logits_readback_pending,
                [&] { return cuda_failure(cudaMemcpyAsync(host_logits, device_logits, bytes, cudaMemcpyDeviceToHost,
                                                          static_cast<cudaStream_t>(config.stream))); },
                [&] { return wait(); })) return f;
        std::string err;
        if (!model.validate_execution_status(config.stream, err)) return bad(err);
        if (!std::all_of(host_logits, host_logits + model.geometry().n_vocab,
                         [](float value) { return std::isfinite(value); })) return bad("nonfinite CUDA logits");
        return {};
    }
    Failure graph_token(int32_t token, int64_t position, uint64_t step,
                        const std::vector<ScheduledCollective>& schedule) {
        Deadline deadline;
        if (auto f = session.operation_deadline(deadline)) return f;
        GraphDeadlineWatchdog watchdog(deadline.end);
        std::string err;
        if (!model.prepare_token(token, position, config.position_base, config.stream, err)) return bad(err);
        const auto stream = static_cast<cudaStream_t>(config.stream);
        if (!decode_exec) {
            // Input staging must finish before NCCL enters its capture session.
            if (auto f = wait()) return f;
            if (!collectives.enter_capture(config.decode_graph_api, config.stream, err)) return bad(err);
            if (auto f = cuda_failure(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal))) return f;
            const auto ready = config.decode_graph_api->check_graph_capture();
            if (ready) err = ready.detail;
            const bool captured = !ready && model.enqueue_prepared_token(device_logits, config.stream, err);
            const bool checked = captured && collectives.check_capture(err);
            const auto ended = cudaStreamEndCapture(stream, &decode_graph);
            if (!checked || ended != cudaSuccess || !decode_graph) {
                collectives.abort_capture("full model stream capture failed");
                return !checked ? bad(err) : cuda_failure(ended);
            }
            if (!collectives.finish_capture(err)) return bad(err);
            if (auto f = cuda_failure(cudaGraphInstantiate(&decode_exec, decode_graph, 0))) return f;
            size_t nodes = 0;
            if (auto f = cuda_failure(cudaGraphGetNodes(decode_graph, nullptr, &nodes))) return f;
            std::fprintf(stderr, "MODEL_GRAPH captured nodes=%zu collectives=%llu\n", nodes,
                         static_cast<unsigned long long>(collectives.expected_capture_collectives()));
        }
        if (!collectives.prepare_replay(schedule, step, deadline, err)) return bad(err);
        graph_pending = true;
        const auto launch = cudaGraphLaunch(decode_exec, stream);
        if (launch != cudaSuccess) {
            graph_unsafe = true;
            collectives.abort_capture("full model graph launch failed");
            return cuda_failure(launch);
        }
        if (!collectives.wait_replay(deadline, err)) {
            graph_unsafe = true;
            return bad(err);
        }
        if (auto f = cuda_failure(cudaMemcpyAsync(host_route_status, collectives.route_status_device(), sizeof(int),
                                                  cudaMemcpyDeviceToHost, stream))) return f;
        if (auto f = wait()) return f;
        graph_pending = false;
        const bool valid = *host_route_status == 0 && model.validate_execution_status(config.stream, err);
        if (!collectives.complete_replay(valid, deadline, err)) return bad(err);
        if (!valid) return bad(err.empty() ? "full model graph route agreement failed" : err);
        if (!model.complete_prepared_token(err)) return bad(err);
        ++graph_replays;
        if (auto failure = finite_logits()) return failure;
        watchdog.complete();
        return {};
    }
    void clear_checkpoint() {
        checkpoint.clear();
        checkpoint_identity = {};
        checkpoint_committed = false;
    }
    std::string checkpoint_receipt() const {
        if (!checkpoint.complete()) return {};
        const auto& id = checkpoint_identity;
        // Rank-local shards differ. Agree identity and logical position, never their bytes.
        return ":checkpoint=" + std::to_string(id.model_id.size()) + ":" + id.model_id + ":" +
               std::to_string(id.request_id.size()) + ":" + id.request_id + ":" +
               std::to_string(id.generation) + ":" + std::to_string(id.slot) + ":" +
               std::to_string(id.max_context) + ":" + std::to_string(id.position_base) + ":" +
               std::to_string(checkpoint.next_position());
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
        command.manifest_hash != p.config.worker.manifest_hash || p.config.worker.rank >= kTpWorld ||
        p.config.worker.slot >= kWorkerMaxSlots || command.slot != p.config.worker.slot)
        return bad("model worker identity/slot/stream/position base is invalid");
    if (command.tokens.size() != command.positions.size()) return bad("model token/position mismatch");
    if (command.phase == StepPhase::checkpoint &&
        (!p.config.checkpoint_max_bytes || !p.config.checkpoint_max_context))
        return bad("checkpoint requires an explicit context capacity and nonzero byte cap");
    for (size_t i = 0; i < command.tokens.size(); ++i) {
        if (command.tokens[i] < 0 || command.tokens[i] >= p.model.geometry().n_vocab ||
            command.positions[i] > uint64_t(std::numeric_limits<int32_t>::max()))
            return bad("model token or position outside supported range");
    }
    if (p.config.prefill_batch_exactnative && command.phase == StepPhase::prefill) {
        for (size_t i = 1; i < command.positions.size(); ++i)
            if (command.positions[i] != command.positions[i - 1] + 1)
                return bad("layer-major prefill requires consecutive token positions");
    }
    return model_forward_schedule(command, p.model.geometry(), p.collectives.sequence(), out,
                                  p.config.batched_expert_collectives, p.config.expert_operands,
                                  p.config.prefill_batch_exactnative, p.config.prefill_expert_batch);
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
    if (p.config.memory_phase)
        if (auto f = p.config.memory_phase(command, uint64_t(p.model.next_position()), true)) return f;
    p.collectives.step(command.step);
    p.collectives.reset_observations();
    p.collectives.expect_schedule(command.expected_collective_schedule);
    p.collectives.preagreed(p.config.preagreed_collectives);
    if (p.config.preagreed_collectives &&
        (command.phase == StepPhase::prefill || command.phase == StepPhase::decode)) {
        std::vector<CollectiveScheduleEntry> schedule;
        schedule.reserve(command.expected_collective_schedule.size());
        for (const auto& entry : command.expected_collective_schedule)
            schedule.push_back({entry.descriptor, entry.route});
        Deadline deadline;
        if (auto f = p.session.operation_deadline(deadline)) return f;
        if (auto f = p.session.agree_operation_schedule(schedule, "model-worker-compute-v1", deadline)) return f;
    }
    std::string err;
    if (command.phase == StepPhase::init) {
        if (p.initialized) return bad("model worker already initialized");
        const auto vocab = p.model.geometry().n_vocab;
        if (vocab <= 0 || uint64_t(vocab) > std::numeric_limits<size_t>::max() / sizeof(float)) return bad("invalid vocabulary size");
        const size_t count = size_t(vocab) * sizeof(float);
        if (auto f = cuda_failure(cudaMalloc(reinterpret_cast<void**>(&p.device_logits), count))) return f;
        if (auto f = cuda_failure(cudaHostAlloc(reinterpret_cast<void**>(&p.host_logits), count, cudaHostAllocDefault))) return f;
        if (auto f = cuda_failure(cudaEventCreateWithFlags(&p.completion, cudaEventDisableTiming))) return f;
        if (p.config.decode_graph_api) {
            if (!p.config.preagreed_collectives || !p.config.batched_expert_collectives ||
                !p.config.expert_operands.compact || p.config.prefill_batch_exactnative || p.config.prefill_expert_batch)
                return bad("full model graph requires singleton preagreed compact batched execution");
            if (auto f = cuda_failure(cudaHostAlloc(reinterpret_cast<void**>(&p.host_route_status), sizeof(int),
                                                      cudaHostAllocDefault))) return f;
        }
        if (!p.model.configure_prefill_batch_exactnative(p.config.prefill_batch_exactnative, err) ||
            !p.model.configure_prefill_expert_batch(p.config.prefill_expert_batch, err) ||
            !p.model.initialize(err) || !p.model.reset(p.config.stream, err)) return bad(err);
        p.initialized = true;
    } else if (command.phase == StepPhase::prefill || command.phase == StepPhase::decode) {
        if (!p.initialized || p.candidate) return bad("forward requires initialized model without pending candidate");
        if (p.request != command.request) {
            p.clear_checkpoint();
            p.request = command.request;
        }
        if (p.config.decode_graph_api && p.graph_warmed) {
            if (command.tokens.size() != 1) return bad("full model graph requires one token per command");
            if (auto f = p.graph_token(command.tokens.front(), int64_t(command.positions.front()), command.step,
                                       command.expected_collective_schedule)) return f;
        } else if (p.config.prefill_batch_exactnative && command.phase == StepPhase::prefill &&
            command.tokens.size() > 1) {
            if (!p.model.tokens(command.tokens, int64_t(command.positions.front()), p.config.position_base,
                                p.device_logits, p.config.stream, err)) return bad(err);
            if (auto f = p.finite_logits()) return f;
        } else {
            for (size_t i = 0; i < command.tokens.size(); ++i) {
                if (!p.model.token(command.tokens[i], int64_t(command.positions[i]), p.config.position_base,
                                   p.device_logits, p.config.stream, err)) return bad(err);
                if (auto f = p.finite_logits()) return f;
            }
            if (p.config.decode_graph_api) p.graph_warmed = true;
        }
        p.candidate = true;
    } else if (command.phase == StepPhase::reset || command.phase == StepPhase::cancel) {
        p.clear_checkpoint();
        p.request.clear();
        if (!p.model.reset(p.config.stream, err)) return bad(err);
        p.candidate = false;
    } else if (command.phase == StepPhase::checkpoint) {
        if (!p.initialized || p.candidate || p.request != command.request)
            return bad("checkpoint requires a committed current request");
        Deadline deadline;
        if (auto f = p.session.operation_deadline(deadline)) return f;
        p.clear_checkpoint();
        p.checkpoint_identity = {p.config.worker.manifest_hash, command.request, command.step, command.slot,
                                 p.config.worker.rank, p.config.checkpoint_max_context, p.config.position_base};
        if (!p.model.capture_checkpoint(p.checkpoint, p.checkpoint_identity, p.config.checkpoint_max_bytes,
                                         p.config.stream, deadline.end, err)) return bad(err);
    } else if (command.phase == StepPhase::rollback) {
        if (!p.checkpoint_committed || !p.checkpoint.complete() || p.request != command.request ||
            p.checkpoint_identity.request_id != command.request || p.checkpoint_identity.slot != command.slot)
            return bad("rollback requires the current request's all-rank checkpoint");
        Deadline deadline;
        if (auto f = p.session.operation_deadline(deadline)) return f;
        if (!p.model.restore_checkpoint(p.checkpoint, p.checkpoint_identity, p.config.stream, deadline.end, err))
            return bad(err);
        p.candidate = false;
        std::fill_n(p.host_logits, size_t(p.model.geometry().n_vocab), 0.0f);
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
    if (p.config.memory_phase)
        if (auto f = p.config.memory_phase(command, uint64_t(p.model.next_position()), false)) return f;
    if (p.config.preagreed_collectives &&
        (command.phase == StepPhase::prefill || command.phase == StepPhase::decode)) {
        if (auto f = p.session.finish_operation_schedule()) return f;
    }
    // Receipt confirms completed command and logical position. It does not claim to checksum model
    // KV/GDN/QSA/PLE bytes; numerical state parity still requires the GPU qualification harness.
    const std::string receipt = "noncrypto-execution-receipt-v1:position=" +
                                std::to_string(p.model.next_position()) + p.checkpoint_receipt();
    if (auto f = p.agree_bytes("model-stage-complete-v1", p.agreed_command + receipt)) return f;
    p.execution_receipt = receipt;
    if (command.phase == StepPhase::checkpoint) p.checkpoint_committed = true;
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
    p.clear_checkpoint();
    p.model.invalidate();
    if (p.config.decode_graph_api) {
        p.graph_unsafe = p.graph_unsafe || p.graph_pending || p.logits_readback_pending || p.collectives.requires_process_exit();
        if (!p.graph_unsafe && !p.release_graph()) p.graph_unsafe = true;
        if (p.graph_unsafe) p.collectives.abort_capture("model worker aborted with pending graph work");
    }
    p.transport.abort("model worker aborted");
    p.session.abort("model worker aborted");
}
void ModelWorkerBackend::discard_session() noexcept {
    auto& p = *impl_;
    p.dead = true;
    p.published = p.candidate = false;
    p.clear_checkpoint();
    p.agreed_command.clear();
    p.execution_receipt.clear();
    p.model.invalidate();
}
const float* ModelWorkerBackend::committed_logits() const {
    return impl_->published && !impl_->dead ? impl_->host_logits : nullptr;
}
size_t ModelWorkerBackend::logits_size() const { return size_t(impl_->model.geometry().n_vocab); }
bool ModelWorkerBackend::requires_process_exit() const noexcept {
    return impl_->graph_unsafe || impl_->graph_pending || impl_->logits_readback_pending ||
           impl_->collectives.requires_process_exit();
}

}  // namespace strata::distributed
#endif
