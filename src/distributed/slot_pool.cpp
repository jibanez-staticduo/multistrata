#include "strata/distributed/limits.hpp"
#include "strata/distributed/slot_pool.hpp"

#include "strata/core/native_head.hpp"
#include "strata/distributed/model.hpp"
#include "strata/kernels/ngram.hpp"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::distributed {
namespace {
Failure bad(std::string_view text) { return Failure::fail(FailureCode::bad_operation, text); }
struct DeviceAllocation {
    void* data = nullptr;
    ~DeviceAllocation() { if (data) cudaFree(data); }
    Failure allocate(uint64_t bytes) {
        if (!bytes || bytes > std::numeric_limits<size_t>::max()) return bad("invalid slot allocation extent");
        const auto status = cudaMalloc(&data, size_t(bytes));
        return status == cudaSuccess ? Failure{} : bad(cudaGetErrorString(status));
    }
};
struct Slot {
    // Destroy Worker/backend/model before releasing any state they borrow.
    DeviceAllocation arena, ple_embedding, ple_scratch;
    core::SessionState state;
    std::vector<float> ple_host;
    std::unique_ptr<ModelForward> model;
    std::unique_ptr<ModelWorkerBackend> backend;
    std::unique_ptr<Worker> worker;
    ~Slot() {
        worker.reset(); backend.reset(); model.reset();
        core::session_release(state);
        for (int64_t i = 0; state.qsa_states && i < state.qsa_alloc; ++i) {
            auto& q = state.qsa_states[state.qsa_ord0 + i];
            if (q.host_step) cudaFreeHost(q.host_step);
            if (q.host_pos) cudaFreeHost(q.host_pos);
        }
        delete[] state.qsa_states;
    }
};

Failure wire_ple(const core::WeightTable& weights, kernels::PleTable& table, Slot& slot) {
    auto get = [&](const char* name) -> const core::WeightRef& {
        const auto* w = weights.find(std::string("blk.1.") + name);
        if (!w || (!w->data && !w->native_data)) throw std::runtime_error("slot PLE tensor is missing");
        return *w;
    };
    auto& state = slot.state;
    const auto& key = get("ple_key.weight");
    const auto& conv = get("ple_conv1d.weight");
    if (conv.bytes != uint64_t(conv.elements) * 2 ||
        (conv.kind != core::WeightKind::F16InF32 &&
         !(conv.kind == core::WeightKind::Verbatim && !conv.quantized())))
        return bad("slot PLE convolution must retain F16 engine storage");
    if (!key.quantized()) state.ple.w.key_bf16 = static_cast<const uint16_t*>(key.data);
    else if (key.native_data) {
        state.ple.w.key_native_data = key.native_data;
        state.ple.w.key_native_type = key.native_type;
        state.ple.w.key_native_q8_1 = key.native_q8_1;
    } else {
        state.ple.w.key_codes = static_cast<const uint8_t*>(key.data);
        state.ple.w.key_scales = reinterpret_cast<const float*>(state.ple.w.key_codes + key.codes_bytes);
    }
    state.ple.w.value_bf16 = static_cast<const uint16_t*>(get("ple_value.weight").data);
    state.ple.w.norm_key = static_cast<const float*>(get("ple_norm_key.weight").data);
    state.ple.w.norm_query = static_cast<const float*>(get("ple_norm_query.weight").data);
    state.ple.w.norm_conv = static_cast<const float*>(get("ple_norm_conv.weight").data);
    state.ple.w.conv1d_f16 = static_cast<const uint16_t*>(conv.data);
    if (auto f = slot.ple_embedding.allocate(kernels::NG_N_EMBD * sizeof(float))) return f;
    if (auto f = slot.ple_scratch.allocate(core::ple_run_scratch_bytes())) return f;
    slot.ple_host.resize(kernels::NG_N_EMBD);
    state.ple.table = &table; state.ple.hist = state.ple_hist;
    state.ple.token = &state.ple_token; state.ple.prev = state.ple_prev;
    state.ple.emb_host = slot.ple_host.data();
    state.ple.emb_dev = static_cast<float*>(slot.ple_embedding.data);
    state.ple.scratch = static_cast<float*>(slot.ple_scratch.data);
    state.ple.consts = kernels::ple_artifact_consts();
    return {};
}
}

struct SlotPool::Impl {
    SlotPoolConfig config;
    SlotPoolResources resources;
    std::vector<std::unique_ptr<Slot>> slots;
    uint64_t cursor = 1, arena_bytes = 0;
    bool initialized = false, failed = false;
    Failure failure;
    Impl(SlotPoolConfig c, SlotPoolResources r) : config(std::move(c)), resources(std::move(r)) {}
    bool pending() const noexcept {
        try {
            if (resources.external_requires_process_exit && resources.external_requires_process_exit()) return true;
        } catch (...) { return true; }
        for (const auto& slot : slots) {
            if (slot->model && slot->model->requires_process_exit()) return true;
            if (slot->backend && slot->backend->requires_process_exit()) return true;
        }
        return false;
    }
    void abort() noexcept {
        failed = true;
        for (const auto& slot : slots) {
            if (slot->backend) { slot->backend->abort(); slot->backend->discard_session(); }
            else if (slot->model) slot->model->invalidate();
        }
        resources.transport.abort("slot pool invalidated");
        resources.session.abort("slot pool invalidated");
    }
    StepResult fail(Failure f) {
        failure = std::move(f); abort();
        return {StepStatus::failed, failure, {}, false};
    }
    Slot& at(uint32_t index) const {
        if (index >= slots.size()) throw std::out_of_range("slot pool index");
        return *slots[index];
    }
    StepResult run(uint32_t index, StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
        if (failed) return {StepStatus::failed, failure ? failure : bad("slot pool was invalidated"), {}, false};
        if (index >= slots.size()) return fail(bad("slot pool command outside configured slots"));
        if (cursor != resources.collectives.sequence()) return fail(bad("slot pool collective cursor drift before command"));
        auto& slot = at(index);
        StepCommand c{config.session_uuid, std::string(request), index, slot.worker->next_step(), phase,
                       {}, {}, config.model_hash, {}};
        if (phase == StepPhase::prefill || phase == StepPhase::decode) { c.tokens = {token}; c.positions = {position}; }
        if (auto f = slot.backend->plan(c, c.expected_collective_schedule)) return fail(std::move(f));
        auto result = slot.worker->execute(c);
        if (result.status != StepStatus::completed && result.status != StepStatus::cancelled)
            return fail(result.failure ? result.failure : bad("slot worker did not complete command"));
        if (cursor != resources.collectives.sequence()) return fail(bad("slot pool collective cursor drift after command"));
        return result;
    }
};
SlotPool::SlotPool(SlotPoolConfig c, SlotPoolResources r) : impl_(std::make_unique<Impl>(std::move(c), std::move(r))) {}
SlotPool::~SlotPool() {
    if (impl_->pending()) {
        // Deliberately quarantine every owner until the caller's exit guard terminates the process.
        (void)impl_.release();
        return;
    }
    if (!impl_->slots.empty() && !impl_->failed) impl_->abort();
}
Failure SlotPool::initialize() {
    auto& p = *impl_;
    if (p.initialized || p.failed) return bad("slot pool cannot initialize twice or after failure");
    const auto& c = p.config;
    if ((c.slots != 1 && c.slots != 4 && c.slots != 8) || c.context < 4 || c.context > kDistributedMaxContext ||
        c.timeout_ms < 100 || c.timeout_ms > 600000 || c.session_uuid.empty() || c.model_hash.empty() ||
        c.rank >= kTpWorld || !p.resources.stream || !p.resources.head.loaded() ||
        p.resources.collectives.sequence() != 1 || core::qsa_kv_resident() != 0 || core::qsa_kv_elastic() ||
        core::qsa_kv_int8() || core::qsa_kv_q4() || core::qsa_kv_hybrid())
        return p.fail(bad("invalid slot pool configuration; resident FP16 identity KV required")).failure;
    try {
        const auto bytes = core::session_bytes(p.resources.geometry, c.context, p.resources.geometry.n_expert_used);
        if (!bytes || bytes > std::numeric_limits<uint64_t>::max() / c.slots)
            return p.fail(bad("slot pool arena byte extent overflow")).failure;
        p.slots.reserve(c.slots);
        for (uint32_t index = 0; index < c.slots; ++index) {
            p.slots.push_back(std::make_unique<Slot>());
            auto& slot = *p.slots.back();
            if (auto f = slot.arena.allocate(bytes)) return p.fail(std::move(f)).failure;
            p.arena_bytes += bytes;
            if (!core::session_init(p.resources.geometry, c.context, p.resources.geometry.n_expert_used, slot.arena.data, slot.state))
                return p.fail(bad("slot pool session initialization failed")).failure;
            if (auto f = wire_ple(p.resources.weights, p.resources.ple_table, slot)) return p.fail(std::move(f)).failure;
            slot.model = std::make_unique<ModelForward>(p.resources.weights, p.resources.geometry, slot.state,
                          int(c.rank), p.resources.collectives, p.resources.experts);
            slot.model->native_head(&p.resources.head);
            WorkerConfig worker{c.session_uuid, c.model_hash, c.rank, index, &p.cursor};
            ModelWorkerConfig backend{worker, p.resources.stream, 0, c.batched_expert_collectives,
                                       c.checkpoint_max_bytes, c.context, c.preagreed_collectives,
                                       c.expert_operands};
            backend.memory_phase = c.memory_phase;
            slot.backend = std::make_unique<ModelWorkerBackend>(backend, *slot.model, p.resources.collectives,
                              p.resources.session, p.resources.transport);
            slot.worker = std::make_unique<Worker>(worker, *slot.backend);
            const auto result = p.run(index, StepPhase::init, "", -1, 0);
            if (result.failure) return result.failure;
        }
        p.initialized = true;
        return {};
    } catch (const std::exception& e) { return p.fail(bad(e.what())).failure; }
      catch (...) { return p.fail(bad("slot pool initialization threw")).failure; }
}
StepResult SlotPool::execute(uint32_t index, StepPhase phase, std::string_view request, int32_t token, uint64_t position) {
    auto& p = *impl_;
    if (!p.initialized) return p.fail(bad("slot pool command before initialization"));
    try { return p.run(index, phase, request, token, position); }
    catch (...) { return p.fail(bad("slot pool backend threw")); }
}
void SlotPool::abort() noexcept { impl_->abort(); }
bool SlotPool::requires_process_exit() const noexcept { return impl_->pending(); }
uint32_t SlotPool::slots() const { return impl_->config.slots; }
uint64_t SlotPool::allocated_session_bytes() const { return impl_->arena_bytes; }
uint64_t SlotPool::collective_sequence() const { return impl_->cursor; }
const float* SlotPool::committed_logits(uint32_t i) const { return impl_->failed ? nullptr : impl_->at(i).backend->committed_logits(); }
size_t SlotPool::logits_size(uint32_t i) const { return impl_->at(i).backend->logits_size(); }
ModelWorkerBackend& SlotPool::execution(uint32_t i) { return *impl_->at(i).backend; }
const core::SessionState& SlotPool::session_state(uint32_t i) const { return impl_->at(i).state; }
const ModelForward& SlotPool::model(uint32_t i) const { return *impl_->at(i).model; }
} // namespace strata::distributed
