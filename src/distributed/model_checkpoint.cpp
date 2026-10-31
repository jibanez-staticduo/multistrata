#include "strata/distributed/model_checkpoint.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <set>
#include <thread>
#include <utility>

#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace strata::distributed {
namespace {
bool fail(std::string& err, const std::string& text) { err = text; return false; }
bool same(const StateEvidenceDescriptor& a, const StateEvidenceDescriptor& b) {
    return a.kind == b.kind && a.name == b.name && a.dtype == b.dtype && a.shape == b.shape &&
           a.layer == b.layer && a.logical_shape == b.logical_shape && a.shard_axis == b.shard_axis &&
           a.axis_indices == b.axis_indices;
}
bool same(const ModelCheckpointIdentity& a, const ModelCheckpointIdentity& b) {
    return a.model_id == b.model_id && a.request_id == b.request_id && a.generation == b.generation &&
           a.slot == b.slot && a.rank == b.rank && a.max_context == b.max_context &&
           a.position_base == b.position_base;
}
bool live(CheckpointDeadline deadline, std::string& err) {
    return std::chrono::steady_clock::now() < deadline || fail(err, "checkpoint deadline expired");
}
bool inventory(const std::vector<StateEvidenceDescriptor>& required, uint64_t position,
               uint64_t cap, uint64_t& total, std::string& err) {
    if (required.empty() || !cap) return fail(err, "checkpoint requires complete inventory and byte cap");
    std::set<std::pair<std::string, std::string>> names;
    total = 0;
    for (const auto& d : required) {
        uint64_t n = 0;
        if (!state_evidence_bytes(d, n, err)) return false;
        if (!names.emplace(d.kind, d.name).second) return fail(err, "duplicate checkpoint state");
        if (d.kind == "kv" && (d.shape.size() != 3 || d.shape[0] != position))
            return fail(err, "checkpoint KV rows disagree with next position");
        if (n > cap - total) return fail(err, "checkpoint exceeds byte cap");
        total += n;
    }
    return true;
}
template<class Array>
bool match(const std::vector<StateEvidenceDescriptor>& required, const std::vector<Array>& arrays,
           std::string& err) {
    if (required.size() != arrays.size()) return fail(err, "incomplete checkpoint state inventory");
    std::set<std::pair<std::string, std::string>> names;
    for (const auto& a : arrays) {
        if (!names.emplace(a.descriptor.kind, a.descriptor.name).second)
            return fail(err, "duplicate checkpoint array");
        if (std::none_of(required.begin(), required.end(), [&](const auto& d) { return same(d, a.descriptor); }))
            return fail(err, "checkpoint state shape or ownership mismatch");
    }
    return true;
}
}

void ModelCheckpoint::clear() noexcept {
    complete_ = false;
    bytes_ = 0;
    next_position_ = 0;
    identity_ = {};
    arrays_.clear();
}

bool ModelCheckpoint::capture(const ModelCheckpointRequest& request,
                              const std::vector<StateEvidenceArray>& sources,
                              CheckpointDeadline deadline, std::string& err) {
    clear();
    err.clear();
    try {
        const auto& id = request.identity;
        uint64_t total = 0;
        if (id.model_id.empty() || id.request_id.empty() || id.rank >= 3 || !id.max_context ||
            id.position_base < 0 || uint64_t(id.position_base) > id.max_context ||
            request.next_position > id.max_context - uint64_t(id.position_base))
            return fail(err, "invalid checkpoint identity or position");
        if (!live(deadline, err) || !inventory(request.required, request.next_position, request.max_bytes, total, err) ||
            !match(request.required, sources, err)) return false;
        for (const auto& a : sources) if (!a.read) return fail(err, "missing checkpoint state reader");
        std::vector<Array> staged;
        staged.reserve(sources.size());
        for (const auto& a : sources) {
            uint64_t n = 0;
            if (!live(deadline, err) || !state_evidence_bytes(a.descriptor, n, err)) return false;
            Array b{a.descriptor, std::vector<uint8_t>(static_cast<std::size_t>(n))};
            if (!a.read(b.bytes.data(), b.bytes.size(), err)) {
                if (err.empty()) err = "checkpoint state reader failed";
                return false;
            }
            if (!live(deadline, err)) return false;
            staged.push_back(std::move(b));
        }
        identity_ = id;
        arrays_ = std::move(staged);
        next_position_ = request.next_position;
        bytes_ = total;
        complete_ = true;
        return true;
    } catch (const std::exception& ex) { clear(); return fail(err, std::string("checkpoint capture: ") + ex.what()); }
      catch (...) { clear(); return fail(err, "checkpoint capture callback threw"); }
}

bool ModelCheckpoint::restore(const ModelCheckpointDestination& dst, CheckpointDeadline deadline,
                              std::string& err) const {
    err.clear();
    bool writing = false;
    const auto invalidate = [&] { if (writing) { try { dst.invalidate(); } catch (...) {} } };
    try {
        uint64_t ignored = 0;
        if (!complete_ || !same(identity_, dst.identity)) return fail(err, "checkpoint identity mismatch or incomplete capture");
        if (!dst.commit_position || !dst.invalidate) return fail(err, "checkpoint requires commit and invalidation callbacks");
        if (!live(deadline, err) || !inventory(dst.required, next_position_, bytes_ ? bytes_ : 1, ignored, err) ||
            !match(dst.required, arrays_, err) || !match(dst.required, dst.arrays, err)) return false;
        for (const auto& a : dst.arrays) if (!a.write) return fail(err, "missing checkpoint state writer");
        for (const auto& a : arrays_) {
            if (!live(deadline, err)) { invalidate(); return false; }
            const auto target = std::find_if(dst.arrays.begin(), dst.arrays.end(), [&](const auto& b) { return same(a.descriptor, b.descriptor); });
            writing = true;
            if (!target->write(a.bytes.data(), a.bytes.size(), err)) {
                if (err.empty()) err = "checkpoint state writer failed";
                invalidate(); return false;
            }
            if (!live(deadline, err)) { invalidate(); return false; }
        }
        if (!dst.commit_position(next_position_, err)) {
            if (err.empty()) err = "checkpoint position commit failed";
            invalidate(); return false;
        }
        if (!live(deadline, err)) { invalidate(); return false; }
        return true;
    } catch (const std::exception& ex) { invalidate(); return fail(err, std::string("checkpoint restore: ") + ex.what()); }
      catch (...) { invalidate(); return fail(err, "checkpoint restore callback threw"); }
}

#ifdef STRATA_MODEL_CHECKPOINT_WITH_CUDA
bool cuda_checkpoint_copy(CudaCheckpointTransfer& transfer, void* host, void* device,
                          std::size_t bytes, bool to_device, void* stream,
                          CheckpointDeadline deadline, std::string& err) {
    if (transfer.poisoned || transfer.pending || !transfer.pinned_staging || !transfer.completion_event ||
        !stream || bytes > transfer.staging_bytes || (bytes && (!host || !device)))
        return fail(err, "invalid or poisoned CUDA checkpoint transfer");
    if (!live(deadline, err)) return false;
    if (!bytes) return true;
    const auto cuda_fail = [&](cudaError_t status) {
        transfer.poisoned = true;
        return fail(err, std::string("CUDA checkpoint copy: ") + cudaGetErrorString(status));
    };
    if (to_device) std::memcpy(transfer.pinned_staging, host, bytes);
    if (!live(deadline, err)) return false;
    transfer.pending = true;
    auto status = cudaMemcpyAsync(to_device ? device : transfer.pinned_staging,
                                  to_device ? transfer.pinned_staging : device, bytes,
                                  to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                                  static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) return cuda_fail(status);
    const auto event = static_cast<cudaEvent_t>(transfer.completion_event);
    status = cudaEventRecord(event, static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) return cuda_fail(status);
    for (;;) {
        status = cudaEventQuery(event);
        if (status == cudaSuccess) break;
        if (status != cudaErrorNotReady) return cuda_fail(status);
        if (!live(deadline, err)) { transfer.poisoned = true; return false; }
        std::this_thread::yield();
    }
    transfer.pending = false;
    if (!live(deadline, err)) { transfer.poisoned = true; return false; }
    if (!to_device) std::memcpy(host, transfer.pinned_staging, bytes);
    if (!live(deadline, err)) { transfer.poisoned = true; return false; }
    return true;
}
#endif
} // namespace strata::distributed
