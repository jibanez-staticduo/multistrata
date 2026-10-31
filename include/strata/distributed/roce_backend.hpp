#pragma once

#if !defined(STRATA_DISTRIBUTED_CUDA)
#error "The native RoCE backend requires the opt-in CUDA distributed build"
#endif

#include "strata/distributed/roce_collective.hpp"
#include "strata/distributed/transport.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace strata::distributed::roce {

inline constexpr uint32_t kProxyAbi = 10;

struct Config {
    uint32_t rank = 0;
    uint32_t world = kTpWorld;
    std::vector<std::string> hca_names;
    // Peer-major route table: world * stripes entries; local-rank entries are -1.
    std::vector<int> route_hca;
    uint32_t stripes = 1;
    int gid_index = 3;
    int traffic_class = 0;
    uint64_t slot_bytes = 4 * 1024 * 1024;
    uint32_t timeout_ms = 10000;
    uint32_t blocks = 1;
    uint32_t threads = 256;
    // Exact source hash is part of the session's approved provenance identity.
    std::string approved_proxy_sha256;
};

// Discover active local HCAs whose selected port-1 GID is populated.
std::vector<std::string> discover_hcas(int gid_index, size_t maximum = 4);

class RoceBackend final : public Backend {
public:
    explicit RoceBackend(Config config);
    ~RoceBackend() override;
    RoceBackend(const RoceBackend&) = delete;
    RoceBackend& operator=(const RoceBackend&) = delete;

    // Setup is deliberately split around the authenticated session exchange:
    // create() allocates/registers local resources; local_blob() is sent to
    // the coordinator; connect() consumes exactly one blob for each rank.
    Failure create();
    Failure local_blob(std::vector<uint8_t>& out) const;
    Failure connect(std::span<const uint8_t> rank_ordered_blobs);
    const std::string& provenance() const { return provenance_; }

    uint32_t world_size() const override { return config_.world; }
    uint32_t local_rank() const override { return config_.rank; }
    Failure raw_all_reduce_sum(const DeviceBuffer& input, const DeviceBuffer& output,
                               uint64_t elements, cudaStream_t stream) override;
    Failure raw_all_gather(const DeviceBuffer& input, const DeviceBuffer& output,
                           uint64_t slot_elements, cudaStream_t stream) override;
    Failure raw_poll(Deadline deadline) override;
    bool raw_async_failure(Failure& out) const override;
    void raw_abort() noexcept override;

    // CUDA graph capture defers execution/completion to graph replay. The
    // wrapper's per-operation wait is a capture-order acknowledgement; callers
    // must call wait_captured_replay() after each graph launch before consuming
    // outputs or releasing buffers.
    bool graph_capture_seen() const { return graph_capture_seen_; }
    Failure wait_captured_replay(Deadline deadline);

    // Once true, a device kernel or NIC may still address backend-owned memory.
    // The process must exit without unwinding those resources.
    bool requires_process_exit() const;

private:
    Failure enqueue(const DeviceBuffer& input, const DeviceBuffer& output,
                    cuda::Kind kind, uint64_t slot_elements, cudaStream_t stream);
    Failure health_locked() const;
    void release_safe() noexcept;

    Config config_;
    cuda::Layout layout_{};
    void* region_ = nullptr;
    void* region_device_ = nullptr;
    void* device_state_ = nullptr;
    void* proxy_ = nullptr;
    void* event_ = nullptr;
    cudaStream_t pending_stream_ = nullptr;
    std::string provenance_;
    Failure sticky_{};
    std::atomic<bool> aborted_{false};
    mutable std::mutex mutex_;
    bool created_ = false;
    bool connected_ = false;
    bool in_flight_ = false;
    bool capture_active_ = false;
    bool graph_capture_seen_ = false;
    bool captured_pending_ = false;
    uint32_t captured_collectives_ = 0;
    uint32_t capture_baseline_ = 0;
    uint32_t last_completed_ = 0;
    cudaStream_t capture_stream_ = nullptr;
    bool capture_replay_pending_ = false;
    bool unsafe_to_destroy_ = false;
};

}  // namespace strata::distributed::roce
