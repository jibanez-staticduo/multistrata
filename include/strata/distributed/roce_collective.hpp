#pragma once

#if !defined(STRATA_DISTRIBUTED_CUDA)
#error "RoCE collective kernels require the CUDA distributed build"
#endif

#include "strata/distributed/transport.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::distributed::roce::cuda {

struct Layout {
    uint64_t recv_off = 0;
    uint64_t flag_off = 0;
    uint64_t send_off = 0;
    uint64_t ctrl_off = 0;
    uint64_t total_bytes = 0;
    uint64_t flag_stride = 0;
    uint64_t slots = 0;
};

enum class Kind : uint32_t { all_reduce_sum = 0, all_gather = 1 };

// These host offsets are shared with proxy.c. The device epoch/counters are
// separate allocations so CUDA graph replay never captures a fixed sequence.
inline constexpr uint32_t kCtrlDoorbellSeq = 0;
inline constexpr uint32_t kCtrlNbytes = 4;
inline constexpr uint32_t kCtrlFailed = 8;
inline constexpr uint32_t kCtrlErrorPeer = 12;
inline constexpr uint32_t kCtrlSlotNbytes = 16;
inline constexpr uint32_t kCtrlErrorLane = 24;
inline constexpr uint32_t kCtrlTimeoutUs = 28;
inline constexpr uint32_t kCtrlWaitedUs = 32;
inline constexpr uint32_t kCtrlCompletedSeq = 36;
inline constexpr uint32_t kCtrlBytes = 128;
inline constexpr uint32_t kPackBytes = 16;
inline constexpr size_t kDeviceStateBytes = 3 * sizeof(uint32_t);

bool layout(uint32_t world, uint64_t slot_bytes, Layout& out);

// Device-visible protocol. All traffic is direct three-rank RoCE. A launch
// updates a device epoch, so graph replay advances sequences dynamically.
cudaError_t launch(const DeviceBuffer& input, const DeviceBuffer& output,
                   Kind kind, uint32_t rank, const Layout& layout,
                   void* host_region_device, uint64_t slot_bytes,
                   void* device_state, uint32_t timeout_us,
                   cudaStream_t stream, uint32_t blocks, uint32_t threads);

}  // namespace strata::distributed::roce::cuda
