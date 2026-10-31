#include "strata/distributed/roce_collective.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace strata::distributed::roce::cuda {
namespace {

constexpr uint32_t kWorld = 3;
constexpr uint32_t kSlots = 2;
constexpr uint32_t kMaxStripes = 4;
constexpr uint32_t kFlagStride = 128;
constexpr uint32_t kPackBytes = 16;
constexpr uint32_t kControlFailed = 8;
constexpr uint32_t kControlPeer = 12;
constexpr uint32_t kControlNbytes = 4;
constexpr uint32_t kControlSlotNbytes = 16;
constexpr uint32_t kControlTimeoutUs = 28;
constexpr uint32_t kControlWaitedUs = 32;
constexpr uint32_t kControlCompleted = 36;

struct DeviceState {
    uint32_t epoch;
    uint32_t error;
    uint32_t sequence;
};

__device__ __forceinline__ void sys_fence() {
    asm volatile("fence.sc.sys;" ::: "memory");
}

__device__ __forceinline__ void gpu_fence() {
    asm volatile("fence.sc.gpu;" ::: "memory");
}

__device__ __forceinline__ uint32_t ld_sys_acquire(const uint32_t* p) {
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t ld_sys_relaxed(const uint32_t* p) {
    uint32_t value;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

__device__ __forceinline__ void st_sys_relaxed(uint32_t* p, uint32_t value) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t atomic_gpu_add(uint32_t* p, uint32_t value) {
    uint32_t prior;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(prior) : "l"(p), "r"(value) : "memory");
    return prior;
}

__device__ __forceinline__ void st_gpu_release(uint32_t* p, uint32_t value) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ void fail(uint8_t* control, DeviceState* state, uint32_t sequence,
                                     uint32_t peer, uint32_t waited) {
    st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlPeer), peer);
    st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlWaitedUs), waited);
    st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlFailed), 1u);
    atomicExch(&state->error, 1u);
    sys_fence();
    atomicExch(&state->sequence, sequence);
    asm volatile("trap;");
}

__device__ __forceinline__ bool wait_flag(uint32_t* flag, uint32_t sequence, uint8_t* control,
                                          DeviceState* state, uint32_t peer, uint32_t timeout_us) {
    const uint64_t start = globaltimer_ns();
    const uint64_t timeout_ticks = static_cast<uint64_t>(timeout_us) * 1000ull;
    for (uint32_t polls = 0;; ++polls) {
        if (ld_sys_acquire(flag) == sequence) return true;
        if ((polls & 63u) == 63u && globaltimer_ns() - start >= timeout_ticks) {
            const uint64_t elapsed = globaltimer_ns() - start;
            fail(control, state, sequence, peer, static_cast<uint32_t>(elapsed / 1000ull));
            return false;
        }
    }
}

template <typename T>
__device__ __forceinline__ T read_value(const T* p) {
    return *p;
}

template <>
__device__ __forceinline__ __nv_bfloat16 read_value(const __nv_bfloat16* p) { return *p; }

template <typename T>
__device__ __forceinline__ T sum3(const T* input, const uint8_t* recv, uint64_t slot_bytes,
                                  uint32_t sequence, uint32_t rank, uint64_t index) {
    float acc = 0.0f;
    for (uint32_t source = 0; source < kWorld; ++source) {
        float value;
        if (source == rank) {
            value = static_cast<float>(read_value(input + index));
        } else {
            const uint8_t* slot = recv + (static_cast<uint64_t>(source) * kSlots + (sequence & 1u)) *
                                              slot_bytes;
            value = static_cast<float>(*reinterpret_cast<const T*>(slot + index * sizeof(T)));
        }
        if (source == 0) acc = value;
        else acc = acc + value;
    }
    return static_cast<T>(acc);
}

template <>
__device__ __forceinline__ __nv_bfloat16 sum3(const __nv_bfloat16* input, const uint8_t* recv,
                                               uint64_t slot_bytes, uint32_t sequence, uint32_t rank,
                                               uint64_t index) {
    float acc = 0.0f;
    for (uint32_t source = 0; source < kWorld; ++source) {
        float value;
        if (source == rank) {
            value = __bfloat162float(input[index]);
        } else {
            const uint8_t* slot = recv + (static_cast<uint64_t>(source) * kSlots + (sequence & 1u)) *
                                              slot_bytes;
            value = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(slot + index * sizeof(__nv_bfloat16)));
        }
        if (source == 0) acc = value;
        else acc = acc + value;
    }
    return __float2bfloat16_rn(acc);
}

template <typename T>
__global__ void collective_kernel(const T* input, T* output, DeviceState* state, uint8_t* region,
                                  uint64_t recv_off, uint64_t flag_off, uint64_t send_off,
                                  uint64_t ctrl_off, uint64_t slot_bytes, uint64_t input_elements,
                                  uint64_t output_elements,
                                  uint32_t rank, uint32_t stripes, uint32_t timeout_us,
                                  Kind kind) {
    uint8_t* control = region + ctrl_off;
    auto* epoch = &state->epoch;
    uint32_t sequence = 0;
    __shared__ uint32_t operation_failed;
    if (threadIdx.x == 0) {
        operation_failed = atomicAdd(&state->error, 0u);
        const uint32_t prior = *epoch;
        if (operation_failed == 0u && prior >= UINT32_MAX - 1u) {
            operation_failed = 1u;
            sequence = prior;
        } else if (operation_failed == 0u) {
            sequence = prior + 1;
            state->sequence = sequence;
        }
    }
    __shared__ uint32_t operation_sequence;
    if (threadIdx.x == 0) {
        operation_sequence = sequence;
        if (operation_failed != 0 && atomicAdd(&state->error, 0u) == 0u) {
            fail(control, state, sequence, UINT32_MAX, 0);
        }
    }
    __syncthreads();
    if (operation_failed != 0) return;
    sequence = operation_sequence;
    const uint64_t nbytes64 = input_elements * sizeof(T);
    if (nbytes64 > slot_bytes || nbytes64 > UINT32_MAX || nbytes64 % kPackBytes != 0) {
        if (threadIdx.x == 0) fail(control, state, sequence, UINT32_MAX, 0);
        return;
    }
    uint8_t* send = region + send_off + (sequence & 1u) * slot_bytes;
    const uint64_t packs = nbytes64 / kPackBytes;
    const uint64_t start_pack = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t pack = start_pack; pack < packs; pack += stride) {
        const auto* source = reinterpret_cast<const uint4*>(input);
        auto* destination = reinterpret_cast<uint4*>(send);
        destination[pack] = source[pack];
    }
    __syncthreads();
    sys_fence();
    __syncthreads();
    if (threadIdx.x == 0) {
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlNbytes), static_cast<uint32_t>(nbytes64));
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlSlotNbytes + (sequence & 1u) * 4u),
                       static_cast<uint32_t>(nbytes64));
        sys_fence();
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control), sequence);
    }

    if (threadIdx.x < (kWorld - 1) * stripes) {
        const uint32_t peer_index = threadIdx.x / stripes;
        const uint32_t lane = threadIdx.x % stripes;
        const uint32_t peer = peer_index >= rank ? peer_index + 1u : peer_index;
        const uint64_t flag_index = ((static_cast<uint64_t>(peer) * kSlots + (sequence & 1u)) *
                                     stripes + lane) * kFlagStride;
        (void)wait_flag(reinterpret_cast<uint32_t*>(region + flag_off + flag_index), sequence,
                        control, state, peer, timeout_us);
    }
    __syncthreads();
    if (threadIdx.x == 0) st_gpu_release(&state->sequence, sequence);
    __syncthreads();

    const uint8_t* recv = region + recv_off;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < output_elements; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        if (kind == Kind::all_reduce_sum) {
            output[i] = sum3(input, recv, slot_bytes, sequence, rank, i);
        } else {
            const uint64_t slot_elems = input_elements;
            const uint32_t src = static_cast<uint32_t>(i / slot_elems);
            const uint64_t offset = i % slot_elems;
            const T* src_ptr = src == rank ? input : reinterpret_cast<const T*>(
                recv + (static_cast<uint64_t>(src) * kSlots + (sequence & 1u)) * slot_bytes);
            output[i] = src_ptr[offset];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        gpu_fence();
        st_gpu_release(epoch, sequence);
        sys_fence();
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlCompleted), sequence);
    }
}

}  // namespace

bool layout(uint32_t world, uint64_t slot_bytes, Layout& out) {
    if (world != kWorld || slot_bytes == 0 || slot_bytes % 4096 != 0 ||
        slot_bytes > (uint64_t{1} << 40)) return false;
    const uint64_t recv_bytes = uint64_t(world) * kSlots * slot_bytes;
    const uint64_t flag_bytes = uint64_t(world) * kSlots * kMaxStripes * kFlagStride;
    if (recv_bytes > UINT64_MAX - flag_bytes) return false;
    out.recv_off = 0;
    out.flag_off = recv_bytes;
    out.send_off = recv_bytes + flag_bytes;
    if (out.send_off > UINT64_MAX - uint64_t(kSlots) * slot_bytes) return false;
    out.ctrl_off = out.send_off + uint64_t(kSlots) * slot_bytes;
    if (out.ctrl_off > UINT64_MAX - kCtrlBytes) return false;
    out.total_bytes = out.ctrl_off + kCtrlBytes;
    out.flag_stride = kFlagStride;
    out.slots = kSlots;
    return true;
}

cudaError_t launch(const DeviceBuffer& input, const DeviceBuffer& output, Kind kind,
                   uint32_t rank, const Layout& region_layout, void* region_device,
                   uint64_t slot_bytes, void* device_state, uint32_t timeout_us,
                   cudaStream_t stream, uint32_t blocks, uint32_t threads) {
    if (rank >= kWorld || region_device == nullptr || device_state == nullptr ||
        stream == nullptr || blocks != 1 || threads < (kWorld - 1) * 1 || timeout_us == 0)
        return cudaErrorInvalidValue;
    auto* state = static_cast<DeviceState*>(device_state);
    auto* region = static_cast<uint8_t*>(region_device);
    const uint32_t stripes = 1;
    switch (input.type) {
        case ScalarType::fp32:
            collective_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(input.data), static_cast<float*>(output.data), state, region,
                region_layout.recv_off, region_layout.flag_off, region_layout.send_off,
                region_layout.ctrl_off, slot_bytes, input.elements, output.elements, rank, stripes,
                timeout_us, kind);
            break;
        case ScalarType::bf16:
            collective_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(output.data), state, region,
                region_layout.recv_off, region_layout.flag_off, region_layout.send_off,
                region_layout.ctrl_off, slot_bytes, input.elements, output.elements, rank, stripes,
                timeout_us, kind);
            break;
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

}  // namespace strata::distributed::roce::cuda
