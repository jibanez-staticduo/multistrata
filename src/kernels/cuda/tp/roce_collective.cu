#include "strata/distributed/roce_collective.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <type_traits>

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

__device__ __forceinline__ void record_failure(uint8_t* control, DeviceState* state,
                                               uint32_t sequence, uint32_t peer,
                                               uint32_t waited) {
    st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlPeer), peer);
    st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlWaitedUs), waited);
    st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlFailed), 1u);
    atomicExch(&state->error, 1u);
    sys_fence();
    atomicExch(&state->sequence, sequence);
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

__device__ __forceinline__ bool wait_flag_cooperative(uint32_t* flag, uint32_t sequence,
                                                       uint32_t timeout_us, uint32_t* failed,
                                                       uint32_t* failed_peer,
                                                       uint32_t* failed_waited,
                                                       uint32_t peer) {
    const uint64_t start = globaltimer_ns();
    const uint64_t timeout_ticks = static_cast<uint64_t>(timeout_us) * 1000ull;
    for (uint32_t polls = 0;; ++polls) {
        if (ld_sys_acquire(flag) == sequence) return true;
        if ((polls & 63u) == 63u && globaltimer_ns() - start >= timeout_ticks) {
            const uint64_t elapsed = globaltimer_ns() - start;
            if (atomicCAS(failed, 0u, 1u) == 0u) {
                *failed_peer = peer;
                *failed_waited = static_cast<uint32_t>(elapsed / 1000ull);
            }
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

__device__ __forceinline__ __nv_bfloat16 sum3_staged(
    const uint8_t* send, const uint8_t* recv, uint64_t slot_bytes,
    uint32_t sequence, uint32_t rank, uint64_t index) {
    float acc = 0.0f;
    for (uint32_t source = 0; source < kWorld; ++source) {
        const uint8_t* slot = source == rank
            ? send
            : recv + (static_cast<uint64_t>(source) * kSlots + (sequence & 1u)) * slot_bytes;
        const float value = __bfloat162float(
            reinterpret_cast<const __nv_bfloat16*>(slot)[index]);
        if (source == 0) acc = value;
        else acc = acc + value;
    }
    return __float2bfloat16_rn(acc);
}

__device__ __forceinline__ uint32_t simple8_ring_part(uint64_t index,
                                                       uint64_t global_first,
                                                       uint64_t global_total) {
    const uint64_t cells = global_total / 4096 + (global_total % 4096 != 0);
    const uint64_t channel_span = (cells / 8 + (cells % 8 != 0)) * 4096;
    const uint64_t global_index = global_first + index;
    const uint64_t channel = global_index / channel_span;
    const uint64_t channel_first = channel * channel_span;
    const uint64_t channel_length = min(channel_span, global_total - channel_first);
    const uint64_t position = global_index - channel_first;
    const uint64_t chunk = min(uint64_t(524288), ((channel_length + 2) / 3 + 3) / 4 * 4);
    const uint64_t loop_first = position / (3 * chunk) * (3 * chunk);
    const uint64_t remaining = channel_length - loop_first;
    const uint64_t live_chunk = remaining < 3 * chunk
        ? ((remaining + 2) / 3 + 3) / 4 * 4
        : chunk;
    return static_cast<uint32_t>((position - loop_first) / live_chunk);
}

__device__ __forceinline__ float sum3_ring(float a, float b, float c, uint32_t part) {
    if (part == 0) return __fadd_rn(__fadd_rn(b, c), a);
    if (part == 1) return __fadd_rn(__fadd_rn(a, c), b);
    return __fadd_rn(__fadd_rn(a, b), c);
}

__device__ __forceinline__ float ring_rank_value(
    const uint8_t* send, const uint8_t* recv, uint64_t slot_bytes,
    uint32_t sequence, uint32_t rank, uint32_t source, uint64_t index) {
    const uint8_t* slot = source == rank
        ? send
        : recv + (static_cast<uint64_t>(source) * kSlots +
                  (sequence & 1u)) * slot_bytes;
    return reinterpret_cast<const float*>(slot)[index];
}

__device__ __forceinline__ float4 ring_rank_values(
    const uint8_t* send, const uint8_t* recv, uint64_t slot_bytes,
    uint32_t sequence, uint32_t rank, uint32_t source, uint64_t pack) {
    const uint8_t* slot = source == rank
        ? send
        : recv + (static_cast<uint64_t>(source) * kSlots +
                  (sequence & 1u)) * slot_bytes;
    return reinterpret_cast<const float4*>(slot)[pack];
}

template <typename T>
__global__ void collective_kernel(const T* input, T* output, DeviceState* state, uint8_t* region,
                                  uint64_t recv_off, uint64_t flag_off, uint64_t send_off,
                                  uint64_t ctrl_off, uint64_t slot_bytes, uint64_t input_elements,
                                  uint64_t output_elements,
                                  uint32_t rank, uint32_t stripes, uint32_t timeout_us,
                                  Kind kind, uint64_t global_first, uint64_t global_total) {
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
        if (kind == Kind::all_reduce_ring_sum) {
            if constexpr (std::is_same_v<T, float>) {
                const uint32_t part = simple8_ring_part(i, global_first, global_total);
                output[i] = sum3_ring(
                    ring_rank_value(send, recv, slot_bytes, sequence, rank, 0, i),
                    ring_rank_value(send, recv, slot_bytes, sequence, rank, 1, i),
                    ring_rank_value(send, recv, slot_bytes, sequence, rank, 2, i), part);
            }
        } else if (kind == Kind::all_reduce_sum) {
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

template <typename T>
__global__ void collective_kernel_cooperative(
    const T* input, T* output, DeviceState* state, uint8_t* region,
    uint64_t recv_off, uint64_t flag_off, uint64_t send_off, uint64_t ctrl_off,
    uint64_t slot_bytes, uint64_t input_elements, uint64_t output_elements,
    uint32_t rank, uint32_t stripes, uint32_t timeout_us, Kind kind,
    uint64_t global_first, uint64_t global_total) {
    auto grid = cooperative_groups::this_grid();
    uint8_t* control = region + ctrl_off;
    auto* epoch = &state->epoch;
    __shared__ uint32_t failed;
    __shared__ uint32_t failed_peer;
    __shared__ uint32_t failed_waited;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        failed = atomicAdd(&state->error, 0u);
        failed_peer = UINT32_MAX;
        failed_waited = 0;
        if (failed == 0u) {
            const uint32_t prior = *epoch;
            if (prior >= UINT32_MAX - 1u) {
                failed = 1u;
                record_failure(control, state, prior, UINT32_MAX, 0);
            } else {
                state->sequence = prior + 1u;
            }
        }
    }
    grid.sync();
    const uint32_t sequence = state->sequence;
    if (atomicAdd(&state->error, 0u) != 0u) return;

    const uint64_t nbytes64 = input_elements * sizeof(T);
    if (nbytes64 > slot_bytes || nbytes64 > UINT32_MAX || nbytes64 % kPackBytes != 0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            record_failure(control, state, sequence, UINT32_MAX, 0);
        }
        grid.sync();
        return;
    }

    uint8_t* send = region + send_off + (sequence & 1u) * slot_bytes;
    const uint64_t packs = nbytes64 / kPackBytes;
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const auto* source = reinterpret_cast<const uint4*>(input);
    auto* destination = reinterpret_cast<uint4*>(send);
    for (uint64_t pack = tid; pack < packs; pack += stride) destination[pack] = source[pack];
    sys_fence();
    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlNbytes),
                       static_cast<uint32_t>(nbytes64));
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlSlotNbytes +
                                                   (sequence & 1u) * 4u),
                       static_cast<uint32_t>(nbytes64));
        sys_fence();
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control), sequence);
    }
    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x < (kWorld - 1) * stripes) {
        const uint32_t peer_index = threadIdx.x / stripes;
        const uint32_t lane = threadIdx.x % stripes;
        const uint32_t peer = peer_index >= rank ? peer_index + 1u : peer_index;
        const uint64_t flag_index = ((static_cast<uint64_t>(peer) * kSlots +
                                      (sequence & 1u)) * stripes + lane) * kFlagStride;
        (void)wait_flag_cooperative(
            reinterpret_cast<uint32_t*>(region + flag_off + flag_index), sequence,
            timeout_us, &failed, &failed_peer, &failed_waited, peer);
    }
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (failed != 0) record_failure(control, state, sequence, failed_peer, failed_waited);
        else st_gpu_release(&state->sequence, sequence);
    }
    grid.sync();
    if (atomicAdd(&state->error, 0u) != 0u) return;

    const uint8_t* recv = region + recv_off;
    if constexpr (std::is_same_v<T, float>) {
        if (kind == Kind::all_reduce_ring_sum) {
            for (uint64_t pack = tid; pack < packs; pack += stride) {
                const uint64_t index = pack * 4;
                const uint32_t part = simple8_ring_part(index, global_first, global_total);
                const float4 a = ring_rank_values(send, recv, slot_bytes, sequence, rank, 0, pack);
                const float4 b = ring_rank_values(send, recv, slot_bytes, sequence, rank, 1, pack);
                const float4 c = ring_rank_values(send, recv, slot_bytes, sequence, rank, 2, pack);
                float4 result;
                result.x = sum3_ring(a.x, b.x, c.x, part);
                result.y = sum3_ring(a.y, b.y, c.y, part);
                result.z = sum3_ring(a.z, b.z, c.z, part);
                result.w = sum3_ring(a.w, b.w, c.w, part);
                reinterpret_cast<float4*>(output)[pack] = result;
            }
        } else if (kind == Kind::all_reduce_sum) {
            for (uint64_t pack = tid; pack < packs; pack += stride) {
                float4 acc{};
                for (uint32_t peer = 0; peer < kWorld; ++peer) {
                    const float4 value = peer == rank
                        ? reinterpret_cast<const float4*>(send)[pack]
                        : reinterpret_cast<const float4*>(
                              recv + (static_cast<uint64_t>(peer) * kSlots +
                                      (sequence & 1u)) * slot_bytes)[pack];
                    if (peer == 0) acc = value;
                    else {
                        acc.x = acc.x + value.x;
                        acc.y = acc.y + value.y;
                        acc.z = acc.z + value.z;
                        acc.w = acc.w + value.w;
                    }
                }
                reinterpret_cast<float4*>(output)[pack] = acc;
            }
        } else {
            const uint64_t output_packs = output_elements / 4u;
            for (uint64_t pack = tid; pack < output_packs; pack += stride) {
                const uint32_t peer = static_cast<uint32_t>(pack / packs);
                const uint64_t offset = pack % packs;
                const uint4* peer_data = peer == rank
                    ? reinterpret_cast<const uint4*>(send)
                    : reinterpret_cast<const uint4*>(
                          recv + (static_cast<uint64_t>(peer) * kSlots +
                                  (sequence & 1u)) * slot_bytes);
                reinterpret_cast<uint4*>(output)[pack] = peer_data[offset];
            }
        }
    } else {
        for (uint64_t i = tid; i < output_elements; i += stride) {
            if (kind == Kind::all_reduce_sum) {
                output[i] = sum3_staged(send, recv, slot_bytes, sequence, rank, i);
            } else {
                const uint64_t slot_elems = input_elements;
                const uint32_t peer = static_cast<uint32_t>(i / slot_elems);
                const uint64_t offset = i % slot_elems;
                const T* peer_data = peer == rank
                    ? reinterpret_cast<const T*>(send)
                    : reinterpret_cast<const T*>(
                          recv + (static_cast<uint64_t>(peer) * kSlots +
                                  (sequence & 1u)) * slot_bytes);
                output[i] = peer_data[offset];
            }
        }
    }
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        gpu_fence();
        st_gpu_release(epoch, sequence);
        sys_fence();
        st_sys_relaxed(reinterpret_cast<uint32_t*>(control + kControlCompleted), sequence);
    }
}

struct CooperativeLimit {
    bool initialized = false;
    uint32_t threads = 0;
    cudaError_t error = cudaSuccess;
    uint32_t max_blocks = 0;
};

CooperativeLimit cooperative_limit(uint32_t threads, bool bf16) {
    static std::mutex mutex;
    static std::array<std::array<std::array<CooperativeLimit, 8>, 2>, 64> cache{};
    std::lock_guard<std::mutex> lock(mutex);
    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return {true, 0, error, 0};
    if (device < 0 || static_cast<size_t>(device) >= cache.size()) {
        return {true, 0, cudaErrorInvalidDevice, 0};
    }
    auto& entries = cache[device][bf16 ? 1 : 0];
    CooperativeLimit* limit = nullptr;
    for (auto& entry : entries) {
        if (entry.initialized && entry.threads == threads) {
            limit = &entry;
            break;
        }
        if (!entry.initialized && limit == nullptr) limit = &entry;
    }
    if (limit != nullptr && limit->initialized && limit->threads == threads) return *limit;

    if (limit == nullptr) return {true, threads, cudaErrorMemoryAllocation, 0};

    int cooperative = 0;
    error = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
    if (error != cudaSuccess) return {true, threads, error, 0};
    if (cooperative == 0) return {true, threads, cudaErrorNotSupported, 0};
    int multiprocessors = 0;
    error = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess) return {true, threads, error, 0};
    int active_blocks = 0;
    if (bf16) {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, collective_kernel_cooperative<__nv_bfloat16>, static_cast<int>(threads), 0);
    } else {
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, collective_kernel_cooperative<float>, static_cast<int>(threads), 0);
    }
    if (error != cudaSuccess) return {true, threads, error, 0};
    *limit = {true, threads, cudaSuccess,
              static_cast<uint32_t>(active_blocks * multiprocessors)};
    return *limit;
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
                   cudaStream_t stream, uint32_t blocks, uint32_t threads,
                   uint64_t global_first, uint64_t global_total) {
    if (rank >= kWorld || region_device == nullptr || device_state == nullptr ||
        stream == nullptr || (blocks != 1 && blocks != 8) ||
        threads < (kWorld - 1) * 1 || timeout_us == 0)
        return cudaErrorInvalidValue;
    if (kind == Kind::all_reduce_ring_sum &&
        (input.type != ScalarType::fp32 || output.type != ScalarType::fp32 ||
         input.elements != output.elements || global_total == 0 || global_total % 8 != 0 ||
         global_first > global_total || input.elements > global_total - global_first ||
         global_first % 4 != 0 || input.elements % 4 != 0))
        return cudaErrorInvalidValue;
    static const bool adaptive=[] {
        const char* value=std::getenv("STRATA_TP_ROCE_ADAPTIVE");return value&&value[0]=='1';
    }();
    if(adaptive&&blocks==8) {
        const uint64_t packs=input.elements*scalar_type_bytes(input.type)/16u;
        const uint64_t needed=(packs+2*threads-1)/(2*threads);
        blocks=needed<=1?1:needed<=2?2:needed<=4?4:8;
    }
    if (blocks > 1) {
        const CooperativeLimit limit = cooperative_limit(threads, input.type == ScalarType::bf16);
        if (limit.error != cudaSuccess) return limit.error;
        if (blocks > limit.max_blocks) return cudaErrorCooperativeLaunchTooLarge;
    }
    auto* state = static_cast<DeviceState*>(device_state);
    auto* region = static_cast<uint8_t*>(region_device);
    const uint32_t stripes = 1;
    switch (input.type) {
        case ScalarType::fp32: {
            if (blocks == 1) {
                collective_kernel<float><<<1, threads, 0, stream>>>(
                    static_cast<const float*>(input.data), static_cast<float*>(output.data), state, region,
                    region_layout.recv_off, region_layout.flag_off, region_layout.send_off,
                    region_layout.ctrl_off, slot_bytes, input.elements, output.elements, rank, stripes,
                    timeout_us, kind, global_first, global_total);
            } else {
                const float* in = static_cast<const float*>(input.data);
                float* out = static_cast<float*>(output.data);
                auto kernel = collective_kernel_cooperative<float>;
                uint32_t cooperative_stripes = stripes;
                uint64_t recv_off = region_layout.recv_off;
                uint64_t flag_off = region_layout.flag_off;
                uint64_t send_off = region_layout.send_off;
                uint64_t ctrl_off = region_layout.ctrl_off;
                uint64_t input_elements = input.elements;
                uint64_t output_elements = output.elements;
                void* args[] = {
                    &in, &out, &state, &region, &recv_off, &flag_off, &send_off, &ctrl_off,
                    &slot_bytes, &input_elements, &output_elements, &rank, &cooperative_stripes,
                    &timeout_us, &kind, &global_first, &global_total
                };
                return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),
                                                   dim3(blocks), dim3(threads), args, 0, stream);
            }
            break;
        }
        case ScalarType::bf16: {
            if (blocks == 1) {
                collective_kernel<__nv_bfloat16><<<1, threads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data), static_cast<__nv_bfloat16*>(output.data), state, region,
                    region_layout.recv_off, region_layout.flag_off, region_layout.send_off,
                    region_layout.ctrl_off, slot_bytes, input.elements, output.elements, rank, stripes,
                    timeout_us, kind, global_first, global_total);
            } else {
                const auto* in = static_cast<const __nv_bfloat16*>(input.data);
                auto* out = static_cast<__nv_bfloat16*>(output.data);
                auto kernel = collective_kernel_cooperative<__nv_bfloat16>;
                uint32_t cooperative_stripes = stripes;
                uint64_t recv_off = region_layout.recv_off;
                uint64_t flag_off = region_layout.flag_off;
                uint64_t send_off = region_layout.send_off;
                uint64_t ctrl_off = region_layout.ctrl_off;
                uint64_t input_elements = input.elements;
                uint64_t output_elements = output.elements;
                void* args[] = {
                    &in, &out, &state, &region, &recv_off, &flag_off, &send_off, &ctrl_off,
                    &slot_bytes, &input_elements, &output_elements, &rank, &cooperative_stripes,
                    &timeout_us, &kind, &global_first, &global_total
                };
                return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel),
                                                   dim3(blocks), dim3(threads), args, 0, stream);
            }
            break;
        }
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

}  // namespace strata::distributed::roce::cuda
