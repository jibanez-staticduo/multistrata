#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels::tp::cuda {
// Original GGML NVFP4 blocks, expert-major. Local FFN width includes zero
// padding only; scales [experts,4] retain the original gate/up/down FP32 tails.
struct Nvfp4Layer {
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    const float* scales = nullptr;
    int hidden = 0, local_ff = 0, experts = 0;
    std::size_t gu_expert_bytes = 0, down_expert_bytes = 0;
    const void* grouped_blobs = nullptr;
    std::size_t grouped_blob_bytes = 0;
    std::size_t grouped_up_offset = 0, grouped_down_offset = 0, grouped_tail_offset = 0;
    int valid_ff = 0; // zero keeps the complete padded width
    bool packed_gate_up = false; // gate/up rows share one expert-major plane
    const void* decode_gu_codes = nullptr; // uint4 gate/up code pairs per 16 values
    const uint16_t* decode_gu_scales = nullptr; // original gate/up scale bytes
};
std::size_t nvfp4_workspace_bytes(int max_rows, int top_k, int hidden, int local_ff);
void nvfp4_pack_decode_gu(const Nvfp4Layer&, void* codes, uint16_t* scales, void* stream);
const char* nvfp4_prefill_dispatch();
// inputs [rows,hidden], IDs/weights [rows,top_k], weighted local sum
// output [rows,hidden]. No allocation, host readback, synchronization or NCCL.
// Invalid device IDs set sticky_status and zero their contribution. The caller
// checks status after completion, before output is committed to clients.
void nvfp4_local_sum(const Nvfp4Layer&, const float* inputs, const int32_t* ids,
                     const float* route_weights, int rows, int top_k, float* output,
                     void* scratch, std::size_t scratch_bytes, int* sticky_status,
                     void* stream, bool prefill);
}
