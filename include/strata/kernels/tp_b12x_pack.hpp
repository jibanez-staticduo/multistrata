#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::tp::cuda {

// Device buffers use the B12X NVFP4 source-native layout. Weight codes are
// copied unchanged; E4M3 block scales are extracted and vec16-swizzled.
struct B12xNvfp4PackSizes {
    std::size_t w1_codes = 0;   // [experts, 2 * intermediate, hidden / 2]
    std::size_t w1_scales = 0;  // [experts, pad128(2 * intermediate), pad4(hidden / 16)]
    std::size_t w1_alpha = 0;   // [experts, 2], gate then up
    std::size_t a1_gscale = 0;  // [experts], dynamic activation scale = 1
    std::size_t w2_codes = 0;   // [experts, hidden, intermediate / 2]
    std::size_t w2_scales = 0;  // [experts, pad128(hidden), pad4(intermediate / 16)]
    std::size_t w2_alpha = 0;   // [experts]
    std::size_t a2_gscale = 0;  // [experts], dynamic activation scale = 1
};

// Sizes are in bytes. Geometry must use complete GGUF NVFP4 blocks.
B12xNvfp4PackSizes b12x_nvfp4_pack_sizes(int experts, int hidden, int intermediate);

void b12x_nvfp4_runtime_alphas(const float* tails, float* gate_up, int experts, void* stream);
void b12x_nvfp4_prepare_ids(const int32_t* ids, int64_t* ids64, int routes,
                          int experts, int* status, void* stream);
// Unswizzled K/32 UE8M0 plus K/16 E4M3 residual; reports any scale loss.
void b12x_nvfp4_a8_scales(const uint8_t* original_swizzled, uint8_t* mx,
                         uint8_t* residual, int experts, int rows, int cols,
                         int* status, void* stream);
void b12x_nvfp4_a16_repack(const uint8_t* codes,const uint8_t* scales,
                          uint32_t* packed_codes,uint8_t* packed_scales,
                          int experts,int rows,int cols,int row_rotation,
                          float scale_factor,void* stream);
void b12x_a16_pack_routes(const int32_t* ids,int routes,int* packed,int* blocks,
                          int* count,int* status,void* stream);
void b12x_a16_global_scales(const float* tails,float* gate_up,float* down,
                            int experts,float gu_factor,float down_factor,void* stream);

// Repack device-resident GGUF type-40 matrices on the supplied CUDA stream.
// A null stream uses CUDA's default stream.
// gate/up are [experts, intermediate, hidden] and down is
// [experts, hidden, intermediate], with 36 bytes per 64-value block.
// tails are [experts, 4] FP32 values {gate, up, down, activation}.
// The returned B12X alpha buffers retain the original gate/up/down tails;
// dynamic input quantization computes its own row scales, so A1/A2 are 1.
// This only enqueues work: it allocates no storage and performs no sync/readback.
void b12x_nvfp4_repack(const void* gate, std::size_t gate_bytes,
                       const void* up, std::size_t up_bytes,
                       const void* down, std::size_t down_bytes,
                       const float* tails, std::size_t tails_bytes,
                       const B12xNvfp4PackSizes& sizes,
                       void* w1_codes, std::size_t w1_codes_bytes,
                       void* w1_scales, std::size_t w1_scales_bytes,
                       float* w1_alpha, std::size_t w1_alpha_bytes,
                       float* a1_gscale, std::size_t a1_gscale_bytes,
                       void* w2_codes, std::size_t w2_codes_bytes,
                       void* w2_scales, std::size_t w2_scales_bytes,
                       float* w2_alpha, std::size_t w2_alpha_bytes,
                       float* a2_gscale, std::size_t a2_gscale_bytes,
                       int experts, int hidden, int intermediate, void* stream);

}  // namespace strata::kernels::tp::cuda
