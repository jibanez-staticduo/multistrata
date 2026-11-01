#include "strata/kernels/tp_b12x_pack.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace pack = strata::kernels::tp::cuda;

namespace {

constexpr int kExperts = 2;
constexpr int kHidden = 192;
constexpr int kIntermediate = 128;
constexpr std::size_t kBlockBytes = 36;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceBuffer {
    void* data = nullptr;
    std::size_t bytes = 0;

    explicit DeviceBuffer(std::size_t size) : bytes(size) {
        check(cudaMalloc(&data, bytes));
    }
    ~DeviceBuffer() {
        if (data) cudaFree(data);
    }
};

std::uint8_t source_byte(int role, int expert, int row, int block, int byte) {
    if (byte < 4) return std::uint8_t(1 + role * 20 + expert * 8 + row % 7 + block * 4 + byte);
    return std::uint8_t(role * 31 + expert * 23 + row * 7 + block * 13 + byte * 3);
}

std::size_t source_row_bytes(int width) {
    return std::size_t(width / 64) * kBlockBytes;
}

std::uint8_t source_code(const std::vector<std::uint8_t>& source, int expert, int row,
                         int rows_per_expert, int width, int code) {
    const int block = code / 64;
    const int in_block = code % 64;
    const int lane = in_block % 16;
    const int sub = in_block / 16;
    const auto offset = std::size_t(expert) * rows_per_expert * source_row_bytes(width) +
                        std::size_t(row) * source_row_bytes(width) +
                        std::size_t(block) * kBlockBytes + 4 + std::size_t(sub * 8 + lane % 8);
    const std::uint8_t packed = source[offset];
    return lane < 8 ? packed & 0x0f : packed >> 4;
}

std::size_t swizzled_index(int expert, int row, int col, int rows_padded, int cols_padded) {
    const int row_block = row / 128;
    const int row_in_block = row % 128;
    const int row_group = row_in_block / 32;
    const int row_lane = row_in_block % 32;
    const int col_block = col / 4;
    const int col_lane = col % 4;
    return std::size_t(expert) * rows_padded * cols_padded +
           (((std::size_t(row_block) * (cols_padded / 4) + col_block) * 32 + row_lane) * 4 +
             row_group) *
                   4 +
               col_lane;
}

void verify_w1_codes(const std::vector<std::uint8_t>& gate, const std::vector<std::uint8_t>& up,
                     const std::vector<std::uint8_t>& packed) {
    const auto row_bytes = std::size_t(kHidden / 2);
    const auto expert_bytes = std::size_t(2 * kIntermediate) * row_bytes;
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int row = 0; row < 2 * kIntermediate; ++row) {
            const int source_row = row < kIntermediate ? row : row - kIntermediate;
            const auto& source = row < kIntermediate ? up : gate;
            for (std::size_t byte = 0; byte < row_bytes; ++byte) {
                const auto lo = source_code(source, expert, source_row, kIntermediate, kHidden, int(byte * 2));
                const auto hi = source_code(source, expert, source_row, kIntermediate, kHidden, int(byte * 2 + 1));
                const std::size_t dst = std::size_t(expert) * expert_bytes +
                                        std::size_t(row) * row_bytes + byte;
                if (packed[dst] != std::uint8_t(lo | (hi << 4)))
                    throw std::runtime_error("packed NVFP4 code mismatch");
            }
        }
    }
}

void verify_w2_codes(const std::vector<std::uint8_t>& down, const std::vector<std::uint8_t>& packed) {
    const auto row_bytes = std::size_t(kIntermediate / 2);
    const auto expert_bytes = std::size_t(kHidden) * row_bytes;
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int row = 0; row < kHidden; ++row) {
            for (std::size_t byte = 0; byte < row_bytes; ++byte) {
                const auto lo = source_code(down, expert, row, kHidden, kIntermediate, int(byte * 2));
                const auto hi = source_code(down, expert, row, kHidden, kIntermediate, int(byte * 2 + 1));
                const std::size_t dst = std::size_t(expert) * expert_bytes +
                                        std::size_t(row) * row_bytes + byte;
                if (packed[dst] != std::uint8_t(lo | (hi << 4)))
                    throw std::runtime_error("packed NVFP4 down code mismatch");
            }
        }
    }
}

}  // namespace

int main() try {
    int device_count = 0;
    const auto device_result = cudaGetDeviceCount(&device_count);
    if (device_result != cudaSuccess || device_count == 0) return 77;

    const auto sizes = pack::b12x_nvfp4_pack_sizes(kExperts, kHidden, kIntermediate);
    const auto gu_source_bytes = std::size_t(kExperts) * kIntermediate * source_row_bytes(kHidden);
    const auto down_source_bytes = std::size_t(kExperts) * kHidden * source_row_bytes(kIntermediate);
    std::vector<std::uint8_t> gate(gu_source_bytes), up(gu_source_bytes), down(down_source_bytes);
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int row = 0; row < kIntermediate; ++row) {
            for (int byte = 0; byte < int(source_row_bytes(kHidden)); ++byte) {
                gate[std::size_t(expert) * kIntermediate * source_row_bytes(kHidden) +
                     std::size_t(row) * source_row_bytes(kHidden) + byte] = source_byte(0, expert, row, byte / 36, byte % 36);
                up[std::size_t(expert) * kIntermediate * source_row_bytes(kHidden) +
                   std::size_t(row) * source_row_bytes(kHidden) + byte] = source_byte(1, expert, row, byte / 36, byte % 36);
            }
        }
        for (int row = 0; row < kHidden; ++row) {
            for (int byte = 0; byte < int(source_row_bytes(kIntermediate)); ++byte) {
                down[std::size_t(expert) * kHidden * source_row_bytes(kIntermediate) +
                     std::size_t(row) * source_row_bytes(kIntermediate) + byte] = source_byte(2, expert, row, byte / 36, byte % 36);
            }
        }
    }

    std::vector<float> tails(kExperts * 4);
    for (int expert = 0; expert < kExperts; ++expert) {
        tails[expert * 4] = 0.25f + expert;
        tails[expert * 4 + 1] = 0.5f + expert;
        tails[expert * 4 + 2] = 0.75f + expert;
        tails[expert * 4 + 3] = 99.f;
    }

    DeviceBuffer d_gate(gate.size()), d_up(up.size()), d_down(down.size()), d_tails(tails.size() * sizeof(float));
    DeviceBuffer d_w1(sizes.w1_codes), d_w1s(sizes.w1_scales), d_w1a(sizes.w1_alpha), d_a1(sizes.a1_gscale);
    DeviceBuffer d_w2(sizes.w2_codes), d_w2s(sizes.w2_scales), d_w2a(sizes.w2_alpha), d_a2(sizes.a2_gscale);
    check(cudaMemcpy(d_gate.data, gate.data(), gate.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_up.data, up.data(), up.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_down.data, down.data(), down.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(d_tails.data, tails.data(), d_tails.bytes, cudaMemcpyHostToDevice));

    pack::b12x_nvfp4_repack(d_gate.data, d_gate.bytes, d_up.data, d_up.bytes,
                            d_down.data, d_down.bytes, static_cast<const float*>(d_tails.data),
                            d_tails.bytes, sizes, d_w1.data, d_w1.bytes, d_w1s.data, d_w1s.bytes,
                            static_cast<float*>(d_w1a.data), d_w1a.bytes,
                            static_cast<float*>(d_a1.data), d_a1.bytes,
                            d_w2.data, d_w2.bytes, d_w2s.data, d_w2s.bytes,
                            static_cast<float*>(d_w2a.data), d_w2a.bytes,
                            static_cast<float*>(d_a2.data), d_a2.bytes,
                            kExperts, kHidden, kIntermediate, nullptr);
    check(cudaStreamSynchronize(nullptr));

    std::vector<std::uint8_t> w1(sizes.w1_codes), w2(sizes.w2_codes);
    std::vector<std::uint8_t> w1_scales(sizes.w1_scales), w2_scales(sizes.w2_scales);
    std::vector<float> w1_alpha(kExperts * 2), w2_alpha(kExperts), a1(kExperts), a2(kExperts);
    check(cudaMemcpy(w1.data(), d_w1.data, w1.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(w2.data(), d_w2.data, w2.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(w1_scales.data(), d_w1s.data, w1_scales.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(w2_scales.data(), d_w2s.data, w2_scales.size(), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(w1_alpha.data(), d_w1a.data, d_w1a.bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(w2_alpha.data(), d_w2a.data, d_w2a.bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(a1.data(), d_a1.data, d_a1.bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(a2.data(), d_a2.data, d_a2.bytes, cudaMemcpyDeviceToHost));
    verify_w1_codes(gate, up, w1);
    verify_w2_codes(down, w2);

    const int w1_rows_padded = 2 * kIntermediate;
    const int w1_cols_padded = (kHidden / 16 + 3) / 4 * 4;
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int row = 0; row < w1_rows_padded; ++row) {
            const int src_row = row < kIntermediate ? row : row - kIntermediate;
            const auto& src = row < kIntermediate ? up : gate;
            for (int col = 0; col < w1_cols_padded; ++col) {
                const auto index = swizzled_index(expert, row, col, w1_rows_padded, w1_cols_padded);
                const std::uint8_t expected = col < kHidden / 16
                    ? src[std::size_t(expert) * kIntermediate * source_row_bytes(kHidden) +
                          std::size_t(src_row) * source_row_bytes(kHidden) +
                          std::size_t(col / 4) * kBlockBytes + col % 4]
                    : 0;
                if (w1_scales[index] != expected) throw std::runtime_error("W1 scale swizzle mismatch");
            }
        }
    }
    const int w2_rows_padded = (kHidden + 127) / 128 * 128;
    const int w2_cols_padded = (kIntermediate / 16 + 3) / 4 * 4;
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int row = 0; row < w2_rows_padded; ++row) {
            for (int col = 0; col < w2_cols_padded; ++col) {
                const auto index = swizzled_index(expert, row, col, w2_rows_padded, w2_cols_padded);
                const std::uint8_t expected = row < kHidden && col < kIntermediate / 16
                    ? down[std::size_t(expert) * kHidden * source_row_bytes(kIntermediate) +
                           std::size_t(row) * source_row_bytes(kIntermediate) +
                           std::size_t(col / 4) * kBlockBytes + col % 4]
                    : 0;
                if (w2_scales[index] != expected) throw std::runtime_error("W2 scale swizzle or padding mismatch");
            }
        }
        if (w1_alpha[expert * 2] != tails[expert * 4] ||
            w1_alpha[expert * 2 + 1] != tails[expert * 4 + 1] ||
            w2_alpha[expert] != tails[expert * 4 + 2] || a1[expert] != 1.f || a2[expert] != 1.f)
            throw std::runtime_error("NVFP4 alpha copy mismatch");
    }
    DeviceBuffer d_compact(kExperts*sizeof(float));
    pack::b12x_nvfp4_runtime_alphas(static_cast<const float*>(d_tails.data),
        static_cast<float*>(d_compact.data),kExperts,nullptr);
    std::vector<float> compact(kExperts);
    check(cudaMemcpy(compact.data(),d_compact.data,d_compact.bytes,cudaMemcpyDeviceToHost));
    for(int e=0;e<kExperts;++e)if(compact[e]!=tails[e*4])
        throw std::runtime_error("compact FC1 alpha differs from original gate tail");
    const int32_t routes[20]={0,1,2,3,4,5,6,7,8,9,0,1,1,-1,512,5,6,7,8,9};
    DeviceBuffer d_ids(sizeof(routes)),d_wide(20*sizeof(int64_t)),d_status(sizeof(int));
    check(cudaMemcpy(d_ids.data,routes,sizeof(routes),cudaMemcpyHostToDevice));
    check(cudaMemset(d_status.data,0,sizeof(int)));
    pack::b12x_nvfp4_prepare_ids(static_cast<const int32_t*>(d_ids.data),
        static_cast<int64_t*>(d_wide.data),20,512,static_cast<int*>(d_status.data),nullptr);
    int status=0;std::vector<int64_t> wide(20);
    check(cudaMemcpy(&status,d_status.data,sizeof(status),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(wide.data(),d_wide.data,d_wide.bytes,cudaMemcpyDeviceToHost));
    if(status!=1||wide[12]!=-1||wide[13]!=-1||wide[14]!=-1)
        throw std::runtime_error("invalid B12X route escaped the sticky guard");
    for(int i=0;i<12;++i)if(wide[i]!=routes[i])
        throw std::runtime_error("valid B12X route changed during widening");
    // Adjacent positive normal scales decompose without rounding; padding is zero.
    std::vector<uint8_t> scales8(size_t(128)*4,0);
    const uint8_t scale_values[4]={56,64,72,80};
    for(int c=0;c<4;++c)scales8[swizzled_index(0,0,c,128,4)]=scale_values[c];
    DeviceBuffer d_scales8(scales8.size()),d_mx(2),d_residual(4);
    check(cudaMemcpy(d_scales8.data,scales8.data(),scales8.size(),cudaMemcpyHostToDevice));
    check(cudaMemset(d_status.data,0,sizeof(int)));
    pack::b12x_nvfp4_a8_scales(static_cast<uint8_t*>(d_scales8.data),
        static_cast<uint8_t*>(d_mx.data),static_cast<uint8_t*>(d_residual.data),
        1,1,64,static_cast<int*>(d_status.data),nullptr);
    uint8_t mx[2]{},residual[4]{};
    check(cudaMemcpy(mx,d_mx.data,2,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(residual,d_residual.data,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&status,d_status.data,4,cudaMemcpyDeviceToHost));
    if(status||mx[0]!=128||mx[1]!=130||residual[0]!=48||residual[1]!=56||
       residual[2]!=48||residual[3]!=56)
        throw std::runtime_error("B12X A8 scale decomposition differs from exact powers");
    scales8[swizzled_index(0,0,0,128,4)]=1;
    scales8[swizzled_index(0,0,1,128,4)]=64;
    check(cudaMemcpy(d_scales8.data,scales8.data(),scales8.size(),cudaMemcpyHostToDevice));
    pack::b12x_nvfp4_a8_scales(static_cast<uint8_t*>(d_scales8.data),
        static_cast<uint8_t*>(d_mx.data),static_cast<uint8_t*>(d_residual.data),
        1,1,64,static_cast<int*>(d_status.data),nullptr);
    check(cudaMemcpy(mx,d_mx.data,2,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(residual,d_residual.data,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&status,d_status.data,4,cudaMemcpyDeviceToHost));
    if(status||mx[0]!=127||residual[0]!=1||residual[1]!=64)
        throw std::runtime_error("recoverable B12X A8 scale pair lost precision");
    scales8[swizzled_index(0,0,1,128,4)]=126;
    check(cudaMemcpy(d_scales8.data,scales8.data(),scales8.size(),cudaMemcpyHostToDevice));
    pack::b12x_nvfp4_a8_scales(static_cast<uint8_t*>(d_scales8.data),
        static_cast<uint8_t*>(d_mx.data),static_cast<uint8_t*>(d_residual.data),
        1,1,64,static_cast<int*>(d_status.data),nullptr);
    check(cudaMemcpy(&status,d_status.data,4,cudaMemcpyDeviceToHost));
    if(status<=0)throw std::runtime_error("lossy B12X A8 scale decomposition was accepted");
    DeviceBuffer d_routes(4224*4),d_blocks(528*4),d_count(4);
    const int32_t unordered[10]={9,1,7,3,5,0,2,4,6,8};
    check(cudaMemcpy(d_ids.data,unordered,sizeof(unordered),cudaMemcpyHostToDevice));
    check(cudaMemset(d_status.data,0,4));
    pack::b12x_a16_pack_routes(static_cast<const int32_t*>(d_ids.data),10,
        static_cast<int*>(d_routes.data),static_cast<int*>(d_blocks.data),
        static_cast<int*>(d_count.data),static_cast<int*>(d_status.data),nullptr);
    int count=0;std::vector<int> route_rows(80),block_experts(10);
    check(cudaMemcpy(&count,d_count.data,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&status,d_status.data,4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(route_rows.data(),d_routes.data,80*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(block_experts.data(),d_blocks.data,10*4,cudaMemcpyDeviceToHost));
    if(count!=80||status)throw std::runtime_error("invalid A16 packed route count");
    for(int e=0;e<10;++e) {
        if(block_experts[e]!=e||unordered[route_rows[e*8]]!=e)
            throw std::runtime_error("A16 route expert order mismatch");
        for(int j=1;j<8;++j)if(route_rows[e*8+j]!=10)
            throw std::runtime_error("A16 route padding does not use live-route sentinel");
    }
    std::puts("B12X NVFP4 pack parity, alpha and route guard passed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "B12X NVFP4 pack parity failed: %s\n", error.what());
    return 1;
}
