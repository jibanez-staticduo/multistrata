#include "strata/kernels/tp_b12x_pack.hpp"

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::kernels::tp::cuda {
namespace {

constexpr std::size_t kNvfp4BlockBytes = 36;
constexpr std::size_t kNvfp4BlockValues = 64;
constexpr std::size_t kScaleValues = 16;
constexpr std::size_t kScaleRows = 128;
constexpr std::size_t kScaleCols = 4;

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::invalid_argument("B12X NVFP4 pack size overflow");
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void check_buffer(const void* ptr, std::size_t actual, std::size_t required, const char* name) {
    if (!ptr || actual < required) throw std::invalid_argument(name);
}

__device__ __forceinline__ std::size_t swizzled_scale_index(
    int expert, int row, int col, int rows_padded, int cols_padded) {
    const int row_block = row / 128;
    const int row_in_block = row % 128;
    const int row_group = row_in_block / 32;
    const int row_lane = row_in_block % 32;
    const int col_block = col / 4;
    const int col_lane = col % 4;
    const std::size_t expert_stride = std::size_t(rows_padded) * cols_padded;
    const std::size_t within = (((std::size_t(row_block) * (cols_padded / 4) + col_block) * 32 +
                                  row_lane) * 4 + row_group) * 4 + col_lane;
    return std::size_t(expert) * expert_stride + within;
}

__device__ __forceinline__ std::uint8_t nvfp4_code(
    const std::uint8_t* source, std::size_t expert_base, std::size_t row_base,
    int code_index) {
    const int block = code_index / 64;
    const int in_block = code_index % 64;
    const int sub = in_block / 16;
    const int lane = in_block % 16;
    const std::size_t byte_index = expert_base + row_base +
        std::size_t(block) * kNvfp4BlockBytes + 4 + std::size_t(sub * 8 + lane % 8);
    const std::uint8_t packed = source[byte_index];
    return lane < 8 ? packed & 0x0f : packed >> 4;
}

__global__ void repack_codes(const std::uint8_t* gate, const std::uint8_t* up,
                             const std::uint8_t* down, std::uint8_t* w1,
                             std::uint8_t* w2, int experts, int hidden, int intermediate,
                             std::size_t w1_row_bytes, std::size_t w2_row_bytes,
                             std::size_t w1_expert_bytes, std::size_t w2_expert_bytes,
                             std::size_t w1_total, std::size_t w2_total) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < w1_total) {
        const std::size_t row_bytes = hidden / 2;
        const int row = int((i / row_bytes) % (2 * intermediate));
        const int expert = int(i / (row_bytes * std::size_t(2 * intermediate)));
        const std::size_t in_row_byte = i % row_bytes;
        const std::uint8_t* source = row < intermediate ? up : gate;
        const int source_row = row < intermediate ? row : row - intermediate;
        const std::size_t expert_base = std::size_t(expert) * w1_expert_bytes;
        const std::size_t row_base = std::size_t(source_row) * w1_row_bytes;
        const int code = int(in_row_byte * 2);
        w1[i] = nvfp4_code(source, expert_base, row_base, code) |
                std::uint8_t(nvfp4_code(source, expert_base, row_base, code + 1) << 4);
    }
    if (i < w2_total) {
        const std::size_t row_bytes = intermediate / 2;
        const int row = int((i / row_bytes) % hidden);
        const int expert = int(i / (row_bytes * std::size_t(hidden)));
        const std::size_t in_row_byte = i % row_bytes;
        const std::size_t expert_base = std::size_t(expert) * w2_expert_bytes;
        const std::size_t row_base = std::size_t(row) * w2_row_bytes;
        const int code = int(in_row_byte * 2);
        w2[i] = nvfp4_code(down, expert_base, row_base, code) |
                std::uint8_t(nvfp4_code(down, expert_base, row_base, code + 1) << 4);
    }
}

__global__ void repack_scales(const std::uint8_t* gate, const std::uint8_t* up,
                              const std::uint8_t* down, std::uint8_t* w1,
                              std::uint8_t* w2, int experts, int hidden, int intermediate,
                              int w1_rows_padded, int w1_cols_padded,
                              int w2_rows_padded, int w2_cols_padded,
                              std::size_t w1_row_bytes, std::size_t w2_row_bytes,
                              std::size_t w1_expert_bytes, std::size_t w2_expert_bytes,
                              std::size_t w1_total, std::size_t w2_total) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < w1_total) {
        const int cols = hidden / int(kScaleValues);
        const std::size_t expert_stride = std::size_t(w1_rows_padded) * w1_cols_padded;
        const int expert = int(i / expert_stride);
        const std::size_t local = i % expert_stride;
        const int local_row = int(local / std::size_t(w1_cols_padded));
        const int scale_col = int(local % std::size_t(w1_cols_padded));
        const std::size_t dst = swizzled_scale_index(expert, local_row, scale_col,
                                                      w1_rows_padded, w1_cols_padded);
        if (local_row < 2 * intermediate && scale_col < cols) {
            const int source_row = local_row < intermediate ? local_row : local_row - intermediate;
            const std::uint8_t* source = local_row < intermediate ? up : gate;
            const std::size_t src = std::size_t(expert) * w1_expert_bytes +
                std::size_t(source_row) * w1_row_bytes +
                std::size_t(scale_col / 4) * kNvfp4BlockBytes + std::size_t(scale_col % 4);
            w1[dst] = source[src];
        } else {
            w1[dst] = 0;
        }
    }
    if (i < w2_total) {
        const int cols = intermediate / int(kScaleValues);
        const std::size_t expert_stride = std::size_t(w2_rows_padded) * w2_cols_padded;
        const int expert = int(i / expert_stride);
        const std::size_t local = i % expert_stride;
        const int local_row = int(local / std::size_t(w2_cols_padded));
        const int scale_col = int(local % std::size_t(w2_cols_padded));
        const std::size_t dst = swizzled_scale_index(expert, local_row, scale_col,
                                                      w2_rows_padded, w2_cols_padded);
        if (local_row < hidden && scale_col < cols) {
            const std::size_t src = std::size_t(expert) * w2_expert_bytes +
                std::size_t(local_row) * w2_row_bytes +
                std::size_t(scale_col / 4) * kNvfp4BlockBytes + std::size_t(scale_col % 4);
            w2[dst] = down[src];
        } else {
            w2[dst] = 0;
        }
    }
}

__global__ void copy_alphas(const float* tails, float* w1_alpha, float* a1,
                            float* w2_alpha, float* a2, int experts) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts) return;
    w1_alpha[std::size_t(e) * 2] = tails[std::size_t(e) * 4];
    w1_alpha[std::size_t(e) * 2 + 1] = tails[std::size_t(e) * 4 + 1];
    w2_alpha[e] = tails[std::size_t(e) * 4 + 2];
    a1[e] = 1.0f;
    a2[e] = 1.0f;
}

void check_launch() {
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void compact_alpha(const float* tails, float* alpha, int experts) {
    const int e=blockIdx.x*blockDim.x+threadIdx.x;
    if(e<experts)alpha[e]=tails[size_t(e)*4];
}
__global__ void a16_global(const float* tails,float* gu,float* down,int experts,float g,float d) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<experts){gu[i]=tails[size_t(i)*4]*g;down[i]=tails[size_t(i)*4+2]*d;}
}
__global__ void prepare_ids(const int32_t* ids, int64_t* ids64, int routes,
                            int experts, int* status) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=routes)return;
    const int e=ids[i];
    bool invalid=e<0||e>=experts;
    for(int j=0;j<i%10;++j)invalid|=ids[i-i%10+j]==e;
    if(invalid)atomicExch(status,1);
    ids64[i]=invalid?-1:e;
}
__device__ float fp8_value(uint8_t x) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(x,__NV_E4M3)));
}
__global__ void a8_scales(const uint8_t* original,uint8_t* mx,uint8_t* residual,
                          int experts,int rows,int cols,int* status) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int pairs=cols/32;
    if(i>=size_t(experts)*rows*pairs)return;
    const int e=i/(size_t(rows)*pairs),row=(i/pairs)%rows,col=(i%pairs)*2;
    const int padded_rows=(rows+127)/128*128,padded_cols=(cols/16+3)/4*4;
    const uint8_t a=original[swizzled_scale_index(e,row,col,padded_rows,padded_cols)];
    const uint8_t b=original[swizzled_scale_index(e,row,col+1,padded_rows,padded_cols)];
    const float x=fp8_value(a),y=fp8_value(b),maximum=fmaxf(x,y);
    int exponent=maximum>0.f?ilogbf(maximum):0;
    float scale=ldexpf(1.f,exponent);
    auto ra=__nv_cvt_float_to_fp8(x/scale,__NV_SATFINITE,__NV_E4M3);
    auto rb=__nv_cvt_float_to_fp8(y/scale,__NV_SATFINITE,__NV_E4M3);
    // A one-bit exponent shift preserves rare small partners without rounding
    // their residual to zero; the original scales must still reconstruct exactly.
    if(fp8_value(ra)*scale!=x||fp8_value(rb)*scale!=y) {
        --exponent;scale*=0.5f;
        ra=__nv_cvt_float_to_fp8(x/scale,__NV_SATFINITE,__NV_E4M3);
        rb=__nv_cvt_float_to_fp8(y/scale,__NV_SATFINITE,__NV_E4M3);
    }
    if(fp8_value(ra)*scale!=x||fp8_value(rb)*scale!=y)atomicCAS(status,0,int(i)+1);
    mx[i]=uint8_t(exponent+127);
    const size_t at=(size_t(e)*rows+row)*(cols/16)+col;
    residual[at]=ra;residual[at+1]=rb;
}
__global__ void a16_codes(const uint8_t* src,uint32_t* dst,int experts,int rows,int cols,int rotation) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(experts)*rows*cols/8)return;
    const int p=i%128,nt=(i/128)%(rows/64),kt=(i/(128*(rows/64)))%(cols/16);
    const int e=i/(size_t(rows)*cols/8),tid=p/4,warp=p%4;
    const int offsets[4]={0,1,8,9},order[8]={0,2,4,6,1,3,5,7};
    uint32_t word=0;
    for(int slot=0;slot<8;++slot) {
        const int j=order[slot],k=(tid%4)*2+offsets[j%4];
        const int row=(nt*64+warp*16+tid/4+(j>=4?8:0)+rotation)%rows;
        const int col=kt*16+k;
        const uint8_t b=src[(size_t(e)*rows+row)*(cols/2)+col/2];
        word|=uint32_t((b>>(4*(col%2)))&15)<<(4*slot);
    }
    dst[i]=word;
}
__global__ void a16_scales(const uint8_t* src,uint8_t* dst,int experts,int rows,int cols,int rotation,float factor) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(experts)*rows*(cols/16))return;
    const int row_permuted=i%rows,k=(i/rows)%(cols/16),e=i/(size_t(rows)*(cols/16));
    // B12X applies its 64-row permutation, then swaps bytes 1 and 2 in groups of four.
    const int within=row_permuted%4;
    const int swapped=row_permuted-within+(within==1?2:within==2?1:within);
    const int perm=(swapped%64)/8+8*(swapped%8);
    const int row=(swapped/64*64+perm+rotation)%rows;
    const int padded_rows=(rows+127)/128*128,padded_cols=(cols/16+3)/4*4;
    const float value=fp8_value(src[swizzled_scale_index(e,row,k,padded_rows,padded_cols)]);
    __half h=__float2half_rn(value*factor);
    h=__hmul(h,__float2half(128.f));
    if(__half2float(h)<2.f)h=__float2half(0.f);
    const uint16_t bits=__half_as_ushort(h)<<1;
    dst[i]=uint8_t(bits>>8);
}
__global__ void a16_routes(const int32_t* ids,int routes,int* packed,int* blocks,int* count,int* status) {
    __shared__ int offsets[513],counts[512];
    const int tid=threadIdx.x;
    int n=0;
    for(int r=0;r<routes;++r)n+=ids[r]==tid;
    counts[tid]=n;
    __syncthreads();
    if(!tid) {
        int offset=0;
        for(int e=0;e<512;++e){offsets[e]=offset;offset+=(counts[e]+7)/8*8;}
        offsets[512]=offset;*count=offset;
    }
    __syncthreads();
    for(int at=tid;at<offsets[512];at+=512)packed[at]=routes;
    for(int at=offsets[tid]/8;at<offsets[tid+1]/8;++at)blocks[at]=tid;
    __syncthreads();
    if(tid<routes) {
        const int e=ids[tid];
        bool invalid=e<0||e>=512;
        for(int j=0;j<tid%10;++j)invalid|=ids[tid-tid%10+j]==e;
        if(invalid)atomicExch(status,1);
        if(e>=0&&e<512) {
            int ordinal=0;for(int r=0;r<tid;++r)ordinal+=ids[r]==e;
            packed[offsets[e]+ordinal]=tid;
        }
    }
}

}  // namespace

void b12x_nvfp4_runtime_alphas(const float* tails,float* gate_up,int experts,void* stream) {
    compact_alpha<<<(experts+255)/256,256,0,(cudaStream_t)stream>>>(tails,gate_up,experts);
    check_launch();
}
void b12x_nvfp4_prepare_ids(const int32_t* ids,int64_t* ids64,int routes,
                           int experts,int* status,void* stream) {
    prepare_ids<<<(routes+255)/256,256,0,(cudaStream_t)stream>>>(ids,ids64,routes,experts,status);
    check_launch();
}
void b12x_nvfp4_a8_scales(const uint8_t* original,uint8_t* mx,uint8_t* residual,
                          int experts,int rows,int cols,int* status,void* stream) {
    if(!original||!mx||!residual||!status||experts<1||rows<1||cols<32||cols%32)
        throw std::invalid_argument("invalid B12X A8 scale conversion");
    a8_scales<<<(size_t(experts)*rows*(cols/32)+255)/256,256,0,(cudaStream_t)stream>>>(
        original,mx,residual,experts,rows,cols,status);
    check_launch();
}
void b12x_nvfp4_a16_repack(const uint8_t* codes,const uint8_t* scales,
                          uint32_t* packed_codes,uint8_t* packed_scales,
                          int experts,int rows,int cols,int rotation,float factor,void* stream) {
    if(!codes||!scales||!packed_codes||!packed_scales||experts<1||rows%64||cols%16||
       rows<64||cols<16||rotation<0||rotation>=rows||!(factor>=1.f))
        throw std::invalid_argument("invalid B12X A16 weight repack");
    const size_t words=size_t(experts)*rows*cols/8;
    a16_codes<<<(words+255)/256,256,0,(cudaStream_t)stream>>>(codes,packed_codes,experts,rows,cols,rotation);
    const size_t n=size_t(experts)*rows*(cols/16);
    a16_scales<<<(n+255)/256,256,0,(cudaStream_t)stream>>>(scales,packed_scales,experts,rows,cols,rotation,factor);
    check_launch();
}
void b12x_a16_pack_routes(const int32_t* ids,int routes,int* packed,int* blocks,
                          int* count,int* status,void* stream) {
    if(!ids||!packed||!blocks||!count||!status||routes<1||routes>400)
        throw std::invalid_argument("invalid B12X A16 route packing");
    a16_routes<<<1,512,0,(cudaStream_t)stream>>>(ids,routes,packed,blocks,count,status);
    check_launch();
}
void b12x_a16_global_scales(const float* tails,float* gu,float* down,int experts,
                            float gu_factor,float down_factor,void* stream) {
    a16_global<<<(experts+255)/256,256,0,(cudaStream_t)stream>>>(tails,gu,down,experts,
        ldexpf(1.f,119)/gu_factor,ldexpf(1.f,119)/down_factor);
    check_launch();
}

B12xNvfp4PackSizes b12x_nvfp4_pack_sizes(int experts, int hidden, int intermediate) {
    if (experts <= 0 || hidden <= 0 || intermediate <= 0 ||
        hidden % int(kNvfp4BlockValues) != 0 || intermediate % int(kNvfp4BlockValues) != 0)
        throw std::invalid_argument("B12X NVFP4 dimensions must be positive multiples of 64");
    const auto e = std::size_t(experts);
    const auto w1_rows = checked_mul(2, std::size_t(intermediate));
    const auto w1_cols = std::size_t(hidden) / kScaleValues;
    const auto w2_rows = std::size_t(hidden);
    const auto w2_cols = std::size_t(intermediate) / kScaleValues;
    B12xNvfp4PackSizes sizes;
    sizes.w1_codes = checked_mul(checked_mul(e, w1_rows), std::size_t(hidden) / 2);
    sizes.w1_scales = checked_mul(checked_mul(e, align_up(w1_rows, kScaleRows)),
                                  align_up(w1_cols, kScaleCols));
    sizes.w1_alpha = checked_mul(checked_mul(e, 2), sizeof(float));
    sizes.a1_gscale = checked_mul(e, sizeof(float));
    sizes.w2_codes = checked_mul(checked_mul(e, w2_rows), std::size_t(intermediate) / 2);
    sizes.w2_scales = checked_mul(checked_mul(e, align_up(w2_rows, kScaleRows)),
                                  align_up(w2_cols, kScaleCols));
    sizes.w2_alpha = checked_mul(e, sizeof(float));
    sizes.a2_gscale = checked_mul(e, sizeof(float));
    return sizes;
}

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
                       int experts, int hidden, int intermediate, void* stream) {
    const auto expected = b12x_nvfp4_pack_sizes(experts, hidden, intermediate);
    if (sizes.w1_codes != expected.w1_codes || sizes.w1_scales != expected.w1_scales ||
        sizes.w1_alpha != expected.w1_alpha || sizes.a1_gscale != expected.a1_gscale ||
        sizes.w2_codes != expected.w2_codes || sizes.w2_scales != expected.w2_scales ||
        sizes.w2_alpha != expected.w2_alpha || sizes.a2_gscale != expected.a2_gscale)
        throw std::invalid_argument("B12X NVFP4 pack geometry or stream mismatch");

    const auto e = std::size_t(experts);
    const auto hidden_blocks = std::size_t(hidden) / kNvfp4BlockValues;
    const auto intermediate_blocks = std::size_t(intermediate) / kNvfp4BlockValues;
    const auto gu_row_bytes = checked_mul(hidden_blocks, kNvfp4BlockBytes);
    const auto down_row_bytes = checked_mul(intermediate_blocks, kNvfp4BlockBytes);
    const auto gu_expert_bytes = checked_mul(std::size_t(intermediate), gu_row_bytes);
    const auto down_expert_bytes = checked_mul(std::size_t(hidden), down_row_bytes);
    check_buffer(gate, gate_bytes, checked_mul(e, gu_expert_bytes), "B12X NVFP4 gate buffer too small");
    check_buffer(up, up_bytes, checked_mul(e, gu_expert_bytes), "B12X NVFP4 up buffer too small");
    check_buffer(down, down_bytes, checked_mul(e, down_expert_bytes), "B12X NVFP4 down buffer too small");
    check_buffer(tails, tails_bytes, checked_mul(checked_mul(e, 4), sizeof(float)), "B12X NVFP4 tails buffer too small");
    check_buffer(w1_codes, w1_codes_bytes, sizes.w1_codes, "B12X NVFP4 W1 code buffer too small");
    check_buffer(w1_scales, w1_scales_bytes, sizes.w1_scales, "B12X NVFP4 W1 scale buffer too small");
    check_buffer(w1_alpha, w1_alpha_bytes, sizes.w1_alpha, "B12X NVFP4 W1 alpha buffer too small");
    check_buffer(a1_gscale, a1_gscale_bytes, sizes.a1_gscale, "B12X NVFP4 A1 scale buffer too small");
    check_buffer(w2_codes, w2_codes_bytes, sizes.w2_codes, "B12X NVFP4 W2 code buffer too small");
    check_buffer(w2_scales, w2_scales_bytes, sizes.w2_scales, "B12X NVFP4 W2 scale buffer too small");
    check_buffer(w2_alpha, w2_alpha_bytes, sizes.w2_alpha, "B12X NVFP4 W2 alpha buffer too small");
    check_buffer(a2_gscale, a2_gscale_bytes, sizes.a2_gscale, "B12X NVFP4 A2 scale buffer too small");

    const auto w1_scale_rows_pad = int(align_up(std::size_t(2) * intermediate, kScaleRows));
    const auto w1_scale_cols_pad = int(align_up(std::size_t(hidden) / kScaleValues, kScaleCols));
    const auto w2_scale_rows_pad = int(align_up(std::size_t(hidden), kScaleRows));
    const auto w2_scale_cols_pad = int(align_up(std::size_t(intermediate) / kScaleValues, kScaleCols));
    const auto w1_scale_total = checked_mul(e, checked_mul(std::size_t(w1_scale_rows_pad),
                                                           std::size_t(w1_scale_cols_pad)));
    const auto w2_scale_total = checked_mul(e, checked_mul(std::size_t(w2_scale_rows_pad),
                                                           std::size_t(w2_scale_cols_pad)));
    const auto blocks = unsigned(256);
    const auto cs = static_cast<cudaStream_t>(stream);

    repack_codes<<<unsigned((std::max(sizes.w1_codes, sizes.w2_codes) + blocks - 1) / blocks),
                   blocks, 0, cs>>>(
        static_cast<const std::uint8_t*>(gate), static_cast<const std::uint8_t*>(up),
        static_cast<const std::uint8_t*>(down), static_cast<std::uint8_t*>(w1_codes),
        static_cast<std::uint8_t*>(w2_codes), experts, hidden, intermediate,
        gu_row_bytes, down_row_bytes, gu_expert_bytes, down_expert_bytes,
        sizes.w1_codes, sizes.w2_codes);
    check_launch();

    repack_scales<<<unsigned((std::max(w1_scale_total, w2_scale_total) + blocks - 1) / blocks),
                    blocks, 0, cs>>>(
        static_cast<const std::uint8_t*>(gate), static_cast<const std::uint8_t*>(up),
        static_cast<const std::uint8_t*>(down), static_cast<std::uint8_t*>(w1_scales),
        static_cast<std::uint8_t*>(w2_scales), experts, hidden, intermediate,
        w1_scale_rows_pad, w1_scale_cols_pad, w2_scale_rows_pad, w2_scale_cols_pad,
        gu_row_bytes, down_row_bytes, gu_expert_bytes, down_expert_bytes,
        w1_scale_total, w2_scale_total);
    check_launch();

    copy_alphas<<<unsigned((experts + 255) / 256), 256, 0, cs>>>(
        tails, w1_alpha, a1_gscale, w2_alpha, a2_gscale, experts);
    check_launch();
}

}  // namespace strata::kernels::tp::cuda
