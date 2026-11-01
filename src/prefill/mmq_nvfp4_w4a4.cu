// src/prefill/mmq_nvfp4_w4a4.cu - NVFP4 MMQ with FP4 activations on Blackwell (W4A4, opt-in: STRATA_PREFILL_NVFP4=w4a4).
//
// RTX 50's FP4 MMA exists only in the arch-specific target (sm_120a), so this is the one translation unit CMake
// builds with 12x -> 12xa (strata_mmq_w4a4); everything else, the default NVFP4 path included, keeps the plain
// architectures and runs on every Blackwell card (sm_121 too). Here mmq.cuh's NVFP4 tile loader and quantize.cu's
// FP4 activation quantizer see BLACKWELL_MMA_AVAILABLE together - they must agree on the activations' layout.
// Their non-static symbols are renamed so they cannot fold with strata_mmq's plain-arch copies.
#include "common.cuh"

#define mul_mat_q_switch_J strata_w4a4_mul_mat_q_switch_J
#define mul_mat_q_case strata_w4a4_mul_mat_q_case
#define quantize_row_q8_1_cuda strata_w4a4_quantize_row_q8_1_cuda
#define quantize_mmq_q8_1_cuda strata_w4a4_quantize_mmq_q8_1_cuda
#define quantize_scatter_mmq_q8_1_cuda strata_w4a4_quantize_scatter_mmq_q8_1_cuda
#define quantize_scatter_mmq_fp4_cuda strata_w4a4_quantize_scatter_mmq_fp4_cuda
#define quantize_mmq_fp4_cuda strata_w4a4_quantize_mmq_fp4_cuda
#include "mmq.cuh"
#include "quantize.cu"
#include "nvfp4_nearest.cuh"
#include "../../include/strata/prefill/moe_mmq.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace {
// 1 when this unit's device code has the FP4 MMA (built for the running card's 12xa); a card without an image
// here (an architecture not in the list, or sm_121 against a 120a build) fails the launch: W4A4 is then off
__global__ void w4a4_probe_kernel(int* out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    *out = 1;
#else
    *out = 0;
#endif
}
}  // namespace

namespace strata::prefill::mmq {
bool nvfp4_nearest_scales() {
    static const bool enabled=[] { const char* e=std::getenv("STRATA_TP_NVFP4_NEAREST");return e&&e[0]=='1'; }();
    return enabled;
}
bool w4a4_available() {
    int* d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    w4a4_probe_kernel<<<1, 1>>>(d);
    const bool ran = cudaGetLastError() == cudaSuccess && cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    (void) cudaGetLastError();
    cudaFree(d);
    return ran && h == 1;
}
void run_nvfp4_w4a4(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_w4a4_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}
void quantize_nvfp4_w4a4(const float* x, const int32_t* ids, void* xq, float* yscale, bool aligned, int64_t cols,
                         int64_t ld, int64_t rows, int64_t padded, cudaStream_t s) {
    if(nvfp4_nearest_scales()) {
        if(aligned)quantize_mmq_nvfp4_nearest<false,true><<<rows,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,s>>>(
            x,ids,xq,yscale,cols,ld,rows*ld,rows*ld,padded,rows,1,0);
        else quantize_mmq_nvfp4_nearest<false,false><<<rows,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,s>>>(
            x,ids,xq,yscale,cols,ld,rows*ld,rows*ld,padded,rows,1,0);
        return;
    }
    strata_w4a4_quantize_mmq_fp4_cuda(x, ids, xq, yscale, GGML_TYPE_NVFP4, aligned, cols, ld, rows * ld, rows * ld,
                                      padded, rows, 1, 1, s);
}
void quantize_nvfp4_scatter(const float* x, const int32_t* slot, void* xq, float* scales,
                            int64_t cols, int64_t ld, int64_t tokens, int k_used, void* stream) {
    if (tokens <= 0) return;
    if (!x || !slot || !xq || !scales || cols <= 0 || cols % 64 || ld < cols || k_used <= 0)
        throw std::invalid_argument("invalid NVFP4 scatter quantization");
    const bool aligned = reinterpret_cast<uintptr_t>(x) % 32 == 0 && size_t(ld) * sizeof(float) % 32 == 0;
    if(nvfp4_nearest_scales()) {
        const int64_t padded=((cols+511)/512)*512;
        if(aligned)quantize_mmq_nvfp4_nearest<true,true><<<tokens,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,(cudaStream_t)stream>>>(
            x,slot,xq,scales,cols,0,ld,0,padded,tokens*k_used,1,k_used);
        else quantize_mmq_nvfp4_nearest<true,false><<<tokens,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,(cudaStream_t)stream>>>(
            x,slot,xq,scales,cols,0,ld,0,padded,tokens*k_used,1,k_used);
        const auto result=cudaGetLastError();if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
        return;
    }
    strata_w4a4_quantize_scatter_mmq_fp4_cuda(x, slot, xq, scales, GGML_TYPE_NVFP4, aligned,
        cols, ld, ((cols + 511) / 512) * 512, tokens, tokens * k_used, k_used,
        static_cast<cudaStream_t>(stream));
    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
}  // namespace strata::prefill::mmq

namespace {
template<int prefill_j>
__global__ void prefill_tile_bounds(const int* bounds, int* tiles, int experts) {
    if (threadIdx.x) return;
    int total = 0;
    tiles[0] = 0;
    for (int e = 0; e < experts; ++e) {
        total += (bounds[e + 1] - bounds[e] + prefill_j - 1) / prefill_j;
        tiles[e + 1] = total;
    }
}
template<int prefill_j>
__global__ void prefill_tiles(const char* __restrict__ weights, size_t expert_bytes,
    const int* __restrict__ bounds, const int* __restrict__ tiles, const int* __restrict__ ids,
    const int* __restrict__ y, const float* __restrict__ scales, int experts, int routes,
    int rows, int cols, int ld, float* __restrict__ dst) {
    constexpr int I = ggml_cuda_mmq_get_I(GGML_TYPE_NVFP4, prefill_j, false);
    const int tile = int(blockIdx.y);
    if (tile >= tiles[experts]) return;
    int first = 0, last = experts;
    while (first + 1 < last) {
        const int middle = (first + last) / 2;
        if (tiles[middle] <= tile) first = middle;
        else last = middle;
    }
    const int lo = bounds[first] + (tile - tiles[first]) * prefill_j;
    const int count = bounds[first + 1] - lo;
    extern __shared__ int shared[];
    const int j = threadIdx.y * 32 + threadIdx.x;
    if (j < prefill_j) shared[j] = j < count ? ids[lo + j] : 0;
    __syncthreads();
    mul_mat_q_process_tile<GGML_TYPE_NVFP4, prefill_j, false, false>(
        weights + size_t(first) * expert_bytes, int(blockIdx.x) * I * (cols / 64),
        y + lo * (sizeof(block_q8_1_mmq) / sizeof(int)), shared, dst + int(blockIdx.x) * I,
        nullptr, scales + lo, cols / 64, routes, ld, rows - int(blockIdx.x) * I - 1,
        count - 1, 0, cols / 64);
}
struct PrefillTileConfig { int threads = 0, shared = 0, rows = 0; };
template<int prefill_j>
PrefillTileConfig prefill_tile_config() {
    static const PrefillTileConfig config = [] {
        PrefillTileConfig result;
        const int device = ggml_cuda_get_device();
        const auto& info = ggml_cuda_info().devices[device];
        const auto c = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, prefill_j, false, info.cc);
        if (c.type == GGML_TYPE_COUNT || c.nthreads != 256) return result;
        const auto bytes = mmq_get_nbytes_shared(c, info.cc);
        if (bytes > info.smpbo || cudaFuncSetAttribute(prefill_tiles<prefill_j>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess) return result;
        result.threads = c.nthreads; result.shared = int(bytes); result.rows = c.I;
        return result;
    }();
    return config;
}
} // namespace
namespace strata::prefill::mmq {
bool nvfp4_prefill_tiles_available() { return prefill_tile_config<128>().threads != 0; }
bool nvfp4_decode_tiles_available() { return prefill_tile_config<8>().threads != 0; }
template<int prefill_j>
void nvfp4_tiles(const Product& p, int32_t* tiles, void* stream) {
    const auto c = prefill_tile_config<prefill_j>();
    if (!c.threads || p.type != 40 || p.n != 512 || p.tab || !p.y_scale || !tiles ||
        p.w_rows % 128 || (p.w_cols != 256 && p.w_cols != 2560) ||
        p.total_rows < 1 || p.total_rows > 81920 || p.max_rows < 1 || p.max_rows > 8192)
        throw std::invalid_argument("unsupported resident NVFP4 tile product");
    const auto cs = static_cast<cudaStream_t>(stream);
    prefill_tile_bounds<prefill_j><<<1,32,0,cs>>>(p.bounds, tiles, p.n);
    // Every nonempty tile consumes at least one route, including skew.
    const int bound = int(std::min((p.total_rows + prefill_j - 1) / prefill_j + p.n - 1,
                                  std::min(p.total_rows, p.n * ((p.max_rows + prefill_j - 1) / prefill_j))));
    prefill_tiles<prefill_j><<<dim3((p.w_rows + c.rows - 1) / c.rows, bound),
        dim3(32,c.threads / 32),c.shared,cs>>>(static_cast<const char*>(p.w),p.expert_bytes,
        p.bounds,tiles,p.ids,static_cast<const int*>(p.xq),p.y_scale,p.n,int(p.total_rows),
        int(p.w_rows),int(p.w_cols),int(p.ld_dst),p.dst);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void nvfp4_prefill_tiles(const Product& p, int32_t* tiles, void* stream) {
    nvfp4_tiles<128>(p,tiles,stream);
}
void nvfp4_decode_tiles(const Product& p, int32_t* tiles, void* stream) {
    nvfp4_tiles<8>(p,tiles,stream);
}
} // namespace strata::prefill::mmq
