// Adapted from the pinned llama.cpp quantize.cu (MIT).
// Opt-in NVFP4 activation scales use nearest E4M3(absmax/6), without a local error search.
// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4_nearest(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy, float * __restrict__ scale,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int n_expert_used) {
#if defined(BLACKWELL_MMA_AVAILABLE)

    const int64_t blocks_per_col = (ne0 + QK_FP4_MMQ - 1) / QK_FP4_MMQ;

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.y % ne2;
        const int64_t i3  = blockIdx.y / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    }
    const float * __restrict__  x_row = x + base_idx;

    float amax = 0.0f;
    if constexpr (use_aligned_float8) {
        for (int64_t i0 = 8 * threadIdx.x; i0 < ne00; i0 += 8 * blockDim.x) {
            const float * x_base = x_row + i0;
            const float8 v = reinterpret_cast<const float8 *>(x_base)[0];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
            amax = fmaxf(amax, fabsf(v.p));
            amax = fmaxf(amax, fabsf(v.q));
            amax = fmaxf(amax, fabsf(v.r));
            amax = fmaxf(amax, fabsf(v.s));
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            amax = fmaxf(amax, fabsf(x_row[i0]));
        }
    }

    amax = warp_reduce_max<WARP_SIZE>(amax);

    __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        warp_amax[warp] = amax;
    }
    __syncthreads();

    if (warp == 0) {
        amax = threadIdx.x < int(CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE) ? warp_amax[lane] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (lane == 0) {
            warp_amax[0] = amax / (6.0f * 448.0f);
            if constexpr (scatter) {
#pragma unroll
                for (int slot = 0; slot < n_expert_used; ++slot) {
                    const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
                    scale[i] = warp_amax[0];
                }
            } else {
                scale[blockIdx.y * ne1 + blockIdx.x] = warp_amax[0];
            }
        }
    }
    __syncthreads();

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;

    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_FP4_MMQ;
        const int sub = (i0_base % QK_FP4_MMQ) / QK_NVFP4_SUB;

        const float row_scale = warp_amax[0];
        const float inv_col_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;

        float vals[QK_NVFP4_SUB];
        if constexpr (use_aligned_float8) {
            const float * x_base = x_row + i0_base;
            const float8 v0 = i0_base +  7 < ne00 ? reinterpret_cast<const float8 *>(x_base)[0]     : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const float8 v1 = i0_base + 15 < ne00 ? reinterpret_cast<const float8 *>(x_base + 8)[0] : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            vals[0] = v0.x; vals[1] = v0.y; vals[2] = v0.z; vals[3] = v0.w;
            vals[4] = v0.p; vals[5] = v0.q; vals[6] = v0.r; vals[7] = v0.s;
            vals[8] = v1.x; vals[9] = v1.y; vals[10] = v1.z; vals[11] = v1.w;
            vals[12] = v1.p; vals[13] = v1.q; vals[14] = v1.r; vals[15] = v1.s;
        } else {
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const int64_t i00 = i0_base + k;
                vals[k] = i00 < ne00 ? x_row[i00] : 0.0f;
            }
        }

        uint32_t q0 = 0;
        uint32_t q1 = 0;

        float amax_sub = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
        }

        const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f);

        uint8_t fp8_code = (uint8_t) first_fp8_code;
        float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
#if CUDART_VERSION >= 12080
        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
        const float s = inv_col_scale * inv_scale;

        __nv_fp4x4_e2m1 q0_lo(make_float4(vals[0] * s, vals[8]  * s, vals[1] * s, vals[9]  * s));
        __nv_fp4x4_e2m1 q0_hi(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
        __nv_fp4x4_e2m1 q1_lo(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
        __nv_fp4x4_e2m1 q1_hi(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));

        const char2 q0_lo_c = *reinterpret_cast<char2 *>(&q0_lo);
        const char2 q0_hi_c = *reinterpret_cast<char2 *>(&q0_hi);
        const char2 q1_lo_c = *reinterpret_cast<char2 *>(&q1_lo);
        const char2 q1_hi_c = *reinterpret_cast<char2 *>(&q1_hi);

        q0 = uint32_t(uint8_t(q0_lo_c.x)) | (uint32_t(uint8_t(q0_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q0_hi_c.x)) << 16) | (uint32_t(uint8_t(q0_hi_c.y)) << 24);
        q1 = uint32_t(uint8_t(q1_lo_c.x)) | (uint32_t(uint8_t(q1_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q1_hi_c.x)) << 16) | (uint32_t(uint8_t(q1_hi_c.y)) << 24);
#else
        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
            q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 0] * inv_col_scale, inv_scale)) << (8 * k);
            q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 8] * inv_col_scale, inv_scale)) << (8 * k + 4);
            q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 4] * inv_col_scale, inv_scale)) << (8 * k);
            q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 12] * inv_col_scale, inv_scale)) << (8 * k + 4);
        }
#endif // CUDART_VERSION >= 12080

        if constexpr (scatter) {
#pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
                block_fp4_mmq * yb = y + (k_block * ne1 + i);
                uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
                yqs[2 * sub + 0] = q0;
                yqs[2 * sub + 1] = q1;
                reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
            }
        } else {
            block_fp4_mmq * yb = y + (blockIdx.y * ((int64_t) blocks_per_col * ne1) + k_block * ne1 + blockIdx.x);
            uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
            yqs[2 * sub + 0] = q0;
            yqs[2 * sub + 1] = q1;
            reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
        }
    }
#else
    GGML_UNUSED_VARS(x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, n_expert_used);
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)

}

