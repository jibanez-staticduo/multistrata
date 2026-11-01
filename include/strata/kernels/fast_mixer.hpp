#pragma once
#include "strata/core/tensor_parallel.hpp"
#include <cstdint>
namespace strata::kernels::tp::cuda {
void mixer_steps(const core::TensorParallelRows&, int rows, int expected_slot,
                  int context, int* steps, int* positions, int heads, int* status, void* stream);
void mixer_steps_batch(const core::TensorParallelRows&, const int* capture_slots, int rows,
                       int context, int* steps, int* positions, int* status, void* stream);
void mixer_rope_positions(const int* steps, int rows, int* keys, int* index, void* stream);
void mixer_gates(const float* alpha, const float* raw_beta, const float* dt,
                  const float* a, float* gate, float* beta, int rows, int heads, void* stream);
void mixer_conv(float* history, const float* qkv, const float* weights, float* h,
                 int rows, int channels, int key_heads, void* stream);
void mixer_conv_commit(float* history, const float* qkv, int rows, int channels, void* stream);
void mixer_recurrence(float* state, const float* h, const float* gate, const float* beta,
                       const float* z, const float* norm, float* y, int rows, int key_heads,
                       int value_heads, void* stream);
void shared_swiglu_f16(const float* gate, const float* up, uint16_t* hidden,
                       int64_t elements, void* stream);
void shared_add(float* local_sum, const float* shared, const float* gate_logits,
                int rows, int width, bool apply_sigmoid, void* stream);
void hc_pack(const FusedGrArgs* args, int rows, float* normalized, uint16_t* packed, void* stream);
void hc_silu_pack(float* lo, uint16_t* packed, int rows, void* stream);
void hc_publish_inject(const FusedGrArgs* args, const float* inject, int rows, void* stream);
void hc_packed_silu_publish(const FusedGrArgs* args, const float* combined,
                            uint16_t* packed, int rows, void* stream);
void transport_bf16_pack(const float* input, uint16_t* output, int64_t count, void* stream);
void transport_bf16_unpack(const uint16_t* input, float* output, int64_t count, void* stream);
void transport_ring_sum(const float* gathered,float* output,uint64_t count,
                         uint64_t first,uint64_t total,void* stream);
void head_local_candidate(const float* logits, float* candidates, int rows, int width, int first, int rank, void* stream);
void head_global_candidate(const float* candidates, int32_t* picks, uint32_t* flags, int rows, int flag_blocks, void* stream);
}
