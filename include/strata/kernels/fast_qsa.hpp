#pragma once
#include "strata/kernels/qsa_decode_attn.hpp"
namespace strata::kernels::tp::cuda::fast_qsa {
using namespace strata::kernels;
uint64_t qsa_decode_attn_scratch_floats(int64_t cap,const QsaShapes&);
void qsa_prompt_attn_set_imma(int);
void qsa_decode_attn_batch(const float*,const QsaAttnPools&,const int32_t*,const int32_t*,
                           int64_t,const QsaShapes&,float*,float*,int64_t,void*);
void qsa_decode_attn_batch_rows(const float*,const QsaAttnPoolsRows&,const int32_t*,const int32_t*,
                               int64_t,const QsaShapes&,float*,float*,int64_t,void*);
bool qsa_prompt_attn_batch(const float*,const QsaAttnPools&,const int32_t*,const int32_t*,
                           int64_t,const QsaShapes&,float*,int64_t,void*);
}
