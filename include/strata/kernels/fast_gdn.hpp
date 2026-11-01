#pragma once
#include <cstddef>
namespace strata::kernels::tp::cuda {
size_t fast_gdn_workspace_bytes(int key_heads);
void fast_gdn_prompt(float* state,const float* h,const float* gate,const float* beta,
                       const float* z,const float* norm,float* output,int rows,int key_heads,
                       void* workspace,size_t bytes,void* stream);
}
