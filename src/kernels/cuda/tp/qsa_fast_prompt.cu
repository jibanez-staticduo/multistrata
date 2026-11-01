// Compile the original tensor-core attention for the local four-head group.
#include "strata/kernels/fast_qsa.hpp"
#define STRATA_QSA_LOCAL_GROUPS 4
#define STRATA_QSA_KERNEL_NAMESPACE strata::kernels::tp::cuda::fast_qsa
#include "../qsa_prompt_attn.cu"
