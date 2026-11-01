// Compile the tuned upstream split-K implementation for four query heads/KV.
#include "strata/kernels/fast_qsa.hpp"
#define STRATA_QSA_LOCAL_GROUPS 4
#define STRATA_QSA_KERNEL_NAMESPACE strata::kernels::tp::cuda::fast_qsa
#include "../qsa_decode_attn.cu"
