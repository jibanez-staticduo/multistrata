#pragma once

namespace strata::kernels::tp::cuda {

struct FastGdnBatchGroups {
    float* state[8] = {};
    float* history[8] = {};
    float* state_prefix[8] = {};
    float* history_prefix[8] = {};
    int first[8] = {};
    int count[8] = {};
    bool snapshot[8] = {};
    int n = 0;
};

// Runs convolution, recurrent state/output, and history commit for up to eight
// independent slot groups. Each group has at most five rows; the compact input
// ranges must fit the configured window. Prefix row i stores state/history after local row i.
void fast_gdn_batch(const FastGdnBatchGroups& groups, const float* qkv, const float* conv_w, float* h,
                    const float* gate, const float* beta, const float* z, const float* norm, float* out,
                    int key_heads, float eps, void* stream);

}  // namespace strata::kernels::tp::cuda
