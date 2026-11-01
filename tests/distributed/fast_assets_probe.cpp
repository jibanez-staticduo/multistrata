#include "strata/distributed/fast_assets.hpp"
#include "strata/core/native_upload.hpp"
#include <cuda_runtime.h>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    if (cudaSetDevice(0) != cudaSuccess) return 77;
    const std::string root = argv[1];
    strata::distributed::FastAssets assets;
    std::string error;
    if (!assets.load(root + "/pack", root + "/qwen-nvfp4-gptq-dense.gguf",
                     root + "/token-embd-bf16.gguf", root + "/ple-fp8.gguf", error)) {
        std::fprintf(stderr, "NVFP4 assets refused: %s\n", error.c_str()); return 1;
    }
    const auto& g = assets.geometry();
    std::printf("NVFP4 assets ready: layers=%lld hidden=%lld ff=%lld experts=%lld vocab=%lld PLE=%d\n",
                (long long)g.n_layers, (long long)g.n_embd, (long long)g.n_ff,
                (long long)g.n_expert, (long long)g.n_vocab, assets.ple().is_open());
}
