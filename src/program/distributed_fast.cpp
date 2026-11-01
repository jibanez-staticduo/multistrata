#include "strata/distributed/fast_config.hpp"
#include "strata/distributed/fast_mixer.hpp"
#include "strata/distributed/nvfp4_experts.hpp"
#include "strata/core/layer.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>

namespace {
using namespace strata;
using namespace strata::distributed;
void require(bool ok, const std::string& error) { if (!ok) throw std::runtime_error(error); }
struct Stream {
    cudaStream_t value = nullptr;
    Stream() {
        if (cudaSetDevice(0) != cudaSuccess || cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("cannot create fast TP CUDA stream");
    }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
class MixerState final : public core::TensorParallelHooks {
public:
    explicit MixerState(FastMixer& mixer) : mixer_(mixer) {}
    bool bind_sessions(core::SessionState* const* sessions, int count, std::string& error) override {
        return mixer_.bind_sessions(sessions, count, error);
    }
    bool reset_slots(const int32_t* ids, int count, void* stream, std::string& error) override {
        return mixer_.reset_slots(ids, count, stream, error);
    }
    bool begin_slots(const int32_t* ids,const int32_t* rows,int count,bool spec,void* stream,std::string& error) override {
        return mixer_.begin_slots(ids,rows,count,spec,stream,error);
    }
    bool commit_slots(const int32_t* ids, const int32_t* keep, int count, void* stream, std::string& error) override {
        return mixer_.commit_slots(ids, keep, count, stream, error);
    }
    bool rollback_slots(const int32_t* ids, const int64_t* positions, int count, void* stream, std::string& error) override {
        return mixer_.rollback_slots(ids, positions, count, stream, error);
    }
    bool validate_execution_status(void* stream, std::string& error) override {
        return mixer_.validate_execution_status(stream, error);
    }
    bool has_fast_hc() const override { return mixer_.has_fast_hc(); }
    bool use_fast_hc(int rows) const override { return mixer_.use_fast_hc(rows); }
    bool has_sharded_shared() const override { return mixer_.has_sharded_shared(); }
    bool shared_expert_local_add(int64_t layer,const core::TensorParallelRows& rows,const float* input,
                                 float* output,void* stream,std::string& error) override {
        return mixer_.shared_expert_local_add(layer,rows,input,output,stream,error);
    }
    bool hyper_connection(const kernels::FusedGrArgs* args,int rows,void* stream,std::string& error) override {
        return mixer_.hyper_connection(args,rows,stream,error);
    }
    bool mixer(int64_t layer, const core::TensorParallelRows& rows, const float* input,
               float* output, void* stream, std::string& error) override {
        return mixer_.run(layer, rows, input, output, stream, error);
    }
    bool experts(int64_t, const core::TensorParallelRows&, const float*, const int32_t*,
                 const float*, int, float*, void*, std::string& error) override {
        error = "mixer state adapter cannot execute experts";
        return false;
    }
private:
    FastMixer& mixer_;
};
}
int main(int argc, char** argv) {
    try {
        auto config = parse_fast_launch(argc, argv);
        // Asset loading prints native diagnostics; preserve JSONL before it starts.
        const int output = config.runtime.rank == 0 ? ::dup(STDOUT_FILENO) : -1;
        if ((config.runtime.rank == 0 && output < 0) || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
            throw std::runtime_error("cannot preserve fast protocol output");
        config.runtime.protocol_output_fd = output;
        Stream stream;
        if(const char* layout=std::getenv("STRATA_TP_MMVQ_LAYOUT");layout&&std::string_view(layout)=="fast")
            kernels::native_mmvq_set_multi_exact(false);
        core::layer_set_native_bf16(true);
        core::layer_set_fused_gr(true);
        core::layer_set_fused_gdn(true);
        core::layer_set_fast_attn(true);
        core::layer_set_fast_select(true);
        core::layer_set_native_flash_attn_short(false);
        kernels::native_gdn_set_enabled(true);
        kernels::native_qsa_set_enabled(true);
        kernels::native_qsa_indexer_set_enabled(true);
        kernels::native_rope_set_enabled(true);
        kernels::native_router_set_enabled(true);
        kernels::native_moe_combine_set_enabled(true);
        kernels::ple_set_native_bf16(true);
        kernels::ple_set_native_postops(true);
        kernels::shared_expert_set_native_bf16(true);
        FastAssets assets;
        std::string error;
        require(assets.load(config.pack, config.dense, config.embedding, config.ple, error), error);
        FastMixer mixer(assets.weights(), assets.geometry(), int(config.runtime.rank));
        require(mixer.initialize({int(config.runtime.prompt_chunk), int(config.runtime.slots), 4}, stream.value, error), error);
        NativeNvfp4Experts experts;
        require(experts.initialize(config.pack, assets.geometry(), int(config.runtime.rank),
                                   int(config.runtime.prompt_chunk), stream.value, error), error);
        MixerState state(mixer);
        const int result = run_fast_runtime(config.runtime, assets,
            [&](auto layer, const auto& rows, auto input, auto out, auto cs, auto& err) {
                return mixer.run(layer, rows, input, out, cs, err);
            },
            [&](auto layer, const auto& rows, auto input, auto ids, auto weights, auto k, auto out, auto cs, auto& err) {
                return experts.local_sum(layer, rows, input, ids, weights, k, out, cs, err);
            }, &state, stream.value,
            [&](auto cs, auto& err) { return experts.validate_execution_status(cs, err); });
        if (output >= 0) ::close(output);
        return result;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fast TP launch refused: %s\n", e.what());
        return 2;
    }
}
