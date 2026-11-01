#include "strata/distributed/fast_execution.hpp"
#include "nccl_api.hpp"
#include "strata/kernels/window_limits.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#if defined(STRATA_DISTRIBUTED_CUDA)
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/prefill/gemm.hpp"
#include <cstdlib>
#include <cstdio>
#endif

namespace strata::distributed {
struct FastTensorParallel::DenseHead {
#if defined(STRATA_DISTRIBUTED_CUDA)
    void* weights=nullptr;
    void* input=nullptr;
    prefill::Gemm gemm;
    ~DenseHead(){if(weights)cudaFree(weights);if(input)cudaFree(input);}
#endif
};
FastTensorParallel::FastTensorParallel(nccl::NcclApi& api, int64_t hidden, Mixer mixer, Experts experts)
    : api_(api), hidden_(hidden), mixer_(std::move(mixer)), experts_(std::move(experts)) {
    if (hidden <= 0 || !mixer_ || !experts_) throw std::invalid_argument("incomplete fast TP providers");
#if defined(STRATA_DISTRIBUTED_CUDA)
    const char* compact=std::getenv("STRATA_TP_COMPACT_HEAD");
    compact_head_enabled_=compact&&compact[0]=='1'&&std::getenv("STRATA_TP_LOGITS_DIR")==nullptr;
    if(const char* type=std::getenv("STRATA_TP_REDUCE_DTYPE");type&&
        (std::string_view(type)=="bf16"||std::string_view(type)=="prefill-bf16")) {
        reduction_prefill_only_=std::string_view(type)=="prefill-bf16";
        const auto result=cudaMalloc(&reduction_scratch_,size_t(8192)*hidden_*sizeof(uint16_t));
        if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
    }
#endif
}
FastTensorParallel::~FastTensorParallel() {
#if defined(STRATA_DISTRIBUTED_CUDA)
    if (head_scratch_) cudaFree(head_scratch_);
    if (head_candidates_) cudaFree(head_candidates_);
    if (reduction_scratch_) cudaFree(reduction_scratch_);
#endif
}
bool FastTensorParallel::begin(void* stream, std::string& error) {
    if (failed_ || stream_ || !stream) { error = "fast TP step is not idle"; return false; }
    const auto f = api_.begin_stream_step(static_cast<cudaStream_t>(stream));
    if (f) { error = f.detail; failed_ = true; return false; }
    stream_ = stream;
    return true;
}
bool FastTensorParallel::finish(Deadline deadline, std::string& error) {
    if (failed_ || !stream_) { error = "no live fast TP step"; return false; }
    const auto f = api_.finish_stream_step(deadline);
    if (f) { error = f.detail; failed_ = true; return false; }
    stream_ = nullptr;
    return true;
}
void FastTensorParallel::abort() noexcept { failed_ = true; api_.comm_abort(); }
bool FastTensorParallel::requires_process_exit() const noexcept { return api_.requires_process_exit(); }
bool FastTensorParallel::bind_sessions(core::SessionState* const* states, int count, std::string& error) {
    return !state_ || state_->bind_sessions(states, count, error);
}
bool FastTensorParallel::reset_slots(const int32_t* slots, int count, void* stream, std::string& error) {
    return !state_ || state_->reset_slots(slots, count, stream, error);
}
bool FastTensorParallel::begin_slots(const int32_t* slots,const int32_t* rows,int count,bool speculative,void* stream,std::string& error) {
    return !state_ || state_->begin_slots(slots,rows,count,speculative,stream,error);
}
bool FastTensorParallel::commit_slots(const int32_t* slots, const int32_t* keep, int count, void* stream, std::string& error) {
    return !state_ || state_->commit_slots(slots, keep, count, stream, error);
}
bool FastTensorParallel::rollback_slots(const int32_t* slots, const int64_t* positions, int count, void* stream, std::string& error) {
    return !state_ || state_->rollback_slots(slots, positions, count, stream, error);
}
bool FastTensorParallel::validate_execution_status(void* stream, std::string& error) {
    return (!state_ || state_->validate_execution_status(stream, error)) && (!validate_ || validate_(stream, error));
}
bool FastTensorParallel::hyper_connection(const kernels::FusedGrArgs* args,int rows,void* stream,std::string& error) {
    return state_ && state_->hyper_connection(args,rows,stream,error);
}
void FastTensorParallel::output_head(const void* weights,int type,size_t row_bytes,int vocab,int rank) {
    if (head_weights_ || !weights || !row_bytes || vocab < 3 || rank < 0 || rank > 2)
        throw std::invalid_argument("invalid TP output head");
    head_weights_=weights;head_type_=type;head_row_bytes_=row_bytes;head_vocab_=vocab;head_rank_=rank;
#if defined(STRATA_DISTRIBUTED_CUDA)
    const auto result=cudaMalloc(&head_scratch_,size_t(kernels::kWindowMaxT)*((vocab+2)/3)*sizeof(float));
    if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
    const auto candidates=cudaMalloc(&head_candidates_,size_t(kernels::kWindowMaxT)*12*sizeof(float));
    if(candidates!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(candidates));
#endif
}
bool FastTensorParallel::output_head_greedy(const void* xq,int32_t* picks,uint32_t* flags,int rows,int blocks,
                                            void* stream,std::string& error) {
#if defined(STRATA_DISTRIBUTED_CUDA)
    if(!compact_head_active_||failed_||!stream_||stream!=stream_||!xq||!picks||rows<1||rows>kernels::kWindowMaxT) {
        error="invalid compact TP head invocation";return false;
    }
    try {
        const int first=int(int64_t(head_vocab_)*head_rank_/3),last=int(int64_t(head_vocab_)*(head_rank_+1)/3);
        const auto* weights=static_cast<const uint8_t*>(head_weights_)+size_t(first)*head_row_bytes_;
        kernels::native_mmvq(head_type_,weights,xq,(float*)head_scratch_,int(hidden_),last-first,rows,stream);
        const auto cs=static_cast<cudaStream_t>(stream);
        auto result=cudaMemsetAsync(head_candidates_,0,size_t(rows)*12*sizeof(float),cs);
        if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
        kernels::tp::cuda::head_local_candidate((float*)head_scratch_,(float*)head_candidates_,rows,last-first,first,head_rank_,stream);
        if(auto f=api_.enqueue_all_reduce_sum(head_candidates_,head_candidates_,uint64_t(rows)*12,ScalarType::fp32,cs)) {
            error=f.detail;failed_=true;return false;
        }
        kernels::tp::cuda::head_global_candidate((float*)head_candidates_,picks,flags,rows,blocks,stream);return true;
    }catch(const std::exception& e){error=e.what();return false;}
#else
    (void)xq;(void)picks;(void)flags;(void)rows;(void)blocks;(void)stream;error="CUDA compact head unavailable";return false;
#endif
}
bool FastTensorParallel::prepare_dense_output_head(void* stream,std::string& error) {
#if defined(STRATA_DISTRIBUTED_CUDA)
    const char* on=std::getenv("STRATA_TP_HEAD_F16");
    if(!on||on[0]!='1')return true;
    if(dense_head_||head_type_!=8||!head_weights_||!stream){error="invalid resident F16 head";return false;}
    try {
        auto d=std::make_unique<DenseHead>();
        const int first=int(int64_t(head_vocab_)*head_rank_/3),last=int(int64_t(head_vocab_)*(head_rank_+1)/3);
        auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
        check(cudaMalloc(&d->weights,size_t(last-first)*hidden_*2));
        check(cudaMalloc(&d->input,size_t(kernels::kWindowMaxT)*hidden_*2));
        if(!d->gemm.init(stream,0,error))return false;
        const auto* weights=static_cast<const uint8_t*>(head_weights_)+size_t(first)*head_row_bytes_;
        kernels::dequant_f16(head_type_,weights,0,last-first,hidden_,(uint16_t*)d->weights,stream);
        check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        dense_head_=std::move(d);return true;
    }catch(const std::exception& e){error=e.what();return false;}
#else
    (void)stream;(void)error;return true;
#endif
}
bool FastTensorParallel::has_dense_output_head(int rows)const {
    return dense_head_&&rows>8&&rows<=kernels::kWindowMaxT;
}
bool FastTensorParallel::dense_output_head(const float* input,float* logits,int32_t* picks,uint32_t* flags,
                                         int rows,int blocks,void* stream,std::string& error) {
#if defined(STRATA_DISTRIBUTED_CUDA)
    if(!has_dense_output_head(rows)||failed_||!stream_||stream!=stream_||!input||
       (compact_head_active_?!picks:!logits)){error="invalid resident F16 head invocation";return false;}
    try {
        const int first=int(int64_t(head_vocab_)*head_rank_/3),last=int(int64_t(head_vocab_)*(head_rank_+1)/3);
        const auto cs=static_cast<cudaStream_t>(stream);
        kernels::f32_to_f16_bulk(input,(uint16_t*)dense_head_->input,int64_t(rows)*hidden_,stream);
        dense_head_->gemm.f16((uint16_t*)dense_head_->input,(uint16_t*)dense_head_->weights,
            (float*)head_scratch_,rows,last-first,hidden_);
        if(compact_head_active_) {
            auto e=cudaMemsetAsync(head_candidates_,0,size_t(rows)*12*sizeof(float),cs);
            if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
            kernels::tp::cuda::head_local_candidate((float*)head_scratch_,(float*)head_candidates_,rows,last-first,first,head_rank_,stream);
            if(auto f=api_.enqueue_all_reduce_sum(head_candidates_,head_candidates_,uint64_t(rows)*12,ScalarType::fp32,cs)) {
                error=f.detail;failed_=true;return false;
            }
            kernels::tp::cuda::head_global_candidate((float*)head_candidates_,picks,flags,rows,blocks,stream);
        } else {
            auto e=cudaMemsetAsync(logits,0,size_t(rows)*head_vocab_*sizeof(float),cs);
            if(e==cudaSuccess)e=cudaMemcpy2DAsync(logits+first,size_t(head_vocab_)*4,head_scratch_,
                size_t(last-first)*4,size_t(last-first)*4,rows,cudaMemcpyDeviceToDevice,cs);
            if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
            if(auto f=api_.enqueue_all_reduce_sum(logits,logits,uint64_t(rows)*head_vocab_,ScalarType::fp32,cs)) {
                error=f.detail;failed_=true;return false;
            }
        }
        return true;
    }catch(const std::exception& e){error=e.what();return false;}
#else
    (void)input;(void)logits;(void)picks;(void)flags;(void)rows;(void)blocks;(void)stream;
    error="CUDA resident F16 head unavailable";return false;
#endif
}
bool FastTensorParallel::output_head(const void* xq,float* logits,int rows,void* stream,std::string& error) {
#if defined(STRATA_DISTRIBUTED_CUDA)
    if (failed_ || !head_weights_ || !head_scratch_ || !xq || !logits || !stream_ || stream != stream_ || rows < 1 || rows > kernels::kWindowMaxT) {
        error="invalid TP output head execution";return false;
    }
    const int first=int(int64_t(head_vocab_)*head_rank_/3);
    const int last=int(int64_t(head_vocab_)*(head_rank_+1)/3);
    const auto cs=static_cast<cudaStream_t>(stream);
    auto result=cudaMemsetAsync(logits,0,size_t(rows)*head_vocab_*sizeof(float),cs);
    if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
    try {
        const auto* weights=static_cast<const uint8_t*>(head_weights_)+size_t(first)*head_row_bytes_;
        kernels::native_mmvq(head_type_,weights,xq,static_cast<float*>(head_scratch_),int(hidden_),last-first,rows,stream);
        result=cudaMemcpy2DAsync(logits+first,size_t(head_vocab_)*sizeof(float),head_scratch_,
            size_t(last-first)*sizeof(float),size_t(last-first)*sizeof(float),rows,cudaMemcpyDeviceToDevice,cs);
        if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
    }catch(const std::exception& e){error=e.what();return false;}
    const auto f=api_.enqueue_all_reduce_sum(logits,logits,uint64_t(rows)*head_vocab_,ScalarType::fp32,cs);
    if(f){error=f.detail;failed_=true;return false;}return true;
#else
    (void)xq;(void)logits;(void)rows;(void)stream;error="CUDA output head unavailable";return false;
#endif
}
bool FastTensorParallel::reduce(const core::TensorParallelRows& rows, float* output,
                                void* stream, std::string& error) {
    if (failed_ || !stream_ || stream != stream_ || !output || rows.count <= 0 ||
        uint64_t(rows.count) > std::numeric_limits<uint64_t>::max() / uint64_t(hidden_)) {
        error = "invalid fast TP reduction shape or stream";
        return false;
    }
#if defined(STRATA_DISTRIBUTED_CUDA)
    if(reduction_scratch_&&(!reduction_prefill_only_||!rows.independent)) {
        if(rows.count>8192){error="BF16 reduction exceeds prompt capacity";return false;}
        const auto n=int64_t(rows.count)*hidden_;
        kernels::tp::cuda::transport_bf16_pack(output,(uint16_t*)reduction_scratch_,n,stream);
        const auto f=api_.enqueue_all_reduce_sum(reduction_scratch_,reduction_scratch_,uint64_t(n),
            ScalarType::bf16,static_cast<cudaStream_t>(stream));
        if(f){error=f.detail;failed_=true;return false;}
        kernels::tp::cuda::transport_bf16_unpack((uint16_t*)reduction_scratch_,output,n,stream);
        return true;
    }
#endif
    const auto f = api_.enqueue_all_reduce_sum(output, output, uint64_t(rows.count) * uint64_t(hidden_),
                                              ScalarType::fp32, static_cast<cudaStream_t>(stream));
    if (f) { error = f.detail; failed_ = true; return false; }
    return true;
}
bool FastTensorParallel::mixer(int64_t layer, const core::TensorParallelRows& rows,
                               const float* input, float* output, void* stream, std::string& error) {
    if (!stream_ || stream != stream_ || !input || !output || layer < 0 || rows.count <= 0) {
        error = "invalid fast TP mixer invocation"; return false;
    }
    if (failed_ || !mixer_(layer, rows, input, output, stream, error)) { failed_ = true; return false; }
#if defined(STRATA_DISTRIBUTED_CUDA)
    if (std::getenv("STRATA_TP_SYNC_DIAG")) {
        const auto result = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        std::fprintf(stderr,"TP diagnostic mixer layer %lld rows %lld: %s\n",(long long)layer,(long long)rows.count,cudaGetErrorString(result));
        if(result!=cudaSuccess){error=cudaGetErrorString(result);failed_=true;return false;}
    }
#endif
    return reduce(rows, output, stream, error);
}
bool FastTensorParallel::experts(int64_t layer, const core::TensorParallelRows& rows,
                                 const float* input, const int32_t* ids, const float* weights, int k,
                                 float* output, void* stream, std::string& error) {
    if (!stream_ || stream != stream_ || !input || !ids || !weights || !output ||
        layer < 0 || rows.count <= 0 || k <= 0) {
        error = "invalid fast TP expert invocation"; return false;
    }
    if (failed_ || !experts_(layer, rows, input, ids, weights, k, output, stream, error)) { failed_ = true; return false; }
    if (has_sharded_shared() && !state_->shared_expert_local_add(layer,rows,input,output,stream,error)) {
        failed_=true;return false;
    }
#if defined(STRATA_DISTRIBUTED_CUDA)
    if (std::getenv("STRATA_TP_SYNC_DIAG")) {
        const auto result = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        std::fprintf(stderr,"TP diagnostic experts layer %lld rows %lld: %s\n",(long long)layer,(long long)rows.count,cudaGetErrorString(result));
        if(result!=cudaSuccess){error=cudaGetErrorString(result);failed_=true;return false;}
    }
#endif
    return reduce(rows, output, stream, error);
}
} // namespace strata::distributed
