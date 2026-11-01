#include "strata/distributed/fast_mixer.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/fast_mixer.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/tp_qsa.hpp"
#include "strata/kernels/fast_qsa.hpp"
#include "strata/kernels/fast_gdn.hpp"
#include "strata/kernels/fast_gdn_batch.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::distributed {
namespace {
namespace k = kernels;
namespace gpu = kernels::tp::cuda;
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void* p=nullptr;
    size_t bytes=0;
    ~Buffer(){if(p)cudaFree(p);}
    void allocate(size_t n){check(cudaMalloc(&p,n));bytes=n;}
    float* f(size_t offset=0)const{return static_cast<float*>(p)+offset;}
    int* i(size_t offset=0)const{return static_cast<int*>(p)+offset;}
    uint16_t* u16(size_t offset=0)const{return static_cast<uint16_t*>(p)+offset;}
};
struct Matrix {
    Buffer data, resident_f16;
    int type=-1, input=0, output=0;
    bool bf16=false;
};
std::vector<int> consecutive(int first,int count) {
    std::vector<int> result(count);for(int i=0;i<count;++i)result[i]=first+i;return result;
}
void append(std::vector<int>& destination,const std::vector<int>& source) {
    destination.insert(destination.end(),source.begin(),source.end());
}
void copy_matrix(Matrix& out,const core::WeightRef* w,const std::vector<int>& outputs,
                   const std::vector<int>& input_runs,void* stream,int q8_input_block=128) {
    if(!w||(!w->native_data&&!w->data)||outputs.empty())throw std::runtime_error("missing fast mixer matrix");
    out.type=w->native_type;out.bf16=!w->native_data;
    const size_t source_row=out.bf16?size_t(w->ne0)*2:k::native_mmvq_weight_bytes(w->native_type,int(w->ne0),1);
    if(out.bf16&&w->kind!=core::WeightKind::Bf16InF32)throw std::runtime_error("fast mixer BF16 matrix form mismatch");
    const void* source=out.bf16?w->data:w->native_data;
    if(q8_input_block <= 0 || q8_input_block % 32 != 0)
        throw std::runtime_error("fast mixer input block must be a positive Q8 multiple");
    out.input=input_runs.empty()?int(w->ne0):int(input_runs.size())*(out.bf16?128:q8_input_block);
    out.output=int(outputs.size());
    const size_t row=out.bf16?size_t(out.input)*2:k::native_mmvq_weight_bytes(out.type,out.input,1);
    out.data.allocate(row*outputs.size());
    for(size_t r=0;r<outputs.size();) {
        const size_t first=r++;
        while(r<outputs.size()&&outputs[r]==outputs[r-1]+1)++r;
        if(outputs[first]<0||outputs[r-1]>=w->ne1)throw std::runtime_error("fast mixer row slice outside matrix");
        auto* dst=static_cast<uint8_t*>(out.data.p)+first*row;
        auto* src=static_cast<const uint8_t*>(source)+size_t(outputs[first])*source_row;
        if(input_runs.empty())check(cudaMemcpy2DAsync(dst,row,src,source_row,row,r-first,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
        else for(size_t b=0;b<input_runs.size();) {
            const size_t begin=b++;
            while(b<input_runs.size()&&input_runs[b]==input_runs[b-1]+1)++b;
            if(out.type!=8&&!out.bf16)throw std::runtime_error("fast mixer input shards require Q8_0 or BF16 blocks");
            const size_t unit=out.bf16?size_t(128)*2:
                k::native_mmvq_weight_bytes(out.type,q8_input_block,1);
            check(cudaMemcpy2DAsync(dst+begin*unit,row,src+size_t(input_runs[begin])*unit,source_row,
                (b-begin)*unit,r-first,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
        }
    }
}
}
struct FastMixer::Impl {
    const core::WeightTable& weights;
    core::ModelGeometry geometry;
    int rank, key_first, key_heads, value_heads, channels, width;
    FastMixerLimits limits{};
    void* stream=nullptr;
    bool ready=false;
    bool shared_sharded=false;
    uint64_t weight_bytes=0;
    std::vector<core::SessionState*> sessions;
    std::vector<int32_t> window_counts;
    bool speculative_window=false;
    struct Layer {
        bool qsa=false;int ordinal=0;
        Matrix qkv,z,alpha,beta,output,q,key,value,index_key,index_query;
        Matrix shared_gate,shared_up,shared_down;
        const uint16_t* shared_gate_input=nullptr;
        Buffer conv,dt,a;
        const float *norm=nullptr,*qnorm=nullptr,*knorm=nullptr,*iqnorm=nullptr,*iknorm=nullptr;
        std::vector<std::unique_ptr<Buffer>> states,histories;
        std::vector<std::unique_ptr<Buffer>> state_prefix,history_prefix;
    };
    std::vector<std::unique_ptr<Layer>> layers;
    Buffer input16,q8,qkv,z,alpha,beta,gate,h,y,normalized,qfull,q,key,value,raw,index_query,index_dead,steps,positions,key_positions,index_positions,ids,scores,attn_scratch,status,keep,gdn_scratch;
    Buffer shared_input_bf16,shared_input_f16,shared_gate_out,shared_up_out,shared_hidden,shared_out,shared_scalar;
    Buffer shared_q8_scratch;
    kernels::NativeSharedWeights shared_native{};
    prefill::Gemm gemm;
    std::unique_ptr<prefill::mmq::Context> dense_mmq;
    Buffer dense_q8,dense_bounds,dense_ids;
    bool hc_bf16=false,hc_large_only=false;
    bool hc_packed_down=false;
    int prompt_qsa_batch=32;
    bool prompt_qsa_fast=false;
    std::map<const uint16_t*,std::unique_ptr<Buffer>> hc_combined_weights;
    Buffer hc_x,hc_hi,hc_lo,hc_lhi,hc_gate,hc_inject,hc_combined;
    Impl(const core::WeightTable& w,const core::ModelGeometry& g,int r):weights(w),geometry(g),rank(r) {
        key_first=r==0?0:r==1?6:11;key_heads=r==0?6:5;
        value_heads=key_heads*3;channels=key_heads*5*128;width=value_heads*128;
    }
    const core::WeightRef* need(const core::LayerView& view,const char* name) {
        auto* w=view.get(name);if(!w)throw std::runtime_error("missing fast mixer tensor "+view.name(name));return w;
    }
    bool use_dense_mmq(const Matrix& matrix,int rows)const {
        return dense_mmq && !matrix.bf16 && matrix.type==8 && rows>=4 &&
               rows<=k::kWindowMaxT && matrix.input%256==0 && matrix.output>=2048;
    }
    void project_mmq(const Matrix& matrix,const float* input,float* out,int rows) {
        prefill::mmq::quantize(input,nullptr,dense_q8.p,8,matrix.input,matrix.input,rows,stream);
        prefill::mmq::Product product;
        product.w=matrix.data.p;product.type=8;product.w_rows=matrix.output;product.w_cols=matrix.input;
        product.expert_bytes=matrix.data.bytes;product.n=1;product.xq=dense_q8.p;
        product.bounds=dense_bounds.i(size_t(rows)*2);product.ids=dense_ids.i();
        product.total_rows=product.max_rows=rows;product.dst=out;product.ld_dst=matrix.output;
        dense_mmq->run(product,stream);
    }
    void project(const Matrix& matrix,const float* input,float* out,int rows) {
        if(matrix.resident_f16.p&&rows>=2) {
            k::f32_to_f16_bulk(input,(uint16_t*)input16.p,size_t(rows)*matrix.input,stream);
            gemm.f16((uint16_t*)input16.p,(const uint16_t*)matrix.resident_f16.p,out,
                      rows,matrix.output,matrix.input);
            return;
        }
        if(use_dense_mmq(matrix,rows)){project_mmq(matrix,input,out,rows);return;}
        if(rows<=k::kWindowMaxT) {
            if(matrix.bf16) {
                for(int first=0;first<rows;first+=8)
                    k::bf16_gemv_fp32_mmvf_multi(input+size_t(first)*matrix.input,matrix.input,
                        (const uint16_t*)matrix.data.p,out+size_t(first)*matrix.output,matrix.output,
                        matrix.input,matrix.output,std::min(8,rows-first),stream);
            }
            else {
                k::native_quantize_q8_1(input,q8.p,matrix.input,rows,stream);
                k::native_mmvq(matrix.type,matrix.data.p,q8.p,out,matrix.input,matrix.output,rows,stream);
            }
        } else {
            if(matrix.bf16) {
                k::f32_to_bf16_bulk(input,(uint16_t*)input16.p,size_t(rows)*matrix.input,stream);
                gemm.bf16((uint16_t*)input16.p,(const uint16_t*)matrix.data.p,out,rows,matrix.output,matrix.input);
            } else {
                k::f32_to_f16_bulk(input,(uint16_t*)input16.p,size_t(rows)*matrix.input,stream);
                gemm.native((uint16_t*)input16.p,matrix.type,matrix.data.p,out,rows,matrix.output,matrix.input);
            }
        }
    }
    void project_shared_input(const float* input,int rows,
                              std::initializer_list<std::pair<const Matrix*,float*>> projections) {
        if(rows>k::kWindowMaxT || projections.size()>8) {
            for(const auto& projection:projections)
                project(*projection.first,input,projection.second,rows);
            return;
        }
        int input_width=-1;
        bool has_native=false;
        for(const auto& projection:projections) {
            const Matrix& matrix=*projection.first;
            if(input_width<0)input_width=matrix.input;
            else if(matrix.input!=input_width)throw std::runtime_error("fast mixer shared projection input mismatch");
            has_native=has_native||(!matrix.bf16&&!(matrix.resident_f16.p&&rows>=2)&&!use_dense_mmq(matrix,rows));
        }
        if(has_native)k::native_quantize_q8_1(input,q8.p,input_width,rows,stream);

        std::array<bool,8> done{};
        size_t i=0;
        for(const auto& projection:projections) {
            const Matrix& matrix=*projection.first;
            if(matrix.bf16||(matrix.resident_f16.p&&rows>=2)||use_dense_mmq(matrix,rows)) {
                project(matrix,input,projection.second,rows);
            } else if(!done[i]) {
                bool paired=false;
                if(matrix.type==8 && rows>=2) {
                    size_t j=0;
                    for(const auto& candidate:projections) {
                        const Matrix& other=*candidate.first;
                        if(j>i && !done[j] && !other.bf16 && !use_dense_mmq(other,rows) && other.type==matrix.type &&
                           other.input==matrix.input && other.output==matrix.output &&
                           k::native_mmvq_pair(matrix.type,matrix.data.p,other.data.p,q8.p,
                                               projection.second,candidate.second,matrix.input,
                                               matrix.output,rows,stream)) {
                            done[j]=true;
                            paired=true;
                            break;
                        }
                        ++j;
                    }
                }
                if(!paired)
                    k::native_mmvq(matrix.type,matrix.data.p,q8.p,projection.second,
                                   matrix.input,matrix.output,rows,stream);
                done[i]=true;
            }
            ++i;
        }
    }
    core::TensorParallelRows subrow(const core::TensorParallelRows& rows,int row) {
        auto part=rows;part.count=1;part.position+=row;
        if(part.positions)part.positions+=row*part.position_stride;
        if(part.slots)part.slots+=row;
        if(part.capture_slots)part.capture_slots+=row;
        return part;
    }
    bool gdn(Layer& layer,const core::TensorParallelRows& rows,const float* input,float* output,std::string& error) {
        const int n=int(rows.count);
        project_shared_input(input,n,{{&layer.qkv,qkv.f()},{&layer.z,z.f()},
                                      {&layer.alpha,alpha.f()},{&layer.beta,beta.f()}});
        gpu::mixer_gates(alpha.f(),beta.f(),layer.dt.f(),layer.a.f(),gate.f(),beta.f(),n,value_heads,stream);
        if(rows.independent) {
            gpu::FastGdnBatchGroups groups{};
            for(int first=0;first<n;) {
                const int slot=rows.capture_slots?rows.capture_slots[first]:rows.slot;
                if(slot<0||size_t(slot)>=sessions.size()||groups.n>=8){error="invalid GDN capture group";return false;}
                int count=1;while(first+count<n && rows.capture_slots && rows.capture_slots[first+count]==slot)++count;
                const int i=groups.n++;groups.first[i]=first;groups.count[i]=count;
                groups.state[i]=layer.states[slot]->f();groups.history[i]=layer.histories[slot]->f();
                groups.state_prefix[i]=layer.state_prefix[slot]->f();groups.history_prefix[i]=layer.history_prefix[slot]->f();
                groups.snapshot[i]=rows.speculative&&count>1;first+=count;
            }
            gpu::fast_gdn_batch(groups,qkv.f(),layer.conv.f(),h.f(),gate.f(),beta.f(),z.f(),layer.norm,
                                y.f(),key_heads,float(geometry.rms_eps),stream);
        } else {
            auto segment=[&](int slot,int first,int count)->bool {
                if(slot<0||size_t(slot)>=sessions.size()){error="invalid prompt slot";return false;}
                gpu::mixer_conv(layer.histories[slot]->f(),qkv.f(size_t(first)*channels),layer.conv.f(),
                    h.f(size_t(first)*channels),count,channels,key_heads,stream);
                gpu::fast_gdn_prompt(layer.states[slot]->f(),h.f(size_t(first)*channels),gate.f(size_t(first)*value_heads),
                    beta.f(size_t(first)*value_heads),z.f(size_t(first)*width),layer.norm,
                    normalized.f(size_t(first)*width),count,key_heads,gdn_scratch.p,gdn_scratch.bytes,stream);
                gpu::mixer_conv_commit(layer.histories[slot]->f(),qkv.f(size_t(first)*channels),count,channels,stream);
                return true;
            };
            if(rows.segment_count) {
                for(int i=0;i<rows.segment_count;++i) {
                    const auto& s=rows.segments[i];
                    if(!segment(s.slot,int(s.first),int(s.count)))return false;
                }
            }else if(!segment(rows.slot,0,n))return false;
        }
        project(layer.output,rows.independent?y.f():normalized.f(),output,n);return true;
    }
    bool qsa(Layer& layer,const core::TensorParallelRows& rows,const float* input,float* output,std::string& error) {
        const int n=int(rows.count);
        project_shared_input(input,n,{{&layer.q,qfull.f()},{&layer.key,key.f()},{&layer.value,value.f()},
                                      {&layer.index_key,raw.f()},{&layer.index_query,index_query.f()}});
        auto shapes=core::qsa_shapes(geometry);shapes.n_head=8;
        const int cap=int(k::qsa_selection_width(k::kTopkMaxCells,shapes));
        const bool batch_attention=rows.independent;
        static const bool batch_prepare_on=[] { const char* e=std::getenv("STRATA_TP_QSA_PREPARE_BATCH");return e&&e[0]=='1'; }();
        const bool batch_prepare=batch_prepare_on&&rows.independent&&rows.capture_slots;
        static const bool batch_append_on=[] { const char* e=std::getenv("STRATA_TP_QSA_APPEND_BATCH");return e&&e[0]=='1'; }();
        const bool batch_append=batch_append_on&&batch_prepare;
        if(batch_prepare) {
            gpu::mixer_steps_batch(rows,rows.capture_slots,n,int(sessions.front()->max_cells),
                steps.i(),positions.i(),status.i(),stream);
            gpu::mixer_rope_positions(steps.i(),n,key_positions.i(),index_positions.i(),stream);
            k::native_qsa_rms_norm_rope(key.f(),256,layer.knorm,key.f(),n*2,256,64,
                float(geometry.rms_eps),k::rope_scaling(),key_positions.i(),stream);
            k::native_qsa_rms_norm_rope(qfull.f(),512,layer.qnorm,q.f(),n*8,256,64,
                float(geometry.rms_eps),k::rope_scaling(),positions.i(),stream);
            k::native_qsa_rms_norm_rope(index_query.f(),128,layer.iqnorm,index_query.f(),n*4,128,64,
                float(geometry.rms_eps),k::rope_scaling(),index_positions.i(),stream);
        }
        if(batch_append) {
            k::KvQ8AppendGroups pools{};
            k::QsaIndexerGroups groups{};
            for(int first=0;first<n;) {
                const int slot=rows.capture_slots[first];
                if(slot<0||size_t(slot)>=sessions.size()||groups.n>=8)
                    throw std::runtime_error("invalid grouped QSA slot");
                int count=1;while(first+count<n&&rows.capture_slots[first+count]==slot)++count;
                if(count>limits.snapshot_depth+1)throw std::runtime_error("grouped QSA prefix exceeds capacity");
                const auto& state=sessions[slot]->qsa_states[layer.ordinal];
                if(state.kv_rot||state.kv_hybrid)throw std::runtime_error("grouped QSA requires unrotated Q8 KV");
                const int i=groups.n++;
                pools.k[i]=state.k_q;pools.v[i]=state.v_q;
                pools.k_scale[i]=state.k_scale;pools.v_scale[i]=state.v_scale;pools.page_table[i]=state.page_table;
                pools.first[i]=groups.first[i]=first;pools.count[i]=groups.count[i]=count;
                groups.buffers[i]={state.idx_tail,state.idx_dead,state.idx_pooled,state.idx_block_pos};
                groups.history_prefix[i]=layer.history_prefix[slot]->f();
                groups.raw_prefix[i]=layer.state_prefix[slot]->f();
                groups.snapshot[i]=rows.speculative&&count>1;
                first+=count;
            }
            pools.n=groups.n;
            k::kv_append_q8_groups(pools,steps.i(),4,key.f(),value.f(),512,n,shapes,stream);
            k::native_qsa_indexer_append_groups(raw.f(),steps.i(),4,index_dead.f(),layer.iknorm,
                float(geometry.rms_eps),groups,shapes,sessions.front()->max_cells,k::rope_scaling(),stream);
        }
        auto prepare=[&](int row,int slot,int count,int64_t position)->bool {
            if(slot<0||size_t(slot)>=sessions.size()){error="invalid QSA capture slot";return false;}
            auto& state=sessions[slot]->qsa_states[layer.ordinal];
            auto part=rows.independent?subrow(rows,row):rows;
            if(!rows.independent) {
                part.position=position;part.count=count;part.positions=nullptr;
                part.slot=slot;part.segments=nullptr;part.segment_count=0;
            }
            auto* step=steps.i(size_t(row)*4);auto* pos=positions.i(size_t(row)*8);
            if(!batch_prepare) {
                gpu::mixer_steps(part,count,slot,int(state.max_cells),step,pos,8,status.i(),stream);
                gpu::mixer_rope_positions(step,count,key_positions.i(),index_positions.i(),stream);
                k::native_qsa_rms_norm_rope(key.f(size_t(row)*512),256,layer.knorm,
                    key.f(size_t(row)*512),count*2,256,64,float(geometry.rms_eps),k::rope_scaling(),key_positions.i(),stream);
                k::native_qsa_rms_norm_rope(qfull.f(size_t(row)*4096),512,layer.qnorm,q.f(size_t(row)*2048),count*8,256,64,
                    float(geometry.rms_eps),k::rope_scaling(),pos,stream);
                // Every token's eight positions are equal; indexer four-head rows
                // are staged contiguously below to avoid an eight-head stride.
                k::native_qsa_rms_norm_rope(index_query.f(size_t(row)*512),128,layer.iqnorm,
                    index_query.f(size_t(row)*512),count*4,128,64,float(geometry.rms_eps),k::rope_scaling(),index_positions.i(),stream);
            }
            if(state.kv_rot||state.kv_hybrid){error="fast QSA requires unrotated INT8 KV";return false;}
            if(!batch_append) {
                k::kv_append_q8_steps(state.k_q,state.v_q,state.k_scale,state.v_scale,state.page_table,step,4,
                    key.f(size_t(row)*512),value.f(size_t(row)*512),512,count,shapes,stream,nullptr);
                k::QsaIndexerBuffers ib{state.idx_tail,state.idx_dead,state.idx_pooled,state.idx_block_pos};
                const bool save=rows.speculative && rows.capture_slots &&
                    ((row>0&&rows.capture_slots[row-1]==slot)||(row+1<n&&rows.capture_slots[row+1]==slot));
                if(save) {
                    int prefix=0;for(int j=row-1;j>=0 && rows.capture_slots && rows.capture_slots[j]==slot;--j)++prefix;
                    if(prefix>limits.snapshot_depth){error="QSA prefix exceeds snapshot capacity";return false;}
                    const size_t stride=size_t(3*128+128+1)*4;
                    auto* destination=static_cast<uint8_t*>(layer.history_prefix[slot]->p)+size_t(prefix)*stride;
                    // Preserve the indexer tail BEFORE this row, so accepted-prefix
                    // restoration can replay only the retained raw keys.
                    check(cudaMemcpyAsync(destination,state.idx_tail,3*128*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                    check(cudaMemcpyAsync(destination+3*128*4,state.idx_dead,128*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                    check(cudaMemcpyAsync(destination+(3*128+128)*4,state.idx_block_pos,4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                }
                k::native_qsa_indexer_append_steps(raw.f(size_t(row)*128),step,4,count,0,layer.iknorm,
                    float(geometry.rms_eps),ib,shapes,state.max_cells,k::rope_scaling(),stream);
                if(save) {
                    int prefix=0;for(int j=row-1;j>=0 && rows.capture_slots && rows.capture_slots[j]==slot;--j)++prefix;
                    check(cudaMemcpyAsync(layer.state_prefix[slot]->f(size_t(prefix)*128),raw.f(size_t(row)*128),128*4,
                        cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                }
            }
            const int64_t blocks=state.max_cells/4+2;
            if(batch_attention){
                if(!batch_append)check(cudaMemcpyAsync(index_dead.f(size_t(row)*128),state.idx_dead,128*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                return true;
            }
            for(int first=0;first<count;first+=prompt_qsa_batch) {
                const int batch=std::min(prompt_qsa_batch,count-first),r=row+first;
                const int64_t active=rows.independent?0:std::min<int64_t>(blocks,(position+first+batch)/4+1);
                const bool identity=prompt_qsa_fast&&!rows.independent&&position+first+batch<=cap;
                if (!identity&&(rows.independent || !k::qsa_block_scores_tc(state.idx_pooled,state.idx_dead,
                        index_query.f(size_t(r)*512),steps.i(size_t(r)*4),batch,blocks,shapes,scores.f(),stream,active))
                    )
                    k::qsa_block_scores(state.idx_pooled,state.idx_dead,index_query.f(size_t(r)*512),steps.i(size_t(r)*4),
                        batch,blocks,shapes,scores.f(),stream,active);
                k::qsa_block_topk(scores.f(),steps.i(size_t(r)*4),batch,blocks,cap,shapes,ids.i(size_t(r)*cap),stream,active);
                if(rows.independent || !gpu::fast_qsa::qsa_prompt_attn_batch(q.f(size_t(r)*2048),
                    core::qsa_attn_pools(state),ids.i(size_t(r)*cap),steps.i(size_t(r)*4),cap,shapes,
                    y.f(size_t(r)*2048),batch,stream)) {
                    for(int offset=0;offset<batch;offset+=32) {
                        const int at=r+offset,small=std::min(32,batch-offset);
                        gpu::fast_qsa::qsa_decode_attn_batch(q.f(size_t(at)*2048),core::qsa_attn_pools(state),
                            ids.i(size_t(at)*cap),steps.i(size_t(at)*4),cap,shapes,attn_scratch.f(),
                            y.f(size_t(at)*2048),small,stream);
                    }
                }
            }
            return true;
        };
        if(rows.independent){
            for(int r=0;r<n;++r)if(!prepare(r,rows.capture_slots?rows.capture_slots[r]:rows.slot,1,0))return false;
            const int64_t blocks=sessions.front()->max_cells/4+2;
            for(int first=0;first<n;first+=8) {
                const int count=std::min(8,n-first);const float* pooled[8]{},*dead[8]{};
                for(int i=0;i<count;++i) {
                    const int slot=rows.capture_slots?rows.capture_slots[first+i]:rows.slot;
                    const auto& state=sessions[slot]->qsa_states[layer.ordinal];pooled[i]=state.idx_pooled;dead[i]=index_dead.f(size_t(first+i)*128);
                }
                if(!k::qsa_block_scores_slots(pooled,dead,index_query.f(size_t(first)*512),steps.i(size_t(first)*4),
                        count,blocks,shapes,scores.f(),stream)) {
                    for(int i=0;i<count;++i)k::qsa_block_scores(pooled[i],dead[i],index_query.f(size_t(first+i)*512),
                        steps.i(size_t(first+i)*4),1,blocks,shapes,scores.f(size_t(i)*blocks),stream);
                }
                k::qsa_block_topk(scores.f(),steps.i(size_t(first)*4),count,blocks,cap,shapes,ids.i(size_t(first)*cap),stream);
            }
            for(int first=0;first<n;first+=k::kAttnRowsMax) {
                const int count=std::min(k::kAttnRowsMax,n-first);k::QsaAttnPoolsRows pools{};pools.n=count;
                for(int i=0;i<count;++i) {
                    const int slot=rows.capture_slots?rows.capture_slots[first+i]:rows.slot;
                    pools.p[i]=core::qsa_attn_pools(sessions[slot]->qsa_states[layer.ordinal]);
                }
                gpu::fast_qsa::qsa_decode_attn_batch_rows(q.f(size_t(first)*2048),pools,ids.i(size_t(first)*cap),
                    steps.i(size_t(first)*4),cap,shapes,attn_scratch.f(),y.f(size_t(first)*2048),count,stream);
            }
        }
        else if(rows.segment_count) {
            for(int i=0;i<rows.segment_count;++i) {
                const auto& s=rows.segments[i];
                if(!prepare(int(s.first),s.slot,int(s.count),s.position))return false;
            }
        }else if(!prepare(0,rows.slot,n,rows.position))return false;
        k::native_qsa_gate_apply(y.f(),qfull.f(),y.f(),n*8,256,stream);
        project(layer.output,y.f(),output,n);return true;
    }
};
FastMixer::FastMixer(const core::WeightTable& w,const core::ModelGeometry& g,int rank):impl_(std::make_unique<Impl>(w,g,rank)){}
FastMixer::~FastMixer()=default;
bool FastMixer::has_sharded_shared()const noexcept{return impl_->shared_sharded;}
bool FastMixer::shared_expert_local_add(int64_t layer,const core::TensorParallelRows& rows,const float* mixed,
                                        float* routed_local_sum,void* stream,std::string& error)try {
    auto& p=*impl_;
    if(!p.ready||!p.shared_sharded||stream!=p.stream||layer<0||size_t(layer)>=p.layers.size()||
       rows.count<1||rows.count>p.limits.max_rows||!mixed||!routed_local_sum)
        throw std::runtime_error("invalid TP shared expert invocation");
    auto& l=*p.layers[size_t(layer)];
    const int n=int(rows.count);
    if(rows.independent) {
        if(n>k::kWindowMaxT)throw std::runtime_error("TP shared decode batch exceeds native MMVQ columns");
        const auto* gate_in=static_cast<const uint16_t*>(l.shared_gate_input);
        kernels::NativeSharedWeights native=p.shared_native;
        native.q8_1=p.shared_q8_scratch.p;
        native.gate_data=l.shared_gate.data.p;
        native.up_data=l.shared_up.data.p;
        native.down_data=l.shared_down.data.p;
        native.gate_type=l.shared_gate.type;
        native.up_type=l.shared_up.type;
        native.down_type=l.shared_down.type;
        kernels::shared_expert_multi(n,mixed,p.shared_input_bf16.u16(),native,gate_in,
            p.shared_gate_out.f(),p.shared_up_out.f(),p.shared_scalar.f(),p.shared_out.f(),
            2560,l.shared_gate.output,stream);
        gpu::shared_add(routed_local_sum,p.shared_out.f(),nullptr,n,2560,false,stream);
        return true;
    }
    // Prompt projections use Prefill's FP16 activation image; the scalar gate uses its BF16 image.
    k::f32_to_f16_bulk(mixed,p.shared_input_f16.u16(),rows.count*2560,stream);
    k::f32_to_bf16_bulk(mixed,p.shared_input_bf16.u16(),rows.count*2560,stream);
    p.gemm.native(p.shared_input_f16.u16(),l.shared_gate.type,l.shared_gate.data.p,
                  p.shared_gate_out.f(),rows.count,l.shared_gate.output,2560);
    p.gemm.native(p.shared_input_f16.u16(),l.shared_up.type,l.shared_up.data.p,
                  p.shared_up_out.f(),rows.count,l.shared_up.output,2560);
    gpu::shared_swiglu_f16(p.shared_gate_out.f(),p.shared_up_out.f(),
                           p.shared_hidden.u16(),rows.count*l.shared_gate.output,stream);
    p.gemm.native(p.shared_hidden.u16(),l.shared_down.type,l.shared_down.data.p,
                  p.shared_out.f(),rows.count,2560,l.shared_down.input);
    p.gemm.bf16(p.shared_input_bf16.u16(),l.shared_gate_input,p.shared_scalar.f(),rows.count,1,2560);
    gpu::shared_add(routed_local_sum,p.shared_out.f(),p.shared_scalar.f(),n,2560,true,stream);
    return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastMixer::has_fast_hc()const{return impl_->hc_bf16;}
bool FastMixer::use_fast_hc(int rows)const{return impl_->hc_bf16&&(!impl_->hc_large_only||rows>=9);}
bool FastMixer::hyper_connection(const k::FusedGrArgs* args,int rows,void* stream,std::string& error)try {
    auto& p=*impl_;
    if(!p.ready||!p.hc_bf16||rows<1||rows>k::kWindowMaxT||stream!=p.stream)
        throw std::runtime_error("invalid TP BF16 HC batch");
    constexpr int N=2560,D=10240,LR=320;
    for(int row=0;row<rows;++row) {
        const auto& a=args[row];
        if(a.R!=args[0].R+size_t(row)*D||a.mixed!=args[0].mixed+size_t(row)*N||
           a.w_norm!=args[0].w_norm||a.w_down!=args[0].w_down||a.w_up!=args[0].w_up||
           a.w_inject!=args[0].w_inject||a.q8_down||a.s_down||!a.w_down||!a.w_up||!a.w_inject)
            throw std::runtime_error("unsupported TP BF16 HC layout");
    }
    gpu::hc_pack(args,rows,p.hc_x.f(),(uint16_t*)p.hc_hi.p,stream);
    if(p.hc_packed_down) {
        const auto it=p.hc_combined_weights.find(args[0].w_down);
        if(it==p.hc_combined_weights.end())throw std::runtime_error("missing packed HC weight");
        p.gemm.bf16((uint16_t*)p.hc_hi.p,it->second->u16(),p.hc_combined.f(),rows,336,D);
        gpu::hc_packed_silu_publish(args,p.hc_combined.f(),(uint16_t*)p.hc_lhi.p,rows,stream);
    } else {
        p.gemm.bf16((uint16_t*)p.hc_hi.p,args[0].w_down,p.hc_lo.f(),rows,LR,D);
        gpu::hc_silu_pack(p.hc_lo.f(),(uint16_t*)p.hc_lhi.p,rows,stream);
    }
    p.gemm.bf16((uint16_t*)p.hc_lhi.p,args[0].w_up,p.hc_gate.f(),rows,D,LR);
    k::native_gr_pre_gated_multi(p.hc_x.f(),p.hc_gate.f(),args[0].mixed,N,4,rows,true,stream);
    if(!p.hc_packed_down) {
        p.gemm.bf16((uint16_t*)p.hc_hi.p,args[0].w_inject,p.hc_inject.f(),rows,4,D);
        gpu::hc_publish_inject(args,p.hc_inject.f(),rows,stream);
    }
    return true;
}catch(const std::exception& e){error=e.what();return false;}
uint64_t fast_mixer_state_bytes_per_slot(const core::ModelGeometry& g,const FastMixerLimits& limits) {
    return uint64_t(g.n_gdn_layers())*(128*18*128+18*128*5/3*3)*4*(1+limits.snapshot_depth);
}
bool FastMixer::initialize(const FastMixerLimits& limits,void* stream,std::string& error)try {
    auto& p=*impl_;const auto& g=p.geometry;
    if(p.ready||p.rank<0||p.rank>2||limits.max_rows<8||limits.max_rows>8192||limits.max_slots<1||limits.max_slots>8||!stream)
        throw std::runtime_error("invalid fast mixer limits");
    p.limits=limits;p.stream=stream;
    if(const char* batch=std::getenv("STRATA_TP_QSA_PROMPT_BATCH");batch&&std::string_view(batch)=="256")
        p.prompt_qsa_batch=256;
    if(const char* fast=std::getenv("STRATA_TP_QSA_PROMPT_FAST");fast&&fast[0]=='1') {
        p.prompt_qsa_fast=true;
        gpu::fast_qsa::qsa_prompt_attn_set_imma(1);
    }
    if(!p.gemm.init(stream,32ll<<20,error))return false;
    const char* shared_env=std::getenv("STRATA_TP_SHARED_SHARD");
    p.shared_sharded=shared_env&&shared_env[0]=='1';
    if(p.shared_sharded && (g.n_embd!=2560||g.n_ff!=640))
        throw std::runtime_error("TP shared sharding requires hidden 2560 and shared FF 640");
    const int shared_first[3]={0,256,448};
    const int shared_width[3]={256,192,192};
    auto value_heads=consecutive(p.key_first,p.key_heads);
    for(int repeat=1;repeat<3;++repeat)append(value_heads,consecutive(repeat*16+p.key_first,p.key_heads));
    auto value_rows=std::vector<int>{};for(int h:value_heads)append(value_rows,consecutive(h*128,128));
    auto conv_rows=consecutive(p.key_first*128,p.key_heads*128);
    append(conv_rows,consecutive((16+p.key_first)*128,p.key_heads*128));
    for(int h:value_heads)append(conv_rows,consecutive(4096+h*128,128));
    const auto all_hidden=consecutive(0,2560);
    int gi=0,qi=0;
    for(int l=0;l<g.n_layers;++l) {
        auto layer=std::make_unique<Impl::Layer>();core::LayerView view(p.weights,l);
        layer->qsa=core::is_qsa_layer(g,l);layer->ordinal=layer->qsa?qi++:gi++;
        auto matrix=[&](Matrix& dst,const char* name,const std::vector<int>& rows,const std::vector<int>& inputs=std::vector<int>{}) {
            copy_matrix(dst,p.need(view,name),rows,inputs,stream);p.weight_bytes+=dst.data.bytes;
        };
        if(!layer->qsa) {
            matrix(layer->qkv,"attn_qkv.weight",conv_rows);matrix(layer->z,"attn_gate.weight",value_rows);
            matrix(layer->alpha,"ssm_alpha.weight",value_heads);matrix(layer->beta,"ssm_beta.weight",value_heads);
            matrix(layer->output,"ssm_out.weight",all_hidden,value_heads);
            auto vector=[&](Buffer& dst,const char* name,const std::vector<int>& rows,int inner) {
                const auto* w=p.need(view,name);dst.allocate(rows.size()*inner*4);
                for(size_t i=0;i<rows.size();++i)check(cudaMemcpyAsync(dst.f(i*inner),static_cast<const float*>(w->data)+rows[i]*inner,
                    inner*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
            };
            vector(layer->conv,"ssm_conv1d.weight",conv_rows,4);vector(layer->dt,"ssm_dt.bias",value_heads,1);vector(layer->a,"ssm_a",value_heads,1);
            layer->norm=static_cast<const float*>(p.need(view,"ssm_norm.weight")->data);
            for(int slot=0;slot<limits.max_slots;++slot) {
                auto state=std::make_unique<Buffer>(),history=std::make_unique<Buffer>();
                state->allocate(size_t(128)*p.value_heads*128*4);history->allocate(size_t(p.channels)*3*4);
                check(cudaMemsetAsync(state->p,0,state->bytes,(cudaStream_t)stream));check(cudaMemsetAsync(history->p,0,history->bytes,(cudaStream_t)stream));
                layer->states.push_back(std::move(state));layer->histories.push_back(std::move(history));
                auto sp=std::make_unique<Buffer>(),hp=std::make_unique<Buffer>();
                sp->allocate(layer->states.back()->bytes*(limits.snapshot_depth+1));
                hp->allocate(layer->histories.back()->bytes*(limits.snapshot_depth+1));
                layer->state_prefix.push_back(std::move(sp));layer->history_prefix.push_back(std::move(hp));
            }
        } else {
            std::vector<int> heads;append(heads,consecutive(p.rank*4,4));append(heads,consecutive(12+p.rank*4,4));
            std::vector<int> qrows,oruns;for(int h:heads){append(qrows,consecutive(h*512,512));append(oruns,consecutive(h*2,2));}
            matrix(layer->q,"attn_q.weight",qrows);matrix(layer->key,"attn_k.weight",consecutive(0,512));
            matrix(layer->value,"attn_v.weight",consecutive(0,512));matrix(layer->output,"attn_output.weight",all_hidden,oruns);
            matrix(layer->index_key,"indexer.k_proj.weight",consecutive(0,128));matrix(layer->index_query,"indexer.q_proj.weight",consecutive(0,512));
            layer->qnorm=(const float*)p.need(view,"attn_q_norm.weight")->data;layer->knorm=(const float*)p.need(view,"attn_k_norm.weight")->data;
            layer->iqnorm=(const float*)p.need(view,"indexer.q_norm.weight")->data;layer->iknorm=(const float*)p.need(view,"indexer.k_norm.weight")->data;
            for(int slot=0;slot<limits.max_slots;++slot) {
                auto sp=std::make_unique<Buffer>(),hp=std::make_unique<Buffer>();
                sp->allocate(size_t(limits.snapshot_depth+1)*128*4);
                hp->allocate(size_t(limits.snapshot_depth+1)*(3*128+128+1)*4);
                layer->state_prefix.push_back(std::move(sp));layer->history_prefix.push_back(std::move(hp));
            }
        }
        if(p.shared_sharded) {
            const auto* gate=p.need(view,"ffn_gate_shexp.weight");
            const auto* up=p.need(view,"ffn_up_shexp.weight");
            const auto* down=p.need(view,"ffn_down_shexp.weight");
            const auto* scalar=p.need(view,"ffn_gate_inp_shexp.weight");
            if(!gate->native_data||!up->native_data||!down->native_data||
               gate->native_type!=8||up->native_type!=8||down->native_type!=8||
               gate->ne0!=g.n_embd||up->ne0!=g.n_embd||gate->ne1!=g.n_ff||up->ne1!=g.n_ff||
               down->ne0!=g.n_ff||down->ne1!=g.n_embd||
               scalar->kind!=core::WeightKind::Bf16InF32||!scalar->data||scalar->elements!=g.n_embd)
                throw std::runtime_error("TP shared sharding requires native Q8_0 projections and a resident BF16 gate");
            const int first=shared_first[p.rank],width=shared_width[p.rank];
            copy_matrix(layer->shared_gate,gate,consecutive(first,width),{},stream);
            copy_matrix(layer->shared_up,up,consecutive(first,width),{},stream);
            std::vector<int> input_blocks;
            input_blocks.reserve(size_t(width/32));
            for(int block=first/32;block<(first+width)/32;++block)input_blocks.push_back(block);
            copy_matrix(layer->shared_down,down,consecutive(0,2560),input_blocks,stream,32);
            layer->shared_gate_input=static_cast<const uint16_t*>(scalar->data);
            p.weight_bytes+=layer->shared_gate.data.bytes+layer->shared_up.data.bytes+layer->shared_down.data.bytes;
        }
        p.layers.push_back(std::move(layer));
    }
    const size_t rows=limits.max_rows;
    const char* dense_f16_env=std::getenv("STRATA_TP_DENSE_F16");
    if(dense_f16_env&&dense_f16_env[0]=='1') {
        // Decode reuses the expansion across windows; original Q8 blocks remain
        // available for small batches. Prefill avoids repeated dequantization.
        for(const auto& layer:p.layers)
            for(auto* matrix:{&layer->qkv,&layer->z,&layer->output,&layer->q,&layer->value}) {
                if(matrix->type!=8||matrix->bf16||matrix->output<1536||matrix->input<1536)continue;
                const size_t elements=size_t(matrix->input)*matrix->output;
                matrix->resident_f16.allocate(elements*sizeof(uint16_t));
                k::dequant_f16(8,matrix->data.p,0,matrix->output,matrix->input,
                    (uint16_t*)matrix->resident_f16.p,stream);
                p.weight_bytes+=matrix->resident_f16.bytes;
            }
    }
    p.input16.allocate(rows*2560*2);p.q8.allocate(k::kWindowMaxT*6144/32*36);
    auto floats=[&](Buffer& b,size_t width){b.allocate(rows*width*4);};
    floats(p.qkv,p.channels);floats(p.z,p.width);floats(p.alpha,p.value_heads);floats(p.beta,p.value_heads);floats(p.gate,p.value_heads);
    floats(p.h,p.channels);floats(p.y,std::max(p.width,2048));floats(p.normalized,p.width);floats(p.qfull,4096);floats(p.q,2048);floats(p.key,512);floats(p.value,512);
    floats(p.raw,128);floats(p.index_query,512);floats(p.index_dead,128);p.steps.allocate(rows*4*4);p.positions.allocate(rows*8*4);
    p.key_positions.allocate(rows*2*4);p.index_positions.allocate(rows*4*4);p.ids.allocate(rows*2051*4);
    p.scores.allocate(size_t(p.prompt_qsa_batch)*(262144/4+2)*4);
    auto shapes=core::qsa_shapes(g);shapes.n_head=8;
    p.attn_scratch.allocate(size_t(32)*k::qsa_decode_attn_scratch_floats(2051,shapes)*4);
    p.status.allocate(4);p.keep.allocate(4);int one=1;
    if(p.shared_sharded) {
        const size_t rows=limits.max_rows;
        p.shared_input_bf16.allocate(rows*size_t(g.n_embd)*sizeof(uint16_t));
        p.shared_input_f16.allocate(rows*size_t(g.n_embd)*sizeof(uint16_t));
        p.shared_gate_out.allocate(rows*256*sizeof(float));
        p.shared_up_out.allocate(rows*256*sizeof(float));
        p.shared_hidden.allocate(rows*256*sizeof(uint16_t));
        p.shared_out.allocate(rows*size_t(g.n_embd)*sizeof(float));
        p.shared_scalar.allocate(rows*sizeof(float));
        p.shared_q8_scratch.allocate(kernels::native_q8_1_bytes(int(g.n_embd),k::kWindowMaxT));
    }
    p.gdn_scratch.allocate(gpu::fast_gdn_workspace_bytes(p.key_heads)+rows*p.width*4);
    if(const char* on=std::getenv("STRATA_TP_HC_BF16");on&&(on[0]=='1'||std::string_view(on)=="large")) {
        p.hc_bf16=true;
        p.hc_large_only=std::string_view(on)=="large";
        p.hc_x.allocate(size_t(k::kWindowMaxT)*10240*4);
        p.hc_hi.allocate(size_t(k::kWindowMaxT)*10240*2);
        p.hc_lo.allocate(size_t(k::kWindowMaxT)*320*4);
        p.hc_lhi.allocate(size_t(k::kWindowMaxT)*320*2);
        p.hc_gate.allocate(size_t(k::kWindowMaxT)*10240*4);
        p.hc_inject.allocate(size_t(k::kWindowMaxT)*4*4);
        if(const char* packed=std::getenv("STRATA_TP_HC_PACKED_DOWN");packed&&packed[0]=='1') {
            p.hc_packed_down=true;
            p.hc_combined.allocate(size_t(k::kWindowMaxT)*336*4);
            for(int l=0;l<g.n_layers;++l)for(const char* half:{"hc_attn_","hc_ffn_"}) {
                core::LayerView view(p.weights,l);
                const auto* down=p.need(view,(std::string(half)+"down.weight").c_str());
                const auto* inject=p.need(view,(std::string(half)+"inject.weight").c_str());
                if(!down->data||!inject->data||down->kind!=core::WeightKind::Bf16InF32||
                   inject->kind!=core::WeightKind::Bf16InF32||down->ne0!=10240||down->ne1!=320||
                   inject->ne0!=10240||inject->ne1!=4)throw std::runtime_error("unsupported packed HC weights");
                auto combined=std::make_unique<Buffer>();combined->allocate(size_t(336)*10240*2);
                check(cudaMemsetAsync(combined->p,0,combined->bytes,(cudaStream_t)stream));
                check(cudaMemcpyAsync(combined->p,down->data,size_t(320)*10240*2,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                check(cudaMemcpyAsync(combined->u16(size_t(320)*10240),inject->data,size_t(4)*10240*2,
                    cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
                p.weight_bytes+=combined->bytes;
                p.hc_combined_weights.emplace((const uint16_t*)down->data,std::move(combined));
            }
        }
    }
    if(const char* on=std::getenv("STRATA_TP_DENSE_MMQ");on&&on[0]=='1') {
        p.dense_mmq=std::make_unique<prefill::mmq::Context>();
        p.dense_q8.allocate(prefill::mmq::q8_bytes(k::kWindowMaxT,6144));
        p.dense_bounds.allocate(size_t(k::kWindowMaxT+1)*2*sizeof(int32_t));
        p.dense_ids.allocate(k::kWindowMaxT*sizeof(int32_t));
        std::array<int32_t,(k::kWindowMaxT+1)*2> bounds{};
        for(int count=1;count<=k::kWindowMaxT;++count)bounds[count*2+1]=count;
        check(cudaMemcpyAsync(p.dense_bounds.p,bounds.data(),sizeof(bounds),cudaMemcpyHostToDevice,(cudaStream_t)stream));
        prefill::mmq::iota(p.dense_ids.i(),k::kWindowMaxT,stream);
        check(cudaMemsetAsync(p.qkv.p,0,p.qkv.bytes,(cudaStream_t)stream));
        // Prime the MMQ allocator for every eligible shape before graph capture.
        for(const auto& layer:p.layers)
            for(const auto* matrix:{&layer->qkv,&layer->z,&layer->alpha,&layer->beta,&layer->output,
                                    &layer->q,&layer->key,&layer->value,&layer->index_key,&layer->index_query})
                for(int count=4;count<=std::min(limits.max_rows,k::kWindowMaxT);++count)
                    if(p.use_dense_mmq(*matrix,count))p.project_mmq(*matrix,p.qkv.f(),p.qfull.f(),count);
    }
    check(cudaMemsetAsync(p.status.p,0,4,(cudaStream_t)stream));check(cudaMemcpyAsync(p.keep.p,&one,4,cudaMemcpyHostToDevice,(cudaStream_t)stream));
    check(cudaStreamSynchronize((cudaStream_t)stream));
    p.ready=true;
    std::fprintf(stderr,"fast mixer rank %d: GDN %d/%d heads, QSA 4+4, sliced output columns, %llu weight bytes\n",p.rank,p.key_heads,p.value_heads,(unsigned long long)p.weight_bytes);
    return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastMixer::bind_sessions(core::SessionState* const* sessions,int count,std::string& error) {
    auto& p=*impl_;if(!p.ready||count!=p.limits.max_slots){error="fast mixer session count mismatch";return false;}
    p.sessions.assign(sessions,sessions+count);p.window_counts.assign(count,0);return true;
}
bool FastMixer::begin_slots(const int32_t* ids,const int32_t* counts,int count,bool speculative,void*,std::string& error) {
    auto& p=*impl_;std::fill(p.window_counts.begin(),p.window_counts.end(),0);
    for(int i=0;i<count;++i) {
        if(ids[i]<0||size_t(ids[i])>=p.window_counts.size()||counts[i]<1||counts[i]>p.limits.snapshot_depth+1) {
            error="invalid fast mixer window layout";return false;
        }
        p.window_counts[ids[i]]=counts[i];
    }
    p.speculative_window=speculative;return true;
}
bool FastMixer::run(int64_t layer,const core::TensorParallelRows& rows,const float* input,float* output,void* stream,std::string& error)try {
    auto& p=*impl_;
    if(!p.ready||stream!=p.stream||layer<0||size_t(layer)>=p.layers.size()||rows.count<1||rows.count>p.limits.max_rows||!input||!output)
        throw std::runtime_error("invalid fast mixer operation");
    if(rows.segment_count) {
        if(rows.independent||!rows.segments||rows.segment_count<1||rows.segment_count>p.limits.max_slots)
            throw std::runtime_error("invalid packed prompt segments");
        int64_t first=0;std::vector<int> slots;
        for(int i=0;i<rows.segment_count;++i) {
            const auto& s=rows.segments[i];
            if(s.first!=first||s.count<1||s.count>rows.count-first||s.position<0||
               s.slot<0||size_t(s.slot)>=p.sessions.size()||
               std::find(slots.begin(),slots.end(),s.slot)!=slots.end())
                throw std::runtime_error("packed prompt segment is outside its slot/row range");
            slots.push_back(s.slot);first+=s.count;
        }
        if(first!=rows.count)throw std::runtime_error("packed prompt segments do not cover rows");
    }
    auto& l=*p.layers[size_t(layer)];return l.qsa?p.qsa(l,rows,input,output,error):p.gdn(l,rows,input,output,error);
}catch(const std::exception& e){error=e.what();return false;}
bool FastMixer::reset_slots(const int32_t* ids,int count,void* stream,std::string& error)try {
    auto& p=*impl_;for(int i=0;i<count;++i) {
        if(ids[i]<0||ids[i]>=p.limits.max_slots)throw std::runtime_error("invalid reset slot");
        for(auto& layer:p.layers)if(!layer->qsa) {
            auto& s=*layer->states[ids[i]];auto& h=*layer->histories[ids[i]];
            check(cudaMemsetAsync(s.p,0,s.bytes,(cudaStream_t)stream));check(cudaMemsetAsync(h.p,0,h.bytes,(cudaStream_t)stream));
        }
    }return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastMixer::commit_slots(const int32_t* ids,const int32_t* keep,int count,void* stream,std::string& error)try {
    auto& p=*impl_;
    for(int i=0;i<count;++i) {
        if(ids[i]<0||ids[i]>=p.limits.max_slots||keep[i]<1||keep[i]>p.limits.snapshot_depth+1)
            throw std::runtime_error("invalid fast mixer prefix commit");
        if(!p.speculative_window || keep[i]==p.window_counts[ids[i]])continue;
        for(auto& layer:p.layers)if(!layer->qsa) {
            auto& state=*layer->states[ids[i]];auto& history=*layer->histories[ids[i]];
            check(cudaMemcpyAsync(state.p,layer->state_prefix[ids[i]]->f(size_t(keep[i]-1)*state.bytes/4),state.bytes,
                cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
            check(cudaMemcpyAsync(history.p,layer->history_prefix[ids[i]]->f(size_t(keep[i]-1)*history.bytes/4),history.bytes,
                cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
        } else if(keep[i]<p.window_counts[ids[i]]) {
            auto& state=p.sessions[ids[i]]->qsa_states[layer->ordinal];
            const size_t stride=size_t(3*128+128+1)*4;
            const auto* snapshot=static_cast<const uint8_t*>(layer->history_prefix[ids[i]]->p)+size_t(keep[i])*stride;
            // Snapshot before the first rejected row equals the indexer state
            // immediately after the accepted prefix. The overwritten spare
            // pooled row is not used for incomplete tails by block_scores.
            check(cudaMemcpyAsync(state.idx_tail,snapshot,3*128*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
            check(cudaMemcpyAsync(state.idx_dead,snapshot+3*128*4,128*4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
            check(cudaMemcpyAsync(state.idx_block_pos,snapshot+(3*128+128)*4,4,cudaMemcpyDeviceToDevice,(cudaStream_t)stream));
        }
    }return true;
}catch(const std::exception& e){error=e.what();return false;}
bool FastMixer::rollback_slots(const int32_t*,const int64_t*,int,void*,std::string& error){error="fast mixer rollback not captured";return false;}
bool FastMixer::validate_execution_status(void* stream,std::string& error) {
    int status=0;auto result=cudaMemcpyAsync(&status,impl_->status.p,4,cudaMemcpyDeviceToHost,(cudaStream_t)stream);
    if(result==cudaSuccess)result=cudaStreamSynchronize((cudaStream_t)stream);
    if(result!=cudaSuccess||status){error=result==cudaSuccess?"invalid device slot/position":cudaGetErrorString(result);return false;}return true;
}
int64_t FastMixer::owned_weight_bytes()const{return int64_t(impl_->weight_bytes);}
bool FastMixer::initialized()const noexcept{return impl_->ready;}
}
