// CPU: g++ -std=c++20 -Wall -Wextra -Werror -Iinclude tests/distributed/test_state_evidence.cpp src/distributed/state_evidence.cpp -o /tmp/test_state_evidence
#include "strata/distributed/state_evidence.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace strata::distributed;
int checks=0, failures=0;
void check(bool value,const char* what) { ++checks; if(!value) { ++failures; std::cerr<<"FAIL: "<<what<<'\n'; } }
StateReadback host_read() {
    return [](void* dst,const void* src,std::size_t n,std::string&) { std::memcpy(dst,src,n); return true; };
}
std::string read_file(const std::filesystem::path& path) {
    std::ifstream f(path,std::ios::binary); return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void reconstruction() {
    // Nonuniform 6/5/5 ownership with THREE value-head runs per rank.
    constexpr int S=3, HV=48;
    std::vector<float> reference(S*HV*S);
    for(std::size_t i=0;i<reference.size();++i) reference[i]=float(i)+0.25f;
    const int starts[]={0,6,11,16};
    std::vector<std::vector<float>> local(3);
    std::vector<StateAxisShard> shards;
    for(int rank=0;rank<3;++rank) {
        std::vector<uint64_t> indices;
        for(int run=0;run<3;++run) for(int h=starts[rank];h<starts[rank+1];++h) indices.push_back(uint64_t(run*16+h));
        for(int row=0;row<S;++row) for(auto h:indices) for(int j=0;j<S;++j)
            local[rank].push_back(reference[(row*HV+h)*S+j]);
        shards.push_back({local[rank].data(),local[rank].size()*sizeof(float),indices});
    }
    std::string error; std::vector<uint8_t> output={123};
    check(reconstruct_state_axis({S,HV,S},1,4,shards,output,error),"three-run recurrent reconstruction");
    check(output.size()==reference.size()*4 && std::memcmp(output.data(),reference.data(),output.size())==0,"all recurrent elements in original order");
    auto missing=shards; missing.pop_back(); const auto before=output;
    check(!reconstruct_state_axis({S,HV,S},1,4,missing,output,error) && output==before,"missing rank rejected without changing output");
    auto duplicate=shards; duplicate[1].indices[0]=duplicate[0].indices[0];
    check(!reconstruct_state_axis({S,HV,S},1,4,duplicate,output,error),"overlapping head rejected");
    auto truncated=shards; --truncated[0].bytes;
    check(!reconstruct_state_axis({S,HV,S},1,4,truncated,output,error),"truncated rank bytes rejected");
    auto outside=shards; outside[0].indices[0]=HV;
    check(!reconstruct_state_axis({S,HV,S},1,4,outside,output,error),"unknown global head rejected");
    check(!reconstruct_state_axis({UINT64_MAX,2},0,8,shards,output,error),"reconstruction arithmetic overflow rejected");

    // q and k contiguous ranges plus the same three value runs, channel-major.
    constexpr int HK=16, CHANNELS=(2*HK+HV)*S, TAPS=3;
    std::vector<float> conv(CHANNELS*TAPS);
    for(std::size_t i=0;i<conv.size();++i) conv[i]=float(9000+i);
    shards.clear(); local.assign(3,{});
    for(int rank=0;rank<3;++rank) {
        std::vector<uint64_t> channels;
        for(int component=0;component<5;++component)
            for(int h=starts[rank];h<starts[rank+1];++h) for(int j=0;j<S;++j)
                channels.push_back(uint64_t((component*HK+h)*S+j));
        for(auto c:channels) for(int tap=0;tap<TAPS;++tap) local[rank].push_back(conv[c*TAPS+tap]);
        shards.push_back({local[rank].data(),local[rank].size()*4,channels});
    }
    check(reconstruct_state_axis({CHANNELS,TAPS},0,4,shards,output,error),"qkv convolution channels reconstruct");
    check(output.size()==conv.size()*4 && std::memcmp(output.data(),conv.data(),output.size())==0,"convolution history retains each tap");
}
void writer(const std::filesystem::path& root) {
    std::string error; uint64_t n=0;
    StateEvidenceDescriptor d{"gdn","layer.0.recurrent","<f4",{2,3,2},0,{},-1,{}};
    check(state_evidence_bytes(d,n,error) && n==48,"exact byte count");
    auto bad=d;bad.dtype="native_float";check(!state_evidence_bytes(bad,n,error),"implicit dtype rejected");
    bad=d;bad.kind="digest";check(!state_evidence_bytes(bad,n,error),"unknown metadata kind rejected");
    bad=d;bad.shape={UINT64_MAX,2};check(!state_evidence_bytes(bad,n,error),"byte overflow rejected");
    bad=d;bad.logical_shape={2,8,2};bad.shard_axis=1;bad.axis_indices={1,1,4};
    check(!state_evidence_bytes(bad,n,error),"duplicate shard ownership rejected");
    std::vector<float> values(12);for(std::size_t i=0;i<values.size();++i) values[i]=float(i);
    std::vector<StateEvidenceArray> arrays={borrowed_state_array(d,values.data(),host_read())};
    StateEvidenceRequest request{"req\"line\n",3,4,48,state_evidence_inventory(arrays)};
    const auto good=root/"good";
    check(write_state_evidence(good.string(),request,arrays,error),"bounded complete capture written");
    const auto json=read_file(good/"manifest.json");
    check(json.find("\"logical_complete\":true")!=std::string::npos && json.find("req\\\"line\\n")==std::string::npos && json.find("\\u000a")!=std::string::npos,"manifest complete and JSON escaped");
    check(read_file(good/"state-0.bin").size()==48,"array file exact shape bytes");
    check(!write_state_evidence(good.string(),request,arrays,error),"existing snapshot never overwritten");
    check(write_state_evidence((root/"nested"/"request-0"/"position-3").string(),request,arrays,error),"nested driver request directories created");
    auto cap=request;cap.max_bytes=47;
    check(!write_state_evidence((root/"cap").string(),cap,arrays,error) && !std::filesystem::exists(root/"cap"),"byte cap fails before creating artifact");
    auto inventory=request;inventory.required.push_back({"kv","layer.3.k","<f2",{4,2,256},3,{},-1,{}});
    check(!write_state_evidence((root/"missing").string(),inventory,arrays,error),"full required arrays missing rejects gate");
    inventory=request;inventory.required[0].shape={2,2,3};
    check(!write_state_evidence((root/"shape").string(),inventory,arrays,error),"same bytes but wrong shape rejected");
    auto failed=arrays;failed[0].read=[](void*,std::size_t,std::string& e){e="readback failed";return false;};
    check(!write_state_evidence((root/"failed").string(),request,failed,error) && !std::filesystem::exists(root/"failed"/"manifest.json"),"readback failure never publishes complete manifest");
    auto shard=arrays;auto& sd=shard[0].descriptor;sd.logical_shape={2,6,2};sd.shard_axis=1;sd.axis_indices={0,2,4};
    auto shard_request=request;shard_request.required=state_evidence_inventory(shard);
    check(write_state_evidence((root/"shard").string(),shard_request,shard,error),"rank local fragment written");
    check(read_file(root/"shard"/"manifest.json").find("\"logical_complete\":false")!=std::string::npos,"rank fragment never declares logical completeness");
    auto wrong=request;wrong.token_count=3;
    check(!write_state_evidence((root/"wrong-pos").string(),wrong,arrays,error),"teacher-forced token count mismatch rejected");
    StateEvidenceDescriptor empty{"kv","layer.3.k","<f2",{0,2,256},3,{},-1,{}};
    std::vector<StateEvidenceArray> reset_arrays={{empty,[](void*,std::size_t n,std::string&){return n==0;}}};
    StateEvidenceRequest reset{"reset-request",0,0,1,state_evidence_inventory(reset_arrays),true};
    check(write_state_evidence((root/"reset").string(),reset,reset_arrays,error),"reset empty loaded KV written explicitly");
    check(read_file(root/"reset"/"manifest.json").find("\"phase\":\"reset\"")!=std::string::npos,"reset phase distinguished from first token");
    reset.reset=false;reset.token_count=1;
    check(!write_state_evidence((root/"empty-token").string(),reset,reset_arrays,error),"empty KV after real token rejected");
}
void paging() {
    // Two heads, 3 physical slots, 4 cells/page, 2 half bits/cell.
    std::vector<uint16_t> pool(3*2*4*2);
    for(std::size_t i=0;i<pool.size();++i) pool[i]=uint16_t(100+i);
    std::vector<uint16_t> result(5*2*2);
    auto reader=paged_kv_state_reader(pool.data(),3,4,2,2,5,{2,0},host_read());
    std::string error;
    check(reader(result.data(),result.size()*2,error),"permuted paged KV loaded 5 cells");
    bool matches=true;
    for(int t=0;t<5;++t)for(int h=0;h<2;++h)for(int d=0;d<2;++d) {
        const int slot=t<4?2:0;
        matches=matches&&result[(t*2+h)*2+d]==pool[((slot*2+h)*4+t%4)*2+d];
    }
    check(matches,"KV shape excludes unused cells and preserves head ordering");
    reader=paged_kv_state_reader(pool.data(),3,4,2,2,5,{2,-1},host_read());
    check(!reader(result.data(),result.size()*2,error),"nonresident loaded KV rejected");
    reader=paged_kv_state_reader(pool.data(),3,0,2,2,5,{2,0},host_read());
    check(!reader(result.data(),result.size()*2,error),"zero page size rejected before division");
}
}
int main() {
    const auto root=std::filesystem::temp_directory_path()/"strata-state-evidence-cpu-test";
    if(std::filesystem::exists(root)) { std::cerr<<"test directory already exists\n";return 2; }
    std::filesystem::create_directory(root);
    reconstruction();writer(root);paging();std::filesystem::remove_all(root);
    std::cout<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
