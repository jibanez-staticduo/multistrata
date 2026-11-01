#include "strata/distributed/fast_runtime.hpp"
#include "strata/distributed/fast_model.hpp"
#include "strata/distributed/fast_service.hpp"
#include "strata/distributed/process_exit_guard.hpp"
#include "strata/kernels/window_limits.hpp"
#include "nccl_api.hpp"
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
#include "fast_roce_api.hpp"
#include "strata/distributed/hybrid_backend.hpp"
#include "roce_control.hpp"
#include "strata/distributed/control_rdma.hpp"
#endif

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <unistd.h>
#include <sstream>
#include <fstream>

namespace strata::distributed {
namespace {
void require(Failure failure) { if (failure) throw std::runtime_error(failure.detail); }
std::string quote(std::string_view text) {
    std::string out = "\"";
    for (const auto c : text) {
        if (c == '"' || c == '\\') out.push_back('\\');
        if (static_cast<unsigned char>(c) < 32) throw std::runtime_error("invalid event string");
        out.push_back(c);
    }
    return out + '"';
}
uint64_t roce_slot_bytes_from_env() {
    const char* value = std::getenv("STRATA_TP_ROCE_SLOT_MIB");
    if (value == nullptr || value[0] == '\0') return 2u << 20;
    const std::string_view mib(value);
    if (mib == "2") return 2u << 20;
    if (mib == "8") return 8u << 20;
    if (mib == "16") return 16u << 20;
    if (mib == "32") return 32u << 20;
    throw std::runtime_error("STRATA_TP_ROCE_SLOT_MIB must be 2, 8, 16 or 32");
}
class ProtocolIo {
public:
    ProtocolIo(bool root, uint32_t timeout, int preserved) : timeout_(timeout) {
        if (root) {
            output_ = ::dup(preserved >= 0 ? preserved : STDOUT_FILENO);
            if (output_ < 0) throw std::runtime_error("cannot preserve protocol output");
            const auto flags = fcntl(output_, F_GETFL);
            if (flags < 0 || fcntl(output_, F_SETFL, flags | O_NONBLOCK)) throw std::runtime_error("cannot bound protocol output");
        }
        if (::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot redirect native diagnostics");
        old_input_ = fcntl(STDIN_FILENO, F_GETFL);
        if (root && (old_input_ < 0 || fcntl(STDIN_FILENO, F_SETFL, old_input_ | O_NONBLOCK)))
            throw std::runtime_error("cannot bound protocol input");
        std::signal(SIGPIPE, SIG_IGN);
    }
    ~ProtocolIo() {
        if (output_ >= 0) ::close(output_);
        if (old_input_ >= 0) fcntl(STDIN_FILENO, F_SETFL, old_input_);
    }
    void emit(std::string text) {
        text.push_back('\n');
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_);
        size_t offset = 0;
        while (offset < text.size()) {
            const auto count = ::write(output_, text.data() + offset, text.size() - offset);
            if (count > 0) { offset += size_t(count); continue; }
            if ((errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) || std::chrono::steady_clock::now() >= end)
                throw std::runtime_error("protocol output failed or exceeded deadline");
            pollfd descriptor{output_, POLLOUT, 0};
            if (::poll(&descriptor, 1, 10) < 0 && errno != EINTR) throw std::runtime_error("protocol output poll failed");
        }
    }
    Failure poll(uint32_t wait_ms, std::string& line, bool& eof) {
        line.clear(); eof = false;
        auto extract = [&] {
            const auto end = pending_.find('\n');
            if (end == std::string::npos) return false;
            line = pending_.substr(0, end); pending_.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        };
        if (extract()) return {};
        if (eof_) { eof = true; return {}; }
        pollfd descriptor{STDIN_FILENO, POLLIN, 0};
        const auto ready = ::poll(&descriptor, 1, int(wait_ms));
        if (ready < 0 && errno != EINTR) return Failure::fail(FailureCode::comm_failed, "protocol input poll failed");
        if (ready <= 0) return {};
        char buffer[8192];
        for(;;) {
        const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
        if (count < 0) return errno == EINTR || errno == EAGAIN ? Failure{} : Failure::fail(FailureCode::comm_failed, "protocol read failed");
        if (!count) {
            eof_ = true;
            if (!pending_.empty()) return Failure::fail(FailureCode::truncated_frame, "protocol input ended before newline");
            eof = true; return {};
        }
        pending_.append(buffer, size_t(count));
        if ((pending_.find('\n') == std::string::npos ? pending_.size() : pending_.find('\n')) > kDistributedMaxLineBytes)
            return Failure::fail(FailureCode::oversized_frame, "protocol input exceeds 4 MiB");
        if(extract())return {};
        // Drain the queued frame before reporting idle: an 8 KiB pipe read
        // often contains only part of a prompt already submitted by HTTP.
        }
    }
private:
    int output_ = -1, old_input_ = -1;
    uint32_t timeout_;
    bool eof_ = false;
    std::string pending_;
};
}
int run_fast_runtime(const FastRuntimeConfig& cfg, FastAssets& assets,
                     FastTensorParallel::Mixer mixer, FastTensorParallel::Experts experts,
                     core::TensorParallelHooks* state_provider, void* stream,
                     std::function<bool(void*, std::string&)> validate) {
    ProtocolIo io(cfg.rank == 0, cfg.timeout_ms, cfg.protocol_output_fd);
    try {
        if (cfg.rank >= 3 || cfg.port < 1024 || cfg.port > 65535 || cfg.context > 262144 ||
            cfg.timeout_ms < 100 || cfg.slots < 1 || cfg.slots > 8 || !stream)
            throw std::runtime_error("invalid final TP runtime configuration");
        Credential credential;
        require(Credential::from_env("STRATA_TP3_CREDENTIAL", credential));
        SystemClock clock;
        nccl::NcclConfig nccl_cfg{cfg.rank, 3, {}, cfg.nccl_library};
        std::string error;
        const char* strict=std::getenv("STRATA_TP_RDMA_ONLY");
        const bool rdma_only=strict&&strict[0]=='1';
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        std::unique_ptr<roce::RoceBackend> roce_backend;
        std::unique_ptr<PendingDmaExitGuard<roce::RoceBackend>> roce_lifetime;
        std::unique_ptr<RoceControlExchange> rdma_control;
#endif
        std::unique_ptr<nccl::NcclApi> api;
        std::string version, provenance;
        if(!rdma_only) {
            api=nccl::make_production_nccl_api(nccl_cfg,error);
            if(!api)throw std::runtime_error(error);
            require(api->identity(version,provenance));
            if(version!=cfg.nccl_version||provenance!=cfg.nccl_provenance)
                throw std::runtime_error("NCCL provenance mismatch");
        }
        const char* transport=std::getenv("STRATA_TP_TRANSPORT");
        const bool use_roce=transport&&std::string_view(transport)=="roce";
        if(rdma_only&&!use_roce)throw std::runtime_error("RDMA-only execution requires RoCE");
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        std::vector<uint8_t> local_blob;
        const uint64_t roce_slot_bytes = use_roce ? roce_slot_bytes_from_env() : 2u << 20;
        if(use_roce) {
            roce::Config c;c.rank=cfg.rank;c.timeout_ms=cfg.timeout_ms;c.slot_bytes=roce_slot_bytes;
            if(const char* blocks=std::getenv("STRATA_TP_ROCE_BLOCKS")) {
                if(std::string_view(blocks)!="8")throw std::runtime_error("invalid RoCE block count");
                c.blocks=8;c.threads=512;
            }
            c.hca_names=roce::discover_hcas(-1,4);c.approved_proxy_sha256=STRATA_ROCE_PROXY_SHA256;
            if(const char* names=std::getenv("STRATA_TP_ROCE_HCAS")) {
                c.hca_names.clear();std::istringstream in(names);std::string name;
                while(std::getline(in,name,','))c.hca_names.push_back(name);
            }
            if(const char* route=std::getenv("STRATA_TP_ROCE_ROUTES")) {
                std::istringstream in(route);std::string number;
                while(std::getline(in,number,',')){size_t used=0;int n=std::stoi(number,&used);if(used!=number.size())throw std::runtime_error("invalid RoCE route");c.route_hca.push_back(n);}
            }
            roce_backend=std::make_unique<roce::RoceBackend>(std::move(c));
            roce_lifetime=std::make_unique<PendingDmaExitGuard<roce::RoceBackend>>(*roce_backend);
            require(roce_backend->create());require(roce_backend->local_blob(local_blob));
            if(rdma_only) {
                api=make_roce_only_api(*roce_backend,roce_slot_bytes);
                require(api->identity(version,provenance));
            }else provenance+=";roce="+roce_backend->provenance();
        }
#else
        if(use_roce)throw std::runtime_error("fast RoCE transport is not built");
#endif
        JoinInfo join;
        join.rank = cfg.rank; join.world = 3; join.session_uuid = cfg.session_uuid;
        join.model_hash = cfg.model_hash; join.config_fingerprint = cfg.config_hash + ":nvfp4-fast-tp3-v1";
        join.config_fingerprint += ":window=" + std::to_string(kernels::kWindowMaxT);
        const char* packed=std::getenv("STRATA_TP_PACKED_PREFILL");
        const bool packed_prefill=packed&&packed[0]=='1';
        join.config_fingerprint+=packed_prefill?":packed-prefill=1":":packed-prefill=0";
        for(const char* name:{"STRATA_TP_DENSE_MMQ","STRATA_TP_HC_BF16","STRATA_TP_MTP_HNORM","STRATA_SPEC_COST",
                             "STRATA_TP_MMVQ_LAYOUT","STRATA_GR_V3","STRATA_TP_PREFILL_TILES",
                             "STRATA_TP_SHARED_SHARD","STRATA_HC_UPMIX","STRATA_TP_DENSE_F16","STRATA_TP_ASYNC_DRAFTS",
                             "STRATA_TP_PLE_IO","STRATA_TP_REDUCE_DTYPE","STRATA_TP_NATIVE_GROUPED",
                             "SPARKNET_ROCE_PROXY_CPU","STRATA_TP_CONTROL_COMPACT","STRATA_TP_ROCE_BLOCKS","STRATA_TP_COMPACT_HEAD",
                             "STRATA_TP_BALANCED_EXPERTS","STRATA_TP_DECODE_FP4","STRATA_TP_QSA_PREPARE_BATCH",
                             "STRATA_TP_QSA_APPEND_BATCH","STRATA_TP_DISTRIBUTED_DRAFTS","STRATA_TP_DRAFT_ASSETS",
                             "STRATA_TP_HC_PACKED_DOWN","STRATA_TP_ROCE_SEGMENT_PREFILL","STRATA_TP_GU_WARP32",
                             "STRATA_TP_PREFILL_QUANT_SCATTER","STRATA_TP_PREFILL_PACKED_GU",
                             "STRATA_TP_DECODE_GU_VECTOR","STRATA_TP_GU_PREFETCH","STRATA_TP_NVFP4_NEAREST",
                             "STRATA_TP_QSA_PROMPT_BATCH","STRATA_TP_QSA_PROMPT_FAST","STRATA_TP_HEAD_F16",
                             "STRATA_TP_ROUTE_ORDER","STRATA_TP_GROUP_REUSE","STRATA_TP_ROCE_ADAPTIVE",
                             "STRATA_TP_B12X_SHA256","STRATA_TP_CONTROL_RDMA","STRATA_TP_RDMA_ONLY",
                             "STRATA_TP_ROCE_RING_ORDER","STRATA_TP_ROCE_SLOT_MIB"}) {
            const char* value=std::getenv(name);const std::string text=value?value:"";
            join.config_fingerprint+=":"+std::string(name)+"="+std::to_string(text.size())+":"+text;
        }
        join.backend_identity = "nccl"; join.nccl_version_expected = version; join.library_provenance = provenance;
        join.nccl_bootstrap_id.assign(256, '0');
        if (!rdma_only && cfg.rank == 0 && !nccl::make_nccl_bootstrap_id(cfg.nccl_library, join.nccl_bootstrap_id, error)) throw std::runtime_error(error);
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        if(use_roce) {
            HybridBootstrap bundle;bundle.roce_blob=std::move(local_blob);
            if(cfg.rank==0)bundle.nccl_id_hex=join.nccl_bootstrap_id;
            require(encode_hybrid_bootstrap(bundle,join.nccl_bootstrap_id));
            join.backend_identity=rdma_only?"roce-only-v1":"nccl+roce-fast-v1";
        }
#endif
        TcpChannelConfig channel_cfg;
        channel_cfg.rank = cfg.rank; channel_cfg.world = 3;
        channel_cfg.self = {cfg.rank0_address, uint16_t(cfg.port)}; channel_cfg.rank0 = channel_cfg.self;
        channel_cfg.credential = credential.secret();
        std::unique_ptr<ControlChannel> channel;
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        if(rdma_only) {
            if(const char* bind=std::getenv("STRATA_TP_RDMA_BIND"))channel_cfg.self.address=bind;
            if(const char* root=std::getenv("STRATA_TP_RDMA_ROOT"))channel_cfg.rank0.address=root;
            channel=make_rdma_control_channel(channel_cfg,clock,error);
        }else
#endif
        channel=make_tcp_control_channel(channel_cfg,clock,error);
        if (!channel) throw std::runtime_error(error);
        SessionConfig session_cfg;
        session_cfg.rank = cfg.rank; session_cfg.session_uuid = cfg.session_uuid; session_cfg.credential = credential;
        session_cfg.approved_nccl_version = version; session_cfg.approved_library_provenance = provenance;
        session_cfg.budgets = {cfg.timeout_ms, cfg.timeout_ms, cfg.timeout_ms, cfg.timeout_ms};
        Session session(session_cfg, join, *channel, clock);
        require(session.start());
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        if(use_roce) {
            std::array<std::string,kTpWorld> fields;
            for(uint32_t peer=0;peer<kTpWorld;++peer)fields[peer]=session.roster().ranks[peer].nccl_bootstrap_id;
            std::vector<uint8_t> blobs;
            require(decode_hybrid_bootstrap_roster(fields,nccl_cfg.bootstrap_id,blobs));
            require(roce_backend->connect(blobs));
        }else
#endif
        require(session.bootstrap_id(0, nccl_cfg.bootstrap_id));
        nccl_cfg.initialize_ms = cfg.timeout_ms;
        require(api->comm_init(nccl_cfg));
        if(rdma_only) {
            if(const char* evidence=std::getenv("NCCL_DEBUG_FILE")) {
                std::ofstream out(evidence,std::ios::app);
                out<<"STRATA_RDMA_ONLY transport=roce bootstrap=rdma_cm control=rdma nccl_initialized=0\n";
            }
        }
#if defined(STRATA_BUILD_ROCE_EXPERIMENTAL)
        if(const char* control=std::getenv("STRATA_TP_CONTROL_RDMA");rdma_only||(control&&control[0]=='1')) {
            if(!use_roce)throw std::runtime_error("RDMA control requires the RoCE transport");
            rdma_control=std::make_unique<RoceControlExchange>(*roce_backend);
            require(session.enable_control_exchange([&](std::string_view local,
                std::vector<std::string>& received,Deadline deadline) {
                return (*rdma_control)(local,received,deadline);
            }));
        }
        if(use_roce&&!rdma_only) {
            const char* segmented=std::getenv("STRATA_TP_ROCE_SEGMENT_PREFILL");
            api=make_fast_roce_api(std::move(api),*roce_backend,roce_slot_bytes,segmented&&segmented[0]=='1');
        }
#endif
        FastTensorParallel tp(*api, assets.geometry().n_embd, std::move(mixer), std::move(experts));
        tp.state_provider(state_provider);
        tp.output_head(assets.head().weights(),assets.head().type(),assets.head().row_bytes(),
                       int(assets.geometry().n_vocab),int(cfg.rank));
        if(!tp.prepare_dense_output_head(stream,error))throw std::runtime_error(error);
        tp.execution_validator(std::move(validate));
        FastModel model(assets.weights(), assets.geometry(), assets.head(), assets.ple(), tp,
                         int(cfg.slots), int(cfg.context), int(cfg.prompt_chunk), stream);
        PendingDmaExitGuard lifetime(tp);
        if (!model.initialize(error)) throw std::runtime_error(error);
        if(cfg.drafts && !model.configure_mtp(cfg.mtp_path,int(cfg.drafts),int(cfg.rank),error)) {
            session.abort("MTP initialization failed");throw std::runtime_error(error);
        }
        if(const char* distributed=std::getenv("STRATA_TP_DISTRIBUTED_DRAFTS");distributed&&distributed[0]=='1') {
            Deadline ready;if(auto f=session.operation_deadline(ready))throw std::runtime_error(f.detail);
            require(session.agree_control_payload("fast-distributed-drafts-ready-v1","owner=slot%3",ready));
        }
        MultiSlotServiceCallbacks callbacks;
        callbacks.poll = [&](uint32_t wait, std::string& line, bool& eof) { return io.poll(wait, line, eof); };
        callbacks.health = [&] {
            Failure failure;
            return api->async_failure(failure) ? failure : Failure{};
        };
        callbacks.emit = [&](std::string_view type, std::string_view request, int32_t token,
                               std::string_view state, std::string_view reason) {
            auto event = type == "rejected" ? "{\"type\":\"rejected\",\"alive_ranks\":[0,1,2],\"request_id\":" +
                quote(request) + ",\"state\":\"ready\",\"reason\":" + quote(reason) + '}' :
                serve_event(type, request, token, cfg.context, state);
            if (type == "ready") event.insert(event.size() - 1, ",\"slots\":" + std::to_string(cfg.slots));
            io.emit(event);
        };
        callbacks.emit_timing = [&](std::string_view type, std::string_view request, const RequestTiming& timing) {
            auto event = serve_event(type, request, -1, cfg.context, "ready");
            event.insert(event.size() - 1, ",\"timing\":" + request_timing_json(timing)); io.emit(event);
        };
        const auto service_failure=run_fast_service(
            {cfg.rank, cfg.slots, int32_t(assets.geometry().n_vocab), cfg.context, 4096, cfg.prompt_chunk,packed_prefill},
            session, model, tp, callbacks, stream, cfg.timeout_ms);
        // A pending-DMA guard can exit before exception diagnostics run.
        if(service_failure) {
            std::fprintf(stderr,"fast TP service rank %u failed: %s\n",cfg.rank,service_failure.detail.c_str());
            std::fflush(stderr);
        }
        require(service_failure);
        require(session.close());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "final TP runtime refused: " << e.what() << '\n';
        return 2;
    }
}
}
