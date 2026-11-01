#include "strata/distributed/fast_assets.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/native_upload.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <set>
#include <cstdlib>
#include <string_view>

namespace strata::distributed {
struct FastAssets::Impl {
    void* arena = nullptr;
    core::WeightTable weights;
    core::ModelGeometry geometry;
    core::NativeDense dense;
    core::NativeHead head;
    core::NativeEmbed embedding;
    kernels::PleTable ple;
    ~Impl() {
        core::set_native_embed(nullptr);
        if (arena) cudaFree(arena);
    }
};
FastAssets::FastAssets() : impl_(std::make_unique<Impl>()) {}
FastAssets::~FastAssets() = default;
bool FastAssets::load(const std::string& pack, const std::string& dense_path,
                       const std::string& embedding_path, const std::string& ple_path,
                       std::string& error) try {
    auto& p = *impl_;
    if (p.arena) { error = "NVFP4 assets are already loaded"; return false; }
    const std::vector<std::string> dense_files{dense_path};
    GgufFile metadata(dense_path);
    if (!(error = read_geometry(metadata, p.geometry)).empty() ||
        !(error = geometry_unsupported(p.geometry)).empty()) return false;
    if (!kernels::cpu::expert_layout_load(pack, p.geometry.n_layers, p.geometry.n_expert, error)) return false;
    const auto& layout = kernels::cpu::expert_layout();
    if (!layout.native || layout.fmt.size() != size_t(p.geometry.n_layers)) {
        error = "NVFP4 native expert metadata is missing"; return false;
    }
    for (const auto& format : layout.fmt)
        if (format.gu_type != 40 || format.d_type != 40 || !format.tail_off) {
            error = "production profile requires original NVFP4 expert codes and scale tails"; return false;
        }
    std::set<std::string> skip;
    if (!core::NativeDense::served_names(dense_files, true, skip, error) ||
        !core::NativeDense::keep_unquantized_ple_key(pack, skip, error)) return false;
    skip.insert("output.weight"); skip.insert("token_embd.weight");
    uint64_t bytes = 0;
    if (!core::WeightTable::pool_bytes(pack, bytes, error, &skip)) return false;
    const auto allocation = cudaMalloc(&p.arena, bytes);
    if (allocation != cudaSuccess) { error = cudaGetErrorString(allocation); return false; }
    if (!p.weights.load(pack, p.arena, bytes, error, &skip) ||
        !p.dense.load(dense_files, p.weights, error, true) ||
        !p.head.load(dense_files, p.geometry.n_embd, p.geometry.n_vocab, error) ||
        !p.embedding.load({embedding_path}, p.geometry.n_embd, p.geometry.n_vocab, error)) return false;
    core::set_native_embed(&p.embedding);
    kernels::PleIoOptions io;
    io.mode = kernels::PleIo::Direct;
    if(const char* mode=std::getenv("STRATA_TP_PLE_IO");mode&&std::string_view(mode)=="ram") {
        io.mode=kernels::PleIo::Mmap;io.lock=true;
    }
    if(!p.ple.open(ple_path,error,io))return false;
    if(io.lock&&!p.ple.locked()){error="TP RAM PLE requires the complete original table to be locked";return false;}
    return true;
} catch (const std::exception& e) { error = e.what(); return false; }
const core::WeightTable& FastAssets::weights() const { return impl_->weights; }
const core::ModelGeometry& FastAssets::geometry() const { return impl_->geometry; }
const core::NativeHead& FastAssets::head() const { return impl_->head; }
kernels::PleTable& FastAssets::ple() { return impl_->ple; }
}
