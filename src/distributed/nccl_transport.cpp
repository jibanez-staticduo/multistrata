// src/distributed/nccl_transport.cpp - the real NCCL library behind the `NcclApi` boundary.
//
// Built only when `STRATA_DISTRIBUTED_CUDA` is set, next to a CUDA runtime and the pinned NCCL headers.  This
// is the only file in this tree that names an NCCL symbol, and it resolves them at run time with
// `dlopen`/`dlsym`, so the CPU build surface never needs the library at link time.
//
// Pinning.  No signature here is written from memory: each resolved function is typed `decltype(&ncclX)`, so a
// call that does not match the pinned header fails to compile.  The header is NCCL 2.30.7, release tag
// `v2.30.7-1` (source commit 73cf112295c33aee2b895f329f592f2a9b4b0f97), rendered with the `sed` rule in that
// tag's own `src/Makefile` (`NCCL_VERSION_CODE` = 23007).  The tarball URL, its SHA-256 and the rendered
// header's SHA-256 are in the task evidence.  The fleet's own NCCL is a rebuild of that same tag with four
// local patches (sparknet/nccl/patches/series, patch head eeacf1c6343f0499e4e7ab20ec7eecfb6f62eda6); those
// patches touch `src/transport/net_ib/p2p.cc`, `src/graph/connect.cc` and `src/enqueue.cc` and no public
// header, so this pinned header is the right compile-time boundary for both builds.
//
// What `identity()` proves, exactly.  Two things:
//   * the version the library reports itself, through `ncclGetVersion`, decoded to "major.minor.patch"; and
//   * `sha256:<hex>` of the bytes of the file that was actually resolved, read whole and hashed here.
// The version is the weaker of the two and is kept only so a refusal can be named.  It cannot tell a patched
// fleet build from an unpatched upstream build carrying the same number, which is why
// `NcclBackend::init()` also requires the hash to equal the approved hash.  A file hash pins every byte, so it
// pins the patch set: two builds that differ by one patched byte differ in this string.  It says nothing about
// the machine, the driver, the HCA firmware, or whether the build carries SM121 code; that stays for the
// hardware window.
//
// Lifecycle, and the documented behaviour this relies on (all from the pinned tag's user guide, published at
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/):
//   * Communicators are created non-blocking (`ncclConfig_t.blocking = 0`).  `usage/communicators.rst`,
//     "Fault Tolerance": "In order to abort NCCL communicators safely, NCCL requires applications to set
//     communicators as nonblocking ... After nonblocking is set, all NCCL calls (except ncclCommDestroy/Abort)
//     become nonblocking so that ncclCommAbort can be called at any point, during initialization,
//     communication or finalizing the communicator.  If NCCL communicators are set blocking, the thread can
//     possibly get stuck inside NCCL calls ... NCCL communicators might hang forever."  That is why a blocked
//     init is not a hang here: `comm_init` polls `ncclCommGetAsyncError` against a deadline, the pattern in
//     `docs/userguide/source/examples.rst` (which stops on `checkTimeout()`), and it holds a handle to abort.
//   * Enqueue is not completion.  `usage/streams.rst`: "The NCCL call returns when the operation has been
//     effectively enqueued to the given stream ... The operation status can be queried using standard CUDA
//     semantics, for example, calling cudaStreamSynchronize or using CUDA events." For ncclInProgress,
//     api/comms.rst first requires async state to become ncclSuccess before any related CUDA call. Only then
//     is the private completion event recorded; polling that event proves device completion.
//   * Late errors come from `ncclCommGetAsyncError`.  `usage/communicators.rst`: "Some communication errors,
//     and in particular network errors, are reported through the ncclCommGetAsyncError function.  Operations
//     experiencing an asynchronous error will usually not progress and never complete."  Every poll asks, and
//     the first answer sticks: `async_failure` must not clear it, or a rank would call itself healthy one
//     step after its peer died.
//   * Abort is not local.  `api/comms.rst`: "All active ranks are required to call this function in order to
//     abort the NCCL communicator successfully."  The session broadcasts ABORT over the control channel for
//     that reason; this file can only do its half.  Kept honest: the same document asks that no thread be
//     inside an NCCL call while `ncclCommAbort` runs.  The transport is one worker thread with one operation
//     in flight, so the normal path (fail, then abort from the thread that failed) satisfies it.  Aborting
//     from a watchdog is serialized with all NCCL calls by the same mutex. Abort itself is a blocking NCCL
//     API and its runtime on failed hardware has not been measured here.
//
// Bootstrap id encoding.  `ncclUniqueId` is 128 opaque bytes; `NcclConfig::bootstrap_id` is a string because it
// travels through the session roster.  The encoding is lowercase hex, 256 characters, no separators, and a
// wrong length is refused before `ncclCommInitRankConfig` is called.  It is never written to a log line.
#ifndef _GNU_SOURCE
// `dladdr`, so the provenance hash is measured on the file the loader actually mapped.
#define _GNU_SOURCE
#endif

#include "distributed/nccl_api.hpp"

#if !defined(STRATA_DISTRIBUTED_CUDA)
// Without the CUDA runtime, the contract's `cudaStream_t` is a stand-in pointer type and no real NCCL stream
// can reach it.  Building this file into that surface would give a transport that cannot enqueue.
#error "src/distributed/nccl_transport.cpp needs -DSTRATA_DISTRIBUTED_CUDA and a CUDA toolkit"
#endif

#include <cuda_runtime.h>
#include <nccl.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace strata::distributed::nccl {

// `detail`, not an anonymous namespace: the probe and the runner call these to check the encoding and the hash
// against `sha256sum` before a rank ever trusts a library with them.
namespace detail {

// ---------------------------------------------------------------- SHA-256, for the library provenance.
//
// Small and self-contained on purpose: the provenance gate runs before anything else in the process has been
// trusted, and it should not need a crypto library the runtime image may not carry.  FIPS 180-4 as published.
namespace sha256_ns {

constexpr std::array<uint32_t, 64> kK = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }

class Sha256 {
 public:
    Sha256() {
        // The initial hash values from FIPS 180-4.
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    }

    void update(const uint8_t* data, size_t len) {
        total_bits_ += static_cast<uint64_t>(len) * 8u;
        for (size_t i = 0; i < len; ++i) {
            buf_[buf_len_++] = data[i];
            if (buf_len_ == buf_.size()) {
                block(buf_.data());
                buf_len_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        // Padding: a set bit, zeros up to 56 mod 64, then the length in bits, big-endian.
        const uint64_t bits = total_bits_;
        const uint8_t one = 0x80u;
        const uint8_t zero = 0x00u;
        update(&one, 1);
        while (buf_len_ != 56) update(&zero, 1);
        std::array<uint8_t, 8> tail{};
        for (int i = 0; i < 8; ++i) tail[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        update(tail.data(), tail.size());

        std::array<uint8_t, 32> out{};
        for (size_t w = 0; w < state_.size(); ++w) {
            for (int b = 0; b < 4; ++b) {
                out[w * 4 + b] = static_cast<uint8_t>(state_[w] >> (24 - 8 * b));
            }
        }
        return out;
    }

 private:
    void block(const uint8_t* p) {
        std::array<uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(p[i * 4]) << 24) | (static_cast<uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(p[i * 4 + 2]) << 8) | static_cast<uint32_t>(p[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + S1 + ch + kK[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = S0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_{};
    std::array<uint8_t, 64> buf_{};
    size_t buf_len_ = 0;
    uint64_t total_bits_ = 0;
};

std::string hex(const uint8_t* data, size_t len) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(kDigits[data[i] >> 4]);
        out.push_back(kDigits[data[i] & 0x0fu]);
    }
    return out;
}

}  // namespace sha256_ns

// Hash a file whole.  False with a reason when it cannot be opened or read: a library we cannot hash is a
// library we cannot approve, so nobody may read false as "good enough".
inline bool sha256_of_file(const std::string& path, std::string& hex_out, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "cannot open '" + path + "' to hash it";
        return false;
    }
    sha256_ns::Sha256 hash;
    std::vector<char> chunk(1u << 20);
    for (;;) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = in.gcount();
        if (got <= 0) break;
        hash.update(reinterpret_cast<const uint8_t*>(chunk.data()), static_cast<size_t>(got));
        if (got < static_cast<std::streamsize>(chunk.size())) break;
    }
    if (!in && !in.eof()) {
        err = "reading '" + path + "' failed while hashing it";
        return false;
    }
    const auto digest = hash.finish();
    hex_out = sha256_ns::hex(digest.data(), digest.size());
    return true;
}

// A dladdr pathname can be replaced after dlopen. Hash an open descriptor only after its inode agrees
// with the mapping containing the resolved symbol, and refuse a file modified while it is being read.
bool mapped_library_hash(const std::string& path, const void* symbol, std::string& hash, std::string& err) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { err = "cannot open mapped NCCL library for provenance"; return false; }
    struct stat before{}, after{};
    bool matches = false;
    if (::fstat(fd, &before) == 0) {
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            unsigned long start = 0, end = 0, offset = 0, inode = 0;
            unsigned dev_major = 0, dev_minor = 0;
            char perms[5]{};
            if (std::sscanf(line.c_str(), "%lx-%lx %4s %lx %x:%x %lu", &start, &end, perms,
                            &offset, &dev_major, &dev_minor, &inode) != 7) continue;
            const auto address = reinterpret_cast<uintptr_t>(symbol);
            if (address >= start && address < end) {
                matches = inode == static_cast<unsigned long>(before.st_ino) &&
                          dev_major == major(before.st_dev) && dev_minor == minor(before.st_dev);
                break;
            }
        }
    }
    bool ok = matches && sha256_of_file("/proc/self/fd/" + std::to_string(fd), hash, err) &&
              ::fstat(fd, &after) == 0 && before.st_size == after.st_size &&
              before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
              before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
              before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
              before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
    ::close(fd);
    if (!ok) err = "NCCL mapped inode differs from the provenance file, or the file changed while hashing";
    return ok;
}

inline std::string hex_of_id(const ncclUniqueId& id) {
    return sha256_ns::hex(reinterpret_cast<const uint8_t*>(id.internal), NCCL_UNIQUE_ID_BYTES);
}

inline bool hex_to_id(const std::string& hex_in, ncclUniqueId& id, std::string& err) {
    if (hex_in.size() != NCCL_UNIQUE_ID_BYTES * 2) {
        err = "the bootstrap id is " + std::to_string(hex_in.size()) + " characters; a 128-byte id is " +
              std::to_string(NCCL_UNIQUE_ID_BYTES * 2) + " of hex";
        return false;
    }
    auto digit = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        const int hi = digit(hex_in[i * 2]);
        const int lo = digit(hex_in[i * 2 + 1]);
        if (hi < 0 || lo < 0) {
            err = "the bootstrap id is not lowercase hex";
            return false;
        }
        id.internal[i] = static_cast<char>((hi << 4) | lo);
    }
    return true;
}

// "major.minor.patch" from an `NCCL_VERSION_CODE`, matching the `NCCL_VERSION(X,Y,Z)` macro of the pinned
// header, whose base changed at 2.9.  Anything that is not the approved string is refused by the gate, so this
// only has to be honest about what the number means.
inline std::string version_string(int code) {
    if (code < 20900) {
        return std::to_string(code / 1000) + "." + std::to_string((code / 100) % 10) + "." +
               std::to_string(code % 100);
    }
    return std::to_string(code / 10000) + "." + std::to_string((code / 100) % 100) + "." +
           std::to_string(code % 100);
}

// The symbols this backend needs.  Each is typed `decltype(&ncclX)`, so the call site is checked against the
// pinned header rather than against what someone remembered about the API.
struct NcclSymbols {
    decltype(&ncclGetVersion) get_version = nullptr;
    decltype(&ncclGetUniqueId) get_unique_id = nullptr;
    decltype(&ncclCommInitRankConfig) comm_init_rank_config = nullptr;
    decltype(&ncclCommCount) comm_count = nullptr;
    decltype(&ncclCommGetAsyncError) comm_get_async_error = nullptr;
    decltype(&ncclGetLastError) get_last_error = nullptr;
    decltype(&ncclGetErrorString) get_error_string = nullptr;
    decltype(&ncclCommAbort) comm_abort = nullptr;
    decltype(&ncclAllReduce) all_reduce = nullptr;
    decltype(&ncclAllGather) all_gather = nullptr;
};

struct CudaCalls {
    decltype(&cudaEventCreateWithFlags) event_create = &cudaEventCreateWithFlags;
    decltype(&cudaEventRecord) event_record = &cudaEventRecord;
    decltype(&cudaEventQuery) event_query = &cudaEventQuery;
    decltype(&cudaEventDestroy) event_destroy = &cudaEventDestroy;
    decltype(&cudaStreamIsCapturing) stream_is_capturing = &cudaStreamIsCapturing;
};

constexpr const char* kSymbolNames[10] = {"ncclGetVersion",
                                         "ncclGetUniqueId",
                                         "ncclCommInitRankConfig",
                                         "ncclCommCount",
                                         "ncclCommGetAsyncError",
                                         "ncclGetLastError",
                                         "ncclGetErrorString",
                                         "ncclCommAbort",
                                         "ncclAllReduce",
                                         "ncclAllGather"};

// `dlsym` hands back `void*`; casting it to a function pointer is what every POSIX loader means by it.
template <typename Fn>
inline Fn resolve_one(void* dl, const char* name) {
    return reinterpret_cast<Fn>(dlsym(dl, name));
}

inline bool resolve_symbols(void* dl, NcclSymbols& out, std::vector<std::string>& missing) {
    bool ok = true;
    auto pick = [&](const char* name, auto NcclSymbols::*member) {
        using Fn = std::remove_reference_t<decltype(std::declval<NcclSymbols>().*member)>;
        Fn fn = resolve_one<Fn>(dl, name);
        out.*member = fn;
        if (fn == nullptr) {
            missing.push_back(name);
            ok = false;
        }
    };
    pick(kSymbolNames[0], &NcclSymbols::get_version);
    pick(kSymbolNames[1], &NcclSymbols::get_unique_id);
    pick(kSymbolNames[2], &NcclSymbols::comm_init_rank_config);
    pick(kSymbolNames[3], &NcclSymbols::comm_count);
    pick(kSymbolNames[4], &NcclSymbols::comm_get_async_error);
    pick(kSymbolNames[5], &NcclSymbols::get_last_error);
    pick(kSymbolNames[6], &NcclSymbols::get_error_string);
    pick(kSymbolNames[7], &NcclSymbols::comm_abort);
    pick(kSymbolNames[8], &NcclSymbols::all_reduce);
    pick(kSymbolNames[9], &NcclSymbols::all_gather);
    return ok;
}

}  // namespace detail

namespace {

using detail::NcclSymbols;

// The production `NcclApi`: one communicator, one completion event, one sticky async failure.
class ProductionNcclApi final : public NcclApi {
 public:
    ProductionNcclApi(NcclConfig config, void* dl, NcclSymbols symbols, std::string resolved_path,
                      std::string provenance, detail::CudaCalls cuda = {})
        : config_(std::move(config)),
          dl_(dl),
          symbols_(symbols),
          resolved_path_(std::move(resolved_path)), cuda_(cuda), provenance_(std::move(provenance)) {}

    ProductionNcclApi(const ProductionNcclApi&) = delete;
    ProductionNcclApi& operator=(const ProductionNcclApi&) = delete;
    // The library stays loaded for the life of the process.  `dlclose` while a communicator from it exists is
    // not something this experiment needs, and an abort in flight is a bad moment to unmap code.
    ~ProductionNcclApi() override { comm_abort(); }

    // `const` because the gate only reads: the version comes from the library, the hash from its file.  The
    // hash is expensive (the fleet library is about 300 MB), so it is computed once and kept.
    Failure identity(std::string& version, std::string& provenance) const override {
        int code = 0;
        const ncclResult_t rc = symbols_.get_version(&code);
        if (rc != ncclSuccess) {
            return Failure::fail(FailureCode::library_mismatch,
                                 "ncclGetVersion returned " + describe_static(rc, nullptr));
        }
        version = detail::version_string(code);
        std::lock_guard<std::mutex> lock(mutex_);
        provenance = provenance_;
        return ok_failure();
    }

    // Where the hashed bytes came from, for the runner's report line.  Not part of the boundary.
    const std::string& resolved_library_path() const { return resolved_path_; }

    bool bootstrap_id(std::string& out, std::string& err) {
        ncclUniqueId id;
        const ncclResult_t rc = symbols_.get_unique_id(&id);
        if (rc != ncclSuccess) { err = "ncclGetUniqueId: " + describe_static(rc, nullptr); return false; }
        out = detail::hex_of_id(id);
        return true;
    }

    // Form the communicator without ever blocking past the caller's patience.
    Failure comm_init(const NcclConfig& config) override {
        // `unique_lock`, not `lock_guard`: this loop must let a watchdog's `abort()` in while it waits.
        std::unique_lock<std::mutex> lock(mutex_);
        if (aborted_) {
            return Failure::fail(FailureCode::aborted, "this backend was already abandoned");
        }
        if (comm_ != nullptr) {
            return Failure::fail(FailureCode::invalid_config, "the communicator was already formed");
        }
        if (config.world == 0 || config.rank >= config.world) {
            return Failure::fail(FailureCode::invalid_config,
                                 "rank " + std::to_string(config.rank) + " of " +
                                     std::to_string(config.world));
        }
        if (config.initialize_ms == 0 || config.initialize_ms > 3600000u) {
            return Failure::fail(FailureCode::invalid_config, "NCCL initialization needs a bounded budget");
        }
        ncclUniqueId id;
        {
            std::string err;
            if (!detail::hex_to_id(config.bootstrap_id, id, err)) {
                return Failure::fail(FailureCode::invalid_config, err);
            }
        }
        // Non-blocking, so an init that cannot finish can still be aborted (see the Fault Tolerance quote at
        // the top of this file).  Everything else stays at NCCL's default: rank-local network settings belong
        // to the environment the runner writes, not to a hard-coded comm config.
        ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
        cfg.blocking = 0;
        cfg.commName = "strata-tp3";

        const ncclResult_t rc = symbols_.comm_init_rank_config(&comm_, static_cast<int>(config.world), id,
                                                               static_cast<int>(config.rank), &cfg);
        if (rc != ncclSuccess && rc != ncclInProgress) {
            return Failure::fail(FailureCode::comm_failed,
                                 "ncclCommInitRankConfig returned " + describe_static(rc, nullptr));
        }
        if (comm_ == nullptr) {
            return Failure::fail(FailureCode::comm_failed, "initialisation gave no communicator handle");
        }
        const auto init_deadline = std::chrono::steady_clock::now() +
                                   std::chrono::milliseconds(config.initialize_ms);
        // Publish the handle before unlocking, and initialize exactly once. Abort and every NCCL call are
        // serialized by the same mutex; the poll loop never retains a stale local handle across an unlock.
        for (;;) {
            if (aborted_) return Failure::fail(FailureCode::aborted, "initialisation was abandoned");
            ncclResult_t state = ncclInProgress;
            const ncclResult_t pr = symbols_.comm_get_async_error(comm_, &state);
            if (pr != ncclSuccess && pr != ncclInProgress) {
                return Failure::fail(FailureCode::comm_failed,
                                     "initialisation reported " + describe_static(pr, nullptr));
            }
            if (state != ncclSuccess && state != ncclInProgress) {
                return Failure::fail(FailureCode::comm_failed,
                                     "initialisation reported " + describe_static(state, nullptr));
            }
            if (pr == ncclSuccess && state == ncclSuccess) break;
            if (std::chrono::steady_clock::now() >= init_deadline) {
                return Failure::fail(FailureCode::deadline_expired, "the communicator init deadline passed");
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(kInitPollMs));
            lock.lock();
        }
        const ncclComm_t comm = comm_;

        // The library's own idea of the world size.  A communicator with the wrong number of ranks means two
        // ranks joined two different sessions, and the sooner that is said, the fewer bytes go the wrong way.
        int ranks = 0;
        if (const ncclResult_t rc = symbols_.comm_count(comm, &ranks); rc != ncclSuccess) {
            symbols_.comm_abort(comm);
            comm_ = nullptr;
            return Failure::fail(FailureCode::comm_failed,
                                 "ncclCommCount returned " + describe_static(rc, nullptr));
        }
        if (ranks != static_cast<int>(config.world)) {
            symbols_.comm_abort(comm);
            comm_ = nullptr;
            return Failure::fail(FailureCode::comm_failed,
                                 "the communicator holds " + std::to_string(ranks) + " ranks, the roster says " +
                                     std::to_string(config.world));
        }

        if (const cudaError_t ce = cuda_.event_create(&event_, cudaEventDisableTiming);
            ce != cudaSuccess) {
            symbols_.comm_abort(comm);
            comm_ = nullptr;
            event_ = nullptr;
            return Failure::fail(FailureCode::comm_failed,
                                 std::string("cudaEventCreateWithFlags: ") + cudaGetErrorString(ce));
        }
        comm_ = comm;
        config_ = config;
        return ok_failure();
    }

    Failure enqueue_all_reduce_sum(const void* send, void* recv, uint64_t elements, ScalarType type,
                                   cudaStream_t stream) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ncclDataType_t dtype = ncclNumTypes;
        if (auto f = ready_locked(stream, elements, type, dtype)) return f;
        if (graph_state_ != GraphState::inactive) {
            return enqueue_graph_locked(lock, send, recv, elements, type, stream, dtype, false);
        }
        const ncclResult_t rc = symbols_.all_reduce(send, recv, static_cast<size_t>(elements), dtype,
                                                   ncclSum, comm_, stream);
        if (!accepted(rc)) {
            return Failure::fail(FailureCode::enqueue_failed,
                                 "ncclAllReduce returned " + describe(rc));
        }
        return begin_completion_locked(stream, rc);
    }

    Failure enqueue_all_gather(const void* send, void* recv, uint64_t slot_elements, ScalarType type,
                               cudaStream_t stream) override {
        std::unique_lock<std::mutex> lock(mutex_);
        ncclDataType_t dtype = ncclNumTypes;
        if (auto f = ready_locked(stream, slot_elements, type, dtype)) return f;
        if (graph_state_ != GraphState::inactive) {
            return enqueue_graph_locked(lock, send, recv, slot_elements, type, stream, dtype, true);
        }
        // NCCL's `sendcount` is one rank's slot; the receive buffer must hold `nranks * sendcount`, which is
        // the wire size `GuardedTransport` already checked.  The padding travels and is read by nobody.
        const ncclResult_t rc = symbols_.all_gather(send, recv, static_cast<size_t>(slot_elements), dtype,
                                                    comm_, stream);
        if (!accepted(rc)) {
            return Failure::fail(FailureCode::enqueue_failed,
                                 "ncclAllGather returned " + describe(rc));
        }
        return begin_completion_locked(stream, rc);
    }

    Failure prepare_graph_session(cudaStream_t stream, uint64_t expected_collectives) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto f = graph_ready_locked()) return f;
        if (stream == nullptr) return Failure::fail(FailureCode::bad_pointer, "a graph session needs a CUDA stream");
        if (expected_collectives == 0) {
            return Failure::fail(FailureCode::bad_extent, "a graph session must contain at least one collective");
        }
        if (graph_state_ != GraphState::inactive || in_flight_) {
            return Failure::fail(FailureCode::in_flight, "NCCL is not idle for graph preparation");
        }
        graph_state_ = GraphState::prepared;
        graph_stream_ = stream;
        graph_expected_ = expected_collectives;
        graph_enqueued_ = 0;
        return ok_failure();
    }

    Failure check_graph_capture() override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (auto f = graph_active_locked()) return f;
        if (graph_state_ != GraphState::prepared && graph_state_ != GraphState::capturing) {
            return Failure::fail(FailureCode::not_ready, "NCCL graph capture is not active");
        }
        if (auto f = await_nccl_success_locked(lock, "graph capture check")) return f;
        if (graph_enqueued_ > graph_expected_) {
            return graph_fail_locked(FailureCode::bad_extent, "graph capture enqueued too many NCCL collectives");
        }
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t ce = cuda_.stream_is_capturing(graph_stream_, &status);
        if (ce != cudaSuccess) {
            return graph_fail_locked(FailureCode::async_error,
                                     std::string("cudaStreamIsCapturing: ") + cudaGetErrorString(ce));
        }
        if (status != cudaStreamCaptureStatusActive) {
            return graph_fail_locked(FailureCode::bad_operation, "the prepared NCCL stream is not being captured");
        }
        graph_state_ = GraphState::capturing;
        return ok_failure();
    }

    Failure finish_graph_capture() override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (auto f = graph_active_locked()) return f;
        if (graph_state_ != GraphState::capturing) {
            return Failure::fail(FailureCode::not_ready, "NCCL graph capture is not active");
        }
        if (auto f = await_nccl_success_locked(lock, "graph capture finish")) return f;
        if (graph_enqueued_ != graph_expected_) {
            return graph_fail_locked(FailureCode::bad_extent,
                                     "graph capture enqueued " + std::to_string(graph_enqueued_) +
                                         " collectives, expected " + std::to_string(graph_expected_));
        }
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t ce = cuda_.stream_is_capturing(graph_stream_, &status);
        if (ce != cudaSuccess) {
            return graph_fail_locked(FailureCode::async_error,
                                     std::string("cudaStreamIsCapturing: ") + cudaGetErrorString(ce));
        }
        if (status != cudaStreamCaptureStatusNone) {
            return graph_fail_locked(FailureCode::bad_operation, "NCCL stream capture has not ended");
        }
        graph_state_ = GraphState::captured;
        return ok_failure();
    }

    Failure begin_captured_replay() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto f = graph_active_locked()) return f;
        if (graph_state_ != GraphState::captured || graph_replay_pending_ || in_flight_) {
            return Failure::fail(FailureCode::bad_operation,
                                 "NCCL graph is not ready for another replay");
        }
        graph_replay_pending_ = true;
        return ok_failure();
    }

    Failure wait_captured_replay(Deadline deadline) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (auto f = graph_active_locked()) return f;
        if (graph_state_ != GraphState::captured || !graph_replay_pending_) {
            return Failure::fail(FailureCode::not_ready, "no captured NCCL replay is pending");
        }
        for (;;) {
            ncclResult_t state = ncclInProgress;
            if (auto f = async_error_unlocked(state)) return graph_fail_locked(f.code, f.detail);
            if (state == ncclSuccess) break;
            if (std::chrono::steady_clock::now() >= deadline.end) {
                return graph_timeout_locked("NCCL did not complete before the graph replay deadline");
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(kInitPollMs));
            lock.lock();
            if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        }
        if (auto f = record_after_locked(graph_stream_, "captured replay")) {
            return graph_fail_locked(f.code, f.detail);
        }
        for (;;) {
            ncclResult_t state = ncclInProgress;
            if (auto f = async_error_unlocked(state)) return graph_fail_locked(f.code, f.detail);
            if (state == ncclInProgress) {
                if (std::chrono::steady_clock::now() >= deadline.end) {
                    return graph_timeout_locked("NCCL became active while completing a graph replay");
                }
            } else {
                const cudaError_t q = cuda_.event_query(event_);
                if (q == cudaSuccess) {
                    graph_replay_pending_ = false;
                    return ok_failure();
                }
                if (q != cudaErrorNotReady) {
                    return graph_fail_locked(FailureCode::async_error,
                                             std::string("cudaEventQuery: ") + cudaGetErrorString(q));
                }
                if (std::chrono::steady_clock::now() >= deadline.end) {
                    return graph_timeout_locked("the device had not finished the graph replay before its deadline");
                }
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(kInitPollMs));
            lock.lock();
            if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        }
    }

    Failure end_graph_session() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto f = graph_active_locked()) return f;
        if ((graph_state_ != GraphState::captured && graph_state_ != GraphState::prepared) ||
            graph_replay_pending_) {
            return Failure::fail(FailureCode::in_flight, "NCCL graph session has pending work");
        }
        graph_state_ = GraphState::inactive;
        graph_stream_ = nullptr;
        graph_expected_ = 0;
        graph_enqueued_ = 0;
        return ok_failure();
    }

    bool requires_process_exit() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return requires_process_exit_;
    }

    Failure poll_completion(Deadline deadline) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        if (graph_state_ != GraphState::inactive) {
            return Failure::fail(FailureCode::bad_operation,
                                 "eager completion polling is unavailable during an NCCL graph session");
        }
        if (comm_ == nullptr) return Failure::fail(FailureCode::not_ready, "no communicator");
        if (!in_flight_ || event_ == nullptr) {
            return Failure::fail(FailureCode::not_ready, "nothing was enqueued to poll");
        }
        // Ask the library first: a network error shows up here and not in the event, and an operation that hit
        // one "will usually not progress and never complete".
        ncclResult_t state = ncclInProgress;
        if (auto f = async_error_unlocked(state)) return f;
        if (state == ncclInProgress) {
            return Failure::fail(FailureCode::deadline_expired, "NCCL is still enqueueing the collective");
        }
        if (!event_recorded_) {
            if (auto f = record_after_locked(pending_stream_, "collective")) return f;
            event_recorded_ = true;
        }
        const cudaError_t q = cuda_.event_query(event_);
        if (q == cudaSuccess) {
            in_flight_ = false;
            return ok_failure();
        }
        if (q == cudaErrorNotReady) {
            // The boundary asks for `deadline_expired` whenever the work is not done; saying which half of the
            // answer it is costs nothing and reads better in a log than the same sentence twice.
            const bool passed = std::chrono::steady_clock::now() >= deadline.end;
            return Failure::fail(FailureCode::deadline_expired,
                                 passed ? "the device had not finished when the deadline passed"
                                        : "the device has not finished yet");
        }
        return Failure::fail(FailureCode::async_error,
                             std::string("cudaEventQuery: ") + cudaGetErrorString(q));
    }

    bool async_failure(Failure& out) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sticky_) {
            out = sticky_failure_;
            return true;
        }
        if (comm_ == nullptr) return false;
        ncclResult_t state = ncclInProgress;
        if (auto f = async_error_unlocked(state)) {
            out = f;
            return true;
        }
        return false;
    }

    void comm_abort() noexcept override {
        cudaEvent_t event = nullptr;
        ncclComm_t comm = nullptr;
        bool preserve_event = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            preserve_event = graph_replay_pending_;
            if (preserve_event) requires_process_exit_ = true;
            else event = event_;
            comm = comm_;
            event_ = nullptr;
            comm_ = nullptr;
            aborted_ = true;
        }
        // Taking the handle out under the lock makes a second call do nothing, which is what the boundary asks
        // for.  `ncclCommAbort` needs no stream and no completed work; it is what stops an in-flight
        // collective.
        if (comm != nullptr) {
            (void)symbols_.comm_abort(comm);
        }
        if (event != nullptr) {
            (void)cuda_.event_destroy(event);
        }
    }

 private:
    enum class GraphState { inactive, prepared, capturing, captured, failed };

    Failure graph_ready_locked() const {
        if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        if (comm_ == nullptr) return Failure::fail(FailureCode::not_ready, "no communicator");
        if (sticky_) return sticky_failure_;
        return ok_failure();
    }

    Failure graph_active_locked() const {
        if (graph_state_ == GraphState::failed) {
            return sticky_ ? sticky_failure_ : Failure::fail(FailureCode::aborted, "NCCL graph session failed");
        }
        return graph_ready_locked();
    }

    Failure graph_fail_locked(FailureCode code, std::string_view detail) {
        if (code == FailureCode::none) code = FailureCode::async_error;
        if (graph_replay_pending_) requires_process_exit_ = true;
        graph_state_ = GraphState::failed;
        sticky_ = true;
        sticky_failure_ = Failure::fail(code, detail);
        return sticky_failure_;
    }

    Failure graph_timeout_locked(std::string_view detail) {
        requires_process_exit_ = true;
        graph_state_ = GraphState::failed;
        sticky_ = true;
        sticky_failure_ = Failure::fail(FailureCode::deadline_expired, detail);
        return sticky_failure_;
    }

    Failure enqueue_graph_locked(std::unique_lock<std::mutex>& lock, const void* send, void* recv,
                                 uint64_t elements, ScalarType type, cudaStream_t stream,
                                 ncclDataType_t dtype, bool gather) {
        if (graph_state_ != GraphState::capturing) {
            return Failure::fail(FailureCode::bad_operation,
                                 "NCCL graph collectives require an active prepared capture");
        }
        if (stream != graph_stream_) {
            return graph_fail_locked(FailureCode::bad_pointer, "a captured NCCL collective used another stream");
        }
        if (auto f = await_nccl_success_locked(lock, "captured collective pre-enqueue")) return f;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t ce = cuda_.stream_is_capturing(stream, &status);
        if (ce != cudaSuccess) {
            return graph_fail_locked(FailureCode::async_error,
                                     std::string("cudaStreamIsCapturing: ") + cudaGetErrorString(ce));
        }
        if (status != cudaStreamCaptureStatusActive) {
            return graph_fail_locked(FailureCode::bad_operation, "prepared NCCL stream is not capturing");
        }
        if (graph_enqueued_ >= graph_expected_) {
            return graph_fail_locked(FailureCode::bad_extent, "graph capture contains too many NCCL collectives");
        }
        const ncclResult_t rc =
            gather ? symbols_.all_gather(send, recv, static_cast<size_t>(elements), dtype, comm_, stream)
                   : symbols_.all_reduce(send, recv, static_cast<size_t>(elements), dtype, ncclSum, comm_, stream);
        if (!accepted(rc)) {
            return graph_fail_locked(FailureCode::enqueue_failed,
                                     std::string(gather ? "ncclAllGather returned " : "ncclAllReduce returned ") +
                                         describe(rc));
        }
        if (rc == ncclInProgress) {
            if (auto f = await_nccl_success_locked(lock, "captured collective enqueue")) return f;
        }
        graph_state_ = GraphState::capturing;
        ++graph_enqueued_;
        return ok_failure();
    }

    Failure await_nccl_success_locked(std::unique_lock<std::mutex>& lock, const char* phase) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(config_.initialize_ms);
        for (;;) {
            ncclResult_t state = ncclInProgress;
            if (auto f = async_error_unlocked(state)) return graph_fail_locked(f.code, f.detail);
            if (state == ncclSuccess) return ok_failure();
            if (std::chrono::steady_clock::now() >= deadline) {
                return graph_fail_locked(FailureCode::deadline_expired,
                                         std::string("NCCL did not reach success during ") + phase);
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(kInitPollMs));
            lock.lock();
            if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        }
    }

    // Everything an enqueue needs before it touches the library.  The caller holds `mutex_`: checking the
    // handle and using it are one step, not two, or an abort between them would be a call on a dead comm.
    Failure ready_locked(cudaStream_t stream, uint64_t elements, ScalarType type,
                         ncclDataType_t& dtype) const {
        if (elements == 0) {
            return Failure::fail(FailureCode::bad_extent, "an empty collective was enqueued");
        }
        // A real stream is required: NCCL puts the work on it, and the stand-in type of the CPU surface can
        // never be one.
        if (stream == nullptr) {
            return Failure::fail(FailureCode::bad_pointer, "a collective needs a CUDA stream");
        }
        switch (type) {
            case ScalarType::fp32: dtype = ncclFloat32; break;   // 4 bytes, NCCL Sum is defined for it
            case ScalarType::bf16: dtype = ncclBfloat16; break;  // 2 bytes, NCCL Sum is defined for it
            default: return Failure::fail(FailureCode::bad_dtype, "NCCL has no name for this element type");
        }
        if (aborted_) return Failure::fail(FailureCode::aborted, "the communicator was abandoned");
        if (comm_ == nullptr) return Failure::fail(FailureCode::not_ready, "no communicator");
        if (graph_state_ == GraphState::failed) {
            return sticky_ ? sticky_failure_ : Failure::fail(FailureCode::aborted, "NCCL graph session failed");
        }
        if (in_flight_) return Failure::fail(FailureCode::in_flight, "a collective is already in flight");
        return ok_failure();
    }

    // `ncclInProgress` from a non-blocking communicator means "accepted, not done".  It is not a failure, and
    // reading it as one would fail every collective on this build surface.
    static bool accepted(ncclResult_t rc) { return rc == ncclSuccess || rc == ncclInProgress; }

    Failure begin_completion_locked(cudaStream_t stream, ncclResult_t rc) {
        in_flight_ = true;
        pending_stream_ = stream;
        event_recorded_ = rc == ncclSuccess;
        return event_recorded_ ? record_after_locked(stream, "collective") : ok_failure();
    }

    Failure record_after_locked(cudaStream_t stream, const char* what) {
        // Called with `mutex_` held, straight after the enqueue it is measuring.
        if (aborted_ || comm_ == nullptr) {
            return Failure::fail(FailureCode::aborted, "the communicator went away mid-enqueue");
        }
        // Recorded after the collective, on the caller's stream: the event is complete exactly when the work
        // queued before it on that stream is complete.
        const cudaError_t ce = cuda_.event_record(event_, stream);
        if (ce != cudaSuccess) {
            return Failure::fail(FailureCode::enqueue_failed,
                                 std::string(what) + ": cudaEventRecord: " + cudaGetErrorString(ce));
        }
        return ok_failure();
    }

    Failure async_error_unlocked(ncclResult_t& state) const {
        if (sticky_) return sticky_failure_;
        state = ncclInProgress;
        const ncclResult_t rc = symbols_.comm_get_async_error(comm_, &state);
        if (rc != ncclSuccess && rc != ncclInProgress) {
            sticky_ = true;
            sticky_failure_ = Failure::fail(FailureCode::async_error,
                                            "ncclCommGetAsyncError returned " + describe_static(rc, nullptr));
            return sticky_failure_;
        }
        if (state != ncclSuccess && state != ncclInProgress) {
            sticky_ = true;
            sticky_failure_ = Failure::fail(FailureCode::async_error,
                                            "the communicator reports " + describe_static(state, nullptr));
            return sticky_failure_;
        }
        if (rc == ncclInProgress) state = ncclInProgress;
        return ok_failure();
    }

    // A named NCCL error, with the communicator's last message when there is one.  Reading `ncclGetLastError`
    // here costs nothing: it does not clear what `ncclCommGetAsyncError` reported.
    std::string describe(ncclResult_t rc) const { return describe_static(rc, comm_); }

    std::string describe_static(ncclResult_t rc, ncclComm_t comm) const {
        std::string out = "unnamed result";
        if (symbols_.get_error_string != nullptr) {
            if (const char* text = symbols_.get_error_string(rc); text != nullptr) out = text;
        }
        if (comm != nullptr && symbols_.get_last_error != nullptr) {
            const char* last = symbols_.get_last_error(comm);
            if (last != nullptr && *last != '\0') out += std::string("; last error: ") + last;
        }
        return out + " (ncclResult " + std::to_string(static_cast<int>(rc)) + ")";
    }

    NcclConfig config_;
    void* dl_ = nullptr;
    NcclSymbols symbols_{};
    std::string resolved_path_;
    detail::CudaCalls cuda_;

    mutable std::mutex mutex_;
    ncclComm_t comm_ = nullptr;
    cudaEvent_t event_ = nullptr;
    cudaStream_t pending_stream_ = nullptr;
    cudaStream_t graph_stream_ = nullptr;
    uint64_t graph_expected_ = 0;
    uint64_t graph_enqueued_ = 0;
    GraphState graph_state_ = GraphState::inactive;
    bool graph_replay_pending_ = false;
    bool requires_process_exit_ = false;
    bool in_flight_ = false;
    bool event_recorded_ = false;
    bool aborted_ = false;
    mutable bool sticky_ = false;
    mutable Failure sticky_failure_{};
    mutable std::string provenance_;
    static constexpr uint32_t kInitPollMs = 5u;
};

}  // namespace

// Load the library this rank was told to load, and only then let anyone ask what it is.
//
// An empty `config.library_path` means the soname `libnccl.so.2` and the loader's own search order.  A full
// path is preferred in production: it is how a rank keeps its distance from whatever the image carries.
// `RTLD_LOCAL` keeps NCCL's symbols out of the global namespace, so a second NCCL in the same process (a
// Python worker that already loaded one) cannot be quietly replaced by ours.
std::unique_ptr<NcclApi> make_production_nccl_api(const NcclConfig& config, std::string& err) {
    err.clear();
    const std::string requested =
        config.library_path.empty() ? std::string("libnccl.so.2") : config.library_path;
    void* dl = dlopen(requested.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (dl == nullptr) {
        const char* msg = dlerror();
        err = "cannot load '" + requested + "': " + (msg != nullptr ? msg : "no reason from the loader");
        return nullptr;
    }

    NcclSymbols symbols;
    std::vector<std::string> missing;
    if (!detail::resolve_symbols(dl, symbols, missing)) {
        std::string list;
        for (size_t i = 0; i < missing.size(); ++i) {
            list += (i == 0 ? "" : ", ") + missing[i];
        }
        dlclose(dl);
        err = "the library at '" + requested + "' is missing " + list +
              "; this build needs every symbol of the pinned NCCL header";
        return nullptr;
    }
    // Where the file really is.  A soname can resolve anywhere, and the provenance hash must be measured on
    // the bytes that were mapped, not on the string the caller happened to pass.
    std::string resolved = requested;
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(symbols.get_version), &info) != 0 && info.dli_fname != nullptr) {
        resolved = info.dli_fname;
    }

    // A library that cannot report its own version is not a library this session can approve.
    int code = 0;
    if (const ncclResult_t rc = symbols.get_version(&code); rc != ncclSuccess) {
        dlclose(dl);
        err = "ncclGetVersion failed in '" + resolved + "' (ncclResult " +
              std::to_string(static_cast<int>(rc)) + ")";
        return nullptr;
    }

    std::string hash;
    if (!detail::mapped_library_hash(resolved, reinterpret_cast<void*>(symbols.get_version), hash, err)) {
        dlclose(dl);
        return nullptr;
    }
    return std::make_unique<ProductionNcclApi>(config, dl, symbols, std::move(resolved), "sha256:" + hash);
}

bool make_nccl_bootstrap_id(const std::string& library_path, std::string& out, std::string& err) {
    out.clear();
    NcclConfig config;
    config.library_path = library_path;
    auto api = make_production_nccl_api(config, err);
    if (!api) return false;
    return static_cast<ProductionNcclApi*>(api.get())->bootstrap_id(out, err);
}

}  // namespace strata::distributed::nccl
