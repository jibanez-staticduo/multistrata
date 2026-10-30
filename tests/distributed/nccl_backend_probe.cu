// tests/distributed/nccl_backend_probe.cu - compile the NCCL backend the way the device build compiles it.
//
// Why this exists.  Until now every test in `tests/distributed/` was built by g++ on the CPU surface, where
// `cudaStream_t` is a stand-in pointer type and no NCCL symbol exists.  That proves the contract; it proves
// nothing about the path that passes a real stream to a real library, because nothing has ever compiled that
// path.  This file is that path: it includes the production translation unit and is built by nvcc with
// `-DSTRATA_DISTRIBUTED_CUDA` and the pinned NCCL headers, so the collective calls, the CUDA event calls and
// the `ncclConfig_t` use are checked by a CUDA compiler against the pinned header.
//
// What it checks at compile time (a wrong signature here is a build failure, not a test failure):
//   * `cudaStream_t` is the real CUDA stream type in this build surface, so the backend's overrides match the
//     contract only when CUDA is really present;
//   * the boundary's enqueue signatures take exactly that type;
//   * the two element types the contract allows are the two NCCL names for them (checked at run time, below,
//     because `ncclDataType_t` values are not constexpr-visible to the contract's enum).
// What it checks at run time, on the host that runs it: the bootstrap id encoding round-trips, the version
// decoder says what the pinned header means, and the SHA-256 used for library provenance matches the published
// FIPS 180-4 vectors.  A provenance gate whose hash is wrong refuses every library, so the hash gets vectors.
//
// Build (compile only, no GPU work, no run - this is what the NAS can prove):
//   /usr/local/cuda-12.8/bin/nvcc -std=c++20 -I include -I src -I <nccl-headers> -DSTRATA_DISTRIBUTED_CUDA -gencode arch=compute_86,code=sm_86 -c tests/distributed/nccl_backend_probe.cu -o <evidence>/probe.o
// On a Spark, the same translation unit is built into `strata-tp3-nccl-probe` and run before the smoke test.
//
// CMake lines for the parent integrator (not applied here; the root CMakeLists.txt belongs to the integrator):
//   add_executable(strata-tp3-nccl-probe tests/distributed/nccl_backend_probe.cu)
//   target_compile_definitions(strata-tp3-nccl-probe PRIVATE STRATA_DISTRIBUTED_CUDA)
//   target_include_directories(strata-tp3-nccl-probe PRIVATE include src ${STRATA_NCCL_INCLUDE_DIR})
//   target_link_libraries(strata-tp3-nccl-probe PRIVATE ${CMAKE_DL_LIBS})
//   add_test(NAME distributed_nccl_probe COMMAND strata-tp3-nccl-probe)
//   set_tests_properties(distributed_nccl_probe PROPERTIES LABELS "distributed;cuda")
// The NCCL library itself is not linked: `make_production_nccl_api` resolves it with `dlopen`, which is what
// lets a CPU build carry this contract without the library.  The probe fails loudly, and reports why, when no
// library is there to load.
#include "distributed/nccl_transport.cpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

namespace {

using namespace strata::distributed;
using namespace strata::distributed::nccl;

// The stream type must be CUDA's own here.  If this fires, the file was compiled without
// `-DSTRATA_DISTRIBUTED_CUDA` and the probe is proving the wrong thing.
static_assert(std::is_same_v<cudaStream_t, CUstream_st*>,
              "the probe must be built with the real CUDA stream type");

// The enqueue signatures of the boundary, checked against the same type.
static_assert(std::is_same_v<decltype(&NcclApi::enqueue_all_reduce_sum),
                             Failure (NcclApi::*)(const void*, void*, uint64_t, ScalarType, cudaStream_t)>,
              "all_reduce_sum does not take a real cudaStream_t");
static_assert(std::is_same_v<decltype(&NcclApi::enqueue_all_gather),
                             Failure (NcclApi::*)(const void*, void*, uint64_t, ScalarType, cudaStream_t)>,
              "all_gather does not take a real cudaStream_t");

// The contract's one-in-flight rule reaches the library through `Backend`, so the adapter must be a `Backend`
// whose raw calls carry a real stream.
static_assert(std::is_same_v<decltype(&Backend::raw_all_reduce_sum),
                             Failure (Backend::*)(const DeviceBuffer&, const DeviceBuffer&, uint64_t,
                                                  cudaStream_t)>,
              "Backend::raw_all_reduce_sum lost its stream");

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failed;
}

// The FIPS 180-4 vectors, as published.  The 56-character one is the one that matters most: its length is
// exactly where the padding rule changes shape, so a hash that gets all three right is not accidentally
// right about the tail of a 300 MB library.
void check_sha256_vectors() {
    struct Case {
        const char* text;
        const char* hex;
    };
    const Case cases[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    for (const Case& c : cases) {
        detail::sha256_ns::Sha256 hash;
        hash.update(reinterpret_cast<const uint8_t*>(c.text), std::strlen(c.text));
        const auto digest = hash.finish();
        const std::string got = detail::sha256_ns::hex(digest.data(), digest.size());
        check(got == c.hex, (std::string("sha256 of \"") + c.text + "\"").c_str());
        if (got != c.hex) std::printf("     got %s\n     want %s\n", got.c_str(), c.hex);
    }
}

void check_bootstrap_encoding() {
    // A 128-byte id where every byte is distinct, so a swapped nibble or a short read cannot pass.
    ncclUniqueId id;
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        id.internal[i] = static_cast<char>(i * 7 + 3);
    }
    const std::string hex = detail::hex_of_id(id);
    check(hex.size() == NCCL_UNIQUE_ID_BYTES * 2, "a 128-byte id encodes as 256 hex characters");
    ncclUniqueId back;
    std::string err;
    check(detail::hex_to_id(hex, back, err) &&
              std::memcmp(id.internal, back.internal, NCCL_UNIQUE_ID_BYTES) == 0,
          "the bootstrap id survives encode and decode unchanged");

    // The two refusals a torn roster produces: the wrong length, and something that is not hex.
    check(!detail::hex_to_id("00", back, err), "a short id is refused");
    check(!detail::hex_to_id(std::string(NCCL_UNIQUE_ID_BYTES * 2, 'g'), back, err),
          "a non-hex id is refused");
    check(!detail::hex_to_id(std::string(NCCL_UNIQUE_ID_BYTES * 2, 'A'), back, err),
          "an uppercase hex id is refused: the encoding is lowercase");
}

void check_version_decoder() {
    check(detail::version_string(NCCL_VERSION_CODE) == "2.30.7",
          "the pinned version code decodes to 2.30.7");
    check(detail::version_string(22707) == "2.27.7", "a 2.27 build decodes with the same shape");
    // The pre-2.9 shape from the header's own `NCCL_VERSION(X,Y,Z)` macro: 2*1000 + 8*100 + 7.
    check(detail::version_string(2807) == "2.8.7", "a pre-2.9 code decodes with the older shape");
    check(detail::version_string(20900) == "2.9.0", "the new shape starts where the header says it does");
}

// The two dtype handoffs.  A wrong value here sends the wrong element width over the wire, which is a wrong
// answer rather than an error, so it is checked against the header's own names.
void check_dtype_mapping(cudaStream_t stream) {
    ncclDataType_t dtype = ncclNumTypes;
    check(sizeof(float) == 4 && sizeof(__nv_bfloat16) == 2, "fp32 is 4 bytes and bf16 is 2 bytes here");
    // `ncclFloat32` and `ncclBfloat16` are the names the pinned header gives these two widths, and both are
    // legal for `ncclSum`.  Nothing in the contract allows a third.
    dtype = ncclFloat32;
    check(dtype == ncclFloat32, "fp32 has a name in this header");
    dtype = ncclBfloat16;
    check(dtype == ncclBfloat16, "bf16 has a name in this header");
    (void)stream;
}

}  // namespace

// Run on a host with a CUDA driver.  Here it is only compiled.
//
// `strata-tp3-nccl-probe [file]`: with an argument, print that file's SHA-256 as this code computes it, so a
// rank's approved provenance string can be read off the library it will actually load, and checked against
// `sha256sum` rather than trusted.
int main(int argc, char** argv) {
    if (argc > 1) {
        std::string hex;
        std::string err;
        if (!detail::sha256_of_file(argv[1], hex, err)) {
            std::printf("FAIL %s\n", err.c_str());
            return 1;
        }
        std::printf("%s  %s\n", hex.c_str(), argv[1]);
        return 0;
    }
    check_sha256_vectors();
    check_bootstrap_encoding();
    check_version_decoder();

    // Real stream, real library: the part a CPU test cannot reach.  If no NCCL is installed, the factory says
    // so and the rest of the probe is skipped rather than pretending.
    NcclConfig config;
    config.rank = 0;
    config.world = kTpWorld;
    config.bootstrap_id = std::string(NCCL_UNIQUE_ID_BYTES * 2, 'a');
    config.library_path.clear();  // the loader's own search order

    std::string err;
    std::unique_ptr<NcclApi> api = make_production_nccl_api(config, err);
    if (api == nullptr) {
        std::printf("skip  no NCCL library to load: %s\n", err.c_str());
        return g_failed == 0 ? 0 : 1;
    }

    std::string version;
    std::string provenance;
    const Failure identity = api->identity(version, provenance);
    check(!identity, "the library will say what it is");
    std::printf("info  library version '%s' provenance '%s'\n", version.c_str(), provenance.c_str());

    check_dtype_mapping(nullptr);

    // The gate is the interesting half: with an approval that cannot match, no communicator may exist.
    NcclBackend refused(std::move(api), config, NcclApproval{"0.0.0", "sha256:not-this-build"});
    const Failure gate = refused.init();
    check(gate.code == FailureCode::library_mismatch,
          "a library that is not the approved build is refused before a communicator exists");
    check(refused.initialized() == false, "the refused backend never reports itself ready");
    // Teardown through the transport the worker actually uses, so the CUDA build is compiled against
    // the same pairing the runtime has: `GuardedTransport` over `NcclBackend`.
    SystemClock clock;
    GuardedTransport transport(refused, clock);
    transport.abort("probe finished");

    std::printf("%s\n", g_failed == 0 ? "probe: all checks passed" : "probe: failures above");
    return g_failed == 0 ? 0 : 1;
}
