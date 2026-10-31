// Exact native MMVQ probe for the TP projection row layout. This checks the
// primitive boundary only; it does not exercise model state, collectives, or GEMM.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace k = strata::kernels;

void ck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreate(&value), "create stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

template<class T>
struct Device {
    T* value = nullptr;
    std::size_t count = 0;

    explicit Device(std::size_t elements) : count(elements) {
        require(elements > 0, "zero-sized device allocation");
        ck(cudaMalloc(&value, elements * sizeof(T)), "allocate device buffer");
    }
    ~Device() { if (value) cudaFree(value); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void upload(const T* source, std::size_t elements) {
        require(elements <= count, "device upload exceeds allocation");
        ck(cudaMemcpy(value, source, elements * sizeof(T), cudaMemcpyHostToDevice), "upload device buffer");
    }
    std::vector<T> read() const {
        std::vector<T> result(count);
        ck(cudaMemcpy(result.data(), value, count * sizeof(T), cudaMemcpyDeviceToHost), "read device buffer");
        return result;
    }
};

struct Matrix {
    std::string label;
    int type = -1;
    int n_in = 0;
    int n_out = 0;
    std::size_t row_bytes = 0;
    int source_row = 0;
    std::vector<uint8_t> weights;
};

struct Codec {
    const char* name;
    int type;
    int block_elements;
    int block_bytes;
    std::array<int, 2> half_scales;
};

constexpr Codec kCodecs[] = {
    {"Q4_0", 2, 32, 18, {0, -1}},
    {"Q5_0", 6, 32, 22, {0, -1}},
    {"Q5_1", 7, 32, 24, {0, 2}},
    {"Q8_0", 8, 32, 34, {0, -1}},
    {"Q3_K", 11, 256, 110, {108, -1}},
    {"Q4_K", 12, 256, 144, {0, 2}},
    {"Q5_K", 13, 256, 176, {0, 2}},
    {"Q6_K", 14, 256, 210, {208, -1}},
    {"IQ4_NL", 20, 32, 18, {0, -1}},
    {"IQ4_XS", 23, 256, 136, {0, -1}},
    {"Q2_0", 42, 64, 18, {0, -1}},
    {"IQ2_XXS", 16, 256, 66, {0, -1}},
    {"IQ2_XS", 17, 256, 74, {0, -1}},
    {"IQ3_XXS", 18, 256, 98, {0, -1}},
    {"IQ3_S", 21, 256, 110, {0, -1}},
    {"IQ2_S", 22, 256, 82, {0, -1}},
    {"IQ1_M", 29, 256, 56, {-1, -1}},
};

int block_elements(int type) {
    for (const auto& codec : kCodecs) if (codec.type == type) return codec.block_elements;
    int elements = 0, bytes = 0;
    require(strata::block_geometry(static_cast<uint32_t>(type), elements, bytes),
            "unsupported GGUF type id " + std::to_string(type));
    return elements;
}

std::size_t row_bytes(int type, int n_in) {
    return k::native_mmvq_weight_bytes(type, n_in, 1);
}

uint16_t finite_half(std::mt19937& rng) {
    const uint32_t bits = rng();
    // Exponents 5..9 produce small, finite, normal fp16 scales.
    return static_cast<uint16_t>(((bits >> 31) << 15) | ((5u + (bits >> 10) % 5u) << 10) |
                                 (bits & 0x3ffu));
}

Matrix random_matrix(const Codec& codec, int n_in, int n_out, uint32_t seed) {
    Matrix matrix;
    matrix.label = codec.name;
    matrix.type = codec.type;
    matrix.n_in = n_in;
    matrix.n_out = n_out;
    matrix.row_bytes = row_bytes(codec.type, n_in);
    matrix.weights.resize(matrix.row_bytes * static_cast<std::size_t>(n_out));

    std::mt19937 rng(seed);
    for (auto& byte : matrix.weights) byte = static_cast<uint8_t>(rng());
    require(matrix.row_bytes % codec.block_bytes == 0, "unexpected codec row geometry");
    for (std::size_t block = 0; block < matrix.weights.size() / codec.block_bytes; ++block) {
        for (int offset : codec.half_scales) {
            if (offset < 0) continue;
            const uint16_t half = finite_half(rng);
            std::memcpy(matrix.weights.data() + block * codec.block_bytes + offset, &half, sizeof(half));
        }
    }
    return matrix;
}

struct Options {
    std::string gguf;
    std::string tensor;
    int row_start = -1;
    int row_count = -1;
};

std::pair<int, int> parse_row_range(const std::string& value) {
    const auto separator = value.find(':');
    require(separator != std::string::npos, "--row-shard must be START:COUNT");
    const auto parse_nonnegative = [](const std::string& part) {
        require(!part.empty(), "empty --row-shard field");
        std::size_t consumed = 0;
        const long parsed = std::stol(part, &consumed);
        require(consumed == part.size() && parsed >= 0 && parsed <= std::numeric_limits<int>::max(),
                "invalid --row-shard field");
        return static_cast<int>(parsed);
    };
    return {parse_nonnegative(value.substr(0, separator)), parse_nonnegative(value.substr(separator + 1))};
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        require(i + 1 < argc || (argument != "--gguf" && argument != "--tensor" && argument != "--row-shard"),
                "missing value for " + argument);
        if (argument == "--gguf") options.gguf = argv[++i];
        else if (argument == "--tensor") options.tensor = argv[++i];
        else if (argument == "--row-shard") {
            const auto range = parse_row_range(argv[++i]);
            options.row_start = range.first;
            options.row_count = range.second;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    require(options.gguf.empty() == options.tensor.empty(), "--gguf and --tensor must be supplied together");
    require(options.gguf.empty() || options.row_count != 0, "--row-shard count must be positive");
    require(options.row_start < 0 || !options.gguf.empty(), "--row-shard requires --gguf and --tensor");
    return options;
}

Matrix read_gguf_matrix(const Options& options) {
    auto model = strata::GgufModel::open(options.gguf);
    std::size_t shard_index = 0;
    const auto* tensor = model.find(options.tensor, &shard_index);
    require(tensor != nullptr, "tensor not found in GGUF shards: " + options.tensor);
    require(tensor->shape.size() == 2, "selected tensor must be a two-dimensional matrix");
    require(tensor->shape[0] <= 16384 && tensor->shape[1] <= 65536, "selected tensor exceeds test shape limits");
    require(tensor->shape[0] <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
                tensor->shape[1] <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
            "selected tensor dimensions exceed native MMVQ limits");
    require(k::native_mmvq_supported(static_cast<int>(tensor->type)),
            std::string("native MMVQ does not support GGUF codec ") + tensor->type_name());
    require(model.in_bounds(*tensor, shard_index), "selected tensor payload is invalid or truncated");

    Matrix matrix;
    matrix.label = options.tensor;
    matrix.type = static_cast<int>(tensor->type);
    matrix.n_in = static_cast<int>(tensor->shape[0]);
    const int full_n_out = static_cast<int>(tensor->shape[1]);
    matrix.row_bytes = row_bytes(matrix.type, matrix.n_in);
    const uint64_t payload_bytes = strata::tensor_payload_bytes(*tensor);
    require(payload_bytes <= 64u * 1024u * 1024u, "selected GGUF matrix exceeds the 64 MiB test limit");
    require(payload_bytes == matrix.row_bytes * static_cast<std::size_t>(full_n_out),
            "GGUF payload size does not match native MMVQ row layout");

    const int first = options.row_start < 0 ? 0 : options.row_start;
    const int count = options.row_count < 0 ? full_n_out - first : options.row_count;
    require(first <= full_n_out && count > 0 && count <= full_n_out - first,
            "--row-shard range exceeds selected tensor");
    matrix.source_row = first;
    matrix.n_out = count;
    matrix.weights.resize(matrix.row_bytes * static_cast<std::size_t>(count));
    const auto* source = model.shard(shard_index).tensor_data(*tensor) +
                         matrix.row_bytes * static_cast<std::size_t>(first);
    std::memcpy(matrix.weights.data(), source, matrix.weights.size());
    std::printf("GGUF tensor=%s codec=%s n_in=%d rows=%d source_row=%d row_bytes=%zu original_bytes=1\n",
                matrix.label.c_str(), tensor->type_name(), matrix.n_in, matrix.n_out, matrix.source_row,
                matrix.row_bytes);
    return matrix;
}

std::vector<std::pair<int, int>> row_partitions(int n_out) {
    if (n_out == 640) return {{0, 256}, {256, 448}, {448, 640}};
    const int first = n_out / 3;
    const int second = (2 * n_out) / 3;
    std::vector<std::pair<int, int>> result;
    if (first > 0) result.emplace_back(0, first);
    if (second > first) result.emplace_back(first, second);
    if (n_out > second) result.emplace_back(second, n_out);
    return result;
}

std::vector<uint8_t> quantized_rows(const float* input, int n_in, int rows, cudaStream_t stream) {
    const std::size_t q_row_bytes = k::native_q8_1_bytes(n_in, 1);
    std::vector<uint8_t> result(static_cast<std::size_t>(rows) * q_row_bytes);
    Device<float> x_device(static_cast<std::size_t>(rows) * n_in);
    Device<uint8_t> q_device(result.size());
    x_device.upload(input, static_cast<std::size_t>(rows) * n_in);
    for (int begin = 0; begin < rows;) {
        const int count = std::min(8, rows - begin);
        k::native_quantize_q8_1(x_device.value + static_cast<std::size_t>(begin) * n_in,
                                q_device.value + static_cast<std::size_t>(begin) * q_row_bytes,
                                n_in, count, stream);
        begin += count;
    }
    ck(cudaStreamSynchronize(stream), "quantize activation rows");
    result = q_device.read();
    return result;
}

void compare_words(const std::vector<uint32_t>& expected, const std::vector<uint32_t>& actual,
                   const std::string& context) {
    require(expected.size() == actual.size(), context + ": output count mismatch");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            char message[256];
            std::snprintf(message, sizeof(message), "%s: output bit mismatch at index %zu (0x%08x vs 0x%08x)",
                          context.c_str(), i, expected[i], actual[i]);
            throw std::runtime_error(message);
        }
    }
}

std::vector<uint32_t> float_bits(const std::vector<float>& values) {
    std::vector<uint32_t> result(values.size());
    std::memcpy(result.data(), values.data(), values.size() * sizeof(float));
    return result;
}

void run_case(const Matrix& matrix, cudaStream_t stream) {
    require(k::native_mmvq_supported(matrix.type), matrix.label + ": unsupported native codec");
    require(matrix.n_in % block_elements(matrix.type) == 0, matrix.label + ": input splits a weight block");
    require(matrix.weights.size() == matrix.row_bytes * static_cast<std::size_t>(matrix.n_out),
            matrix.label + ": source byte count mismatch");

    constexpr int kRows = 10;
    constexpr int kChunk = 8;
    std::mt19937 rng(static_cast<uint32_t>(matrix.type * 7919 + matrix.n_in * 17 + matrix.n_out));
    std::normal_distribution<float> normal(0.f, 0.75f);
    std::vector<float> input(static_cast<std::size_t>(kRows) * matrix.n_in);
    for (float& value : input) value = normal(rng);

    // Quantize the complete input together and through independent one-row calls.
    const auto q_multi = quantized_rows(input.data(), matrix.n_in, kRows, stream);
    const std::size_t q_row_bytes = k::native_q8_1_bytes(matrix.n_in, 1);
    std::vector<uint8_t> q_single(static_cast<std::size_t>(kRows) * q_row_bytes);

    Device<uint8_t> q_device(q_multi.size());
    q_device.upload(q_multi.data(), q_multi.size());
    Device<uint8_t> q_single_device(q_single.size());
    Device<uint8_t> w_device(matrix.weights.size());
    w_device.upload(matrix.weights.data(), matrix.weights.size());
    Device<float> x_device(input.size());
    x_device.upload(input.data(), input.size());
    Device<float> multi_output(static_cast<std::size_t>(8) * matrix.n_out);
    Device<float> single_output_all(static_cast<std::size_t>(kRows) * matrix.n_out);

    k::native_mmvq_set_multi_exact(true);
    k::native_mmvq_set_b6_rows(0);
    require(k::native_mmvq_multi_exact(), "native_mmvq_multi_exact(true) did not take effect");
    for (int row = 0; row < kRows; ++row) {
        auto* q = q_single_device.value + static_cast<std::size_t>(row) * q_row_bytes;
        k::native_quantize_q8_1(x_device.value + static_cast<std::size_t>(row) * matrix.n_in,
                                q, matrix.n_in, 1, stream);
        k::native_mmvq(matrix.type, w_device.value, q,
                       single_output_all.value + static_cast<std::size_t>(row) * matrix.n_out,
                       matrix.n_in, matrix.n_out, 1, stream);
    }
    ck(cudaStreamSynchronize(stream), "run separate-row quantization and MMVQ references");
    q_single = q_single_device.read();
    const std::vector<uint32_t> reference_all = float_bits(single_output_all.read());

    const auto partitions = row_partitions(matrix.n_out);
    for (int columns = 2; columns <= 8; ++columns) {
        require(std::memcmp(q_multi.data(), q_single.data(), static_cast<std::size_t>(columns) * q_row_bytes) == 0,
                matrix.label + ": batched Q8_1 bytes differ from separate one-row quantization");

        k::native_mmvq(matrix.type, w_device.value, q_device.value, multi_output.value,
                       matrix.n_in, matrix.n_out, columns, stream);
        ck(cudaStreamSynchronize(stream), "run multi-column native MMVQ");
        auto got = float_bits(multi_output.read());
        got.resize(static_cast<std::size_t>(columns) * matrix.n_out);
        std::vector<uint32_t> expected_prefix(static_cast<std::size_t>(columns) * matrix.n_out);
        for (int column = 0; column < columns; ++column) {
            std::copy_n(reference_all.begin() + static_cast<std::size_t>(column) * matrix.n_out, matrix.n_out,
                        expected_prefix.begin() + static_cast<std::size_t>(column) * matrix.n_out);
        }
        compare_words(expected_prefix, got, matrix.label + " ncols=" + std::to_string(columns));

        // Run the same multi-column call independently on each contiguous
        // output-row shard and put its results back at the original row indices.
        std::vector<uint32_t> row_mapped(static_cast<std::size_t>(columns) * matrix.n_out);
        for (const auto& [begin, end] : partitions) {
            const int shard_rows = end - begin;
            const std::size_t shard_bytes = matrix.row_bytes * static_cast<std::size_t>(shard_rows);
            Device<uint8_t> shard_weights(shard_bytes);
            shard_weights.upload(matrix.weights.data() + matrix.row_bytes * static_cast<std::size_t>(begin),
                                 shard_bytes);
            Device<float> shard_output(static_cast<std::size_t>(columns) * shard_rows);
            k::native_mmvq(matrix.type, shard_weights.value, q_device.value, shard_output.value,
                           matrix.n_in, shard_rows, columns, stream);
            ck(cudaStreamSynchronize(stream), "run row-sharded multi-column MMVQ");
            const auto shard_bits = float_bits(shard_output.read());
            for (int column = 0; column < columns; ++column) {
                std::copy_n(shard_bits.begin() + static_cast<std::size_t>(column) * shard_rows, shard_rows,
                            row_mapped.begin() + static_cast<std::size_t>(column) * matrix.n_out + begin);
            }
        }
        compare_words(expected_prefix, row_mapped, matrix.label + " row-mapped ncols=" + std::to_string(columns));
    }

    // A ten-row sequence crosses both requested boundaries: 1 + 9 is executed
    // as 1 + 8 + 1, while 8 + remainder is executed as 8 + 2.
    require(std::memcmp(q_multi.data(), q_single.data(), q_single.size()) == 0,
            matrix.label + ": all-row Q8_1 bytes differ from separate one-row quantization");

    Device<float> chunked(static_cast<std::size_t>(kRows) * matrix.n_out);
    for (int begin = 0; begin < kRows;) {
        const int count = std::min(kChunk, kRows - begin);
        const std::size_t q_offset = static_cast<std::size_t>(begin) * q_row_bytes;
        const std::size_t y_offset = static_cast<std::size_t>(begin) * matrix.n_out;
        k::native_mmvq(matrix.type, w_device.value,
                       static_cast<const uint8_t*>(q_device.value) + q_offset,
                       chunked.value + y_offset,
                       matrix.n_in, matrix.n_out, count, stream);
        begin += count;
    }
    ck(cudaStreamSynchronize(stream), "run chunked native MMVQ");
    compare_words(reference_all, float_bits(chunked.read()), matrix.label + " chunks=8+remainder");

    // Explicitly model the common first-token plus remaining-batch schedule.
    Device<float> first_then_rest(static_cast<std::size_t>(kRows) * matrix.n_out);
    int start = 0;
    while (start < kRows) {
        const int count = start == 0 ? 1 : std::min(kChunk, kRows - start);
        const std::size_t q_offset = static_cast<std::size_t>(start) * q_row_bytes;
        const std::size_t y_offset = static_cast<std::size_t>(start) * matrix.n_out;
        k::native_mmvq(matrix.type, w_device.value,
                       static_cast<const uint8_t*>(q_device.value) + q_offset,
                       first_then_rest.value + y_offset,
                       matrix.n_in, matrix.n_out, count, stream);
        start += count;
    }
    ck(cudaStreamSynchronize(stream), "run first-row plus batch MMVQ");
    compare_words(reference_all, float_bits(first_then_rest.read()), matrix.label + " chunks=1+N");

    const auto weights_after = w_device.read();
    require(weights_after == matrix.weights, matrix.label + ": native MMVQ modified original weight bytes");
    std::printf("PASS projection codec=%s type=%d shape=%dx%d row_shards=%zu ncols=2..8 chunks=1+N,8+remainder "
                "weight_bytes_unchanged=1\n",
                matrix.label.c_str(), matrix.type, matrix.n_in, matrix.n_out, partitions.size());
}

void run_synthetic_cases(cudaStream_t stream) {
    for (std::size_t i = 0; i < std::size(kCodecs); ++i) {
        const auto& codec = kCodecs[i];
        run_case(random_matrix(codec, 2560, 640, static_cast<uint32_t>(0x1000 + i)), stream);
        run_case(random_matrix(codec, 6144, 2560, static_cast<uint32_t>(0x2000 + i)), stream);
        if (codec.block_elements == 32) {
            run_case(random_matrix(codec, 640, 2560, static_cast<uint32_t>(0x3000 + i)), stream);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            std::fprintf(stderr, "SKIP: no usable CUDA device for native MMVQ projection test\n");
            return 77;
        }

        Stream stream;
        k::native_mmvq_set_multi_exact(true);
        require(k::native_mmvq_multi_exact(), "could not enable exact native multi-column mode");
        if (!options.gguf.empty()) {
            run_case(read_gguf_matrix(options), stream.value);
        } else {
            run_synthetic_cases(stream.value);
        }
        ck(cudaDeviceSynchronize(), "finish native MMVQ projection test");
        std::printf("tp_projection_multicolumn: ok; this validates native Q8_1/MMVQ projection parity only\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "tp_projection_multicolumn: %s\n", error.what());
        return 2;
    }
}
