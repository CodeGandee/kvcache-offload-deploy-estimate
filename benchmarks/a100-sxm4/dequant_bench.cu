#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status_ = (call);                                           \
        if (status_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(status_));                          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__device__ __forceinline__ float e4m3fn_to_float(std::uint8_t value) {
    const unsigned sign = value >> 7;
    const unsigned exponent = (value >> 3) & 0xf;
    const unsigned mantissa = value & 0x7;
    float magnitude;
    if (exponent == 0) {
        magnitude = static_cast<float>(mantissa) * 0.001953125f;
    } else {
        const unsigned bits = ((exponent + 120) << 23) | (mantissa << 20);
        magnitude = __uint_as_float(bits);
    }
    return sign ? -magnitude : magnitude;
}

__global__ void initialize(std::uint8_t* source, float* scales,
                           std::size_t values) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < values) {
        source[index] = static_cast<std::uint8_t>((index * 37 + 11) % 120);
    }
    if (index < values / 128) {
        scales[index] = 1.0f + static_cast<float>(index & 7) * 0.03125f;
    }
}

__global__ void dequantize_e4m3fn(const std::uint8_t* source,
                                  const float* scales,
                                  __nv_bfloat16* output,
                                  std::size_t values) {
    const std::size_t first =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (first >= values) {
        return;
    }
    const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(source + first);
    const float scale = __ldg(scales + first / 128);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const auto encoded = static_cast<std::uint8_t>(packed >> (lane * 8));
        output[first + lane] = __float2bfloat16(e4m3fn_to_float(encoded) * scale);
    }
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int main() {
    constexpr std::size_t values = 128ull * 1024 * 1024;
    constexpr int iterations = 20;
    constexpr int rounds = 7;
    constexpr int threads = 256;
    constexpr double useful_bytes_per_value = 1.0 + 4.0 / 128.0 + 2.0;

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count != 2) {
        std::fprintf(stderr, "expected exactly two visible CUDA devices\n");
        return 2;
    }

    std::printf("{\"implementation\":\"custom_cuda_e4m3fn_block128_to_bf16\","
                "\"values\":%zu,\"devices\":[", values);
    for (int device = 0; device < device_count; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        std::uint8_t* source = nullptr;
        float* scales = nullptr;
        __nv_bfloat16* output = nullptr;
        CUDA_CHECK(cudaMalloc(&source, values));
        CUDA_CHECK(cudaMalloc(&scales, values / 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&output, values * sizeof(__nv_bfloat16)));

        const int init_blocks = static_cast<int>((values + threads - 1) / threads);
        initialize<<<init_blocks, threads>>>(source, scales, values);
        CUDA_CHECK(cudaGetLastError());
        const int blocks = static_cast<int>((values / 4 + threads - 1) / threads);
        for (int warmup = 0; warmup < 5; ++warmup) {
            dequantize_e4m3fn<<<blocks, threads>>>(source, scales, output, values);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> milliseconds;
        for (int round = 0; round < rounds; ++round) {
            cudaEvent_t start;
            cudaEvent_t end;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&end));
            CUDA_CHECK(cudaEventRecord(start));
            for (int iteration = 0; iteration < iterations; ++iteration) {
                dequantize_e4m3fn<<<blocks, threads>>>(source, scales, output, values);
            }
            CUDA_CHECK(cudaEventRecord(end));
            CUDA_CHECK(cudaEventSynchronize(end));
            float total_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));
            milliseconds.push_back(total_ms / iterations);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(end));
        }

        const double latency_ms = median(milliseconds);
        const double gvalues = static_cast<double>(values) / (latency_ms / 1000.0) / 1e9;
        const double useful_gbps = gvalues * useful_bytes_per_value;
        if (device != 0) {
            std::printf(",");
        }
        std::printf("{\"logical_device\":%d,\"latency_ms\":%.9g,"
                    "\"gvalues_per_second\":%.9g,\"useful_gbps\":%.9g}",
                    device, latency_ms, gvalues, useful_gbps);

        CUDA_CHECK(cudaFree(source));
        CUDA_CHECK(cudaFree(scales));
        CUDA_CHECK(cudaFree(output));
    }
    std::printf("]}\n");
    return 0;
}
