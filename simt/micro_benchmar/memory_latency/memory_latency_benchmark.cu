
// sm89_memory_latency_benchmark.cu
// SM89 / Ada dependent memory-latency microbenchmark.
// Build: nvcc -O3 -arch=sm_89 sm89_memory_latency_benchmark.cu -o sm89_memory_latency_benchmark
// Run:   ./sm89_memory_latency_benchmark
// SASS:  cuobjdump --dump-sass sm89_memory_latency_benchmark

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

static constexpr std::size_t NODE_STRIDE_BYTES = 128;

__device__ __forceinline__ uint32_t ld_ca_u32(const uint32_t* p) {
    uint32_t out;
    asm volatile("ld.global.ca.u32 %0, [%1];\n" : "=r"(out) : "l"(p) : "memory");
    return out;
}

__device__ __forceinline__ uint32_t ld_cg_u32(const uint32_t* p) {
    uint32_t out;
    asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(out) : "l"(p) : "memory");
    return out;
}

__global__ __launch_bounds__(1)
void latency_l1_kernel(const uint32_t* __restrict__ chain,
                       uint32_t start_idx,
                       int warmup_iters,
                       int measure_iters,
                       unsigned long long* cycles_out,
                       uint32_t* final_idx_out) {
    uint32_t idx = start_idx;
    for (int i = 0; i < warmup_iters; ++i) idx = ld_ca_u32(chain + idx);
    asm volatile("" : "+r"(idx) :: "memory");
    const unsigned long long t0 = clock64();
    for (int i = 0; i < measure_iters; ++i) idx = ld_ca_u32(chain + idx);
    const unsigned long long t1 = clock64();
    cycles_out[0] = t1 - t0;
    final_idx_out[0] = idx;
}

__global__ __launch_bounds__(1)
void latency_cg_kernel(const uint32_t* __restrict__ chain,
                       uint32_t start_idx,
                       int warmup_iters,
                       int measure_iters,
                       unsigned long long* cycles_out,
                       uint32_t* final_idx_out) {
    uint32_t idx = start_idx;
    for (int i = 0; i < warmup_iters; ++i) idx = ld_cg_u32(chain + idx);
    asm volatile("" : "+r"(idx) :: "memory");
    const unsigned long long t0 = clock64();
    for (int i = 0; i < measure_iters; ++i) idx = ld_cg_u32(chain + idx);
    const unsigned long long t1 = clock64();
    cycles_out[0] = t1 - t0;
    final_idx_out[0] = idx;
}

__global__ void l2_thrash_kernel(uint32_t* buf, std::size_t words) {
    const std::size_t tid = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
    for (std::size_t i = tid; i < words; i += stride) {
        uint32_t v = buf[i];
        buf[i] = v + 1u;
    }
}

struct ChainHost {
    std::vector<uint32_t> data;
    uint32_t start_idx = 0;
    std::size_t node_count = 0;
};

static ChainHost build_random_chain(std::size_t working_set_bytes, uint32_t seed = 12345) {
    if (working_set_bytes < 2 * NODE_STRIDE_BYTES) {
        std::fprintf(stderr, "Working set too small.\n");
        std::exit(EXIT_FAILURE);
    }
    working_set_bytes = (working_set_bytes / NODE_STRIDE_BYTES) * NODE_STRIDE_BYTES;
    const std::size_t words = working_set_bytes / sizeof(uint32_t);
    const std::size_t stride_words = NODE_STRIDE_BYTES / sizeof(uint32_t);
    const std::size_t nodes = working_set_bytes / NODE_STRIDE_BYTES;
    if (words > (std::size_t)UINT32_MAX) {
        std::fprintf(stderr, "Working set too large for 32-bit indices.\n");
        std::exit(EXIT_FAILURE);
    }

    ChainHost chain;
    chain.data.resize(words, 0u);
    chain.node_count = nodes;

    std::vector<uint32_t> perm(nodes);
    for (uint32_t i = 0; i < (uint32_t)nodes; ++i) perm[i] = i;
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (std::size_t i = 0; i < nodes; ++i) {
        const uint32_t cur_node = perm[i];
        const uint32_t next_node = perm[(i + 1) % nodes];
        const std::size_t cur_word = (std::size_t)cur_node * stride_words;
        const std::size_t next_word = (std::size_t)next_node * stride_words;
        chain.data[cur_word] = (uint32_t)next_word;
    }
    chain.start_idx = (uint32_t)((std::size_t)perm[0] * stride_words);
    return chain;
}

enum class Path { L1_CA, L2_CG, DRAM_CG };

static const char* path_name(Path p) {
    switch (p) {
        case Path::L1_CA: return "L1 / ld.ca";
        case Path::L2_CG: return "L2 / ld.cg";
        case Path::DRAM_CG: return "DRAM / ld.cg";
    }
    return "unknown";
}

static void thrash_l2(uint32_t* d_flush, std::size_t flush_bytes) {
    if (!d_flush || flush_bytes == 0) return;
    constexpr int threads = 256;
    constexpr int blocks = 256;
    const std::size_t words = flush_bytes / sizeof(uint32_t);
    l2_thrash_kernel<<<blocks, threads>>>(d_flush, words);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static double run_latency_test(std::size_t working_set_bytes,
                               Path path,
                               int measure_iters,
                               int warmup_rounds,
                               uint32_t* d_flush,
                               std::size_t flush_bytes) {
    ChainHost h_chain = build_random_chain(working_set_bytes);
    const std::size_t alloc_bytes = h_chain.data.size() * sizeof(uint32_t);

    uint32_t* d_chain = nullptr;
    unsigned long long* d_cycles = nullptr;
    uint32_t* d_final = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chain, alloc_bytes));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_final, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_chain, h_chain.data.data(), alloc_bytes, cudaMemcpyHostToDevice));

    int warmup_iters = 0;
    if (path == Path::L1_CA || path == Path::L2_CG) {
        warmup_iters = (int)(h_chain.node_count * (std::size_t)warmup_rounds);
    }
    if (path == Path::DRAM_CG) {
        thrash_l2(d_flush, flush_bytes);
        warmup_iters = 0;
    }

    if (path == Path::L1_CA) {
        latency_l1_kernel<<<1, 1>>>(d_chain, h_chain.start_idx, warmup_iters,
                                    measure_iters, d_cycles, d_final);
    } else {
        latency_cg_kernel<<<1, 1>>>(d_chain, h_chain.start_idx, warmup_iters,
                                    measure_iters, d_cycles, d_final);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long total_cycles = 0;
    uint32_t final_idx = 0;
    CUDA_CHECK(cudaMemcpy(&total_cycles, d_cycles, sizeof(total_cycles), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&final_idx, d_final, sizeof(final_idx), cudaMemcpyDeviceToHost));

    const double cycles_per_load = (double)total_cycles / (double)measure_iters;
    std::printf("%-14s | WS=%9.2f MB | nodes=%9zu | total=%12llu cyc | latency=%9.2f cyc/load | final=%u\n",
                path_name(path), working_set_bytes / 1024.0 / 1024.0,
                h_chain.node_count, total_cycles, cycles_per_load, final_idx);

    CUDA_CHECK(cudaFree(d_chain));
    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_final));
    return cycles_per_load;
}

int main() {
    constexpr int DEVICE = 0;
    constexpr int MEASURE_ITERS = 200000;
    constexpr int WARMUP_ROUNDS = 2;

    CUDA_CHECK(cudaSetDevice(DEVICE));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, DEVICE));

    std::printf("============================================================\n");
    std::printf("SM89 dependent memory-latency microbenchmark\n");
    std::printf("============================================================\n");
    std::printf("GPU                 : %s\n", prop.name);
    std::printf("Compute capability  : %d.%d\n", prop.major, prop.minor);
    std::printf("L2 cache            : %.2f MB\n", prop.l2CacheSize / 1024.0 / 1024.0);
    std::printf("Reported SM clock   : %.3f GHz\n", prop.clockRate / 1e6);
    std::printf("Measurement iters   : %d\n", MEASURE_ITERS);
    std::printf("Pointer-node stride : %zu B\n", NODE_STRIDE_BYTES);
    std::printf("============================================================\n\n");

    if (!(prop.major == 8 && prop.minor == 9)) {
        std::printf("WARNING: current device is not SM89.\n\n");
    }

    // Allocate a buffer used to disturb L2 before the DRAM test.
    std::size_t flush_bytes = std::max<std::size_t>(
        256ULL << 20, (std::size_t)prop.l2CacheSize * 4ULL);
    uint32_t* d_flush = nullptr;
    cudaError_t flush_err = cudaMalloc(&d_flush, flush_bytes);
    if (flush_err != cudaSuccess) {
        std::fprintf(stderr,
                     "WARNING: could not allocate %.2f MB L2-thrash buffer: %s\n"
                     "DRAM test will proceed without explicit L2 thrashing.\n\n",
                     flush_bytes / 1024.0 / 1024.0, cudaGetErrorString(flush_err));
        d_flush = nullptr;
        flush_bytes = 0;
        cudaGetLastError();
    } else {
        CUDA_CHECK(cudaMemset(d_flush, 0, flush_bytes));
    }

    std::printf("[1] L1 latency test\n");
    run_latency_test(16ULL << 10, Path::L1_CA, MEASURE_ITERS,
                     WARMUP_ROUNDS, d_flush, flush_bytes);

    std::printf("\n[2] L2 latency test\n");
    std::size_t l2_ws = std::max<std::size_t>(
        1ULL << 20, (std::size_t)prop.l2CacheSize / 8ULL);
    run_latency_test(l2_ws, Path::L2_CG, MEASURE_ITERS,
                     WARMUP_ROUNDS, d_flush, flush_bytes);

    std::printf("\n[3] DRAM / device-memory latency test\n");
    std::size_t dram_ws = std::max<std::size_t>(
        256ULL << 20, (std::size_t)prop.l2CacheSize * 4ULL);
    run_latency_test(dram_ws, Path::DRAM_CG, MEASURE_ITERS,
                     0, d_flush, flush_bytes);

    // Optional working-set sweep: uncomment as needed.
    /*
    std::printf("\n[4] Optional ld.cg working-set sweep\n");
    const std::size_t sweep_sizes[] = {
        64ULL << 10, 256ULL << 10,
        1ULL << 20, 2ULL << 20, 4ULL << 20, 8ULL << 20,
        16ULL << 20, 32ULL << 20, 64ULL << 20,
        128ULL << 20, 256ULL << 20, 512ULL << 20
    };
    for (std::size_t ws : sweep_sizes) {
        run_latency_test(ws, Path::L2_CG, MEASURE_ITERS,
                         WARMUP_ROUNDS, d_flush, flush_bytes);
    }
    */

    if (d_flush) CUDA_CHECK(cudaFree(d_flush));
    std::printf("\nDone.\n");
    return 0;
}
