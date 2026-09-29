// h20_memory_bandwidth_optimized.cu
//
// Optimized read-bandwidth microbenchmark for NVIDIA H20 / Hopper-style hierarchy.
//
// Measures:
//   L1  : ld.global.ca
//   L2  : ld.global.cg
//   HBM : ld.global.cg
//
// Key optimization versus the previous version:
//   For every individual load instruction, lanes in a warp access contiguous u32
//   addresses. ILP is created by accessing different contiguous segments:
//
//      lane 0 : base + 0,        base + T,        base + 2T, ...
//      lane 1 : base + 1,        base + T + 1,    base + 2T + 1, ...
//      ...
//      lane31 : base + 31,       base + T + 31,   base + 2T + 31, ...
//
//   where T is blockDim.x for cache tests, or total_threads for HBM.
//   Thus each warp's ld.global.u32 is naturally coalesced.
//
// Thread sweep:
//   128 / 256 / 512 / 1024 threads/block.
//
// Launch policy for each thread setting:
//   L1:  1 block per SM, 16KB private slice per SM.
//   L2:  1 block per SM, 32MB total unique working set split across blocks.
//   HBM: 4 blocks per SM, 1GB streaming working set, repeated 16 times.
//
// Build:
//   nvcc -O3 -lineinfo -arch=sm_90 h20_memory_bandwidth_optimized.cu -o h20_bw
//
// Run:
//   ./h20_bw
//
// Output:
//   console + h20_memory_bandwidth_thread_sweep.csv
//
// Notes:
//   1. cudaGetDeviceProperties().multiProcessorCount is used as the SM count.
//   2. Exact block-to-SM pinning is not guaranteed by CUDA.
//   3. The reported bandwidth is useful payload bytes issued by load instructions.
//      It is not the physical fabric traffic after cache-line/sector effects.
//   4. Use Nsight Compute / hardware PMU counters to verify L1/L2 hit rates when possible.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(expr) do {                                           \
    cudaError_t _cuda_err = (expr);                                     \
    if (_cuda_err != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(_cuda_err)); \
        std::exit(EXIT_FAILURE);                                        \
    }                                                                   \
} while (0)

static constexpr size_t KB = 1024ULL;
static constexpr size_t MB = 1024ULL * 1024ULL;
static constexpr size_t GB = 1024ULL * 1024ULL * 1024ULL;

static constexpr int THREAD_CONFIGS[] = {128, 256, 512, 1024};
static constexpr int ILP = 8;

// Default test parameters.
static constexpr size_t L1_SLICE_BYTES = 16 * KB; // per SM
static constexpr size_t L2_TOTAL_BYTES = 32 * MB; // total unique WS, comfortably below ~60MB L2
static constexpr size_t HBM_TOTAL_BYTES = 1 * GB; // >> L2
static constexpr int L1_REPEAT = 32768;
static constexpr int L2_REPEAT = 1024;
static constexpr int HBM_REPEAT = 16;
static constexpr int HBM_BLOCKS_PER_SM = 4;

// -----------------------------------------------------------------------------
// PTX load helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__
uint32_t ld_ca_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile(
        "ld.global.ca.u32 %0, [%1];"
        : "=r"(v)
        : "l"(p)
        : "memory");
    return v;
}

__device__ __forceinline__
uint32_t ld_cg_u32(const uint32_t* p)
{
    uint32_t v;
    asm volatile(
        "ld.global.cg.u32 %0, [%1];"
        : "=r"(v)
        : "l"(p)
        : "memory");
    return v;
}

__device__ __forceinline__
uint32_t read_smid()
{
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// -----------------------------------------------------------------------------
// Optional SM probe: records actual SM IDs observed during a short probe.
// The benchmark still uses multiProcessorCount as the authoritative SM count.
// -----------------------------------------------------------------------------

__global__
void probe_smid_kernel(uint32_t* smids)
{
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
    }
}

static void probe_observed_sms(int sm_count)
{
    const int blocks = sm_count * 8;
    const int threads = 32;

    uint32_t* d_smids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_smids, static_cast<size_t>(blocks) * sizeof(uint32_t)));

    probe_smid_kernel<<<blocks, threads>>>(d_smids);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_smids(blocks);
    CUDA_CHECK(cudaMemcpy(h_smids.data(),
                          d_smids,
                          static_cast<size_t>(blocks) * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_smids));

    std::vector<int> seen(sm_count, 0);
    for (uint32_t id : h_smids) {
        if (id < static_cast<uint32_t>(sm_count)) {
            seen[id] = 1;
        }
    }

    int observed = 0;
    for (int x : seen) observed += x;

    std::printf("Reported SM count : %d\n", sm_count);
    std::printf("Observed SM count : %d\n", observed);
}

// -----------------------------------------------------------------------------
// Cache-resident bandwidth kernel.
//
// MODE_L1=true:
//   slice_id = SMID, so each SM accesses a private slice.
//
// MODE_L1=false:
//   slice_id = blockIdx.x, so each block accesses a unique L2 slice.
//   All slices together form L2_TOTAL_BYTES.
//
// Coalescing:
//   For a fixed k, lanes 0..31 access adjacent uint32_t elements.
// -----------------------------------------------------------------------------

template <bool USE_CA, bool MODE_L1>
__global__
void cache_read_bw_kernel(const uint32_t* __restrict__ src,
                          size_t elements_per_slice,
                          int num_slices,
                          int repeat,
                          uint32_t* __restrict__ sink)
{
    int slice_id;

    if constexpr (MODE_L1) {
        slice_id = static_cast<int>(read_smid()) % num_slices;
    } else {
        slice_id = static_cast<int>(blockIdx.x) % num_slices;
    }

    const uint32_t* base_ptr =
        src + static_cast<size_t>(slice_id) * elements_per_slice;

    const size_t lane = static_cast<size_t>(threadIdx.x);
    const size_t block_threads = static_cast<size_t>(blockDim.x);

    uint32_t a0=0, a1=0, a2=0, a3=0;
    uint32_t a4=0, a5=0, a6=0, a7=0;

    const size_t tile = block_threads * ILP;

    for (int r = 0; r < repeat; ++r) {
        for (size_t base = 0; base < elements_per_slice; base += tile) {
            const size_t i0 = base + lane + 0 * block_threads;
            const size_t i1 = base + lane + 1 * block_threads;
            const size_t i2 = base + lane + 2 * block_threads;
            const size_t i3 = base + lane + 3 * block_threads;
            const size_t i4 = base + lane + 4 * block_threads;
            const size_t i5 = base + lane + 5 * block_threads;
            const size_t i6 = base + lane + 6 * block_threads;
            const size_t i7 = base + lane + 7 * block_threads;

            if constexpr (USE_CA) {
                if (i0 < elements_per_slice) a0 ^= ld_ca_u32(base_ptr + i0);
                if (i1 < elements_per_slice) a1 ^= ld_ca_u32(base_ptr + i1);
                if (i2 < elements_per_slice) a2 ^= ld_ca_u32(base_ptr + i2);
                if (i3 < elements_per_slice) a3 ^= ld_ca_u32(base_ptr + i3);
                if (i4 < elements_per_slice) a4 ^= ld_ca_u32(base_ptr + i4);
                if (i5 < elements_per_slice) a5 ^= ld_ca_u32(base_ptr + i5);
                if (i6 < elements_per_slice) a6 ^= ld_ca_u32(base_ptr + i6);
                if (i7 < elements_per_slice) a7 ^= ld_ca_u32(base_ptr + i7);
            } else {
                if (i0 < elements_per_slice) a0 ^= ld_cg_u32(base_ptr + i0);
                if (i1 < elements_per_slice) a1 ^= ld_cg_u32(base_ptr + i1);
                if (i2 < elements_per_slice) a2 ^= ld_cg_u32(base_ptr + i2);
                if (i3 < elements_per_slice) a3 ^= ld_cg_u32(base_ptr + i3);
                if (i4 < elements_per_slice) a4 ^= ld_cg_u32(base_ptr + i4);
                if (i5 < elements_per_slice) a5 ^= ld_cg_u32(base_ptr + i5);
                if (i6 < elements_per_slice) a6 ^= ld_cg_u32(base_ptr + i6);
                if (i7 < elements_per_slice) a7 ^= ld_cg_u32(base_ptr + i7);
            }
        }
    }

    // Every thread writes a result, preventing the compiler from deleting
    // non-thread0 loads as dead code.
    const size_t gtid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    sink[gtid] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}

// -----------------------------------------------------------------------------
// HBM streaming kernel.
//
// For each individual load k, adjacent warp lanes access adjacent addresses.
// ILP is across segments separated by total_threads.
// -----------------------------------------------------------------------------

__global__
void hbm_read_bw_kernel(const uint32_t* __restrict__ src,
                        size_t elements,
                        int repeat,
                        uint32_t* __restrict__ sink)
{
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t total_threads =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    const size_t tile = total_threads * ILP;

    uint32_t a0=0, a1=0, a2=0, a3=0;
    uint32_t a4=0, a5=0, a6=0, a7=0;

    for (int r = 0; r < repeat; ++r) {
        for (size_t base = 0; base < elements; base += tile) {
            const size_t i0 = base + tid + 0 * total_threads;
            const size_t i1 = base + tid + 1 * total_threads;
            const size_t i2 = base + tid + 2 * total_threads;
            const size_t i3 = base + tid + 3 * total_threads;
            const size_t i4 = base + tid + 4 * total_threads;
            const size_t i5 = base + tid + 5 * total_threads;
            const size_t i6 = base + tid + 6 * total_threads;
            const size_t i7 = base + tid + 7 * total_threads;

            if (i0 < elements) a0 ^= ld_cg_u32(src + i0);
            if (i1 < elements) a1 ^= ld_cg_u32(src + i1);
            if (i2 < elements) a2 ^= ld_cg_u32(src + i2);
            if (i3 < elements) a3 ^= ld_cg_u32(src + i3);
            if (i4 < elements) a4 ^= ld_cg_u32(src + i4);
            if (i5 < elements) a5 ^= ld_cg_u32(src + i5);
            if (i6 < elements) a6 ^= ld_cg_u32(src + i6);
            if (i7 < elements) a7 ^= ld_cg_u32(src + i7);
        }
    }

    sink[tid] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

struct Result {
    std::string level;
    std::string load_policy;
    int blocks;
    int threads;
    size_t working_set_bytes;
    int repeat;
    double time_ms;
    double bytes_read;
    double bandwidth_gbps;
};

static size_t round_down(size_t x, size_t align)
{
    return (x / align) * align;
}

template <bool USE_CA, bool MODE_L1>
static Result run_cache_test(const char* level,
                             const char* policy,
                             int sm_count,
                             int threads,
                             size_t requested_total_ws,
                             size_t requested_slice_ws,
                             int repeat)
{
    const int blocks = sm_count;

    size_t slice_bytes = 0;
    int num_slices = blocks;

    if constexpr (MODE_L1) {
        slice_bytes = requested_slice_ws;
        num_slices = sm_count;
    } else {
        // Divide the requested total L2 working set into one disjoint slice/block.
        const size_t min_align =
            static_cast<size_t>(threads) * sizeof(uint32_t);

        slice_bytes =
            round_down(requested_total_ws / static_cast<size_t>(blocks),
                       min_align);

        if (slice_bytes == 0) {
            std::fprintf(stderr, "L2 slice became zero; total working set too small.\n");
            std::exit(EXIT_FAILURE);
        }
    }

    const size_t total_ws =
        slice_bytes * static_cast<size_t>(num_slices);

    const size_t elements_per_slice =
        slice_bytes / sizeof(uint32_t);

    uint32_t* d_src = nullptr;
    uint32_t* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, total_ws));
    CUDA_CHECK(cudaMalloc(
        &d_sink,
        static_cast<size_t>(blocks) * threads * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(d_src, 0x5A, total_ws));
    CUDA_CHECK(cudaMemset(
        d_sink, 0,
        static_cast<size_t>(blocks) * threads * sizeof(uint32_t)));

    // Untimed warmup: populate target cache.
    cache_read_bw_kernel<USE_CA, MODE_L1>
        <<<blocks, threads>>>(
            d_src,
            elements_per_slice,
            num_slices,
            1,
            d_sink);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt));

    cache_read_bw_kernel<USE_CA, MODE_L1>
        <<<blocks, threads>>>(
            d_src,
            elements_per_slice,
            num_slices,
            repeat,
            d_sink);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));

    // Each block reads exactly one full slice per repeat.
    const double bytes_read =
        static_cast<double>(blocks) *
        static_cast<double>(slice_bytes) *
        static_cast<double>(repeat);

    const double gbps =
        bytes_read / (static_cast<double>(ms) * 1.0e-3) / 1.0e9;

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_sink));

    return Result{
        level,
        policy,
        blocks,
        threads,
        total_ws,
        repeat,
        ms,
        bytes_read,
        gbps
    };
}

static Result run_hbm_test(int sm_count,
                           int threads,
                           size_t total_ws,
                           int repeat)
{
    const int blocks = sm_count * HBM_BLOCKS_PER_SM;

    // Keep element count aligned to uint32_t.
    total_ws = round_down(total_ws, sizeof(uint32_t));
    const size_t elements = total_ws / sizeof(uint32_t);

    uint32_t* d_src = nullptr;
    uint32_t* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, total_ws));
    CUDA_CHECK(cudaMalloc(
        &d_sink,
        static_cast<size_t>(blocks) * threads * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(d_src, 0xA5, total_ws));
    CUDA_CHECK(cudaMemset(
        d_sink, 0,
        static_cast<size_t>(blocks) * threads * sizeof(uint32_t)));

    // Untimed pass.
    hbm_read_bw_kernel<<<blocks, threads>>>(
        d_src, elements, 1, d_sink);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt));

    hbm_read_bw_kernel<<<blocks, threads>>>(
        d_src, elements, repeat, d_sink);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));

    const double bytes_read =
        static_cast<double>(total_ws) *
        static_cast<double>(repeat);

    const double gbps =
        bytes_read / (static_cast<double>(ms) * 1.0e-3) / 1.0e9;

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_sink));

    return Result{
        "HBM",
        "ld.global.cg",
        blocks,
        threads,
        total_ws,
        repeat,
        ms,
        bytes_read,
        gbps
    };
}

static void print_result(const Result& r)
{
    std::printf(
        "%-4s %-12s blocks=%4d threads=%4d "
        "WS=%8.2f MB repeat=%6d time=%8.3f ms "
        "read=%8.2f GB BW=%10.2f GB/s\n",
        r.level.c_str(),
        r.load_policy.c_str(),
        r.blocks,
        r.threads,
        r.working_set_bytes / 1024.0 / 1024.0,
        r.repeat,
        r.time_ms,
        r.bytes_read / 1.0e9,
        r.bandwidth_gbps);
}

int main(int argc, char** argv)
{
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int sm_count = prop.multiProcessorCount;

    std::printf("GPU               : %s\n", prop.name);
    std::printf("SM count          : %d\n", sm_count);
    std::printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    std::printf("thread configs    : 128, 256, 512, 1024\n");
    std::printf("ILP               : %d\n\n", ILP);

    if (prop.maxThreadsPerBlock < 128) {
        std::fprintf(stderr,
            "ERROR: device supports only %d threads/block; need at least 128.\n",
            prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    probe_observed_sms(sm_count);
    std::printf("\n");

    // Optional CLI argument: HBM working set in MiB.
    size_t hbm_ws = HBM_TOTAL_BYTES;
    if (argc >= 2) {
        const unsigned long long mib = std::strtoull(argv[1], nullptr, 10);
        if (mib > 0) hbm_ws = static_cast<size_t>(mib) * MB;
    }

    std::vector<Result> results;

    for (int threads : THREAD_CONFIGS) {
        if (threads > prop.maxThreadsPerBlock) {
            std::printf(
                "Skip %d threads/block: device maxThreadsPerBlock=%d\n",
                threads,
                prop.maxThreadsPerBlock);
            continue;
        }

        std::printf("\n============================================================\n");
        std::printf("Testing threads/block = %d\n", threads);
        std::printf("============================================================\n");

        // L1:
        // one 16KB private slice per SM; ld.global.ca.
        Result l1 = run_cache_test<true, true>(
            "L1",
            "ld.global.ca",
            sm_count,
            threads,
            0,
            L1_SLICE_BYTES,
            L1_REPEAT);
        results.push_back(l1);
        print_result(l1);

        // L2:
        // 32MB total unique working set, split across all active blocks;
        // ld.global.cg bypasses L1.
        Result l2 = run_cache_test<false, false>(
            "L2",
            "ld.global.cg",
            sm_count,
            threads,
            L2_TOTAL_BYTES,
            0,
            L2_REPEAT);
        results.push_back(l2);
        print_result(l2);

        // HBM:
        // 1GB default streaming set, 4 blocks/SM.
        Result hbm = run_hbm_test(
            sm_count,
            threads,
            hbm_ws,
            HBM_REPEAT);
        results.push_back(hbm);
        print_result(hbm);
    }

    std::printf("\nResults\n");
    std::printf("-------\n");
    for (const auto& r : results) {
        print_result(r);
    }

    std::ofstream csv("h20_memory_bandwidth_thread_sweep.csv");
    csv << "level,load_policy,blocks,threads_per_block,"
           "working_set_bytes,repeat,time_ms,bytes_read,bandwidth_GBps\n";

    for (const auto& r : results) {
        csv << r.level << ","
            << r.load_policy << ","
            << r.blocks << ","
            << r.threads << ","
            << r.working_set_bytes << ","
            << r.repeat << ","
            << r.time_ms << ","
            << r.bytes_read << ","
            << r.bandwidth_gbps << "\n";
    }

    std::printf("\nSaved: h20_memory_bandwidth_thread_sweep.csv\n");
    std::printf("Tip: run './h20_bw 2048' to use a 2GiB HBM working set.\n");

    return 0;
}
