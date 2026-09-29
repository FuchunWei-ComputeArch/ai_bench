
// memory_bandwidth_benchmark_l2_ca_regenerated.cu
//
// Hierarchy assumption:
//   L1: 32 KB / SM
//   L2: 1 MB / 4 SM
//   L3: 32 MB / GPU
//   HBM: device memory
//
// Load policy:
//   L1  -> ld.global.ca
//   L2  -> ld.global.cg
//   L3  -> ld.global.cg
//   HBM -> ld.global.cg
//
// Benchmark threads/block:
//   L1/L2/L3/HBM -> 1024
//
// Threads:
//   L1: 1024 threads/block
//   L2: 256 threads/block
//   L3: 256 threads/block
//   HBM: 256 threads/block
//
// Build:
//   nvcc -O3 -lineinfo memory_bandwidth_benchmark_l2_ca_regenerated.cu -o memory_bw

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>

#define CUDA_CHECK(expr) do {                                      \
    cudaError_t _cuda_err = (expr);                                \
    if (_cuda_err != cudaSuccess) {                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                     __FILE__, __LINE__,                            \
                     cudaGetErrorString(_cuda_err));                \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

static constexpr size_t KB = 1024ULL;
static constexpr size_t MB = 1024ULL * 1024ULL;
static constexpr int ILP = 8;

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
// Probe actual SM IDs used by the device/runtime.
// Launch many blocks, record %smid, then count unique IDs on host.
// -----------------------------------------------------------------------------
__global__
void probe_smid_kernel(uint32_t* smids)
{
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = read_smid();
    }
}

static int probe_active_sm_count(int reported_sm_count)
{
    // Launch multiple blocks per reported SM so scheduler has enough work
    // to distribute blocks across the device.
    const int probe_blocks = reported_sm_count * 8;
    const int probe_threads = 32;

    uint32_t* d_smids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_smids,
                          static_cast<size_t>(probe_blocks) *
                          sizeof(uint32_t)));

    probe_smid_kernel<<<probe_blocks, probe_threads>>>(d_smids);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_smids(probe_blocks);
    CUDA_CHECK(cudaMemcpy(h_smids.data(),
                          d_smids,
                          static_cast<size_t>(probe_blocks) *
                          sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_smids));

    std::vector<int> seen(reported_sm_count, 0);

    for (uint32_t id : h_smids) {
        if (id < static_cast<uint32_t>(reported_sm_count)) {
            seen[id] = 1;
        }
    }

    int active_sm_count = 0;
    for (int v : seen) {
        active_sm_count += v;
    }

    std::printf("Reported SM count : %d\n", reported_sm_count);
    std::printf("Observed SM count : %d\n", active_sm_count);

    std::printf("Observed SM IDs   : ");
    for (int i = 0; i < reported_sm_count; ++i) {
        if (seen[i]) {
            std::printf("%d ", i);
        }
    }
    std::printf("\n\n");

    return active_sm_count;
}

template <bool USE_CA, int MODE>
__global__
void cache_bw_kernel(
    const uint32_t* __restrict__ src,
    size_t elements_per_slice,
    int num_slices,
    int repeat,
    unsigned long long* checksum)
{
    uint32_t smid = read_smid();

    int slice_id = 0;
    if constexpr (MODE == 0) {
        // L1: one independent slice per SM
        slice_id = static_cast<int>(smid) % num_slices;
    } else {
        // L2/L3: one slice per 4-SM group
        slice_id = static_cast<int>(smid / 4) % num_slices;
    }

    const uint32_t* base =
        src + static_cast<size_t>(slice_id) * elements_per_slice;

    size_t lane = threadIdx.x;
    size_t stride = blockDim.x;

    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t a4=0,a5=0,a6=0,a7=0;

    // Warmup
    for (size_t i = lane * ILP;
         i + ILP - 1 < elements_per_slice;
         i += stride * ILP)
    {
        if constexpr (USE_CA) {
            a0 ^= ld_ca_u32(base+i+0);
            a1 ^= ld_ca_u32(base+i+1);
            a2 ^= ld_ca_u32(base+i+2);
            a3 ^= ld_ca_u32(base+i+3);
            a4 ^= ld_ca_u32(base+i+4);
            a5 ^= ld_ca_u32(base+i+5);
            a6 ^= ld_ca_u32(base+i+6);
            a7 ^= ld_ca_u32(base+i+7);
        } else {
            a0 ^= ld_cg_u32(base+i+0);
            a1 ^= ld_cg_u32(base+i+1);
            a2 ^= ld_cg_u32(base+i+2);
            a3 ^= ld_cg_u32(base+i+3);
            a4 ^= ld_cg_u32(base+i+4);
            a5 ^= ld_cg_u32(base+i+5);
            a6 ^= ld_cg_u32(base+i+6);
            a7 ^= ld_cg_u32(base+i+7);
        }
    }

    __syncthreads();

    for (int r = 0; r < repeat; ++r) {
        for (size_t i = lane * ILP;
             i + ILP - 1 < elements_per_slice;
             i += stride * ILP)
        {
            if constexpr (USE_CA) {
                a0 ^= ld_ca_u32(base+i+0);
                a1 ^= ld_ca_u32(base+i+1);
                a2 ^= ld_ca_u32(base+i+2);
                a3 ^= ld_ca_u32(base+i+3);
                a4 ^= ld_ca_u32(base+i+4);
                a5 ^= ld_ca_u32(base+i+5);
                a6 ^= ld_ca_u32(base+i+6);
                a7 ^= ld_ca_u32(base+i+7);
            } else {
                a0 ^= ld_cg_u32(base+i+0);
                a1 ^= ld_cg_u32(base+i+1);
                a2 ^= ld_cg_u32(base+i+2);
                a3 ^= ld_cg_u32(base+i+3);
                a4 ^= ld_cg_u32(base+i+4);
                a5 ^= ld_cg_u32(base+i+5);
                a6 ^= ld_cg_u32(base+i+6);
                a7 ^= ld_cg_u32(base+i+7);
            }
        }
    }

    if (threadIdx.x == 0) {
        checksum[blockIdx.x] =
            static_cast<unsigned long long>(
                a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7);
    }
}

__global__
void hbm_bw_kernel(
    const uint32_t* __restrict__ src,
    size_t elements,
    unsigned long long* checksum)
{
    size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_threads =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    uint32_t a0=0,a1=0,a2=0,a3=0;
    uint32_t a4=0,a5=0,a6=0,a7=0;

    for (size_t i = tid * ILP;
         i + ILP - 1 < elements;
         i += total_threads * ILP)
    {
        a0 ^= ld_cg_u32(src+i+0);
        a1 ^= ld_cg_u32(src+i+1);
        a2 ^= ld_cg_u32(src+i+2);
        a3 ^= ld_cg_u32(src+i+3);
        a4 ^= ld_cg_u32(src+i+4);
        a5 ^= ld_cg_u32(src+i+5);
        a6 ^= ld_cg_u32(src+i+6);
        a7 ^= ld_cg_u32(src+i+7);
    }

    if (threadIdx.x == 0) {
        checksum[blockIdx.x] =
            static_cast<unsigned long long>(
                a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7);
    }
}

template <bool USE_CA, int MODE>
static void run_cache_test(
    const char* name,
    int blocks,
    int threads,
    size_t ws_per_slice,
    int num_slices,
    int repeat)
{
    size_t total_ws =
        ws_per_slice * static_cast<size_t>(num_slices);

    uint32_t* d_src = nullptr;
    unsigned long long* d_checksum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, total_ws));
    CUDA_CHECK(cudaMalloc(
        &d_checksum,
        static_cast<size_t>(blocks) * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_src, 1, total_ws));

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    size_t elements_per_slice =
        ws_per_slice / sizeof(uint32_t);

    // Untimed warmup launch
    cache_bw_kernel<USE_CA, MODE><<<blocks, threads>>>(
        d_src,
        elements_per_slice,
        num_slices,
        1,
        d_checksum);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_evt));

    cache_bw_kernel<USE_CA, MODE><<<blocks, threads>>>(
        d_src,
        elements_per_slice,
        num_slices,
        repeat,
        d_checksum);

    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));

    double bytes_read =
        static_cast<double>(blocks) *
        static_cast<double>(ws_per_slice) *
        static_cast<double>(repeat);

    double bw_gbps =
        bytes_read / (static_cast<double>(ms) * 1e-3) / 1e9;

    std::printf(
        "%-6s blocks=%d threads=%d slice=%.2f KB totalWS=%.2f MB "
        "time=%.3f ms BW=%.2f GB/s\n",
        name,
        blocks,
        threads,
        ws_per_slice / 1024.0,
        total_ws / 1024.0 / 1024.0,
        ms,
        bw_gbps);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_checksum));
}

static void run_hbm_test(
    int blocks,
    int threads,
    size_t total_ws)
{
    uint32_t* d_src = nullptr;
    unsigned long long* d_checksum = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src, total_ws));
    CUDA_CHECK(cudaMalloc(
        &d_checksum,
        static_cast<size_t>(blocks) * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_src, 1, total_ws));

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    size_t elements =
        total_ws / sizeof(uint32_t);

    hbm_bw_kernel<<<blocks, threads>>>(
        d_src,
        elements,
        d_checksum);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_evt));

    hbm_bw_kernel<<<blocks, threads>>>(
        d_src,
        elements,
        d_checksum);

    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));

    double bw_gbps =
        static_cast<double>(total_ws) /
        (static_cast<double>(ms) * 1e-3) /
        1e9;

    std::printf(
        "HBM    blocks=%d threads=%d totalWS=%.2f MB "
        "time=%.3f ms BW=%.2f GB/s\n",
        blocks,
        threads,
        total_ws / 1024.0 / 1024.0,
        ms,
        bw_gbps);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_checksum));
}

int main()
{
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int reported_sm_count = prop.multiProcessorCount;

    std::printf("GPU: %s\n", prop.name);

    if (prop.maxThreadsPerBlock < 1024) {
        std::fprintf(stderr,
            "ERROR: maxThreadsPerBlock=%d, but benchmark requires 1024 threads/block.\n",
            prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    // First probe/collect actual SM IDs observed by a scheduling probe.
    int observed_sm_count =
        probe_active_sm_count(reported_sm_count);

    // Normally this should equal multiProcessorCount.
    // Fall back to the reported value if the probe result is unexpected.
    const int sm_count =
        (observed_sm_count > 0)
            ? observed_sm_count
            : reported_sm_count;

    const int l2_groups = (sm_count + 3) / 4;

    std::printf("SM count used by benchmark: %d\n", sm_count);
    std::printf("L2 groups: %d\n\n", l2_groups);

    // L1:
    // 16 KB per SM, ld.global.ca, 1024 threads/block
    run_cache_test<true, 0>(
        "L1-ca",
        sm_count,
        1024,
        16 * KB,
        sm_count,
        200);

    // L2:
    // 512 KB per 4-SM group, ld.global.cg
    run_cache_test<false, 1>(
        "L2-cg",
        sm_count,
        1024,
        512 * KB,
        l2_groups,
        100);

    // L3:
    // 2 MB per 4-SM group, ld.global.cg
    // limit total working set to <= 16 MB
    int l3_groups = l2_groups;
    while (static_cast<size_t>(l3_groups) * 2 * MB > 16 * MB) {
        --l3_groups;
    }
    if (l3_groups < 1) l3_groups = 1;

    run_cache_test<false, 2>(
        "L3-cg",
        sm_count,
        1024,
        2 * MB,
        l3_groups,
        40);

    // HBM:
    // working set >> 32MB L3
    run_hbm_test(
        sm_count * 4,
        1024,
        256 * MB);

    return 0;
}

