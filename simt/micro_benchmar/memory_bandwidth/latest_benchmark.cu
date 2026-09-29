// simt_hbm_bandwidth.cu
//
// CUDA implementation of the provided SIMT HBM access pattern.
//
// Core access rule:
//   idx = block_start + i * blockDim.x + threadIdx.x
//
// Therefore, for every load instruction, adjacent threads access adjacent
// elements, which is suitable for coalesced global-memory/HBM traffic.
//
// Default sweep:
//   threads/block = 128, 256, 512, 1024
//   blocks          = SM_count * 4
//   total x+y WS    = 2048 MiB by default
//   kernel repeats  = 8
//   timing trials   = 5
//
// Build:
//   nvcc -O3 -lineinfo -arch=sm_90 simt_hbm_bandwidth.cu -o simt_hbm_bw
//
// Run:
//   ./simt_hbm_bw
//
// Optional:
//   ./simt_hbm_bw <total_read_working_set_MiB> <kernel_repeat> <trials> <blocks_per_sm>
//
// Example:
//   ./simt_hbm_bw 4096 8 7 4
//
// Output:
//   simt_hbm_bandwidth_results.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do {                                           \
    cudaError_t _cuda_err = (expr);                                     \
    if (_cuda_err != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                     __FILE__, __LINE__, cudaGetErrorString(_cuda_err)); \
        std::exit(EXIT_FAILURE);                                        \
    }                                                                   \
} while (0)

static constexpr size_t KiB = 1024ULL;
static constexpr size_t MiB = 1024ULL * 1024ULL;
static constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;

static constexpr int THREAD_CONFIGS[] = {128, 256, 512, 1024};
static constexpr int ILP = 8;

// -----------------------------------------------------------------------------
// Device kernels
// -----------------------------------------------------------------------------

__global__ void InitArray(float* p, size_t n, float value)
{
    size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        p[i] = value;
    }
}


// Based directly on the user's SIMT sample.
//
// Every block owns a disjoint contiguous region:
//
//   block_region = blockDim.x * data_num_per_thread elements
//
// In one step, thread 0..N-1 access:
//
//   base + 0
//   base + 1
//   ...
//   base + N-1
//
// so warp/lane accesses are contiguous.
//
// Eight independent accumulators are used to increase memory-level parallelism
// and reduce a single long arithmetic dependency chain.
__global__ void SimtHbmReadBw(const float* __restrict__ x,
                              const float* __restrict__ y,
                              float* __restrict__ z,
                              int data_num_per_thread,
                              int kernel_repeat)
{
    const size_t tid = static_cast<size_t>(threadIdx.x);
    const size_t threads = static_cast<size_t>(blockDim.x);

    const size_t start_idx =
        static_cast<size_t>(blockIdx.x) *
        threads *
        static_cast<size_t>(data_num_per_thread);

    float s0 = 1.0f;
    float s1 = 2.0f;
    float s2 = 3.0f;
    float s3 = 4.0f;
    float s4 = 5.0f;
    float s5 = 6.0f;
    float s6 = 7.0f;
    float s7 = 8.0f;

    for (int rep = 0; rep < kernel_repeat; ++rep) {

        int i = 0;

        for (; i + ILP <= data_num_per_thread; i += ILP) {

            const size_t base =
                start_idx +
                static_cast<size_t>(i) * threads +
                tid;

            // For each individual load below, adjacent threads access adjacent
            // float elements. This is the key coalescing property.
            s0 += x[base + 0 * threads] + y[base + 0 * threads];
            s1 += x[base + 1 * threads] + y[base + 1 * threads];
            s2 += x[base + 2 * threads] + y[base + 2 * threads];
            s3 += x[base + 3 * threads] + y[base + 3 * threads];
            s4 += x[base + 4 * threads] + y[base + 4 * threads];
            s5 += x[base + 5 * threads] + y[base + 5 * threads];
            s6 += x[base + 6 * threads] + y[base + 6 * threads];
            s7 += x[base + 7 * threads] + y[base + 7 * threads];
        }

        // Tail path if data_num_per_thread is not divisible by 8.
        for (; i < data_num_per_thread; ++i) {
            const size_t idx =
                start_idx +
                static_cast<size_t>(i) * threads +
                tid;

            s0 += x[idx] + y[idx];
        }
    }

    // Every thread writes one result so the compiler cannot eliminate the loads.
    const size_t out_idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    z[out_idx] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
}


// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

struct Result {
    int threads_per_block;
    int blocks;
    int blocks_per_sm;
    int sm_count;
    int data_num_per_thread;
    int kernel_repeat;
    int trials;

    size_t elements_per_array;
    size_t bytes_per_array;
    size_t read_bytes_per_kernel;

    double median_ms;
    double min_ms;
    double max_ms;

    double bandwidth_GBps;
    double bandwidth_GiBps;
};


static double Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n & 1) {
        return static_cast<double>(values[n / 2]);
    }

    return 0.5 *
        (static_cast<double>(values[n / 2 - 1]) +
         static_cast<double>(values[n / 2]));
}


static size_t AlignDown(size_t x, size_t a)
{
    return (x / a) * a;
}


static Result RunOneConfig(int sm_count,
                           int max_threads_per_block,
                           int threads,
                           int blocks_per_sm,
                           size_t requested_total_xy_bytes,
                           int kernel_repeat,
                           int trials)
{
    if (threads > max_threads_per_block) {
        std::fprintf(stderr,
            "Skip %d threads/block: device limit is %d\n",
            threads,
            max_threads_per_block);
        return {};
    }

    const int blocks = sm_count * blocks_per_sm;

    // requested_total_xy_bytes means x+y combined.
    //
    // x elements:
    //   blocks * threads * data_num_per_thread
    //
    // total input bytes:
    //   2 * elements_per_array * sizeof(float)
    //
    const size_t denominator =
        static_cast<size_t>(2) *
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads) *
        sizeof(float);

    size_t dpt =
        requested_total_xy_bytes / denominator;

    // Keep ILP loop clean.
    dpt = AlignDown(dpt, ILP);

    if (dpt < static_cast<size_t>(ILP)) {
        dpt = ILP;
    }

    if (dpt > static_cast<size_t>(INT32_MAX)) {
        std::fprintf(stderr,
            "data_num_per_thread is too large for int kernel argument.\n");
        std::exit(EXIT_FAILURE);
    }

    const int data_num_per_thread =
        static_cast<int>(dpt);

    const size_t elements_per_array =
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads) *
        static_cast<size_t>(data_num_per_thread);

    const size_t bytes_per_array =
        elements_per_array * sizeof(float);

    const size_t actual_total_xy_bytes =
        2ULL * bytes_per_array;

    const size_t z_elements =
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads);

    const size_t z_bytes =
        z_elements * sizeof(float);

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes_per_array));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_per_array));
    CUDA_CHECK(cudaMalloc(&d_z, z_bytes));

    // Initialize outside timed region.
    const int init_threads = 256;
    int init_blocks =
        static_cast<int>((elements_per_array + init_threads - 1) /
                         init_threads);

    // Avoid absurdly large init grids; grid-stride loop handles the rest.
    init_blocks = std::min(init_blocks, sm_count * 32);

    InitArray<<<init_blocks, init_threads>>>(
        d_x, elements_per_array, 1.0f);
    InitArray<<<init_blocks, init_threads>>>(
        d_y, elements_per_array, 2.0f);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup one full HBM pass.
    SimtHbmReadBw<<<blocks, threads>>>(
        d_x,
        d_y,
        d_z,
        data_num_per_thread,
        1);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;

    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    std::vector<float> times_ms;
    times_ms.reserve(trials);

    for (int t = 0; t < trials; ++t) {

        CUDA_CHECK(cudaEventRecord(start_evt));

        SimtHbmReadBw<<<blocks, threads>>>(
            d_x,
            d_y,
            d_z,
            data_num_per_thread,
            kernel_repeat);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop_evt));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));

        float ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms,
            start_evt,
            stop_evt));

        times_ms.push_back(ms);
    }

    const double median_ms =
        Median(times_ms);

    const double min_ms =
        *std::min_element(times_ms.begin(), times_ms.end());

    const double max_ms =
        *std::max_element(times_ms.begin(), times_ms.end());

    // x + y are both read once per logical iteration.
    const size_t read_bytes_per_kernel =
        actual_total_xy_bytes *
        static_cast<size_t>(kernel_repeat);

    const double seconds =
        median_ms * 1.0e-3;

    // Decimal GB/s, conventionally used for vendor memory bandwidth.
    const double bandwidth_GBps =
        static_cast<double>(read_bytes_per_kernel) /
        seconds /
        1.0e9;

    const double bandwidth_GiBps =
        static_cast<double>(read_bytes_per_kernel) /
        seconds /
        static_cast<double>(GiB);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    Result r{};

    r.threads_per_block = threads;
    r.blocks = blocks;
    r.blocks_per_sm = blocks_per_sm;
    r.sm_count = sm_count;
    r.data_num_per_thread = data_num_per_thread;
    r.kernel_repeat = kernel_repeat;
    r.trials = trials;

    r.elements_per_array = elements_per_array;
    r.bytes_per_array = bytes_per_array;
    r.read_bytes_per_kernel = read_bytes_per_kernel;

    r.median_ms = median_ms;
    r.min_ms = min_ms;
    r.max_ms = max_ms;

    r.bandwidth_GBps = bandwidth_GBps;
    r.bandwidth_GiBps = bandwidth_GiBps;

    return r;
}


static void PrintResult(const Result& r)
{
    if (r.threads_per_block == 0) {
        return;
    }

    const double array_MiB =
        static_cast<double>(r.bytes_per_array) /
        static_cast<double>(MiB);

    const double total_input_MiB =
        2.0 * array_MiB;

    const double timed_read_GB =
        static_cast<double>(r.read_bytes_per_kernel) /
        1.0e9;

    std::printf(
        "threads=%4d  blocks=%4d  dpt=%6d  "
        "x+y WS=%8.2f MiB  timedRead=%7.2f GB  "
        "median=%8.3f ms  BW=%9.2f GB/s\n",
        r.threads_per_block,
        r.blocks,
        r.data_num_per_thread,
        total_input_MiB,
        timed_read_GB,
        r.median_ms,
        r.bandwidth_GBps);
}


int main(int argc, char** argv)
{
    int device = 0;

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        device));

    size_t free_bytes = 0;
    size_t total_bytes = 0;

    CUDA_CHECK(cudaMemGetInfo(
        &free_bytes,
        &total_bytes));

    const int sm_count =
        prop.multiProcessorCount;

    size_t requested_total_xy_bytes =
        2048ULL * MiB;

    int kernel_repeat = 8;
    int trials = 5;
    int blocks_per_sm = 4;

    if (argc >= 2) {
        requested_total_xy_bytes =
            static_cast<size_t>(
                std::strtoull(argv[1], nullptr, 10)) *
            MiB;
    }

    if (argc >= 3) {
        kernel_repeat =
            std::max(1, std::atoi(argv[2]));
    }

    if (argc >= 4) {
        trials =
            std::max(1, std::atoi(argv[3]));
    }

    if (argc >= 5) {
        blocks_per_sm =
            std::max(1, std::atoi(argv[4]));
    }

    // x+y allocations dominate. Keep some free memory for runtime/context/z.
    const size_t safe_limit =
        static_cast<size_t>(
            static_cast<double>(free_bytes) * 0.70);

    if (requested_total_xy_bytes > safe_limit) {
        std::printf(
            "Requested x+y working set %.2f MiB exceeds 70%% of free GPU memory.\n"
            "Capping to %.2f MiB.\n",
            requested_total_xy_bytes / static_cast<double>(MiB),
            safe_limit / static_cast<double>(MiB));

        requested_total_xy_bytes =
            safe_limit;
    }

    std::printf("GPU                 : %s\n", prop.name);
    std::printf("SM count            : %d\n", sm_count);
    std::printf("maxThreadsPerBlock  : %d\n", prop.maxThreadsPerBlock);
    std::printf("global memory       : %.2f GiB\n",
                total_bytes / static_cast<double>(GiB));
    std::printf("free memory         : %.2f GiB\n",
                free_bytes / static_cast<double>(GiB));
    std::printf("target x+y WS       : %.2f MiB\n",
                requested_total_xy_bytes / static_cast<double>(MiB));
    std::printf("blocks/SM           : %d\n", blocks_per_sm);
    std::printf("kernel repeat       : %d\n", kernel_repeat);
    std::printf("timing trials       : %d\n\n", trials);

    std::vector<Result> results;

    for (int threads : THREAD_CONFIGS) {

        if (threads > prop.maxThreadsPerBlock) {
            std::printf(
                "Skip %d threads/block: unsupported by device.\n",
                threads);
            continue;
        }

        Result r =
            RunOneConfig(
                sm_count,
                prop.maxThreadsPerBlock,
                threads,
                blocks_per_sm,
                requested_total_xy_bytes,
                kernel_repeat,
                trials);

        results.push_back(r);

        PrintResult(r);
    }

    std::ofstream csv(
        "simt_hbm_bandwidth_results.csv");

    csv <<
        "gpu,sm_count,threads_per_block,blocks,blocks_per_sm,"
        "data_num_per_thread,kernel_repeat,trials,"
        "elements_per_array,bytes_per_array,"
        "read_bytes_per_timed_kernel,"
        "median_ms,min_ms,max_ms,"
        "bandwidth_GBps,bandwidth_GiBps\n";

    for (const Result& r : results) {

        csv <<
            '"' << prop.name << '"' << ","
            << r.sm_count << ","
            << r.threads_per_block << ","
            << r.blocks << ","
            << r.blocks_per_sm << ","
            << r.data_num_per_thread << ","
            << r.kernel_repeat << ","
            << r.trials << ","
            << r.elements_per_array << ","
            << r.bytes_per_array << ","
            << r.read_bytes_per_kernel << ","
            << r.median_ms << ","
            << r.min_ms << ","
            << r.max_ms << ","
            << r.bandwidth_GBps << ","
            << r.bandwidth_GiBps << "\n";
    }

    csv.close();

    std::printf(
        "\nSaved CSV: simt_hbm_bandwidth_results.csv\n");

    if (!results.empty()) {

        auto best =
            std::max_element(
                results.begin(),
                results.end(),
                [](const Result& a, const Result& b) {
                    return a.bandwidth_GBps <
                           b.bandwidth_GBps;
                });

        std::printf(
            "Best configuration: %d threads/block, %.2f GB/s\n",
            best->threads_per_block,
            best->bandwidth_GBps);
    }

    return 0;
}
