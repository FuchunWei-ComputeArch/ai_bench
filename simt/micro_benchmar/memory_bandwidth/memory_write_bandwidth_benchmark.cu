// sm90_hbm_write_bandwidth_policy_ilp.cu
//
// NVIDIA Hopper / SM90 HBM write-bandwidth benchmark.
//
// Configurable:
//   - total write working set
//   - kernel repeat count
//   - blocks per SM
//   - global-store cache policy
//   - instruction-level parallelism (ILP)
//   - timing trials
//
// Thread sweep:
//   128 / 256 / 512 / 1024 threads per block
//
// Runtime arguments:
//   argv[1] = total_data_MiB   (default 1024)
//   argv[2] = kernel_repeat    (default 8)
//   argv[3] = blocks_per_sm    (default 4)
//   argv[4] = store_policy     (default cs)
//            wb | cg | cs | noalloc | wt
//            aliases: 0 | 1 | 2 | 3 | 4
//   argv[5] = ilp              (default 2)
//            supported: 1 | 2 | 4 | 8 | 16
//   argv[6] = trials           (default 5)
//
// Example:
//   ./sm90_hbm_write_bw 1024 8 4 cs 8 5
//
// Build:
//   nvcc -O3 -lineinfo -arch=sm_90 sm90_hbm_write_bandwidth_policy_ilp.cu -o sm90_hbm_write_bw
//
// Output:
//   sm90_hbm_write_bandwidth_policy_ilp_results.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

static constexpr size_t MiB = 1024ULL * 1024ULL;
static constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;

static constexpr int THREAD_CONFIGS[] = {128, 256, 512, 1024};

// -----------------------------------------------------------------------------
// Store policies
// -----------------------------------------------------------------------------

enum class StorePolicy : int {
    WB = 0,
    CG = 1,
    CS = 2,
    NOALLOC = 3,
    WT = 4
};

static const char* StorePolicyName(StorePolicy p)
{
    switch (p) {
        case StorePolicy::WB:      return "wb";
        case StorePolicy::CG:      return "cg";
        case StorePolicy::CS:      return "cs";
        case StorePolicy::NOALLOC: return "noalloc";
        case StorePolicy::WT:      return "wt";
    }
    return "unknown";
}

static bool ParseStorePolicy(const char* s, StorePolicy* out)
{
    if (!s || !out) return false;

    if (std::strcmp(s, "wb") == 0 || std::strcmp(s, "0") == 0) {
        *out = StorePolicy::WB;
        return true;
    }
    if (std::strcmp(s, "cg") == 0 || std::strcmp(s, "1") == 0) {
        *out = StorePolicy::CG;
        return true;
    }
    if (std::strcmp(s, "cs") == 0 || std::strcmp(s, "2") == 0) {
        *out = StorePolicy::CS;
        return true;
    }
    if (std::strcmp(s, "noalloc") == 0 ||
        std::strcmp(s, "no_allocate") == 0 ||
        std::strcmp(s, "3") == 0) {
        *out = StorePolicy::NOALLOC;
        return true;
    }
    if (std::strcmp(s, "wt") == 0 || std::strcmp(s, "4") == 0) {
        *out = StorePolicy::WT;
        return true;
    }

    return false;
}

static bool IsSupportedILP(int ilp)
{
    return ilp == 1 || ilp == 2 || ilp == 4 || ilp == 8 || ilp == 16;
}

// -----------------------------------------------------------------------------
// SM90 global-store helpers
// -----------------------------------------------------------------------------

template <StorePolicy POLICY>
__device__ __forceinline__
void st_global_policy_f32(float* p, float value)
{
    if constexpr (POLICY == StorePolicy::WB) {
        asm volatile(
            "st.global.wb.f32 [%0], %1;"
            :
            : "l"(p), "f"(value)
            : "memory");
    }
    else if constexpr (POLICY == StorePolicy::CG) {
        asm volatile(
            "st.global.cg.f32 [%0], %1;"
            :
            : "l"(p), "f"(value)
            : "memory");
    }
    else if constexpr (POLICY == StorePolicy::CS) {
        asm volatile(
            "st.global.cs.f32 [%0], %1;"
            :
            : "l"(p), "f"(value)
            : "memory");
    }
    else if constexpr (POLICY == StorePolicy::NOALLOC) {
        asm volatile(
            "st.global.L1::no_allocate.f32 [%0], %1;"
            :
            : "l"(p), "f"(value)
            : "memory");
    }
    else if constexpr (POLICY == StorePolicy::WT) {
        asm volatile(
            "st.global.wt.f32 [%0], %1;"
            :
            : "l"(p), "f"(value)
            : "memory");
    }
}

// -----------------------------------------------------------------------------
// Write-bandwidth kernel
//
// For a fixed ILP lane k:
//   thread 0 -> base + k*blockDim.x + 0
//   thread 1 -> base + k*blockDim.x + 1
//   ...
//   thread31 -> base + k*blockDim.x + 31
//
// Hence every store instruction remains warp-coalesced.
//
// ILP is a compile-time template parameter selected by host-side runtime
// dispatch. This lets the compiler fully unroll the independent stores.
// -----------------------------------------------------------------------------

template <StorePolicy POLICY, int ILP>
__global__ __launch_bounds__(1024)
void SimtHbmWriteBw(float* __restrict__ z,
                    int data_num_per_thread,
                    int kernel_repeat)
{
    static_assert(ILP >= 1, "ILP must be >= 1");

    const size_t tid =
        static_cast<size_t>(threadIdx.x);

    const size_t threads =
        static_cast<size_t>(blockDim.x);

    const size_t start_idx =
        static_cast<size_t>(blockIdx.x) *
        threads *
        static_cast<size_t>(data_num_per_thread);

    const float value_base =
        2.0f + static_cast<float>(threadIdx.x & 31) * 0.001f;

    for (int rep = 0; rep < kernel_repeat; ++rep) {

        int i = 0;

        // Main unrolled ILP loop.
        for (; i + ILP <= data_num_per_thread; i += ILP) {

            const size_t base =
                start_idx +
                static_cast<size_t>(i) * threads +
                tid;

            #pragma unroll
            for (int k = 0; k < ILP; ++k) {

                const size_t idx =
                    base +
                    static_cast<size_t>(k) * threads;

                const float value =
                    value_base + static_cast<float>(k) * 0.0001f;

                st_global_policy_f32<POLICY>(
                    z + idx,
                    value);
            }
        }

        // Tail path. In normal use data_num_per_thread is aligned to ILP,
        // so this usually executes zero iterations.
        for (; i < data_num_per_thread; ++i) {

            const size_t idx =
                start_idx +
                static_cast<size_t>(i) * threads +
                tid;

            st_global_policy_f32<POLICY>(
                z + idx,
                value_base);
        }
    }
}

// -----------------------------------------------------------------------------
// Host-side launch dispatch
// -----------------------------------------------------------------------------

template <StorePolicy POLICY, int ILP>
static void LaunchWriteKernel(int blocks,
                              int threads,
                              float* d_z,
                              int data_num_per_thread,
                              int kernel_repeat)
{
    SimtHbmWriteBw<POLICY, ILP><<<blocks, threads>>>(
        d_z,
        data_num_per_thread,
        kernel_repeat);
}

template <StorePolicy POLICY>
static void DispatchILP(int ilp,
                        int blocks,
                        int threads,
                        float* d_z,
                        int data_num_per_thread,
                        int kernel_repeat)
{
    switch (ilp) {
        case 1:
            LaunchWriteKernel<POLICY, 1>(
                blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 2:
            LaunchWriteKernel<POLICY, 2>(
                blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 4:
            LaunchWriteKernel<POLICY, 4>(
                blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 8:
            LaunchWriteKernel<POLICY, 8>(
                blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 16:
            LaunchWriteKernel<POLICY, 16>(
                blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        default:
            std::fprintf(stderr,
                "Unsupported ILP=%d. Supported: 1,2,4,8,16\n",
                ilp);
            std::exit(EXIT_FAILURE);
    }
}

static void DispatchWriteKernel(StorePolicy policy,
                                int ilp,
                                int blocks,
                                int threads,
                                float* d_z,
                                int data_num_per_thread,
                                int kernel_repeat)
{
    switch (policy) {
        case StorePolicy::WB:
            DispatchILP<StorePolicy::WB>(
                ilp, blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::CG:
            DispatchILP<StorePolicy::CG>(
                ilp, blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::CS:
            DispatchILP<StorePolicy::CS>(
                ilp, blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::NOALLOC:
            DispatchILP<StorePolicy::NOALLOC>(
                ilp, blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::WT:
            DispatchILP<StorePolicy::WT>(
                ilp, blocks, threads, d_z,
                data_num_per_thread, kernel_repeat);
            break;
    }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

struct Result {
    StorePolicy policy = StorePolicy::CS;

    int ilp = 0;
    int sm_count = 0;
    int threads_per_block = 0;
    int blocks_per_sm = 0;
    int blocks = 0;
    int data_num_per_thread = 0;
    int kernel_repeat = 0;
    int trials = 0;

    size_t requested_data_bytes = 0;
    size_t actual_data_bytes = 0;
    size_t timed_write_bytes = 0;

    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    double bandwidth_GBps = 0.0;
    double bandwidth_GiBps = 0.0;
};

static double Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n & 1U) {
        return static_cast<double>(values[n / 2]);
    }

    return 0.5 *
        (static_cast<double>(values[n / 2 - 1]) +
         static_cast<double>(values[n / 2]));
}

static int CalculateDataNumPerThread(size_t requested_data_bytes,
                                     int blocks,
                                     int threads,
                                     int ilp)
{
    const size_t denominator =
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads) *
        sizeof(float);

    size_t dpt =
        requested_data_bytes / denominator;

    // Align to ILP so the hot loop is entirely the unrolled ILP path.
    dpt =
        (dpt / static_cast<size_t>(ilp)) *
        static_cast<size_t>(ilp);

    if (dpt < static_cast<size_t>(ilp)) {
        dpt = static_cast<size_t>(ilp);
    }

    if (dpt > static_cast<size_t>(INT32_MAX)) {
        std::fprintf(stderr,
            "data_num_per_thread exceeds int range.\n");
        std::exit(EXIT_FAILURE);
    }

    return static_cast<int>(dpt);
}

static Result RunOneConfig(int sm_count,
                           int max_threads_per_block,
                           int threads,
                           int blocks_per_sm,
                           size_t requested_data_bytes,
                           int kernel_repeat,
                           int trials,
                           StorePolicy policy,
                           int ilp)
{
    if (threads > max_threads_per_block) {
        std::fprintf(stderr,
            "Skip %d threads/block: device max is %d.\n",
            threads,
            max_threads_per_block);
        return {};
    }

    const int blocks =
        sm_count * blocks_per_sm;

    const int data_num_per_thread =
        CalculateDataNumPerThread(
            requested_data_bytes,
            blocks,
            threads,
            ilp);

    const size_t actual_data_bytes =
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads) *
        static_cast<size_t>(data_num_per_thread) *
        sizeof(float);

    const size_t timed_write_bytes =
        actual_data_bytes *
        static_cast<size_t>(kernel_repeat);

    float* d_z = nullptr;

    CUDA_CHECK(cudaMalloc(&d_z, actual_data_bytes));

    // Untimed warmup.
    DispatchWriteKernel(
        policy,
        ilp,
        blocks,
        threads,
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

    for (int trial = 0; trial < trials; ++trial) {

        CUDA_CHECK(cudaEventRecord(start_evt));

        DispatchWriteKernel(
            policy,
            ilp,
            blocks,
            threads,
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
        *std::min_element(
            times_ms.begin(),
            times_ms.end());

    const double max_ms =
        *std::max_element(
            times_ms.begin(),
            times_ms.end());

    const double seconds =
        median_ms * 1.0e-3;

    const double bandwidth_GBps =
        static_cast<double>(timed_write_bytes) /
        seconds /
        1.0e9;

    const double bandwidth_GiBps =
        static_cast<double>(timed_write_bytes) /
        seconds /
        static_cast<double>(GiB);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    CUDA_CHECK(cudaFree(d_z));

    Result r;

    r.policy = policy;
    r.ilp = ilp;
    r.sm_count = sm_count;
    r.threads_per_block = threads;
    r.blocks_per_sm = blocks_per_sm;
    r.blocks = blocks;
    r.data_num_per_thread = data_num_per_thread;
    r.kernel_repeat = kernel_repeat;
    r.trials = trials;

    r.requested_data_bytes = requested_data_bytes;
    r.actual_data_bytes = actual_data_bytes;
    r.timed_write_bytes = timed_write_bytes;

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

    std::printf(
        "policy=%-7s ILP=%2d threads=%4d blocks=%4d blocks/SM=%2d "
        "dpt=%6d WS=%8.2f MiB repeat=%4d "
        "write=%8.2f GB median=%8.3f ms BW=%9.2f GB/s\n",
        StorePolicyName(r.policy),
        r.ilp,
        r.threads_per_block,
        r.blocks,
        r.blocks_per_sm,
        r.data_num_per_thread,
        r.actual_data_bytes / static_cast<double>(MiB),
        r.kernel_repeat,
        r.timed_write_bytes / 1.0e9,
        r.median_ms,
        r.bandwidth_GBps);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

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

    // Defaults.
    size_t requested_data_bytes =
        1024ULL * MiB;

    int kernel_repeat = 8;
    int blocks_per_sm = 4;
    StorePolicy policy = StorePolicy::CS;
    int ilp = 2;
    int trials = 5;

    if (argc >= 2) {
        requested_data_bytes =
            static_cast<size_t>(
                std::strtoull(argv[1], nullptr, 10)) *
            MiB;
    }

    if (argc >= 3) {
        kernel_repeat =
            std::max(1, std::atoi(argv[2]));
    }

    if (argc >= 4) {
        blocks_per_sm =
            std::max(1, std::atoi(argv[3]));
    }

    if (argc >= 5) {
        if (!ParseStorePolicy(argv[4], &policy)) {
            std::fprintf(stderr,
                "ERROR: unknown store policy '%s'.\n"
                "Supported: wb, cg, cs, noalloc, wt "
                "(or 0,1,2,3,4).\n",
                argv[4]);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 6) {
        ilp = std::atoi(argv[5]);

        if (!IsSupportedILP(ilp)) {
            std::fprintf(stderr,
                "ERROR: unsupported ILP=%d. "
                "Supported: 1,2,4,8,16.\n",
                ilp);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 7) {
        trials =
            std::max(1, std::atoi(argv[6]));
    }

    // Leave memory headroom for runtime/context.
    const size_t safe_limit =
        static_cast<size_t>(
            static_cast<double>(free_bytes) * 0.80);

    if (requested_data_bytes > safe_limit) {
        std::printf(
            "Requested %.2f MiB exceeds 80%% of free GPU memory; "
            "capping to %.2f MiB.\n",
            requested_data_bytes / static_cast<double>(MiB),
            safe_limit / static_cast<double>(MiB));

        requested_data_bytes = safe_limit;
    }

    const int sm_count =
        prop.multiProcessorCount;

    std::printf("GPU                  : %s\n", prop.name);
    std::printf("compute capability   : %d.%d\n", prop.major, prop.minor);
    std::printf("SM count             : %d\n", sm_count);
    std::printf("maxThreadsPerBlock   : %d\n", prop.maxThreadsPerBlock);
    std::printf("global memory        : %.2f GiB\n",
                total_bytes / static_cast<double>(GiB));
    std::printf("free memory          : %.2f GiB\n",
                free_bytes / static_cast<double>(GiB));
    std::printf("requested data       : %.2f MiB\n",
                requested_data_bytes / static_cast<double>(MiB));
    std::printf("kernel repeat        : %d\n", kernel_repeat);
    std::printf("blocks/SM            : %d\n", blocks_per_sm);
    std::printf("store policy         : %s\n",
                StorePolicyName(policy));
    std::printf("ILP                  : %d\n", ilp);
    std::printf("trials               : %d\n\n", trials);

    if (prop.major != 9) {
        std::fprintf(stderr,
            "WARNING: benchmark targets SM90/Hopper; "
            "detected compute capability %d.%d.\n\n",
            prop.major,
            prop.minor);
    }

    std::vector<Result> results;

    for (int threads : THREAD_CONFIGS) {

        if (threads > prop.maxThreadsPerBlock) {
            std::printf(
                "Skip %d threads/block: unsupported.\n",
                threads);
            continue;
        }

        Result r =
            RunOneConfig(
                sm_count,
                prop.maxThreadsPerBlock,
                threads,
                blocks_per_sm,
                requested_data_bytes,
                kernel_repeat,
                trials,
                policy,
                ilp);

        results.push_back(r);
        PrintResult(r);
    }

    std::ofstream csv(
        "sm90_hbm_write_bandwidth_policy_ilp_results.csv");

    csv <<
        "gpu,cc_major,cc_minor,store_policy,ilp,sm_count,"
        "threads_per_block,blocks_per_sm,blocks,"
        "data_num_per_thread,kernel_repeat,trials,"
        "requested_data_bytes,actual_data_bytes,"
        "timed_write_bytes,median_ms,min_ms,max_ms,"
        "bandwidth_GBps,bandwidth_GiBps\n";

    for (const Result& r : results) {
        csv <<
            '"' << prop.name << '"' << ","
            << prop.major << ","
            << prop.minor << ","
            << StorePolicyName(r.policy) << ","
            << r.ilp << ","
            << r.sm_count << ","
            << r.threads_per_block << ","
            << r.blocks_per_sm << ","
            << r.blocks << ","
            << r.data_num_per_thread << ","
            << r.kernel_repeat << ","
            << r.trials << ","
            << r.requested_data_bytes << ","
            << r.actual_data_bytes << ","
            << r.timed_write_bytes << ","
            << r.median_ms << ","
            << r.min_ms << ","
            << r.max_ms << ","
            << r.bandwidth_GBps << ","
            << r.bandwidth_GiBps << "\n";
    }

    csv.close();

    std::printf(
        "\nSaved CSV: "
        "sm90_hbm_write_bandwidth_policy_ilp_results.csv\n");

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
            "Best: policy=%s ILP=%d %d threads/block -> %.2f GB/s\n",
            StorePolicyName(policy),
            ilp,
            best->threads_per_block,
            best->bandwidth_GBps);
    }

    return 0;
}
