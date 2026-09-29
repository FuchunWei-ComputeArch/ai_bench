// sm90_z_eq_x_plus_y_bandwidth.cu
//
// NVIDIA Hopper / SM90 combined read+write bandwidth benchmark.
//
// Kernel operation:
//     z = x + y
//
// Each thread processes data_num_per_thread elements.  For each ILP lane,
// adjacent threads access adjacent float elements, preserving coalescing.
//
// Thread sweep:
//     128 / 256 / 512 / 1024 threads per block
//
// Runtime arguments:
//   argv[1] = array_data_MiB   : size of EACH of x/y/z arrays (default 1024)
//   argv[2] = kernel_repeat    : full-array repeats inside one kernel (default 8)
//   argv[3] = blocks_per_sm    : launched blocks per SM (default 4)
//   argv[4] = ilp              : 1/2/4/8/16 (default 2)
//   argv[5] = load_policy      : ca/cg or 1/0 (default cg)
//   argv[6] = store_policy     : wb/cg/cs/noalloc/wt or 0/1/2/3/4 (default cs)
//   argv[7] = trials           : timing trials, median reported (default 5)
//
// Example:
//   ./sm90_zxy_bw 1024 8 4 8 cg cs 5
//
// Build:
//   nvcc -O3 -lineinfo -arch=sm_90 sm90_z_eq_x_plus_y_bandwidth.cu -o sm90_zxy_bw
//
// Bandwidth accounting per full pass:
//   read bytes  = 2 * actual_array_bytes
//   write bytes = 1 * actual_array_bytes
//   total bytes = 3 * actual_array_bytes
//
// Timed bytes multiply the above by kernel_repeat.

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
// Policy definitions
// -----------------------------------------------------------------------------

enum class LoadPolicy : int {
    CG = 0,
    CA = 1
};

enum class StorePolicy : int {
    WB = 0,
    CG = 1,
    CS = 2,
    NOALLOC = 3,
    WT = 4
};

static const char* LoadPolicyName(LoadPolicy p)
{
    return p == LoadPolicy::CA ? "ca" : "cg";
}

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

static bool ParseLoadPolicy(const char* s, LoadPolicy* out)
{
    if (!s || !out) return false;

    if (std::strcmp(s, "ca") == 0 || std::strcmp(s, "1") == 0) {
        *out = LoadPolicy::CA;
        return true;
    }

    if (std::strcmp(s, "cg") == 0 || std::strcmp(s, "0") == 0) {
        *out = LoadPolicy::CG;
        return true;
    }

    return false;
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
// SM90 load/store helpers
// -----------------------------------------------------------------------------

template <LoadPolicy POLICY>
__device__ __forceinline__
float ld_global_policy_f32(const float* p)
{
    float v;

    if constexpr (POLICY == LoadPolicy::CA) {
        asm volatile(
            "ld.global.ca.f32 %0, [%1];"
            : "=f"(v)
            : "l"(p)
            : "memory");
    } else {
        asm volatile(
            "ld.global.cg.f32 %0, [%1];"
            : "=f"(v)
            : "l"(p)
            : "memory");
    }

    return v;
}

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
// Combined read + write kernel: z = x + y
//
// Addressing for ILP lane k:
//
//   idx = start_idx
//       + (i + k) * blockDim.x
//       + threadIdx.x
//
// Therefore, for a fixed instruction/lane k, warp threads touch consecutive
// float elements:
//
//   lane 0 -> idx + 0
//   lane 1 -> idx + 1
//   ...
//   lane31 -> idx + 31
//
// This gives coalesced x-load, y-load, and z-store streams.
// -----------------------------------------------------------------------------

template <LoadPolicy LP, StorePolicy SP, int ILP>
__global__ __launch_bounds__(1024)
void SimtZEqXPlusY(const float* __restrict__ x,
                   const float* __restrict__ y,
                   float* __restrict__ z,
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

    for (int rep = 0; rep < kernel_repeat; ++rep) {
        int i = 0;

        // Main ILP path.
        for (; i + ILP <= data_num_per_thread; i += ILP) {

            const size_t base =
                start_idx +
                static_cast<size_t>(i) * threads +
                tid;

            // Separate arrays preserve independent memory operations.
            float vx[ILP];
            float vy[ILP];

            #pragma unroll
            for (int k = 0; k < ILP; ++k) {
                const size_t idx =
                    base +
                    static_cast<size_t>(k) * threads;

                vx[k] = ld_global_policy_f32<LP>(x + idx);
                vy[k] = ld_global_policy_f32<LP>(y + idx);
            }

            #pragma unroll
            for (int k = 0; k < ILP; ++k) {
                const size_t idx =
                    base +
                    static_cast<size_t>(k) * threads;

                const float value = vx[k] + vy[k];

                st_global_policy_f32<SP>(
                    z + idx,
                    value);
            }
        }

        // Tail path; normally absent because host aligns dpt to ILP.
        for (; i < data_num_per_thread; ++i) {

            const size_t idx =
                start_idx +
                static_cast<size_t>(i) * threads +
                tid;

            const float a =
                ld_global_policy_f32<LP>(x + idx);

            const float b =
                ld_global_policy_f32<LP>(y + idx);

            st_global_policy_f32<SP>(
                z + idx,
                a + b);
        }
    }
}

// -----------------------------------------------------------------------------
// Launch dispatch
// -----------------------------------------------------------------------------

template <LoadPolicy LP, StorePolicy SP, int ILP>
static void LaunchKernel(int blocks,
                         int threads,
                         const float* d_x,
                         const float* d_y,
                         float* d_z,
                         int data_num_per_thread,
                         int kernel_repeat)
{
    SimtZEqXPlusY<LP, SP, ILP><<<blocks, threads>>>(
        d_x,
        d_y,
        d_z,
        data_num_per_thread,
        kernel_repeat);
}

template <LoadPolicy LP, StorePolicy SP>
static void DispatchILP(int ilp,
                        int blocks,
                        int threads,
                        const float* d_x,
                        const float* d_y,
                        float* d_z,
                        int data_num_per_thread,
                        int kernel_repeat)
{
    switch (ilp) {
        case 1:
            LaunchKernel<LP, SP, 1>(
                blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 2:
            LaunchKernel<LP, SP, 2>(
                blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 4:
            LaunchKernel<LP, SP, 4>(
                blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 8:
            LaunchKernel<LP, SP, 8>(
                blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case 16:
            LaunchKernel<LP, SP, 16>(
                blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        default:
            std::fprintf(stderr,
                "Unsupported ILP=%d. Supported: 1,2,4,8,16\n",
                ilp);
            std::exit(EXIT_FAILURE);
    }
}

template <LoadPolicy LP>
static void DispatchStorePolicy(StorePolicy sp,
                                int ilp,
                                int blocks,
                                int threads,
                                const float* d_x,
                                const float* d_y,
                                float* d_z,
                                int data_num_per_thread,
                                int kernel_repeat)
{
    switch (sp) {
        case StorePolicy::WB:
            DispatchILP<LP, StorePolicy::WB>(
                ilp, blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::CG:
            DispatchILP<LP, StorePolicy::CG>(
                ilp, blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::CS:
            DispatchILP<LP, StorePolicy::CS>(
                ilp, blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::NOALLOC:
            DispatchILP<LP, StorePolicy::NOALLOC>(
                ilp, blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;

        case StorePolicy::WT:
            DispatchILP<LP, StorePolicy::WT>(
                ilp, blocks, threads, d_x, d_y, d_z,
                data_num_per_thread, kernel_repeat);
            break;
    }
}

static void DispatchKernel(LoadPolicy lp,
                           StorePolicy sp,
                           int ilp,
                           int blocks,
                           int threads,
                           const float* d_x,
                           const float* d_y,
                           float* d_z,
                           int data_num_per_thread,
                           int kernel_repeat)
{
    if (lp == LoadPolicy::CA) {
        DispatchStorePolicy<LoadPolicy::CA>(
            sp, ilp, blocks, threads,
            d_x, d_y, d_z,
            data_num_per_thread,
            kernel_repeat);
    } else {
        DispatchStorePolicy<LoadPolicy::CG>(
            sp, ilp, blocks, threads,
            d_x, d_y, d_z,
            data_num_per_thread,
            kernel_repeat);
    }
}

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

__global__ void InitXY(float* x,
                       float* y,
                       size_t elements)
{
    const size_t tid =
        static_cast<size_t>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const size_t stride =
        static_cast<size_t>(gridDim.x) *
        blockDim.x;

    for (size_t i = tid; i < elements; i += stride) {
        x[i] = 1.0f + static_cast<float>(i & 255ULL) * 0.001f;
        y[i] = 2.0f + static_cast<float>(i & 127ULL) * 0.001f;
    }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

struct Result {
    LoadPolicy load_policy = LoadPolicy::CG;
    StorePolicy store_policy = StorePolicy::CS;

    int ilp = 0;
    int sm_count = 0;
    int threads_per_block = 0;
    int blocks_per_sm = 0;
    int blocks = 0;
    int data_num_per_thread = 0;
    int kernel_repeat = 0;
    int trials = 0;

    size_t requested_array_bytes = 0;
    size_t actual_array_bytes = 0;

    size_t timed_read_bytes = 0;
    size_t timed_write_bytes = 0;
    size_t timed_total_bytes = 0;

    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    double read_GBps = 0.0;
    double write_GBps = 0.0;
    double total_GBps = 0.0;
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

static int CalculateDataNumPerThread(size_t requested_array_bytes,
                                     int blocks,
                                     int threads,
                                     int ilp)
{
    const size_t denominator =
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads) *
        sizeof(float);

    size_t dpt =
        requested_array_bytes / denominator;

    // Keep the timed kernel entirely in the unrolled ILP path.
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
                           size_t requested_array_bytes,
                           int kernel_repeat,
                           int trials,
                           int ilp,
                           LoadPolicy load_policy,
                           StorePolicy store_policy)
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
            requested_array_bytes,
            blocks,
            threads,
            ilp);

    const size_t actual_elements =
        static_cast<size_t>(blocks) *
        static_cast<size_t>(threads) *
        static_cast<size_t>(data_num_per_thread);

    const size_t actual_array_bytes =
        actual_elements * sizeof(float);

    // Per repeat:
    //   x read = actual_array_bytes
    //   y read = actual_array_bytes
    //   z write = actual_array_bytes
    const size_t timed_read_bytes =
        2ULL *
        actual_array_bytes *
        static_cast<size_t>(kernel_repeat);

    const size_t timed_write_bytes =
        actual_array_bytes *
        static_cast<size_t>(kernel_repeat);

    const size_t timed_total_bytes =
        timed_read_bytes +
        timed_write_bytes;

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, actual_array_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, actual_array_bytes));
    CUDA_CHECK(cudaMalloc(&d_z, actual_array_bytes));

    // Initialize x/y outside the timed region.
    const int init_threads = 256;
    const int init_blocks =
        std::min(
            65535,
            std::max(
                1,
                static_cast<int>(
                    (actual_elements + init_threads - 1) /
                    init_threads)));

    InitXY<<<init_blocks, init_threads>>>(
        d_x,
        d_y,
        actual_elements);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Untimed warmup.
    DispatchKernel(
        load_policy,
        store_policy,
        ilp,
        blocks,
        threads,
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

    for (int trial = 0; trial < trials; ++trial) {

        CUDA_CHECK(cudaEventRecord(start_evt));

        DispatchKernel(
            load_policy,
            store_policy,
            ilp,
            blocks,
            threads,
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
        *std::min_element(
            times_ms.begin(),
            times_ms.end());

    const double max_ms =
        *std::max_element(
            times_ms.begin(),
            times_ms.end());

    const double seconds =
        median_ms * 1.0e-3;

    Result r;

    r.load_policy = load_policy;
    r.store_policy = store_policy;
    r.ilp = ilp;
    r.sm_count = sm_count;
    r.threads_per_block = threads;
    r.blocks_per_sm = blocks_per_sm;
    r.blocks = blocks;
    r.data_num_per_thread = data_num_per_thread;
    r.kernel_repeat = kernel_repeat;
    r.trials = trials;

    r.requested_array_bytes = requested_array_bytes;
    r.actual_array_bytes = actual_array_bytes;

    r.timed_read_bytes = timed_read_bytes;
    r.timed_write_bytes = timed_write_bytes;
    r.timed_total_bytes = timed_total_bytes;

    r.median_ms = median_ms;
    r.min_ms = min_ms;
    r.max_ms = max_ms;

    r.read_GBps =
        static_cast<double>(timed_read_bytes) /
        seconds /
        1.0e9;

    r.write_GBps =
        static_cast<double>(timed_write_bytes) /
        seconds /
        1.0e9;

    r.total_GBps =
        static_cast<double>(timed_total_bytes) /
        seconds /
        1.0e9;

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    return r;
}

static void PrintResult(const Result& r)
{
    if (r.threads_per_block == 0) {
        return;
    }

    std::printf(
        "LD=%-2s ST=%-7s ILP=%2d threads=%4d blocks=%4d blocks/SM=%2d "
        "dpt=%7d array=%8.2f MiB repeat=%4d "
        "median=%8.3f ms READ=%9.2f GB/s WRITE=%9.2f GB/s TOTAL=%9.2f GB/s\n",
        LoadPolicyName(r.load_policy),
        StorePolicyName(r.store_policy),
        r.ilp,
        r.threads_per_block,
        r.blocks,
        r.blocks_per_sm,
        r.data_num_per_thread,
        r.actual_array_bytes / static_cast<double>(MiB),
        r.kernel_repeat,
        r.median_ms,
        r.read_GBps,
        r.write_GBps,
        r.total_GBps);
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
    size_t requested_array_bytes =
        1024ULL * MiB;

    int kernel_repeat = 8;
    int blocks_per_sm = 4;
    int ilp = 2;
    LoadPolicy load_policy = LoadPolicy::CG;
    StorePolicy store_policy = StorePolicy::CS;
    int trials = 5;

    if (argc >= 2) {
        requested_array_bytes =
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
        ilp = std::atoi(argv[4]);

        if (!IsSupportedILP(ilp)) {
            std::fprintf(stderr,
                "ERROR: unsupported ILP=%d. "
                "Supported: 1,2,4,8,16.\n",
                ilp);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 6) {
        if (!ParseLoadPolicy(
                argv[5],
                &load_policy)) {
            std::fprintf(stderr,
                "ERROR: unknown load policy '%s'. "
                "Supported: ca, cg (or 1,0).\n",
                argv[5]);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 7) {
        if (!ParseStorePolicy(
                argv[6],
                &store_policy)) {
            std::fprintf(stderr,
                "ERROR: unknown store policy '%s'. "
                "Supported: wb,cg,cs,noalloc,wt "
                "(or 0,1,2,3,4).\n",
                argv[6]);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 8) {
        trials =
            std::max(1, std::atoi(argv[7]));
    }

    // Three arrays x/y/z are allocated.  Leave runtime/context headroom.
    const size_t safe_per_array =
        static_cast<size_t>(
            static_cast<double>(free_bytes) * 0.75 / 3.0);

    if (requested_array_bytes > safe_per_array) {
        std::printf(
            "Requested %.2f MiB per array is too large for current free memory; "
            "capping to %.2f MiB per array.\n",
            requested_array_bytes / static_cast<double>(MiB),
            safe_per_array / static_cast<double>(MiB));

        requested_array_bytes = safe_per_array;
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
    std::printf("array data requested : %.2f MiB EACH for x/y/z\n",
                requested_array_bytes / static_cast<double>(MiB));
    std::printf("total allocation req : %.2f MiB\n",
                3.0 * requested_array_bytes /
                static_cast<double>(MiB));
    std::printf("kernel repeat        : %d\n", kernel_repeat);
    std::printf("blocks/SM            : %d\n", blocks_per_sm);
    std::printf("ILP                  : %d\n", ilp);
    std::printf("load policy          : ld.global.%s\n",
                LoadPolicyName(load_policy));
    std::printf("store policy         : %s\n",
                StorePolicyName(store_policy));
    std::printf("trials               : %d\n\n",
                trials);

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
                requested_array_bytes,
                kernel_repeat,
                trials,
                ilp,
                load_policy,
                store_policy);

        results.push_back(r);
        PrintResult(r);
    }

    std::ofstream csv(
        "sm90_z_eq_x_plus_y_bandwidth_results.csv");

    csv <<
        "gpu,cc_major,cc_minor,load_policy,store_policy,ilp,"
        "sm_count,threads_per_block,blocks_per_sm,blocks,"
        "data_num_per_thread,kernel_repeat,trials,"
        "requested_array_bytes,actual_array_bytes,"
        "timed_read_bytes,timed_write_bytes,timed_total_bytes,"
        "median_ms,min_ms,max_ms,"
        "read_GBps,write_GBps,total_GBps\n";

    for (const Result& r : results) {
        csv <<
            '"' << prop.name << '"' << ","
            << prop.major << ","
            << prop.minor << ","
            << LoadPolicyName(r.load_policy) << ","
            << StorePolicyName(r.store_policy) << ","
            << r.ilp << ","
            << r.sm_count << ","
            << r.threads_per_block << ","
            << r.blocks_per_sm << ","
            << r.blocks << ","
            << r.data_num_per_thread << ","
            << r.kernel_repeat << ","
            << r.trials << ","
            << r.requested_array_bytes << ","
            << r.actual_array_bytes << ","
            << r.timed_read_bytes << ","
            << r.timed_write_bytes << ","
            << r.timed_total_bytes << ","
            << r.median_ms << ","
            << r.min_ms << ","
            << r.max_ms << ","
            << r.read_GBps << ","
            << r.write_GBps << ","
            << r.total_GBps << "\n";
    }

    csv.close();

    std::printf(
        "\nSaved CSV: sm90_z_eq_x_plus_y_bandwidth_results.csv\n");

    if (!results.empty()) {
        auto best =
            std::max_element(
                results.begin(),
                results.end(),
                [](const Result& a, const Result& b) {
                    return a.total_GBps <
                           b.total_GBps;
                });

        std::printf(
            "Best aggregate: %d threads/block -> "
            "READ %.2f GB/s, WRITE %.2f GB/s, TOTAL %.2f GB/s\n",
            best->threads_per_block,
            best->read_GBps,
            best->write_GBps,
            best->total_GBps);
    }

    return 0;
}
