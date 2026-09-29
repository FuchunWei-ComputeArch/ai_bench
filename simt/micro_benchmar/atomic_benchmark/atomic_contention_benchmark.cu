// sm90_atomic_contention_benchmark.cu
//
// NVIDIA Hopper / SM90 atomic contention benchmark.
//
// Purpose:
//   Measure throughput when many SIMT threads perform atomic operations on the
//   same global-memory address.
//
// Supported PTX operation classes:
//   ATOM : atom.global.add.u32  (returns old value)
//   RED  : red.global.add.u32   (does not return old value)
//
// Default contention:
//   hot_addresses = 1
//   => every thread in every block updates counters[0].
//
// Thread sweep:
//   128 / 256 / 512 / 1024 threads per block
//
// Runtime arguments:
//   argv[1] = atomic_type            atom | red | 0 | 1   (default red)
//   argv[2] = atomic_ops_per_thread  operations/thread/repeat (default 1024)
//   argv[3] = kernel_repeat          full operation repeats (default 8)
//   argv[4] = blocks_per_sm          launched blocks per SM (default 4)
//   argv[5] = ilp                    1/2/4/8/16 (default 2)
//   argv[6] = hot_addresses          number of contended counters (default 1)
//   argv[7] = trials                 timing trials (default 5)
//
// Example: all threads contend on one global address
//   ./sm90_atomic_bw red 1024 8 4 4 1 5
//   ./sm90_atomic_bw atom 1024 8 4 4 1 5
//
// Build:
//   nvcc -O3 -lineinfo -arch=sm_90 sm90_atomic_contention_benchmark.cu -o sm90_atomic_bw
//
// Main metric:
//   GAtomicOps/s = total atomic operations / kernel time / 1e9
//
// Important:
//   This is an atomic-operation throughput benchmark, not ordinary memory
//   bandwidth.  With hot_addresses=1, serialization at the destination is
//   intentionally the dominant effect.

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

static constexpr int THREAD_CONFIGS[] = {128, 256, 512, 1024};

// -----------------------------------------------------------------------------
// Atomic type
// -----------------------------------------------------------------------------

enum class AtomicType : int {
    ATOM = 0,
    RED  = 1
};

static const char* AtomicTypeName(AtomicType t)
{
    return t == AtomicType::ATOM ? "atom" : "red";
}

static bool ParseAtomicType(const char* s, AtomicType* out)
{
    if (!s || !out) return false;

    if (std::strcmp(s, "atom") == 0 ||
        std::strcmp(s, "ATOM") == 0 ||
        std::strcmp(s, "0") == 0) {
        *out = AtomicType::ATOM;
        return true;
    }

    if (std::strcmp(s, "red") == 0 ||
        std::strcmp(s, "RED") == 0 ||
        std::strcmp(s, "1") == 0) {
        *out = AtomicType::RED;
        return true;
    }

    return false;
}

static bool IsSupportedILP(int ilp)
{
    return ilp == 1 ||
           ilp == 2 ||
           ilp == 4 ||
           ilp == 8 ||
           ilp == 16;
}

// -----------------------------------------------------------------------------
// PTX atomic helpers
// -----------------------------------------------------------------------------

template <AtomicType TYPE>
__device__ __forceinline__
uint32_t atomic_add_u32(uint32_t* p, uint32_t value)
{
    if constexpr (TYPE == AtomicType::ATOM) {
        uint32_t old;

        asm volatile(
            "atom.global.add.u32 %0, [%1], %2;"
            : "=r"(old)
            : "l"(p), "r"(value)
            : "memory");

        return old;
    }
    else {
        asm volatile(
            "red.global.add.u32 [%0], %1;"
            :
            : "l"(p), "r"(value)
            : "memory");

        // RED has no return value.
        return 0;
    }
}

// -----------------------------------------------------------------------------
// Contention kernel
//
// For hot_addresses == 1:
//     every thread -> counters[0]
//
// For hot_addresses > 1:
//     threads are distributed across several hot counters:
//
//     hot_index = global_thread_id % hot_addresses
//
// This allows contention scaling studies while preserving the same kernel.
// -----------------------------------------------------------------------------

template <AtomicType TYPE, int ILP>
__global__ __launch_bounds__(1024)
void AtomicContentionKernel(uint32_t* __restrict__ counters,
                            uint32_t* __restrict__ atom_sink,
                            int atomic_ops_per_thread,
                            int kernel_repeat,
                            int hot_addresses)
{
    static_assert(ILP >= 1, "ILP must be >= 1");

    const uint64_t global_tid =
        static_cast<uint64_t>(blockIdx.x) *
        static_cast<uint64_t>(blockDim.x) +
        static_cast<uint64_t>(threadIdx.x);

    const uint32_t hot_index =
        static_cast<uint32_t>(
            global_tid %
            static_cast<uint64_t>(hot_addresses));

    uint32_t* target =
        counters + hot_index;

    // ATOM returns old values.  We keep independent sink accumulators so the
    // returned values remain architecturally consumed without creating one
    // long dependency chain across every atomic instruction.
    uint32_t sink[ILP];

    #pragma unroll
    for (int k = 0; k < ILP; ++k) {
        sink[k] = 0;
    }

    for (int rep = 0; rep < kernel_repeat; ++rep) {

        int i = 0;

        // Main ILP path.
        for (; i + ILP <= atomic_ops_per_thread; i += ILP) {

            #pragma unroll
            for (int k = 0; k < ILP; ++k) {

                const uint32_t old =
                    atomic_add_u32<TYPE>(
                        target,
                        1u);

                if constexpr (TYPE == AtomicType::ATOM) {
                    sink[k] += old;
                }
            }
        }

        // Tail path. Normally absent because host aligns operation count to ILP.
        for (; i < atomic_ops_per_thread; ++i) {

            const uint32_t old =
                atomic_add_u32<TYPE>(
                    target,
                    1u);

            if constexpr (TYPE == AtomicType::ATOM) {
                sink[0] += old;
            }
        }
    }

    // Only ATOM has returned values.  Write one sink per thread after the
    // timed atomic loop so the compiler cannot discard the return-data path.
    if constexpr (TYPE == AtomicType::ATOM) {

        uint32_t total = 0;

        #pragma unroll
        for (int k = 0; k < ILP; ++k) {
            total += sink[k];
        }

        atom_sink[global_tid] = total;
    }
}

// -----------------------------------------------------------------------------
// Launch dispatch
// -----------------------------------------------------------------------------

template <AtomicType TYPE, int ILP>
static void LaunchAtomicKernel(int blocks,
                               int threads,
                               uint32_t* d_counters,
                               uint32_t* d_atom_sink,
                               int atomic_ops_per_thread,
                               int kernel_repeat,
                               int hot_addresses)
{
    AtomicContentionKernel<TYPE, ILP><<<blocks, threads>>>(
        d_counters,
        d_atom_sink,
        atomic_ops_per_thread,
        kernel_repeat,
        hot_addresses);
}

template <AtomicType TYPE>
static void DispatchILP(int ilp,
                        int blocks,
                        int threads,
                        uint32_t* d_counters,
                        uint32_t* d_atom_sink,
                        int atomic_ops_per_thread,
                        int kernel_repeat,
                        int hot_addresses)
{
    switch (ilp) {
        case 1:
            LaunchAtomicKernel<TYPE, 1>(
                blocks, threads, d_counters, d_atom_sink,
                atomic_ops_per_thread, kernel_repeat, hot_addresses);
            break;

        case 2:
            LaunchAtomicKernel<TYPE, 2>(
                blocks, threads, d_counters, d_atom_sink,
                atomic_ops_per_thread, kernel_repeat, hot_addresses);
            break;

        case 4:
            LaunchAtomicKernel<TYPE, 4>(
                blocks, threads, d_counters, d_atom_sink,
                atomic_ops_per_thread, kernel_repeat, hot_addresses);
            break;

        case 8:
            LaunchAtomicKernel<TYPE, 8>(
                blocks, threads, d_counters, d_atom_sink,
                atomic_ops_per_thread, kernel_repeat, hot_addresses);
            break;

        case 16:
            LaunchAtomicKernel<TYPE, 16>(
                blocks, threads, d_counters, d_atom_sink,
                atomic_ops_per_thread, kernel_repeat, hot_addresses);
            break;

        default:
            std::fprintf(
                stderr,
                "Unsupported ILP=%d. Supported: 1,2,4,8,16\n",
                ilp);
            std::exit(EXIT_FAILURE);
    }
}

static void DispatchAtomicKernel(AtomicType type,
                                 int ilp,
                                 int blocks,
                                 int threads,
                                 uint32_t* d_counters,
                                 uint32_t* d_atom_sink,
                                 int atomic_ops_per_thread,
                                 int kernel_repeat,
                                 int hot_addresses)
{
    if (type == AtomicType::ATOM) {
        DispatchILP<AtomicType::ATOM>(
            ilp,
            blocks,
            threads,
            d_counters,
            d_atom_sink,
            atomic_ops_per_thread,
            kernel_repeat,
            hot_addresses);
    }
    else {
        DispatchILP<AtomicType::RED>(
            ilp,
            blocks,
            threads,
            d_counters,
            d_atom_sink,
            atomic_ops_per_thread,
            kernel_repeat,
            hot_addresses);
    }
}

// -----------------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------------

struct Result {
    AtomicType atomic_type = AtomicType::RED;

    int ilp = 0;
    int sm_count = 0;
    int threads_per_block = 0;
    int blocks_per_sm = 0;
    int blocks = 0;
    int atomic_ops_per_thread = 0;
    int kernel_repeat = 0;
    int hot_addresses = 0;
    int trials = 0;

    uint64_t total_threads = 0;
    uint64_t total_atomic_ops = 0;

    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;

    double GAtomicOps = 0.0;
    double ns_per_atomic_effective = 0.0;

    uint32_t counter0 = 0;
};

static double Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n & 1U) {
        return static_cast<double>(
            values[n / 2]);
    }

    return 0.5 *
        (static_cast<double>(values[n / 2 - 1]) +
         static_cast<double>(values[n / 2]));
}

static int AlignOpsToILP(int requested_ops,
                         int ilp)
{
    int ops =
        (requested_ops / ilp) * ilp;

    if (ops < ilp) {
        ops = ilp;
    }

    return ops;
}

static Result RunOneConfig(int sm_count,
                           int max_threads_per_block,
                           int threads,
                           int blocks_per_sm,
                           int atomic_ops_per_thread,
                           int kernel_repeat,
                           int ilp,
                           int hot_addresses,
                           int trials,
                           AtomicType atomic_type)
{
    if (threads > max_threads_per_block) {
        std::fprintf(
            stderr,
            "Skip %d threads/block: device max is %d.\n",
            threads,
            max_threads_per_block);
        return {};
    }

    const int blocks =
        sm_count * blocks_per_sm;

    const uint64_t total_threads =
        static_cast<uint64_t>(blocks) *
        static_cast<uint64_t>(threads);

    atomic_ops_per_thread =
        AlignOpsToILP(
            atomic_ops_per_thread,
            ilp);

    const uint64_t total_atomic_ops =
        total_threads *
        static_cast<uint64_t>(atomic_ops_per_thread) *
        static_cast<uint64_t>(kernel_repeat);

    uint32_t* d_counters = nullptr;
    uint32_t* d_atom_sink = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_counters,
        static_cast<size_t>(hot_addresses) *
        sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(
        &d_atom_sink,
        static_cast<size_t>(total_threads) *
        sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(
        d_counters,
        0,
        static_cast<size_t>(hot_addresses) *
        sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(
        d_atom_sink,
        0,
        static_cast<size_t>(total_threads) *
        sizeof(uint32_t)));

    // Warmup.
    //
    // Use a small operation count to warm the execution path without adding a
    // huge offset to the counters.
    DispatchAtomicKernel(
        atomic_type,
        ilp,
        blocks,
        threads,
        d_counters,
        d_atom_sink,
        ilp,
        1,
        hot_addresses);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;

    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    std::vector<float> times_ms;
    times_ms.reserve(trials);

    uint32_t last_counter0 = 0;

    for (int trial = 0; trial < trials; ++trial) {

        // Reset outside timed region.
        CUDA_CHECK(cudaMemset(
            d_counters,
            0,
            static_cast<size_t>(hot_addresses) *
            sizeof(uint32_t)));

        CUDA_CHECK(cudaEventRecord(start_evt));

        DispatchAtomicKernel(
            atomic_type,
            ilp,
            blocks,
            threads,
            d_counters,
            d_atom_sink,
            atomic_ops_per_thread,
            kernel_repeat,
            hot_addresses);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop_evt));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));

        float ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms,
            start_evt,
            stop_evt));

        times_ms.push_back(ms);

        // Validation read is outside timed region.
        CUDA_CHECK(cudaMemcpy(
            &last_counter0,
            d_counters,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
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

    const double GAtomicOps =
        static_cast<double>(total_atomic_ops) /
        seconds /
        1.0e9;

    // This is reciprocal aggregate throughput, not single-operation latency.
    const double ns_per_atomic_effective =
        median_ms * 1.0e6 /
        static_cast<double>(total_atomic_ops);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    CUDA_CHECK(cudaFree(d_counters));
    CUDA_CHECK(cudaFree(d_atom_sink));

    Result r;

    r.atomic_type = atomic_type;
    r.ilp = ilp;
    r.sm_count = sm_count;
    r.threads_per_block = threads;
    r.blocks_per_sm = blocks_per_sm;
    r.blocks = blocks;
    r.atomic_ops_per_thread = atomic_ops_per_thread;
    r.kernel_repeat = kernel_repeat;
    r.hot_addresses = hot_addresses;
    r.trials = trials;

    r.total_threads = total_threads;
    r.total_atomic_ops = total_atomic_ops;

    r.median_ms = median_ms;
    r.min_ms = min_ms;
    r.max_ms = max_ms;

    r.GAtomicOps = GAtomicOps;
    r.ns_per_atomic_effective = ns_per_atomic_effective;
    r.counter0 = last_counter0;

    return r;
}

static void PrintResult(const Result& r)
{
    if (r.threads_per_block == 0) {
        return;
    }

    std::printf(
        "type=%-4s ILP=%2d threads=%4d blocks=%4d blocks/SM=%2d "
        "hot=%3d ops/thread=%6d repeat=%4d "
        "totalOps=%12llu median=%9.3f ms "
        "throughput=%9.3f GAtomicOps/s "
        "effective=%9.4f ns/op counter0=%u\n",
        AtomicTypeName(r.atomic_type),
        r.ilp,
        r.threads_per_block,
        r.blocks,
        r.blocks_per_sm,
        r.hot_addresses,
        r.atomic_ops_per_thread,
        r.kernel_repeat,
        static_cast<unsigned long long>(
            r.total_atomic_ops),
        r.median_ms,
        r.GAtomicOps,
        r.ns_per_atomic_effective,
        r.counter0);
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

    AtomicType atomic_type =
        AtomicType::RED;

    int atomic_ops_per_thread = 1024;
    int kernel_repeat = 8;
    int blocks_per_sm = 4;
    int ilp = 2;
    int hot_addresses = 1;
    int trials = 5;

    if (argc >= 2) {
        if (!ParseAtomicType(
                argv[1],
                &atomic_type)) {
            std::fprintf(
                stderr,
                "ERROR: unknown atomic type '%s'. "
                "Supported: atom/red or 0/1.\n",
                argv[1]);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 3) {
        atomic_ops_per_thread =
            std::max(
                1,
                std::atoi(argv[2]));
    }

    if (argc >= 4) {
        kernel_repeat =
            std::max(
                1,
                std::atoi(argv[3]));
    }

    if (argc >= 5) {
        blocks_per_sm =
            std::max(
                1,
                std::atoi(argv[4]));
    }

    if (argc >= 6) {
        ilp =
            std::atoi(argv[5]);

        if (!IsSupportedILP(ilp)) {
            std::fprintf(
                stderr,
                "ERROR: unsupported ILP=%d. "
                "Supported: 1,2,4,8,16.\n",
                ilp);
            return EXIT_FAILURE;
        }
    }

    if (argc >= 7) {
        hot_addresses =
            std::max(
                1,
                std::atoi(argv[6]));
    }

    if (argc >= 8) {
        trials =
            std::max(
                1,
                std::atoi(argv[7]));
    }

    const int sm_count =
        prop.multiProcessorCount;

    std::printf("GPU                  : %s\n", prop.name);
    std::printf("compute capability   : %d.%d\n", prop.major, prop.minor);
    std::printf("SM count             : %d\n", sm_count);
    std::printf("maxThreadsPerBlock   : %d\n", prop.maxThreadsPerBlock);
    std::printf("atomic type          : %s.global.add.u32\n",
                AtomicTypeName(atomic_type));
    std::printf("atomic ops/thread    : %d\n",
                atomic_ops_per_thread);
    std::printf("kernel repeat        : %d\n",
                kernel_repeat);
    std::printf("blocks/SM            : %d\n",
                blocks_per_sm);
    std::printf("ILP                  : %d\n",
                ilp);
    std::printf("hot addresses        : %d\n",
                hot_addresses);
    std::printf("trials               : %d\n\n",
                trials);

    if (hot_addresses == 1) {
        std::printf(
            "Contention mode       : ALL GPU threads update the SAME address.\n\n");
    } else {
        std::printf(
            "Contention mode       : threads distributed across %d hot addresses.\n\n",
            hot_addresses);
    }

    if (prop.major != 9) {
        std::fprintf(
            stderr,
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
                atomic_ops_per_thread,
                kernel_repeat,
                ilp,
                hot_addresses,
                trials,
                atomic_type);

        results.push_back(r);
        PrintResult(r);
    }

    std::ofstream csv(
        "sm90_atomic_contention_results.csv");

    csv <<
        "gpu,cc_major,cc_minor,atomic_type,ilp,"
        "sm_count,threads_per_block,blocks_per_sm,blocks,"
        "hot_addresses,atomic_ops_per_thread,kernel_repeat,trials,"
        "total_threads,total_atomic_ops,"
        "median_ms,min_ms,max_ms,"
        "GAtomicOps_per_s,effective_ns_per_op,counter0\n";

    for (const Result& r : results) {
        csv <<
            '"' << prop.name << '"' << ","
            << prop.major << ","
            << prop.minor << ","
            << AtomicTypeName(r.atomic_type) << ","
            << r.ilp << ","
            << r.sm_count << ","
            << r.threads_per_block << ","
            << r.blocks_per_sm << ","
            << r.blocks << ","
            << r.hot_addresses << ","
            << r.atomic_ops_per_thread << ","
            << r.kernel_repeat << ","
            << r.trials << ","
            << r.total_threads << ","
            << r.total_atomic_ops << ","
            << r.median_ms << ","
            << r.min_ms << ","
            << r.max_ms << ","
            << r.GAtomicOps << ","
            << r.ns_per_atomic_effective << ","
            << r.counter0 << "\n";
    }

    csv.close();

    std::printf(
        "\nSaved CSV: sm90_atomic_contention_results.csv\n");

    if (!results.empty()) {

        auto best =
            std::max_element(
                results.begin(),
                results.end(),
                [](const Result& a, const Result& b) {
                    return a.GAtomicOps <
                           b.GAtomicOps;
                });

        std::printf(
            "Best: type=%s ILP=%d %d threads/block -> "
            "%.3f GAtomicOps/s\n",
            AtomicTypeName(atomic_type),
            ilp,
            best->threads_per_block,
            best->GAtomicOps);
    }

    return 0;
}
