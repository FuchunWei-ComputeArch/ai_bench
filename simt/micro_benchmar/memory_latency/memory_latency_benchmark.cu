// unknown_simt_memory_hierarchy_probe.cu
//
// Black-box memory hierarchy probe for a SIMT-compatible GPU-like device.
//
// Goals:
//   - Do NOT assume the number of cache levels.
//   - Do NOT require NVIDIA-specific cache-bypass instructions.
//   - Infer hierarchy candidates from latency cliffs vs. working-set size.
//   - Use random dependent pointer chasing to suppress memory-level parallelism.
//   - Sweep working-set size, node spacing, and random seeds.
//   - Save raw results to CSV.
//
// Build example:
//   nvcc -O3 -lineinfo unknown_simt_memory_hierarchy_probe.cu \
//        -o unknown_simt_memory_hierarchy_probe
//
// Run:
//   ./unknown_simt_memory_hierarchy_probe
//
// Output:
//   unknown_simt_memory_hierarchy_results.csv
//
// Important:
//   The detected capacities are EFFECTIVE capacities. Associativity,
//   replacement policy, address mapping, cache-line/sector size and sharing
//   topology can shift the observed transitions.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
do {                                                                         \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                     __FILE__, __LINE__, cudaGetErrorString(_err));          \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

static constexpr int WARMUP_ROUNDS  = 3;
static constexpr int MEASURE_ROUNDS = 30;

static const std::vector<std::size_t> kStrides = {
    4
};

static const std::vector<uint32_t> kSeeds = {
    1
};

static const std::vector<std::size_t> kWorkingSets = {
       4ULL << 10,
       8ULL << 10,
      12ULL << 10,
      16ULL << 10,
      24ULL << 10,
      32ULL << 10,
      48ULL << 10,
      64ULL << 10,
      96ULL << 10,
     128ULL << 10,
     192ULL << 10,
     256ULL << 10,
     384ULL << 10,
     512ULL << 10,
     768ULL << 10,
       1ULL << 20,
       2ULL << 20,
       3ULL << 20,
       4ULL << 20,
       6ULL << 20,
       8ULL << 20,
      12ULL << 20,
      16ULL << 20,
      24ULL << 20,
      32ULL << 20,
      48ULL << 20,
      64ULL << 20,
      96ULL << 20,
     128ULL << 20,
     192ULL << 20,
     256ULL << 20
};

__device__ __forceinline__
uint32_t dependent_load_u32(const volatile uint32_t* p)
{
    return *p;
}
__device__ __forceinline__
uint32_t load_cacheable_u32(const volatile uint32_t* p){
    uint32_t v;
	asm volatile(
        "ld.global.ca.u32 %0, [%1];"
        : "=r"(v)
        : "l"(p)
        : "memory"
    );
    return v;
}


__global__ __launch_bounds__(1)
void pointer_chase_kernel(const volatile uint32_t* chain,
                          uint32_t start_idx,
                          int warmup_iters,
                          int measure_iters,
                          unsigned long long* cycles_out,
                          uint32_t* final_idx_out)
{
    uint32_t idx = start_idx;

    for (int i = 0; i < warmup_iters; ++i) {
        idx = dependent_load_u32(chain + idx);
    }

    asm volatile("" : "+r"(idx) :: "memory");

    unsigned long long t0 = clock64();

    for (int i = 0; i < measure_iters; ++i) {
		idx = load_cacheable_u32(chain + idx);
    }

    unsigned long long t1 = clock64();

    cycles_out[0] = t1 - t0;
    final_idx_out[0] = idx;
}

struct HostChain {
    std::vector<uint32_t> data;
    uint32_t start_idx = 0;
    std::size_t node_count = 0;
    std::size_t ws_bytes = 0;
};

static HostChain build_random_chain(std::size_t working_set_bytes,
                                    std::size_t node_stride_bytes,
                                    uint32_t seed)
{
    if (node_stride_bytes < sizeof(uint32_t) ||
        node_stride_bytes % sizeof(uint32_t) != 0) {
        std::fprintf(stderr,
                     "Invalid stride: %zu bytes\n",
                     node_stride_bytes);
        std::exit(EXIT_FAILURE);
    }

    working_set_bytes =
        (working_set_bytes / node_stride_bytes) * node_stride_bytes;

    if (working_set_bytes < 2 * node_stride_bytes) {
        std::fprintf(stderr,
                     "Working set too small for stride.\n");
        std::exit(EXIT_FAILURE);
    }

    const std::size_t stride_words =
        node_stride_bytes / sizeof(uint32_t);

    const std::size_t nodes =
        working_set_bytes / node_stride_bytes;

    const std::size_t total_words =
        nodes * stride_words;

    if (total_words > static_cast<std::size_t>(UINT32_MAX)) {
        std::fprintf(stderr,
                     "Working set too large for uint32_t indices.\n");
        std::exit(EXIT_FAILURE);
    }

    HostChain h;
    h.data.assign(total_words, 0u);
    h.node_count = nodes;
    h.ws_bytes = working_set_bytes;

    std::vector<uint32_t> perm(nodes);
    for (uint32_t i = 0; i < static_cast<uint32_t>(nodes); ++i)
        perm[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (std::size_t i = 0; i < nodes; ++i) {
        const uint32_t cur_node  = perm[i];
        const uint32_t next_node = perm[(i + 1) % nodes];

        const std::size_t cur_word =
            static_cast<std::size_t>(cur_node) * stride_words;

        const std::size_t next_word =
            static_cast<std::size_t>(next_node) * stride_words;

        h.data[cur_word] =
            static_cast<uint32_t>(next_word);
    }

    h.start_idx =
        static_cast<uint32_t>(
            static_cast<std::size_t>(perm[0]) * stride_words);

    return h;
}

struct Result {
    std::size_t ws_bytes;
    std::size_t stride_bytes;
    uint32_t seed;
    int warmup_rounds;
    int measure_accesses;
    unsigned long long total_cycles;
    double cycles_per_access;
};

static Result run_one(std::size_t ws,
                      std::size_t stride,
                      uint32_t seed)
{
    HostChain h =
        build_random_chain(ws, stride, seed);

    const std::size_t warmup_iters_sz =
        h.node_count *
        static_cast<std::size_t>(WARMUP_ROUNDS);

    const std::size_t measure_iters_sz =
        h.node_count *
        static_cast<std::size_t>(MEASURE_ROUNDS);

    if (warmup_iters_sz > static_cast<std::size_t>(INT32_MAX) ||
        measure_iters_sz > static_cast<std::size_t>(INT32_MAX)) {
        std::fprintf(stderr,
                     "Too many iterations. Reduce rounds or working set.\n");
        std::exit(EXIT_FAILURE);
    }

    const int warmup_iters =
        static_cast<int>(warmup_iters_sz);

    const int measure_iters =
        static_cast<int>(measure_iters_sz);

    uint32_t* d_chain = nullptr;
    unsigned long long* d_cycles = nullptr;
    uint32_t* d_final = nullptr;

    const std::size_t alloc_bytes =
        h.data.size() * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc(&d_chain, alloc_bytes));
    CUDA_CHECK(cudaMalloc(&d_cycles,
                          sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_final,
                          sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_chain,
                          h.data.data(),
                          alloc_bytes,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    pointer_chase_kernel<<<1, 1>>>(
        d_chain,
        h.start_idx,
        warmup_iters,
        measure_iters,
        d_cycles,
        d_final);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long total_cycles = 0;
    uint32_t final_idx = 0;

    CUDA_CHECK(cudaMemcpy(&total_cycles,
                          d_cycles,
                          sizeof(total_cycles),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&final_idx,
                          d_final,
                          sizeof(final_idx),
                          cudaMemcpyDeviceToHost));

    volatile uint32_t sink = final_idx;
    (void)sink;

    Result r;
    r.ws_bytes = h.ws_bytes;
    r.stride_bytes = stride;
    r.seed = seed;
    r.warmup_rounds = WARMUP_ROUNDS;
    r.measure_accesses = measure_iters;
    r.total_cycles = total_cycles;
    r.cycles_per_access =
        static_cast<double>(total_cycles) /
        static_cast<double>(measure_iters);

    CUDA_CHECK(cudaFree(d_chain));
    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_final));

    return r;
}

static void save_csv(const std::vector<Result>& results,
                     const std::string& filename)
{
    std::ofstream ofs(filename);

    if (!ofs) {
        std::fprintf(stderr,
                     "Cannot open %s\n",
                     filename.c_str());
        std::exit(EXIT_FAILURE);
    }

    ofs << "working_set_bytes,"
           "working_set_kb,"
           "working_set_mb,"
           "stride_bytes,"
           "seed,"
           "warmup_rounds,"
           "measure_accesses,"
           "total_cycles,"
           "cycles_per_access\n";

    ofs << std::fixed << std::setprecision(6);

    for (const auto& r : results) {
        ofs << r.ws_bytes << ","
            << r.ws_bytes / 1024.0 << ","
            << r.ws_bytes / 1024.0 / 1024.0 << ","
            << r.stride_bytes << ","
            << r.seed << ","
            << r.warmup_rounds << ","
            << r.measure_accesses << ","
            << r.total_cycles << ","
            << r.cycles_per_access << "\n";
    }
}

struct AggPoint {
    std::size_t ws_bytes;
    double median_cycles;
};

static std::vector<AggPoint>
aggregate_for_stride(const std::vector<Result>& results,
                     std::size_t stride)
{
    std::vector<AggPoint> out;

    for (std::size_t ws : kWorkingSets) {
        std::vector<double> vals;

        for (const auto& r : results) {
            if (r.stride_bytes == stride &&
                r.ws_bytes == ws) {
                vals.push_back(r.cycles_per_access);
            }
        }

        if (vals.empty())
            continue;

        std::sort(vals.begin(), vals.end());

        double median = 0.0;

        if (vals.size() & 1) {
            median = vals[vals.size() / 2];
        } else {
            median =
                0.5 * (vals[vals.size()/2 - 1] +
                       vals[vals.size()/2]);
        }

        out.push_back({ws, median});
    }

    return out;
}

static std::vector<std::size_t>
detect_candidate_cliffs(const std::vector<AggPoint>& pts,
                        double ratio_threshold = 1.35)
{
    std::vector<std::size_t> cliffs;

    for (std::size_t i = 1; i < pts.size(); ++i) {
        const double prev = pts[i - 1].median_cycles;
        const double cur  = pts[i].median_cycles;

        if (prev <= 0.0)
            continue;

        const double ratio = cur / prev;

        if (ratio >= ratio_threshold) {
            cliffs.push_back(pts[i].ws_bytes);
        }
    }

    return cliffs;
}

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::size_t free_mem = 0;
    std::size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    std::printf("============================================================\n");
    std::printf("Black-box SIMT Memory Hierarchy Probe\n");
    std::printf("============================================================\n");
    std::printf("Device                   : %s\n", prop.name);
    std::printf("Compute capability       : %d.%d\n",
                prop.major, prop.minor);
    std::printf("Visible device memory    : %.2f GB\n",
                total_mem / 1024.0 / 1024.0 / 1024.0);
    std::printf("Free device memory       : %.2f GB\n",
                free_mem / 1024.0 / 1024.0 / 1024.0);
    std::printf("Reported L2 (reference)  : %.2f MB\n",
                prop.l2CacheSize / 1024.0 / 1024.0);
    std::printf("Shared memory / SM       : %.2f KB\n",
                prop.sharedMemPerMultiprocessor / 1024.0);
    std::printf("Registers / SM           : %d x 32-bit\n",
                prop.regsPerMultiprocessor);
    std::printf("SM count                 : %d\n",
                prop.multiProcessorCount);
    std::printf("============================================================\n");

    std::vector<Result> results;

    for (std::size_t stride : kStrides) {
        std::printf("\n---------------- stride = %zu B ----------------\n",
                    stride);

        for (std::size_t ws : kWorkingSets) {
            if (ws > free_mem / 2) {
                std::printf(
                    "skip WS=%8.2f MB (insufficient free memory margin)\n",
                    ws / 1024.0 / 1024.0);
                continue;
            }

            for (uint32_t seed : kSeeds) {
                Result r = run_one(ws, stride, seed);
                results.push_back(r);

                std::printf(
                    "WS=%9.3f MB  stride=%4zu B  seed=%2u  "
                    "latency=%9.2f cycles/access\n",
                    r.ws_bytes / 1024.0 / 1024.0,
                    r.stride_bytes,
                    r.seed,
                    r.cycles_per_access);
            }
        }
    }

    const std::string csv_file =
        "unknown_simt_memory_hierarchy_results.csv";

    save_csv(results, csv_file);

    const std::size_t analysis_stride =
        kStrides.back();

    std::vector<AggPoint> agg =
        aggregate_for_stride(results, analysis_stride);

    std::vector<std::size_t> cliffs =
        detect_candidate_cliffs(agg, 1.35);

    std::printf("\n============================================================\n");
    std::printf("Candidate memory-hierarchy boundaries\n");
    std::printf("Analysis stride: %zu B\n",
                analysis_stride);
    std::printf("============================================================\n");

    if (cliffs.empty()) {
        std::printf(
            "No strong cliff detected by the simple 1.35x rule.\n"
            "Inspect the CSV and add denser working-set points around bends.\n");
    } else {
        for (std::size_t i = 0; i < cliffs.size(); ++i) {
            if (cliffs[i] < (1ULL << 20)) {
                std::printf(
                    "Candidate M%zu -> M%zu boundary near %.2f KB\n",
                    i, i + 1,
                    cliffs[i] / 1024.0);
            } else {
                std::printf(
                    "Candidate M%zu -> M%zu boundary near %.2f MB\n",
                    i, i + 1,
                    cliffs[i] / 1024.0 / 1024.0);
            }
        }
    }

    std::printf("\nCSV saved to: %s\n",
                csv_file.c_str());

    std::printf(
        "\nHow to interpret:\n"
        "  1) Plot cycles/access vs log2(working_set_bytes).\n"
        "  2) Stable plateaus are candidate memory levels M0, M1, M2...\n"
        "  3) Upward transitions are effective capacity boundaries.\n"
        "  4) Compare all stride curves to identify spatial/cache-line effects.\n"
        "  5) Re-run with denser WS points near each transition.\n"
        "  6) Add multi-block interference tests to determine private/shared scope.\n"
        "  7) Inspect machine code to verify one strict dependent load chain.\n");

    return 0;
}
