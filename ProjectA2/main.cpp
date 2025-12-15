/**
 * Project A2: Dense vs Sparse Matrix Multiplication (SIMD + Multithreading)
 * 
 * This program implements and benchmarks:
 * - Dense GEMM (C = A * B) with cache tiling and SIMD
 * - Sparse CSR-SpMM (C = A_sparse * B) with SIMD optimization
 * - Multithreading support via OpenMP
 * - Performance analysis with GFLOP/s and memory bandwidth
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cblas.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <cstring>
#include <time.h>
#include <sched.h>
#include <x86intrin.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

using namespace std;
using namespace std::chrono;

// Global output file stream
ofstream g_logfile;

// ============================================================================
// High-Precision Timing Utilities
// ============================================================================

// Get high-precision timestamp in seconds using CLOCK_MONOTONIC
inline double get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Read CPU timestamp counter (TSC) for cycle-accurate timing
inline uint64_t rdtsc() {
    return __rdtsc();
}

// Get CPU frequency estimate (cycles per second)
double estimate_cpu_frequency() {
    const double measurement_time = 0.1; // 100ms
    
    uint64_t tsc_start = rdtsc();
    double time_start = get_timestamp();
    
    // Busy wait for measurement period
    while (get_timestamp() - time_start < measurement_time);
    
    uint64_t tsc_end = rdtsc();
    double time_end = get_timestamp();
    
    double elapsed_time = time_end - time_start;
    uint64_t elapsed_cycles = tsc_end - tsc_start;
    
    return elapsed_cycles / elapsed_time;
}

// ============================================================================
// Performance Counter Integration
// ============================================================================

class PerfCounters {
private:
    int fd_cycles;
    int fd_instructions;
    int fd_cache_misses;
    int fd_cache_refs;
    int fd_llc_misses;
    int fd_llc_refs;
    bool enabled;
    
    static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                               int cpu, int group_fd, unsigned long flags) {
        return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    }
    
    int setup_counter(uint32_t type, uint64_t config) {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = type;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        
        int fd = perf_event_open(&pe, 0, -1, -1, 0);
        return fd;
    }
    
public:
    struct Counters {
        long long cycles;
        long long instructions;
        long long cache_misses;
        long long cache_refs;
        long long llc_misses;
        long long llc_refs;
        double ipc;  // Instructions per cycle
        double cache_miss_rate;
        double llc_miss_rate;
    };
    
    PerfCounters() : enabled(false), fd_cycles(-1), fd_instructions(-1), 
                     fd_cache_misses(-1), fd_cache_refs(-1), 
                     fd_llc_misses(-1), fd_llc_refs(-1) {
        fd_cycles = setup_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        fd_instructions = setup_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
        fd_cache_misses = setup_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
        fd_cache_refs = setup_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
        
        // LLC (Last Level Cache) counters
        struct perf_event_attr pe_llc_miss, pe_llc_ref;
        memset(&pe_llc_miss, 0, sizeof(struct perf_event_attr));
        memset(&pe_llc_ref, 0, sizeof(struct perf_event_attr));
        
        pe_llc_miss.type = PERF_TYPE_HW_CACHE;
        pe_llc_miss.size = sizeof(struct perf_event_attr);
        pe_llc_miss.config = (PERF_COUNT_HW_CACHE_LL) | 
                             (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                             (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        pe_llc_miss.disabled = 1;
        pe_llc_miss.exclude_kernel = 1;
        pe_llc_miss.exclude_hv = 1;
        fd_llc_misses = perf_event_open(&pe_llc_miss, 0, -1, -1, 0);
        
        pe_llc_ref.type = PERF_TYPE_HW_CACHE;
        pe_llc_ref.size = sizeof(struct perf_event_attr);
        pe_llc_ref.config = (PERF_COUNT_HW_CACHE_LL) | 
                            (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                            (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
        pe_llc_ref.disabled = 1;
        pe_llc_ref.exclude_kernel = 1;
        pe_llc_ref.exclude_hv = 1;
        fd_llc_refs = perf_event_open(&pe_llc_ref, 0, -1, -1, 0);
        
        enabled = (fd_cycles >= 0 && fd_instructions >= 0 && 
                  fd_cache_misses >= 0 && fd_cache_refs >= 0);
    }
    
    ~PerfCounters() {
        if (fd_cycles >= 0) close(fd_cycles);
        if (fd_instructions >= 0) close(fd_instructions);
        if (fd_cache_misses >= 0) close(fd_cache_misses);
        if (fd_cache_refs >= 0) close(fd_cache_refs);
        if (fd_llc_misses >= 0) close(fd_llc_misses);
        if (fd_llc_refs >= 0) close(fd_llc_refs);
    }
    
    void start() {
        if (!enabled) return;
        ioctl(fd_cycles, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_instructions, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_cache_misses, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd_cache_refs, PERF_EVENT_IOC_RESET, 0);
        if (fd_llc_misses >= 0) ioctl(fd_llc_misses, PERF_EVENT_IOC_RESET, 0);
        if (fd_llc_refs >= 0) ioctl(fd_llc_refs, PERF_EVENT_IOC_RESET, 0);
        
        ioctl(fd_cycles, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(fd_instructions, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(fd_cache_misses, PERF_EVENT_IOC_ENABLE, 0);
        ioctl(fd_cache_refs, PERF_EVENT_IOC_ENABLE, 0);
        if (fd_llc_misses >= 0) ioctl(fd_llc_misses, PERF_EVENT_IOC_ENABLE, 0);
        if (fd_llc_refs >= 0) ioctl(fd_llc_refs, PERF_EVENT_IOC_ENABLE, 0);
    }
    
    Counters stop() {
        Counters c = {0};
        if (!enabled) return c;
        
        ioctl(fd_cycles, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_instructions, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_cache_misses, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_cache_refs, PERF_EVENT_IOC_DISABLE, 0);
        if (fd_llc_misses >= 0) ioctl(fd_llc_misses, PERF_EVENT_IOC_DISABLE, 0);
        if (fd_llc_refs >= 0) ioctl(fd_llc_refs, PERF_EVENT_IOC_DISABLE, 0);
        
        read(fd_cycles, &c.cycles, sizeof(long long));
        read(fd_instructions, &c.instructions, sizeof(long long));
        read(fd_cache_misses, &c.cache_misses, sizeof(long long));
        read(fd_cache_refs, &c.cache_refs, sizeof(long long));
        if (fd_llc_misses >= 0) read(fd_llc_misses, &c.llc_misses, sizeof(long long));
        if (fd_llc_refs >= 0) read(fd_llc_refs, &c.llc_refs, sizeof(long long));
        
        c.ipc = (c.cycles > 0) ? (double)c.instructions / c.cycles : 0.0;
        c.cache_miss_rate = (c.cache_refs > 0) ? 
                           (double)c.cache_misses / c.cache_refs * 100.0 : 0.0;
        c.llc_miss_rate = (c.llc_refs > 0) ? 
                         (double)c.llc_misses / c.llc_refs * 100.0 : 0.0;
        
        return c;
    }
    
    bool is_enabled() const { return enabled; }
};

// Macro to print to both console and file
#define LOG(x) do { cout << x; if (g_logfile.is_open()) g_logfile << x; } while(0)

// ============================================================================
// CSR (Compressed Sparse Row) Matrix Format
// ============================================================================
struct CSRMatrix {
    int rows;
    int cols;
    int nnz;  // number of non-zeros
    vector<double> values;
    vector<int> col_indices;
    vector<int> row_ptr;
    
    CSRMatrix(int r, int c) : rows(r), cols(c), nnz(0) {
        row_ptr.resize(rows + 1, 0);
    }
};

// ============================================================================
// Dense Matrix (Row-major layout)
// ============================================================================
struct DenseMatrix {
    int rows;
    int cols;
    vector<double> data;
    
    DenseMatrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    double& operator()(int i, int j) {
        return data[i * cols + j];
    }
    
    const double& operator()(int i, int j) const {
        return data[i * cols + j];
    }
};

// ============================================================================
// Matrix Generation Functions
// ============================================================================

// Generate random dense matrix
void generateDenseMatrix(DenseMatrix& mat, double min_val = 0.0, double max_val = 1.0, int seed = 42) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(min_val, max_val);
    
    for (auto& val : mat.data) {
        val = dist(gen);
    }
}

// Convert dense matrix to CSR format with given density
void denseToCSR(const DenseMatrix& dense, CSRMatrix& csr, double density, int seed = 42) {
    mt19937 gen(seed);
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    uniform_real_distribution<double> val_dist(0.0, 1.0);
    
    csr.values.clear();
    csr.col_indices.clear();
    csr.row_ptr[0] = 0;
    
    for (int i = 0; i < csr.rows; i++) {
        for (int j = 0; j < csr.cols; j++) {
            if (prob_dist(gen) < density) {
                csr.values.push_back(val_dist(gen));
                csr.col_indices.push_back(j);
            }
        }
        csr.row_ptr[i + 1] = csr.values.size();
    }
    csr.nnz = csr.values.size();
}

// ============================================================================
// Dense GEMM: C = A * B (with cache tiling and SIMD)
// ============================================================================

// Scalar version - explicitly disable SIMD auto-vectorization
__attribute__((optimize("no-tree-vectorize")))
void denseGEMM_naive(const DenseMatrix& A, const DenseMatrix& B, DenseMatrix& C) {
    #pragma omp parallel for
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            double sum = 0.0;
            #pragma GCC ivdep
            #pragma clang loop vectorize(disable)
            for (int k = 0; k < A.cols; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// Cache-tiled GEMM with SIMD optimization
void denseGEMM_tiled(const DenseMatrix& A, const DenseMatrix& B, DenseMatrix& C, int tile_size = 64) {
    const int m = A.rows;
    const int n = B.cols;
    const int k = A.cols;
    
    // Initialize C to zero
    fill(C.data.begin(), C.data.end(), 0.0);
    
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < m; ii += tile_size) {
        for (int jj = 0; jj < n; jj += tile_size) {
            for (int kk = 0; kk < k; kk += tile_size) {
                // Tile boundaries
                int i_end = min(ii + tile_size, m);
                int j_end = min(jj + tile_size, n);
                int k_end = min(kk + tile_size, k);
                
                // Process tile
                for (int i = ii; i < i_end; i++) {
                    for (int kk_inner = kk; kk_inner < k_end; kk_inner++) {
                        double a_val = A(i, kk_inner);
                        for (int j = jj; j < j_end; j++) {
                            C(i, j) += a_val * B(kk_inner, j);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Sparse CSR-SpMM: C = A_sparse * B (SIMD-optimized)
// ============================================================================

// Scalar version - explicitly disable SIMD auto-vectorization
__attribute__((optimize("no-tree-vectorize")))
void csrSpMM(const CSRMatrix& A, const DenseMatrix& B, DenseMatrix& C) {
    const int n = B.cols;
    
    // Initialize C to zero
    fill(C.data.begin(), C.data.end(), 0.0);
    
    #pragma omp parallel for
    for (int i = 0; i < A.rows; i++) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];
        
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            
            // Process non-zeros in row i of A
            #pragma GCC ivdep
            #pragma clang loop vectorize(disable)
            for (int idx = row_start; idx < row_end; idx++) {
                int k = A.col_indices[idx];
                double a_val = A.values[idx];
                sum += a_val * B(k, j);
            }
            
            C(i, j) = sum;
        }
    }
}

// SIMD-optimized CSR-SpMM (processes multiple columns of B at once)
void csrSpMM_simd(const CSRMatrix& A, const DenseMatrix& B, DenseMatrix& C) {
    const int n = B.cols;
    
    // Initialize C to zero
    fill(C.data.begin(), C.data.end(), 0.0);
    
    #pragma omp parallel for
    for (int i = 0; i < A.rows; i++) {
        int row_start = A.row_ptr[i];
        int row_end = A.row_ptr[i + 1];
        
        // Process columns of B
        int j = 0;
        
#ifdef __AVX512F__
        // Process 8 columns at a time with AVX-512
        for (; j + 7 < n; j += 8) {
            __m512d sum = _mm512_setzero_pd();
            
            for (int idx = row_start; idx < row_end; idx++) {
                int k = A.col_indices[idx];
                __m512d a_val = _mm512_set1_pd(A.values[idx]);
                __m512d b_vals = _mm512_loadu_pd(&B.data[k * n + j]);
                sum = _mm512_fmadd_pd(a_val, b_vals, sum);
            }
            
            _mm512_storeu_pd(&C.data[i * n + j], sum);
        }
#elif defined(__AVX2__)
        // Process 4 columns at a time with AVX2
        for (; j + 3 < n; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            
            for (int idx = row_start; idx < row_end; idx++) {
                int k = A.col_indices[idx];
                __m256d a_val = _mm256_set1_pd(A.values[idx]);
                __m256d b_vals = _mm256_loadu_pd(&B.data[k * n + j]);
                sum = _mm256_fmadd_pd(a_val, b_vals, sum);
            }
            
            _mm256_storeu_pd(&C.data[i * n + j], sum);
        }
#endif
        
        // Process remaining columns
        for (; j < n; j++) {
            double sum = 0.0;
            for (int idx = row_start; idx < row_end; idx++) {
                int k = A.col_indices[idx];
                sum += A.values[idx] * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// ============================================================================
// Performance Measurement and Validation
// ============================================================================

// OpenBLAS reference GEMM: C = A * B
void openblasGEMM(const DenseMatrix& A, const DenseMatrix& B, DenseMatrix& C) {
    // Initialize C to zero
    fill(C.data.begin(), C.data.end(), 0.0);
    
    // cblas_dgemm parameters:
    // CblasRowMajor: row-major layout
    // CblasNoTrans: no transpose on A or B
    // m, n, k: dimensions
    // alpha = 1.0, beta = 0.0: C = 1.0 * A * B + 0.0 * C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A.rows, B.cols, A.cols,
                1.0, A.data.data(), A.cols,
                B.data.data(), B.cols,
                0.0, C.data.data(), C.cols);
}

double relativeError(const DenseMatrix& A, const DenseMatrix& B) {
    double max_diff = 0.0;
    double max_val = 0.0;
    
    for (size_t i = 0; i < A.data.size(); i++) {
        max_diff = max(max_diff, abs(A.data[i] - B.data[i]));
        max_val = max(max_val, abs(A.data[i]));
    }
    
    return max_diff / (max_val + 1e-10);
}

double maxAbsoluteError(const DenseMatrix& A, const DenseMatrix& B) {
    double max_diff = 0.0;
    
    for (size_t i = 0; i < A.data.size(); i++) {
        max_diff = max(max_diff, abs(A.data[i] - B.data[i]));
    }
    
    return max_diff;
}

struct BenchmarkResult {
    double median;
    double min;
    double max;
    double mean;
    double stddev;
    PerfCounters::Counters perf;  // Hardware counters
};

template<typename Func>
BenchmarkResult benchmarkFunction(Func func, int num_runs = 5) {
    vector<double> times;
    PerfCounters perf;
    PerfCounters::Counters perf_data = {0};
    
    // Warmup and determine if we need more iterations
    func();
    double warmup_start = get_timestamp();
    func();
    double warmup_end = get_timestamp();
    double warmup_time = warmup_end - warmup_start;
    
    // For very fast operations, run many iterations to reduce noise
    // Adaptive: aim for 50ms total measurement time for sub-millisecond operations
    int inner_iterations = 1;
    const double min_measurement_time = 0.050;  // 50ms minimum for reliable timing
    if (warmup_time < min_measurement_time && warmup_time > 0) {
        inner_iterations = max(1, (int)ceil(min_measurement_time / warmup_time));
        // Cap at 50000 iterations for very tiny operations
        inner_iterations = min(inner_iterations, 50000);
    }
    
    for (int run = 0; run < num_runs; run++) {
        // Measure with perf counters on median run
        if (run == num_runs / 2) {
            perf.start();
        }
        
        // Use high-precision clock_gettime for nanosecond accuracy
        double start = get_timestamp();
        for (int iter = 0; iter < inner_iterations; iter++) {
            func();
        }
        double end = get_timestamp();
        
        if (run == num_runs / 2) {
            perf_data = perf.stop();
        }
        
        double elapsed = end - start;
        times.push_back(elapsed / inner_iterations);  // Average per iteration
    }
    
    // Calculate statistics
    sort(times.begin(), times.end());
    
    BenchmarkResult result;
    result.median = times[num_runs / 2];
    result.min = times[0];
    result.max = times[num_runs - 1];
    
    // Calculate mean
    double sum = 0.0;
    for (double t : times) sum += t;
    result.mean = sum / num_runs;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double t : times) {
        double diff = t - result.mean;
        variance += diff * diff;
    }
    result.stddev = sqrt(variance / num_runs);
    result.perf = perf_data;
    
    return result;
}

double computeGFLOPS(double time_sec, long long flops) {
    return (flops / 1e9) / time_sec;
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

void runExperiment(int m, int k, int n, double density, bool validate = true) {
    LOG("\n" << string(70, '=') << "\n");
    LOG("Experiment: m=" << m << ", k=" << k << ", n=" << n 
         << ", density=" << fixed << setprecision(2) << (density * 100) << "%\n");
    LOG(string(70, '=') << "\n");
    
    // Generate matrices
    DenseMatrix A_dense(m, k);
    DenseMatrix B(k, n);
    DenseMatrix C_dense(m, n);
    DenseMatrix C_sparse(m, n);
    DenseMatrix C_sparse_simd(m, n);
    DenseMatrix C_reference(m, n);
    
    generateDenseMatrix(A_dense, 0.0, 1.0, 42);
    generateDenseMatrix(B, 0.0, 1.0, 43);
    
    CSRMatrix A_sparse(m, k);
    denseToCSR(A_dense, A_sparse, density, 44);
    
    LOG("CSR non-zeros: " << A_sparse.nnz << " / " << (m * k) 
         << " (" << (100.0 * A_sparse.nnz / (m * k)) << "%)\n");
    LOG("Avg non-zeros per row: " << (double)A_sparse.nnz / m << "\n");
    
    // Compute arithmetic intensity
    double dense_ai = (2.0 * m * n * k) / (sizeof(double) * (m * k + k * n + m * n));
    double sparse_ai = (2.0 * A_sparse.nnz * n) / (sizeof(double) * (A_sparse.nnz + 2 * A_sparse.nnz + k * n + m * n));
    LOG("Arithmetic Intensity - Dense: " << fixed << setprecision(2) << dense_ai << " FLOPs/byte\n");
    LOG("Arithmetic Intensity - Sparse: " << setprecision(2) << sparse_ai << " FLOPs/byte\n");
    
    // Validation against OpenBLAS reference
    if (validate) {
        LOG("\nValidation against OpenBLAS:\n");
        
        // Warmup run for OpenBLAS
        openblasGEMM(A_dense, B, C_reference);
        
        // Benchmark OpenBLAS properly with multiple runs
        auto ref_result = benchmarkFunction([&]() {
            openblasGEMM(A_dense, B, C_reference);
        });
        
        long long dense_flops = 2LL * m * n * k;
        double ref_gflops = computeGFLOPS(ref_result.median, dense_flops);
        
        LOG("  OpenBLAS GEMM: " << fixed << setprecision(6) << ref_result.median << " s, "
            << setprecision(2) << ref_gflops << " GFLOP/s\n");        // Validate dense GEMM
        denseGEMM_tiled(A_dense, B, C_dense, 64);
        double dense_error = relativeError(C_dense, C_reference);
        double dense_abs_error = maxAbsoluteError(C_dense, C_reference);
        LOG("  Dense GEMM error: " << scientific << setprecision(2) 
             << dense_error << " (relative), " << dense_abs_error << " (absolute)\n");
        
        if (dense_error < 1e-10) {
            LOG("  ✓ Dense GEMM validated successfully\n");
        } else if (dense_error < 1e-6) {
            LOG("  ⚠ Dense GEMM within acceptable tolerance\n");
        } else {
            LOG("  ✗ WARNING: Dense GEMM error exceeds tolerance\n");
        }
        
        // Validate sparse implementations (they compute A_sparse * B where A_sparse ≈ A_dense)
        // Generate corresponding reference by using the same sparse pattern
        DenseMatrix A_sparse_dense(m, k);
        fill(A_sparse_dense.data.begin(), A_sparse_dense.data.end(), 0.0);
        for (int i = 0; i < A_sparse.rows; i++) {
            for (int idx = A_sparse.row_ptr[i]; idx < A_sparse.row_ptr[i + 1]; idx++) {
                int j = A_sparse.col_indices[idx];
                A_sparse_dense(i, j) = A_sparse.values[idx];
            }
        }
        DenseMatrix C_sparse_ref(m, n);
        openblasGEMM(A_sparse_dense, B, C_sparse_ref);
        
        csrSpMM(A_sparse, B, C_sparse);
        double sparse_error = relativeError(C_sparse, C_sparse_ref);
        double sparse_abs_error = maxAbsoluteError(C_sparse, C_sparse_ref);
        LOG("  CSR-SpMM (scalar) error: " << scientific << setprecision(2) 
             << sparse_error << " (relative), " << sparse_abs_error << " (absolute)\n");
        
        if (sparse_error < 1e-10) {
            LOG("  ✓ CSR-SpMM (scalar) validated successfully\n");
        } else if (sparse_error < 1e-6) {
            LOG("  ⚠ CSR-SpMM (scalar) within acceptable tolerance\n");
        } else {
            LOG("  ✗ WARNING: CSR-SpMM (scalar) error exceeds tolerance\n");
        }
        
        csrSpMM_simd(A_sparse, B, C_sparse_simd);
        double sparse_simd_error = relativeError(C_sparse_simd, C_sparse_ref);
        double sparse_simd_abs_error = maxAbsoluteError(C_sparse_simd, C_sparse_ref);
        LOG("  CSR-SpMM (SIMD) error: " << scientific << setprecision(2) 
             << sparse_simd_error << " (relative), " << sparse_simd_abs_error << " (absolute)\n");
        
        if (sparse_simd_error < 1e-10) {
            LOG("  ✓ CSR-SpMM (SIMD) validated successfully\n");
        } else if (sparse_simd_error < 1e-6) {
            LOG("  ⚠ CSR-SpMM (SIMD) within acceptable tolerance\n");
        } else {
            LOG("  ✗ WARNING: CSR-SpMM (SIMD) error exceeds tolerance\n");
        }
        
        LOG(fixed);
    }
    
    // Calculate FLOPs
    long long dense_flops = 2LL * m * n * k;
    long long sparse_flops = 2LL * A_sparse.nnz * n;
    
    // Benchmark Dense GEMM
    LOG("\n" << string(70, '-') << "\n");
    LOG("Performance Benchmarks:\n");
    LOG(string(70, '-') << "\n");
    LOG("\nDense GEMM (tiled):\n");
    auto dense_result = benchmarkFunction([&]() {
        denseGEMM_tiled(A_dense, B, C_dense, 64);
    });
    double dense_gflops = computeGFLOPS(dense_result.median, dense_flops);
    LOG("  Time: " << fixed << setprecision(6) << dense_result.median 
        << " s (±" << setprecision(6) << dense_result.stddev << " s)\n");
    LOG("  GFLOP/s: " << setprecision(2) << dense_gflops << "\n");
    if (dense_result.perf.cycles > 0) {
        LOG("  IPC: " << setprecision(2) << dense_result.perf.ipc << "\n");
        LOG("  Cache miss rate: " << setprecision(2) << dense_result.perf.cache_miss_rate << "%\n");
        LOG("  LLC miss rate: " << setprecision(2) << dense_result.perf.llc_miss_rate << "%\n");
    }
    
    // Benchmark CSR-SpMM
    LOG("\nCSR-SpMM (scalar):\n");
    auto sparse_result = benchmarkFunction([&]() {
        csrSpMM(A_sparse, B, C_sparse);
    });
    double sparse_gflops = computeGFLOPS(sparse_result.median, sparse_flops);
    double cpnz = (sparse_result.median * omp_get_max_threads() * 2.4e9) / A_sparse.nnz; // Assuming 2.4GHz
    LOG("  Time: " << fixed << setprecision(6) << sparse_result.median 
        << " s (±" << setprecision(6) << sparse_result.stddev << " s)\n");
    LOG("  GFLOP/s: " << setprecision(2) << sparse_gflops << "\n");
    LOG("  CPNZ: " << setprecision(2) << cpnz << " cycles/nonzero\n");
    if (sparse_result.perf.cycles > 0) {
        LOG("  IPC: " << setprecision(2) << sparse_result.perf.ipc << "\n");
        LOG("  Cache miss rate: " << setprecision(2) << sparse_result.perf.cache_miss_rate << "%\n");
        LOG("  LLC miss rate: " << setprecision(2) << sparse_result.perf.llc_miss_rate << "%\n");
    }
    
    // Benchmark CSR-SpMM SIMD
    LOG("\nCSR-SpMM (SIMD):\n");
    auto sparse_simd_result = benchmarkFunction([&]() {
        csrSpMM_simd(A_sparse, B, C_sparse_simd);
    });
    double sparse_simd_gflops = computeGFLOPS(sparse_simd_result.median, sparse_flops);
    double cpnz_simd = (sparse_simd_result.median * omp_get_max_threads() * 2.4e9) / A_sparse.nnz;
    LOG("  Time: " << fixed << setprecision(6) << sparse_simd_result.median 
        << " s (±" << setprecision(6) << sparse_simd_result.stddev << " s)\n");
    LOG("  GFLOP/s: " << setprecision(2) << sparse_simd_gflops << "\n");
    LOG("  CPNZ: " << setprecision(2) << cpnz_simd << " cycles/nonzero\n");
    LOG("  Speedup vs scalar: " << setprecision(2) << (sparse_result.median / sparse_simd_result.median) << "x\n");
    if (sparse_simd_result.perf.cycles > 0) {
        LOG("  IPC: " << setprecision(2) << sparse_simd_result.perf.ipc << "\n");
        LOG("  Cache miss rate: " << setprecision(2) << sparse_simd_result.perf.cache_miss_rate << "%\n");
        LOG("  LLC miss rate: " << setprecision(2) << sparse_simd_result.perf.llc_miss_rate << "%\n");
    }
    
    // Speedup comparison
    LOG("\nSpeedup Analysis:\n");
    LOG("  Dense vs Sparse (scalar): " << setprecision(2) 
         << (dense_result.median / sparse_result.median) << "x\n");
    LOG("  Dense vs Sparse (SIMD): " << setprecision(2) 
         << (dense_result.median / sparse_simd_result.median) << "x\n");
}

// ============================================================================
// Sweep Functions for Comprehensive Analysis
// ============================================================================

// Sweep over density with fixed matrix size
void densitySweep(int m, int k, int n, const vector<double>& densities, bool validate = false) {
    LOG("\n" << string(70, '=') << "\n");
    LOG("DENSITY SWEEP: Fixed size m=" << m << ", k=" << k << ", n=" << n << "\n");
    LOG(string(70, '=') << "\n");
    
    for (double density : densities) {
        runExperiment(m, k, n, density, validate);
    }
}

// Sweep over matrix sizes with fixed density
void sizeSweep(const vector<tuple<int,int,int>>& sizes, double density, bool validate = false) {
    LOG("\n" << string(70, '=') << "\n");
    LOG("SIZE SWEEP: Fixed density=" << fixed << setprecision(2) << (density * 100) << "%\n");
    LOG(string(70, '=') << "\n");
    
    for (const auto& [m, k, n] : sizes) {
        runExperiment(m, k, n, density, validate);
    }
}

// Sweep over different matrix structures (square, tall-skinny, fat)
void structureSweep(int base_size, double density, bool validate = false) {
    LOG("\n" << string(70, '=') << "\n");
    LOG("STRUCTURE SWEEP: Base size=" << base_size << ", density=" 
         << fixed << setprecision(2) << (density * 100) << "%\n");
    LOG(string(70, '=') << "\n");
    
    // Square
    LOG("\n--- Square Matrix ---\n");
    runExperiment(base_size, base_size, base_size, density, validate);
    
    // Tall-skinny (many rows, few columns)
    LOG("\n--- Tall-Skinny Matrix ---\n");
    runExperiment(base_size * 2, base_size, base_size / 2, density, validate);
    
    // Fat (few rows, many columns)
    LOG("\n--- Fat Matrix ---\n");
    runExperiment(base_size / 2, base_size, base_size * 2, density, validate);
    
    // Very tall-skinny
    LOG("\n--- Very Tall-Skinny Matrix ---\n");
    runExperiment(base_size * 4, base_size, base_size / 4, density, validate);
}

// Combined sweep: size × density grid
void gridSweep(const vector<int>& sizes, const vector<double>& densities, bool validate = false) {
    LOG("\n" << string(70, '=') << "\n");
    LOG("GRID SWEEP: " << sizes.size() << " sizes × " << densities.size() << " densities\n");
    LOG(string(70, '=') << "\n");
    
    for (int size : sizes) {
        for (double density : densities) {
            runExperiment(size, size, size, density, validate);
        }
    }
}

void printUsage(const char* program_name) {
    cout << "Usage: " << program_name << " [mode] [options]\n\n";
    cout << "Modes:\n";
    cout << "  default           - Run preset experiments\n";
    cout << "  density           - Sweep over densities\n";
    cout << "  size              - Sweep over matrix sizes\n";
    cout << "  structure         - Sweep over matrix structures\n";
    cout << "  grid              - Full grid sweep (size × density)\n";
    cout << "  custom m k n d    - Single experiment with custom parameters\n\n";
    cout << "Examples:\n";
    cout << "  " << program_name << " default\n";
    cout << "  " << program_name << " density\n";
    cout << "  " << program_name << " size\n";
    cout << "  " << program_name << " custom 1024 1024 1024 0.05\n";
}

int main(int argc, char* argv[]) {
    // Open log file with timestamp
    auto now = system_clock::now();
    auto now_time = system_clock::to_time_t(now);
    stringstream filename;
    filename << "benchmark_results_" << now_time << ".txt";
    
    g_logfile.open(filename.str());
    if (!g_logfile.is_open()) {
        cerr << "Warning: Could not open log file " << filename.str() << endl;
    } else {
        cout << "Logging results to: " << filename.str() << "\n\n";
    }
    
    cout << "Dense vs Sparse Matrix Multiplication Benchmark\n";
    cout << "================================================\n\n";
    
    if (g_logfile.is_open()) {
        g_logfile << "Dense vs Sparse Matrix Multiplication Benchmark\n";
        g_logfile << "================================================\n\n";
    }
    
    // System information
    cout << "System Configuration:\n";
    cout << "  OpenMP threads: " << omp_get_max_threads() << "\n";
    
    if (g_logfile.is_open()) {
        g_logfile << "System Configuration:\n";
        g_logfile << "  OpenMP threads: " << omp_get_max_threads() << "\n";
    }
    
#ifdef __AVX512F__
    cout << "  SIMD: AVX-512 (8 doubles/vector)\n";
    if (g_logfile.is_open()) g_logfile << "  SIMD: AVX-512 (8 doubles/vector)\n";
#elif defined(__AVX2__)
    cout << "  SIMD: AVX2 (4 doubles/vector)\n";
    if (g_logfile.is_open()) g_logfile << "  SIMD: AVX2 (4 doubles/vector)\n";
#elif defined(__AVX__)
    cout << "  SIMD: AVX (4 doubles/vector)\n";
    if (g_logfile.is_open()) g_logfile << "  SIMD: AVX (4 doubles/vector)\n";
#else
    cout << "  SIMD: None (scalar only)\n";
    if (g_logfile.is_open()) g_logfile << "  SIMD: None (scalar only)\n";
#endif
    
    // Display timing precision info
    cout << "  Timing: CLOCK_MONOTONIC (nanosecond precision)\n";
    if (g_logfile.is_open()) g_logfile << "  Timing: CLOCK_MONOTONIC (nanosecond precision)\n";
    
    // Check perf availability
    PerfCounters perf_test;
    if (perf_test.is_enabled()) {
        cout << "  Perf counters: Available\n";
        if (g_logfile.is_open()) g_logfile << "  Perf counters: Available\n";
    } else {
        cout << "  Perf counters: Unavailable (run: sudo sysctl -w kernel.perf_event_paranoid=1)\n";
        if (g_logfile.is_open()) g_logfile << "  Perf counters: Unavailable\n";
    }
    
    // Parse command-line arguments
    string mode = (argc > 1) ? argv[1] : "default";
    
    if (mode == "help" || mode == "-h" || mode == "--help") {
        printUsage(argv[0]);
        return 0;
    }
    
    if (mode == "custom" && argc == 6) {
        // Custom single experiment: m k n density
        int m = atoi(argv[2]);
        int k = atoi(argv[3]);
        int n = atoi(argv[4]);
        double density = atof(argv[5]);
        
        runExperiment(m, k, n, density, true);
        
    } else if (mode == "density") {
        // Density sweep: 0.1% to 50%
        vector<double> densities = {0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50};
        densitySweep(1024, 1024, 1024, densities, false);
        
    } else if (mode == "size") {
        // Size sweep with fixed density
        vector<tuple<int,int,int>> sizes = {
            {256, 256, 256},
            {512, 512, 512},
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096}
        };
        sizeSweep(sizes, 0.05, false);
        
    } else if (mode == "structure") {
        // Structure sweep
        structureSweep(1024, 0.05, false);
        
    } else if (mode == "grid") {
        // Full grid sweep
        vector<int> sizes = {512, 1024, 2048};
        vector<double> densities = {0.01, 0.05, 0.10, 0.20};
        gridSweep(sizes, densities, false);
        
    } else {
        // Default: preset experiments
        LOG("\n--- Running Default Experiments ---\n");
        
        // Small matrix (fits in cache)
        runExperiment(512, 512, 512, 0.05);
        
        // Medium matrix
        runExperiment(1024, 1024, 1024, 0.10);
        
        // Larger matrix with lower density
        runExperiment(2048, 2048, 512, 0.02);
        
        // Large matrix with high density
        runExperiment(4048, 4048, 512, 0.15);

        // Rectangular (tall-skinny)
        runExperiment(2048, 512, 128, 0.05);
    }
    
    cout << "\n" << string(70, '=') << "\n";
    cout << "Benchmark Complete!\n";
    cout << string(70, '=') << "\n";
    
    if (g_logfile.is_open()) {
        g_logfile << "\n" << string(70, '=') << "\n";
        g_logfile << "Benchmark Complete!\n";
        g_logfile << "======================================================================\n";
        g_logfile.close();
        cout << "\nResults saved to: " << filename.str() << "\n";
    }
    
    return 0;
}
