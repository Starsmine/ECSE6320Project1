#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <emmintrin.h>  // SSE2 intrinsics
#include <fstream>
#include <cstdlib>
#include <random>
#include <string>
#include <algorithm>
#ifdef _WIN32
  #include <malloc.h> // for _aligned_malloc/_aligned_free
#endif

static bool use_aligned_loads = true; // toggled per-test
static bool csv_header_written = false;

// Helper functions for aligned memory
static inline void* aligned_malloc_bytes(size_t bytes, size_t align) {
#ifdef _WIN32
    return _aligned_malloc(bytes, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
#endif
}

static inline void aligned_free_bytes(void* p) {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

template <typename T>
static T* aligned_alloc_array(size_t nelems, size_t align = 64) {
    return reinterpret_cast<T*>(aligned_malloc_bytes(nelems * sizeof(T), align));
}

template <typename T>
static void aligned_free_array(T* p) {
    aligned_free_bytes(reinterpret_cast<void*>(p));
}

// Load operations with alignment handling
static inline __m128 load_ps128(const float* p) { return use_aligned_loads ? _mm_load_ps(p) : _mm_loadu_ps(p); }
static inline __m128d load_pd128(const double* p) { return use_aligned_loads ? _mm_load_pd(p) : _mm_loadu_pd(p); }
static inline __m128i load_si128(const int32_t* p) { return use_aligned_loads ? _mm_load_si128((const __m128i*)p) : _mm_loadu_si128((const __m128i*)p); }

static inline __m256 load_ps256(const float* p) { return use_aligned_loads ? _mm256_load_ps(p) : _mm256_loadu_ps(p); }
static inline __m256d load_pd256(const double* p) { return use_aligned_loads ? _mm256_load_pd(p) : _mm256_loadu_pd(p); }
static inline __m256i load_si256(const int32_t* p) { return use_aligned_loads ? _mm256_load_si256((const __m256i*)p) : _mm256_loadu_si256((const __m256i*)p); }

static inline __m512 load_ps512(const float* p) { return use_aligned_loads ? _mm512_load_ps(p) : _mm512_loadu_ps(p); }
static inline __m512d load_pd512(const double* p) { return use_aligned_loads ? _mm512_load_pd(p) : _mm512_loadu_pd(p); }
static inline __m512i load_si512(const int32_t* p) { return use_aligned_loads ? _mm512_load_si512((const void*)p) : _mm512_loadu_si512((const void*)p); }

template <typename T>
constexpr const char* dtype_name() {
    if (std::is_same<T, float>::value)  return "float32";
    if (std::is_same<T, double>::value) return "float64";
    if (std::is_same<T, int32_t>::value) return "int32";
    return "unknown";
}

template <typename T>
void log_per_trial(const std::string &filename,
                   const std::string &kernel,
                   const std::string &dtype,
                   size_t n_buf,
                   int trial,
                   double time_ms,
                   double gflops,
                   T result,
                   const std::string &pattern,
                   size_t stride)
{
    std::ofstream f(filename, std::ios::app);
    if (!f) {
        std::cerr << "Failed to open CSV file: " << filename << "\n";
        return;
    }
    if (!csv_header_written) {
        f << "Kernel,Type,N_buf,Pattern,Stride,Trial,Time(ms),GFLOP/s,Result\n";
        csv_header_written = true;
    }
    f << kernel << "," << dtype << "," << n_buf << "," << pattern << ","
      << stride << "," << trial << "," << time_ms << "," << gflops << ","
      << result << "\n";
}

// Scalar implementation
template <typename T>
T dotp_scalar(const T* x, const T* y, size_t n) {
    T sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum = std::fma(x[i], y[i], sum);  // Use FMA for better precision
    }
    return sum;
}

// SSE2 implementation with optimized horizontal sum
template <typename T>
T dotp_sse2(const T* x, const T* y, size_t n);

template <>
float dotp_sse2<float>(const float* x, const float* y, size_t n) {
    __m128 sum = _mm_setzero_ps();
    size_t i = 0;
    
    // Main loop - process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        __m128 xvec = load_ps128(x + i);
        __m128 yvec = load_ps128(y + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(xvec, yvec));
    }
    
    // Optimized horizontal sum using SSE3 instructions
    __m128 shuf = _mm_movehdup_ps(sum);    // Broadcast elements 1,3 to 0,2
    __m128 sums = _mm_add_ps(sum, shuf);   // Add pairs
    shuf = _mm_movehl_ps(shuf, sums);      // High half -> low half
    sums = _mm_add_ss(sums, shuf);         // Add last pair
    float final_sum;
    _mm_store_ss(&final_sum, sums);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_sum = std::fma(x[i], y[i], final_sum);
    }
    
    return final_sum;
}

template <>
double dotp_sse2<double>(const double* x, const double* y, size_t n) {
    __m128d sum = _mm_setzero_pd();
    size_t i = 0;
    
    // Main loop - process 2 elements at a time
    for (; i + 1 < n; i += 2) {
        __m128d xvec = load_pd128(x + i);
        __m128d yvec = load_pd128(y + i);
        sum = _mm_add_pd(sum, _mm_mul_pd(xvec, yvec));
    }
    
    // Horizontal sum
    __m128d high = _mm_unpackhi_pd(sum, sum);
    __m128d result = _mm_add_sd(sum, high);
    double final_sum;
    _mm_store_sd(&final_sum, result);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_sum = std::fma(x[i], y[i], final_sum);
    }
    
    return final_sum;
}

// AVX2 implementation with FMA
template <typename T>
T dotp_avx2(const T* x, const T* y, size_t n);

template <>
float dotp_avx2<float>(const float* x, const float* y, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    // Main loop - process 8 elements at a time using FMA
    for (; i + 7 < n; i += 8) {
        __m256 xvec = load_ps256(x + i);
        __m256 yvec = load_ps256(y + i);
        sum = _mm256_fmadd_ps(xvec, yvec, sum);
    }
    
    // Optimized horizontal sum for AVX
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(low, high);
    
    // Now sum the remaining 4 floats using SSE
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    
    float final_sum;
    _mm_store_ss(&final_sum, sums);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_sum = std::fma(x[i], y[i], final_sum);
    }
    
    return final_sum;
}

template <>
double dotp_avx2<double>(const double* x, const double* y, size_t n) {
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    
    // Main loop - process 4 elements at a time using FMA
    for (; i + 3 < n; i += 4) {
        __m256d xvec = load_pd256(x + i);
        __m256d yvec = load_pd256(y + i);
        sum = _mm256_fmadd_pd(xvec, yvec, sum);
    }
    
    // Optimized horizontal sum for AVX
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d sum128 = _mm_add_pd(low, high);
    
    // Sum the remaining 2 doubles
    __m128d high64 = _mm_unpackhi_pd(sum128, sum128);
    __m128d result = _mm_add_sd(sum128, high64);
    
    double final_sum;
    _mm_store_sd(&final_sum, result);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_sum = std::fma(x[i], y[i], final_sum);
    }
    
    return final_sum;
}

// AVX-512 implementation with built-in reduction
template <typename T>
T dotp_avx512(const T* x, const T* y, size_t n);

template <>
float dotp_avx512<float>(const float* x, const float* y, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    
    // Main loop - process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        __m512 xvec = load_ps512(x + i);
        __m512 yvec = load_ps512(y + i);
        sum = _mm512_fmadd_ps(xvec, yvec, sum);
    }
    
    // Process remaining elements with masking
    if (i < n) {
        __mmask16 mask = (1ULL << (n - i)) - 1;
        __m512 xvec = _mm512_maskz_load_ps(mask, x + i);
        __m512 yvec = _mm512_maskz_load_ps(mask, y + i);
        sum = _mm512_maskz_fmadd_ps(mask, xvec, yvec, sum);
    }
    
    // Use efficient built-in reduction
    return _mm512_reduce_add_ps(sum);
}

template <>
double dotp_avx512<double>(const double* x, const double* y, size_t n) {
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;
    
    // Main loop - process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        __m512d xvec = load_pd512(x + i);
        __m512d yvec = load_pd512(y + i);
        sum = _mm512_fmadd_pd(xvec, yvec, sum);
    }
    
    // Process remaining elements with masking
    if (i < n) {
        __mmask8 mask = (1ULL << (n - i)) - 1;
        __m512d xvec = _mm512_maskz_load_pd(mask, x + i);
        __m512d yvec = _mm512_maskz_load_pd(mask, y + i);
        sum = _mm512_maskz_fmadd_pd(mask, xvec, yvec, sum);
    }
    
    // Use efficient built-in reduction
    return _mm512_reduce_add_pd(sum);
}

// Int32 implementations with 64-bit accumulation to prevent overflow
template <>
int32_t dotp_scalar<int32_t>(const int32_t* x, const int32_t* y, size_t n) {
    int64_t sum = 0;  // Use 64-bit accumulator to prevent overflow
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<int64_t>(x[i]) * y[i];
    }
    return static_cast<int32_t>(sum);  // Convert back to int32_t
}

template <>
int32_t dotp_sse2<int32_t>(const int32_t* x, const int32_t* y, size_t n) {
    __m128i sum_lo = _mm_setzero_si128();  // Lower 32 bits of products
    __m128i sum_hi = _mm_setzero_si128();  // Upper 32 bits of products
    size_t i = 0;
    
    // Process 4 elements at a time
    for (; i + 3 < n; i += 4) {
        __m128i xvec = load_si128(x + i);
        __m128i yvec = load_si128(y + i);
        
        // Multiply with 64-bit results
        __m128i lo = _mm_mul_epi32(xvec, yvec);          // Multiply even indices
        __m128i hi = _mm_mul_epi32(_mm_srli_si128(xvec, 4), 
                                  _mm_srli_si128(yvec, 4)); // Multiply odd indices
        
        sum_lo = _mm_add_epi64(sum_lo, lo);
        sum_hi = _mm_add_epi64(sum_hi, hi);
    }
    
    // Horizontal sum with 64-bit accumulation
    __m128i sum = _mm_add_epi64(sum_lo, sum_hi);
    int64_t final_sum = _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_sum += static_cast<int64_t>(x[i]) * y[i];
    }
    
    return static_cast<int32_t>(final_sum);
}

template <>
int32_t dotp_avx2<int32_t>(const int32_t* x, const int32_t* y, size_t n) {
    __m256i sum_lo = _mm256_setzero_si256();
    __m256i sum_hi = _mm256_setzero_si256();
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        __m256i xvec = load_si256(x + i);
        __m256i yvec = load_si256(y + i);
        
        // Multiply with 64-bit results
        __m256i lo = _mm256_mul_epi32(xvec, yvec);          // Multiply even indices
        __m256i hi = _mm256_mul_epi32(_mm256_srli_si256(xvec, 4), 
                                     _mm256_srli_si256(yvec, 4)); // Multiply odd indices
        
        sum_lo = _mm256_add_epi64(sum_lo, lo);
        sum_hi = _mm256_add_epi64(sum_hi, hi);
    }
    
    // Horizontal sum with 64-bit accumulation
    __m256i sum = _mm256_add_epi64(sum_lo, sum_hi);
    __m128i sum128 = _mm_add_epi64(_mm256_extracti128_si256(sum, 0),
                                  _mm256_extracti128_si256(sum, 1));
    int64_t final_sum = _mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1);
    
    // Handle remaining elements
    for (; i < n; ++i) {
        final_sum += static_cast<int64_t>(x[i]) * y[i];
    }
    
    return static_cast<int32_t>(final_sum);
}

template <>
int32_t dotp_avx512<int32_t>(const int32_t* x, const int32_t* y, size_t n) {
    __m512i sum_lo = _mm512_setzero_si512();
    __m512i sum_hi = _mm512_setzero_si512();
    size_t i = 0;
    
    // Process 16 elements at a time
    for (; i + 15 < n; i += 16) {
        __m512i xvec = load_si512(x + i);
        __m512i yvec = load_si512(y + i);
        
        // Multiply with 64-bit results
        __m512i lo = _mm512_mul_epi32(xvec, yvec);          // Multiply even indices
        __m512i hi = _mm512_mul_epi32(_mm512_srli_epi64(xvec, 32), 
                                     _mm512_srli_epi64(yvec, 32)); // Multiply odd indices
        
        sum_lo = _mm512_add_epi64(sum_lo, lo);
        sum_hi = _mm512_add_epi64(sum_hi, hi);
    }
    
    // Process remaining elements with masking
    if (i < n) {
        __mmask16 mask = (1ULL << (n - i)) - 1;
        __m512i xvec = _mm512_maskz_load_epi32(mask, x + i);
        __m512i yvec = _mm512_maskz_load_epi32(mask, y + i);
        
        __m512i lo = _mm512_mul_epi32(xvec, yvec);
        __m512i hi = _mm512_mul_epi32(_mm512_srli_epi64(xvec, 32),
                                     _mm512_srli_epi64(yvec, 32));
        
        sum_lo = _mm512_add_epi64(sum_lo, lo);
        sum_hi = _mm512_add_epi64(sum_hi, hi);
    }
    
    // Horizontal sum using AVX-512 reduction
    __m512i sum = _mm512_add_epi64(sum_lo, sum_hi);
    int64_t final_sum = _mm512_reduce_add_epi64(sum);
    
    return static_cast<int32_t>(final_sum);
}

// Strided dot product wrapper
template <typename T>
T dotp_strided_wrapper(T (*simd_func)(const T*, const T*, size_t),
                      const T* x, const T* y, size_t touched_actual, size_t stride)
{
    if (stride == 1) {
        return simd_func(x, y, touched_actual);
    }
    
    T sum = 0;
    for (size_t k = 0; k < touched_actual; ++k) {
        size_t i = k * stride;
        sum = std::fma(x[i], y[i], sum);
    }
    return sum;
}

// Gather dot product wrapper
template <typename T>
T dotp_gather_wrapper(const T* x, const T* y, const size_t* idx, size_t m)
{
    T sum = 0;
    for (size_t k = 0; k < m; ++k) {
        size_t i = idx[k];
        sum = std::fma(x[i], y[i], sum);
    }
    return sum;
}

template <typename T>
void sweep_benchmark(const char* label, T (*func)(const T*, const T*, size_t)) {
    std::vector<size_t> sizes = {
        1 << 10,  // 1 KB
        1 << 12,  // 4 KB
        1 << 14,  // 16 KB
        1 << 16,  // 64 KB
        1 << 18,  // 256 KB
        1 << 20,  // 1 MB
        1 << 22,  // 4 MB
        1 << 24,   // 16 MB
        1 << 26,   // 64 MB
        1 << 28,   // 256 MB
    };

    const int trials = 10;
    const size_t ALIGN = 64;

    for (bool mis : {false, true}) {
        use_aligned_loads = !mis;
        for (size_t n : sizes) {
            // allocate one extra element so we can create a misaligned view by byte-offset
            T* base_x = aligned_alloc_array<T>(n + 1, ALIGN);
            T* base_y = aligned_alloc_array<T>(n + 1, ALIGN);
            if (!base_x || !base_y) {
                std::cerr << "aligned_alloc failed for N=" << n << "\n";
                aligned_free_array(base_x);
                aligned_free_array(base_y);
                continue;
            }

            // Initialize with more interesting data pattern
            for (size_t i = 0; i < n + 1; ++i) {
                base_x[i] = T(1.0 + std::sin(i * 0.1));
                base_y[i] = T(0.5 + std::cos(i * 0.1));
            }

            T* x = base_x;
            T* y = base_y;
            size_t effective_n = n;
            std::string lbl = label;
            if (mis) {
                // create misaligned view by offsetting by one element
                x = reinterpret_cast<T*>(reinterpret_cast<char*>(base_x) + sizeof(T));
                y = reinterpret_cast<T*>(reinterpret_cast<char*>(base_y) + sizeof(T));
                effective_n = (n > 0) ? n - 1 : 0;
                lbl += " (misaligned)";
            }

            if (effective_n == 0) {
                aligned_free_array(base_x);
                aligned_free_array(base_y);
                continue;
            }

            for (int t = 0; t < trials; ++t) {
                // Warm-up
                T result = func(x, y, effective_n);

                auto start = std::chrono::high_resolution_clock::now();
                result = func(x, y, effective_n);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                double seconds = elapsed.count();
                double flops = 2.0 * static_cast<double>(effective_n); // multiply and add per element
                double gflops = (seconds > 0.0) ? (flops / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;

                std::cout << lbl << " | N = " << n << (mis ? " (misaligned)" : "") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  Result = " << result << "\n\n";

                log_per_trial<T>("dotp_benchmark_v2.csv",
                                lbl,
                                dtype_name<T>(),
                                n,
                                t + 1,
                                time_ms,
                                gflops,
                                result,
                                "contiguous",
                                1);
            }

            aligned_free_array(base_x);
            aligned_free_array(base_y);
        }
    }
}

template <typename T>
void sweep_stride_benchmark(const char* kernel_label,
                          T (*simd_func)(const T*, const T*, size_t),
                          int trials = 10)
{
    std::vector<size_t> ms = { 1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
    std::vector<size_t> strides = { 1, 2, 4, 8, 16 };
    const size_t ALIGN = 64;

    for (bool mis : {false, true}) {
        use_aligned_loads = !mis;
        for (size_t m : ms) {
            size_t max_stride = *std::max_element(strides.begin(), strides.end());
            size_t buf_elems = m * max_stride + 1;
            T* base_x = aligned_alloc_array<T>(buf_elems, ALIGN);
            T* base_y = aligned_alloc_array<T>(buf_elems, ALIGN);
            
            if (!base_x || !base_y) {
                std::cerr << "aligned_alloc failed for m=" << m << "\n";
                aligned_free_array(base_x);
                aligned_free_array(base_y);
                continue;
            }

            // Initialize with more interesting pattern
            for (size_t i = 0; i < buf_elems; ++i) {
                base_x[i] = T(1.0 + std::sin(i * 0.1));
                base_y[i] = T(0.5 + std::cos(i * 0.1));
            }

            T* x = base_x;
            T* y = base_y;
            std::string mis_suf = "";
            if (mis) {
                x = reinterpret_cast<T*>(reinterpret_cast<char*>(base_x) + sizeof(T));
                y = reinterpret_cast<T*>(reinterpret_cast<char*>(base_y) + sizeof(T));
                mis_suf = " (misaligned)";
            }

            // Stride tests
            for (size_t stride : strides) {
                size_t touched_actual = std::min(m, buf_elems / stride);
                if (touched_actual == 0) continue;

                double flops_per_trial = 2.0 * double(touched_actual);
                std::string label = std::string(kernel_label) + " stride=" + std::to_string(stride) + mis_suf;

                for (int t = 0; t < trials; ++t) {
                    // Warm-up
                    T result = dotp_strided_wrapper(simd_func, x, y, touched_actual, stride);

                    auto start = std::chrono::high_resolution_clock::now();
                    result = dotp_strided_wrapper(simd_func, x, y, touched_actual, stride);
                    auto end = std::chrono::high_resolution_clock::now();

                    double seconds = std::chrono::duration<double>(end - start).count();
                    double gflops = (seconds > 0.0) ? (flops_per_trial / (seconds * 1e9)) : 0.0;
                    double time_ms = seconds * 1e3;

                    std::cout << label << " | buf_elems = " << buf_elems << ":\n";
                    std::cout << "  Time: " << time_ms << " ms\n";
                    std::cout << "  GFLOP/s: " << gflops << "\n";
                    std::cout << "  Result = " << result << "\n\n";

                    log_per_trial<T>("dotp_benchmark_v2.csv",
                                    label,
                                    dtype_name<T>(),
                                    buf_elems,
                                    t + 1,
                                    time_ms,
                                    gflops,
                                    result,
                                    "strided",
                                    stride);
                }
            }

            // Gather test
            std::vector<size_t> idx(m);
            std::mt19937_64 rng(12345);
            for (size_t k = 0; k < m; ++k) {
                idx[k] = (k * max_stride) % (buf_elems - 1);
            }
            std::shuffle(idx.begin(), idx.end(), rng);

            for (int t = 0; t < trials; ++t) {
                T result = dotp_gather_wrapper(x, y, idx.data(), m);

                auto start = std::chrono::high_resolution_clock::now();
                result = dotp_gather_wrapper(x, y, idx.data(), m);
                auto end = std::chrono::high_resolution_clock::now();

                double seconds = std::chrono::duration<double>(end - start).count();
                double gflops = (2.0 * m) / (seconds * 1e9);
                double time_ms = seconds * 1e3;

                log_per_trial<T>("dotp_benchmark_v2.csv",
                                std::string(kernel_label) + mis_suf,
                                dtype_name<T>(),
                                buf_elems,
                                t + 1,
                                time_ms,
                                gflops,
                                result,
                                "gather",
                                0);
            }

            aligned_free_array(base_x);
            aligned_free_array(base_y);
        }
    }
}

int main() {
    // Clear the benchmark file
    std::ofstream f("dotp_benchmark_v2.csv", std::ios::trunc);
    f.close();
    csv_header_written = false;

    // Float32 benchmarks
    std::cout << "\n=== Float32 Benchmarks ===\n";
    sweep_benchmark("Scalar float32", dotp_scalar<float>);
    sweep_benchmark("SSE2 float32", dotp_sse2<float>);
    sweep_benchmark("AVX2 float32", dotp_avx2<float>);
    sweep_benchmark("AVX512 float32", dotp_avx512<float>);

    // Float64 benchmarks
    std::cout << "\n=== Float64 Benchmarks ===\n";
    sweep_benchmark("Scalar float64", dotp_scalar<double>);
    sweep_benchmark("SSE2 float64", dotp_sse2<double>);
    sweep_benchmark("AVX2 float64", dotp_avx2<double>);
    sweep_benchmark("AVX512 float64", dotp_avx512<double>);

    // Strided benchmarks
    std::cout << "\n=== Strided Float32 Benchmarks ===\n";
    sweep_stride_benchmark("Scalar float32", dotp_scalar<float>);
    sweep_stride_benchmark("SSE2 float32", dotp_sse2<float>);
    sweep_stride_benchmark("AVX2 float32", dotp_avx2<float>);
    sweep_stride_benchmark("AVX512 float32", dotp_avx512<float>);

    std::cout << "\n=== Strided Float64 Benchmarks ===\n";
    sweep_stride_benchmark("Scalar float64", dotp_scalar<double>);
    sweep_stride_benchmark("SSE2 float64", dotp_sse2<double>);
    sweep_stride_benchmark("AVX2 float64", dotp_avx2<double>);
    sweep_stride_benchmark("AVX512 float64", dotp_avx512<double>);

    // Int32 benchmarks
    std::cout << "\n=== Int32 Benchmarks ===\n";
    sweep_benchmark("Scalar int32", dotp_scalar<int32_t>);
    sweep_benchmark("SSE2 int32", dotp_sse2<int32_t>);
    sweep_benchmark("AVX2 int32", dotp_avx2<int32_t>);
    sweep_benchmark("AVX512 int32", dotp_avx512<int32_t>);

    // Strided Int32 benchmarks
    std::cout << "\n=== Strided Int32 Benchmarks ===\n";
    sweep_stride_benchmark("Scalar int32", dotp_scalar<int32_t>);
    sweep_stride_benchmark("SSE2 int32", dotp_sse2<int32_t>);
    sweep_stride_benchmark("AVX2 int32", dotp_avx2<int32_t>);
    sweep_stride_benchmark("AVX512 int32", dotp_avx512<int32_t>);

    return 0;
}