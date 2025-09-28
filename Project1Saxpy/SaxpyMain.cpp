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

// SSE (128-bit)
static inline __m128 load_ps128(const float* p) { return use_aligned_loads ? _mm_load_ps(p) : _mm_loadu_ps(p); }
static inline void store_ps128(float* p, __m128 v) { if (use_aligned_loads) _mm_store_ps(p, v); else _mm_storeu_ps(p, v); }
static inline __m128d load_pd128(const double* p) { return use_aligned_loads ? _mm_load_pd(p) : _mm_loadu_pd(p); }
static inline void store_pd128(double* p, __m128d v) { if (use_aligned_loads) _mm_store_pd(p, v); else _mm_storeu_pd(p, v); }
static inline __m128i load_si128(const int32_t* p) { return use_aligned_loads ? _mm_load_si128((const __m128i*)p) : _mm_loadu_si128((const __m128i*)p); }
static inline void store_si128(int32_t* p, __m128i v) { if (use_aligned_loads) _mm_store_si128((__m128i*)p, v); else _mm_storeu_si128((__m128i*)p, v); }

// AVX (256-bit)
static inline __m256 load_ps256(const float* p) { return use_aligned_loads ? _mm256_load_ps(p) : _mm256_loadu_ps(p); }
static inline void store_ps256(float* p, __m256 v) { if (use_aligned_loads) _mm256_store_ps(p, v); else _mm256_storeu_ps(p, v); }
static inline __m256d load_pd256(const double* p) { return use_aligned_loads ? _mm256_load_pd(p) : _mm256_loadu_pd(p); }
static inline void store_pd256(double* p, __m256d v) { if (use_aligned_loads) _mm256_store_pd(p, v); else _mm256_storeu_pd(p, v); }
static inline __m256i load_si256(const int32_t* p) { return use_aligned_loads ? _mm256_load_si256((const __m256i*)p) : _mm256_loadu_si256((const __m256i*)p); }
static inline void store_si256(int32_t* p, __m256i v) { if (use_aligned_loads) _mm256_store_si256((__m256i*)p, v); else _mm256_storeu_si256((__m256i*)p, v); }

// AVX-512 (512-bit)
static inline __m512 load_ps512(const float* p) { return use_aligned_loads ? _mm512_load_ps(p) : _mm512_loadu_ps(p); }
static inline void store_ps512(float* p, __m512 v) { if (use_aligned_loads) _mm512_store_ps(p, v); else _mm512_storeu_ps(p, v); }
static inline __m512d load_pd512(const double* p) { return use_aligned_loads ? _mm512_load_pd(p) : _mm512_loadu_pd(p); }
static inline void store_pd512(double* p, __m512d v) { if (use_aligned_loads) _mm512_store_pd(p, v); else _mm512_storeu_pd(p, v); }
static inline __m512i load_si512(const int32_t* p) { return use_aligned_loads ? _mm512_load_si512((const __m512i*)p) : _mm512_loadu_si512((const __m512i*)p); }
static inline void store_si512(int32_t* p, __m512i v) { if (use_aligned_loads) _mm512_store_si512((__m512i*)p, v); else _mm512_storeu_si512((__m512i*)p, v); }

// Note: AVX-512 masked load/store (maskz/mask_store) are unaligned-safe; keep using them for tails.

static bool csv_header_written = false;
static bool fast_mode = false;

template <typename T>
constexpr const char* dtype_name() {
    if (std::is_same<T, float>::value)  return "float32";
    if (std::is_same<T, double>::value) return "float64";
    if (std::is_same<T, int32_t>::value) return "int32";
    return "unknown";
}

// Replace your existing log_per_trial with this version (add Pattern and Stride)
template <typename T>
void log_per_trial(const std::string &filename,
                   const std::string &kernel,
                   const std::string &dtype,
                   size_t n,
                   int trial,
                   double time_ms,
                   double gflops,
                   T y0,
                   bool aligned,
                   const std::string &pattern,
                   size_t stride)
{
    std::ofstream f(filename, std::ios::app);
    if (!f) {
        std::cerr << "Failed to open CSV file: " << filename << "\n";
        return;
    }

    if (!csv_header_written) {
        f << "Kernel,Type,N,Trial,Time(ms),GFLOP/s,y0,Aligned,Pattern,Stride\n";
        csv_header_written = true;
    }

        // Write CSV row with quoted kernel and pattern fields
        f << '"' << kernel << '"' << ','
            << dtype << ',' << n << ',' << trial << ','
            << time_ms << ',' << gflops << ',' << y0 << ','
            << (aligned ? "aligned" : "misaligned") << ','
            << '"' << pattern << '"' << ',' << stride << '\n';

    f.close();
}
// -- Aligned allocation helpers (platform-specific)
static void* aligned_malloc_bytes(size_t bytes, size_t align) {
#ifdef _WIN32
    return _aligned_malloc(bytes, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
#endif
}

static void aligned_free_bytes(void* p) {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

    template <typename T>
    static T* aligned_alloc_array(size_t nelems, size_t align = 64) {
        size_t bytes = nelems * sizeof(T);
        return reinterpret_cast<T*>(aligned_malloc_bytes(bytes, align));
    }
    
    template <typename T>
    static void aligned_free_array(T* p) {
        aligned_free_bytes(reinterpret_cast<void*>(p));
    }    


// --- safe strided wrapper (unchanged semantics, but caller must pass touched_actual) ---
template <typename T>
void saxpy_strided_wrapper(void (*simd_func)(T,const T*,T*,size_t),
                           T a, const T* x, T* y, size_t touched_actual, size_t stride)
{
    if (stride == 1) {
        // contiguous: call SIMD implementation directly
        simd_func(a, x, y, touched_actual);
        return;
    }
    // strided scalar loop (touch touched_actual elements at distance stride)
    for (size_t k = 0; k < touched_actual; ++k) {
        size_t i = k * stride;
        y[i] = a * x[i] + y[i];
    }
}

    
// Gather-like wrapper that uses an index array (always scalar here)
template <typename T>
void saxpy_gather_wrapper(T a, const T* x, T* y, const size_t* idx, size_t m)
{
    for (size_t k = 0; k < m; ++k) {
        size_t i = idx[k];
        y[i] = a * x[i] + y[i];
    }
}

// New: gather-read variant: read from scattered x (at idx) and write contiguous y[0..m-1]
template <typename T>
void saxpy_gather_read_xidx_write_contig_y(T a, const T* x, T* y, const size_t* idx, size_t m)
{
    for (size_t k = 0; k < m; ++k) {
        size_t i = idx[k];
        y[k] = a * x[i] + y[k];
    }
}

// Vectorized gather-read for float using AVX2 _mm256_i32gather_ps
template <>
void saxpy_gather_read_xidx_write_contig_y<float>(float a, const float* x, float* y, const size_t* idx, size_t m) {
    size_t k = 0;
    __m256 a_vec = _mm256_set1_ps(a);
    for (; k + 7 < m; k += 8) {
        // load 8 32-bit indices
        __m256i idxv = _mm256_loadu_si256((const __m256i*)&idx[k]);
        __m256 xg = _mm256_i32gather_ps(x, idxv, 4);
        __m256 yv = _mm256_loadu_ps(&y[k]);
        __m256 r = _mm256_fmadd_ps(a_vec, xg, yv);
        _mm256_storeu_ps(&y[k], r);
    }
    for (; k < m; ++k) y[k] = a * x[idx[k]] + y[k];
}

// Vectorized gather-read for double using AVX2 _mm256_i64gather_pd
template <>
void saxpy_gather_read_xidx_write_contig_y<double>(double a, const double* x, double* y, const size_t* idx, size_t m) {
    size_t k = 0;
    __m256d a_vec = _mm256_set1_pd(a);
    for (; k + 3 < m; k += 4) {
        // load 4 64-bit indices
        __m256i idxv = _mm256_loadu_si256((const __m256i*)&idx[k]);
        __m256d xg = _mm256_i64gather_pd(x, idxv, 8);
        __m256d yv = _mm256_loadu_pd(&y[k]);
        __m256d r = _mm256_fmadd_pd(a_vec, xg, yv);
        _mm256_storeu_pd(&y[k], r);
    }
    for (; k < m; ++k) y[k] = a * x[idx[k]] + y[k];
}

// New: scatter-write variant: read contiguous x[0..m-1] and write to scattered y at idx
template <typename T>
void saxpy_scatter_write_yidx_read_contig_x(T a, const T* x, T* y, const size_t* idx, size_t m)
{
    for (size_t k = 0; k < m; ++k) {
        size_t i = idx[k];
        y[i] = a * x[k] + y[i];
    }
}


template <typename T>
void saxpy_scalar(T a, const T* x, T* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

template <>
void saxpy_scalar<int32_t>(int32_t a, const int32_t* x, int32_t* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}


template <typename T>
void saxpy_avx(T a, const T* x, T* y, size_t n);

template <>
void saxpy_avx<float>(float a, const float* x, float* y, size_t n) {
    __m256 a_vec = _mm256_set1_ps(a);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = load_ps256(&x[i]);
        __m256 y_vec = load_ps256(&y[i]);
        __m256 result = _mm256_fmadd_ps(a_vec, x_vec, y_vec);
        store_ps256(&y[i], result);

    }
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

template <>
void saxpy_avx<double>(double a, const double* x, double* y, size_t n) {
    __m256d a_vec = _mm256_set1_pd(a);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d x_vec = load_pd256(&x[i]);
        __m256d y_vec = load_pd256(&y[i]);
        __m256d result = _mm256_fmadd_pd(a_vec, x_vec, y_vec);
        store_pd256(&y[i], result);

    }
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

template <>
void saxpy_avx<int32_t>(int32_t a, const int32_t* x, int32_t* y, size_t n) {
    __m256i a_vec = _mm256_set1_epi32(a);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256i x_vec = load_si256(&x[i]);
        __m256i y_vec = load_si256(&y[i]);
        __m256i prod = _mm256_mullo_epi32(a_vec, x_vec);
        __m256i result = _mm256_add_epi32(prod, y_vec);
       store_si256(&y[i], result);
    }
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}


template <typename T>
void saxpy_avx512(T a, const T* x, T* y, size_t n);

template <>
void saxpy_avx512<float>(float a, const float* x, float* y, size_t n) {
    __m512 a_vec = _mm512_set1_ps(a);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = load_ps512(&x[i]);
        __m512 y_vec = load_ps512(&y[i]);
        __m512 result = _mm512_fmadd_ps(a_vec, x_vec, y_vec);
        store_ps512(&y[i], result);
    }
    if (i < n) {
        __mmask16 mask = (1 << (n - i)) - 1;
        __m512 x_vec = _mm512_maskz_load_ps(mask, &x[i]);
        __m512 y_vec = _mm512_maskz_load_ps(mask, &y[i]);
        __m512 result = _mm512_maskz_fmadd_ps(mask, a_vec, x_vec, y_vec);
        _mm512_mask_storeu_ps(&y[i], mask, result);
    }
}

template <>
void saxpy_avx512<double>(double a, const double* x, double* y, size_t n) {
    __m512d a_vec = _mm512_set1_pd(a);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m512d x_vec = load_pd512(&x[i]);
        __m512d y_vec = load_pd512(&y[i]);
        __m512d result = _mm512_fmadd_pd(a_vec, x_vec, y_vec);
        store_pd512(&y[i], result);
    }
    if (i < n) {
        __mmask8 mask = (1 << (n - i)) - 1;
        __m512d x_vec = _mm512_maskz_load_pd(mask, &x[i]);
        __m512d y_vec = _mm512_maskz_load_pd(mask, &y[i]);
        __m512d result = _mm512_maskz_fmadd_pd(mask, a_vec, x_vec, y_vec);
        _mm512_mask_storeu_pd(&y[i], mask, result);
    }
}

template <>
void saxpy_avx512<int32_t>(int32_t a, const int32_t* x, int32_t* y, size_t n) {
    __m512i a_vec = _mm512_set1_epi32(a);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512i x_vec = load_si512(&x[i]);
        __m512i y_vec = load_si512(&y[i]);
        __m512i prod = _mm512_mullo_epi32(a_vec, x_vec);
        __m512i result = _mm512_add_epi32(prod, y_vec);
        store_si512(&y[i], result);
    }

    // Masked tail
    if (i < n) {
        __mmask16 mask = (1 << (n - i)) - 1;
        __m512i x_vec = _mm512_maskz_load_epi32(mask, &x[i]);
        __m512i y_vec = _mm512_maskz_load_epi32(mask, &y[i]);
        __m512i prod = _mm512_mullo_epi32(a_vec, x_vec);
        __m512i result = _mm512_add_epi32(prod, y_vec);
        _mm512_mask_storeu_epi32(&y[i], mask, result);
    }
}


template <typename T>
void saxpy_sse2(T a, const T* x, T* y, size_t n);

// float32 version
template <>
void saxpy_sse2<float>(float a, const float* x, float* y, size_t n) {
    __m128 a_vec = _mm_set1_ps(a);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = load_ps128(&x[i]);
        __m128 y_vec = load_ps128(&y[i]);
        __m128 result = _mm_add_ps(_mm_mul_ps(a_vec, x_vec), y_vec);
        store_ps128(&y[i], result);
    }
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

// float64 version
template <>
void saxpy_sse2<double>(double a, const double* x, double* y, size_t n) {
    __m128d a_vec = _mm_set1_pd(a);
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        __m128d x_vec = load_pd128(&x[i]);
        __m128d y_vec = load_pd128(&y[i]);
        __m128d result = _mm_add_pd(_mm_mul_pd(a_vec, x_vec), y_vec);
        store_pd128(&y[i], result);
    }
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}

template <>
void saxpy_sse2<int32_t>(int32_t a, const int32_t* x, int32_t* y, size_t n) {
    __m128i a_vec = _mm_set1_epi32(a);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = load_si128(&x[i]);
        __m128i y_vec = load_si128(&y[i]);
        __m128i prod = _mm_mullo_epi32(a_vec, x_vec);
        __m128i result = _mm_add_epi32(prod, y_vec);
        store_si128(&y[i], result);
    }
    for (; i < n; ++i) y[i] = a * x[i] + y[i];
}


template <typename T>
void sweep_benchmark(const char* label, void (*func)(T, const T*, T*, size_t)) {
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
        //1 << 30   // 1 GB //unnessesary and slow
    };

    const int trials = 10;
    const size_t ALIGN = 64; // alignment used for aligned runs
    for (bool misalign : {false, true}) {
        use_aligned_loads = !misalign;
        for (size_t n : sizes) {
            // allocate aligned buffers with one extra element to allow a 1-element offset
            T *x_base = aligned_alloc_array<T>(n + 1, ALIGN);
            T *y_base = aligned_alloc_array<T>(n + 1, ALIGN);
            if (!x_base || !y_base) {
                std::cerr << "aligned_alloc failed for N=" << n << "\n";
                aligned_free_array(x_base);
                aligned_free_array(y_base);
                continue;
            }

            // initialize arrays with non-trivial values
            for (size_t i = 0; i < n; ++i) {
                x_base[i] = T(1.0);
                y_base[i] = T(0.5);
            }

            T *x = x_base;
            T *y = y_base;
            size_t effective_n = n;
            if (misalign) {
                // offset by one element to create a misaligned view
                x = reinterpret_cast<T *>(reinterpret_cast<char *>(x_base) + sizeof(T));
                y = reinterpret_cast<T *>(reinterpret_cast<char *>(y_base) + sizeof(T));
                effective_n = (n > 0) ? n - 1 : 0;
            }

            T a = T(2.0);

            for (int t = 0; t < trials; ++t) {
                // reset y for each trial to avoid accumulation effects
                for (size_t i = 0; i < effective_n; ++i) y[i] = T(0.5);

                // optional warm-up; keep consistent across trials
                func(a, x, y, effective_n);

                auto start = std::chrono::high_resolution_clock::now();
                func(a, x, y, effective_n);
                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                double seconds = elapsed.count();
                double flops = 2.0 * static_cast<double>(effective_n);
                double gflops = (seconds > 0.0) ? (flops / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;

                std::cout << label << " | N = " << n << (misalign ? " (misaligned)" : "") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  y[0] = " << y[0] << "\n\n";

                log_per_trial<T>("saxpy_benchmark.csv",
                                label,
                                dtype_name<T>(),
                                effective_n,   // use processed element count for contiguous runs
                                t + 1,
                                time_ms,
                                gflops,
                                y[0],
                                !misalign,
                                "contiguous",
                                1);               
            }

        // free aligned memory
        aligned_free_array(x_base);
        aligned_free_array(y_base);

     }
    }
}

// --- Sweep helper for stride & gather tests ---
// sizes: number of touched elements (m) for each test; total buffer size = m * max_stride
template <typename T>
void sweep_stride_and_gather(const char* kernel_label,
                             void (*simd_func)(T,const T*,T*,size_t),
                             int trials = 10)
{
    // choose m values (number of touched elements) you want to test (e.g., same sizes as before but smaller count)
    std::vector<size_t> ms = { 1<<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24};
    if (fast_mode) ms = { 1<<12, 1<<16, 1<<20 }; // small subset for quick runs

    // strides in elements: 1=contiguous, then 2,4,8
    std::vector<size_t> strides = { 1, 2, 4, 8 };

    const size_t ALIGN = 64;
    std::mt19937_64 rng(12345);

    for (size_t m : ms) {
        size_t max_stride = 1;
        for (size_t s : strides) if (s > max_stride) max_stride = s;
        size_t buf_elems = m * max_stride + 1; // +1 safety
        T* x_base = aligned_alloc_array<T>(buf_elems, ALIGN);
        T* y_base = aligned_alloc_array<T>(buf_elems, ALIGN);
        if (!x_base || !y_base) {
            std::cerr << "aligned_alloc failed for m=" << m << "\n";
            aligned_free_array(x_base);
            aligned_free_array(y_base);
            continue;
        }

        // Initialize whole buffer
        for (size_t i = 0; i < buf_elems; ++i) { x_base[i] = T(1.0); y_base[i] = T(0.5); }

        T a = T(2.0);

         // Strided tests
         for (size_t stride : strides) {
            // compute effective buffer pointers (keep aligned base so wrappers can use aligned loads if stride==1)
            const T* x = x_base;
            T* y = y_base;

            // compute safe touched count so (touched_actual-1)*stride < buf_elems
            // avoid overflow and ensure at least one element touched
            size_t touched_requested = m;
            size_t touched_max = 0;
            if (stride == 0) touched_max = 0;
            else touched_max = buf_elems / stride;            // floor(buf_elems / stride)
            size_t touched_actual = std::min(touched_requested, touched_max);
            if (touched_actual == 0) {
                std::cerr << "Skipping stride test: buf_elems=" << buf_elems << " too small for stride " << stride << " (m=" << m << ")\n";
                continue;
            }

            double flops_per_trial = 2.0 * double(touched_actual); // used to compute GFLOP/s
            std::string label = std::string(kernel_label) + " stride=" + std::to_string(stride);

            for (int t = 0; t < trials; ++t) {
                // reset y for trial (only the actually touched indices)
                for (size_t k = 0; k < touched_actual; ++k) {
                    size_t i = k * stride;
                    y[i] = T(0.5);
                }

                // warm-up
                saxpy_strided_wrapper<T>(simd_func, a, x, y, touched_actual, stride);

                auto start = std::chrono::high_resolution_clock::now();
                saxpy_strided_wrapper<T>(simd_func, a, x, y, touched_actual, stride);
                auto end = std::chrono::high_resolution_clock::now();

                double seconds = std::chrono::duration<double>(end - start).count();
                double gflops = (seconds > 0.0) ? (flops_per_trial / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;

                std::cout << label << " | buf_elems = " << buf_elems << (!use_aligned_loads ? "" : " (misaligned)") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  Strided\n";
                std::cout << "  y[0] = " << y[0] << "\n\n";

                // Log buffer footprint in N, and Stride in elements
                log_per_trial<T>("saxpy_benchmark.csv",
                                 label,
                                 dtype_name<T>(),
                                 buf_elems,
                                 t + 1,
                                 time_ms,
                                 gflops,
                                 y[0],
                                 use_aligned_loads,
                                 "strided",
                                 stride);
            }
        }


        // Gather-like test: build index array (random or strided-permuted)
        // We'll construct random indices that are in range [0, buf_elems-1], without replacement if possible.
        std::vector<size_t> idx(m);
        // simple pseudo-random unique indices via reservoir-like fill of spaced positions:
        // ensure indices are distributed across the buffer
        for (size_t k = 0; k < m; ++k) idx[k] = (k * max_stride) % (buf_elems - 1);

        // also create a shuffled version for more random gathers
        std::shuffle(idx.begin(), idx.end(), rng);

        // We'll run three gather/scatter variants and log them separately:
        // 1) in-place gather-like (y[idx[k]] = a*x[idx[k]] + y[idx[k]])
        // 2) gather-read: read scattered x[idx[k]] and write contiguous y[0..m-1]
        // 3) scatter-write: read contiguous x[0..m-1] and write scattered y[idx[k]]
        double flops_per_trial_g = 2.0 * double(m);

        // Variant 1: in-place gather (existing)
        {
            std::string g_label = std::string(kernel_label) + " gather-inplace";
            for (int t = 0; t < trials; ++t) {
                for (size_t k = 0; k < m; ++k) y_base[idx[k]] = T(0.5);
                // warm-up
                saxpy_gather_wrapper<T>(a, x_base, y_base, idx.data(), m);
                auto start = std::chrono::high_resolution_clock::now();
                saxpy_gather_wrapper<T>(a, x_base, y_base, idx.data(), m);
                auto end = std::chrono::high_resolution_clock::now();
                double seconds = std::chrono::duration<double>(end - start).count();
                double gflops = (seconds > 0.0) ? (flops_per_trial_g / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;
                std::cout << g_label << " | buf_elems = " << buf_elems << (!use_aligned_loads ? " (misaligned)" : "") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  y[idx[0]] = " << y_base[idx[0]] << "\n\n";
                log_per_trial<T>("saxpy_benchmark.csv", g_label, dtype_name<T>(), buf_elems, t + 1, time_ms, gflops, y_base[idx[0]], use_aligned_loads, "gather-inplace", 0);
            }
        }

        // Variant 2: gather-read (read x[idx] -> write contig y[0..m-1])
        {
            std::string g_label = std::string(kernel_label) + " gather-read";
            for (int t = 0; t < trials; ++t) {
                // reset destination contiguous y[0..m-1]
                for (size_t k = 0; k < m; ++k) y_base[k] = T(0.5);
                // warm-up
                saxpy_gather_read_xidx_write_contig_y<T>(a, x_base, y_base, idx.data(), m);
                auto start = std::chrono::high_resolution_clock::now();
                saxpy_gather_read_xidx_write_contig_y<T>(a, x_base, y_base, idx.data(), m);
                auto end = std::chrono::high_resolution_clock::now();
                double seconds = std::chrono::duration<double>(end - start).count();
                double gflops = (seconds > 0.0) ? (flops_per_trial_g / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;
                std::cout << g_label << " | buf_elems = " << buf_elems << (!use_aligned_loads ? " (misaligned)" : "") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  y[0] = " << y_base[0] << "\n\n";
                log_per_trial<T>("saxpy_benchmark.csv", g_label, dtype_name<T>(), buf_elems, t + 1, time_ms, gflops, y_base[0], use_aligned_loads, "gather-read", 0);
            }

            // If available, also run a vectorized gather-read variant for float/double
            if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                std::string v_label = std::string(kernel_label) + " gather-read-avx2";
                for (int t = 0; t < trials; ++t) {
                    for (size_t k = 0; k < m; ++k) y_base[k] = T(0.5);
                    // warm-up (vectorized specialization will be used for float/double)
                    saxpy_gather_read_xidx_write_contig_y<T>(a, x_base, y_base, idx.data(), m);
                    auto start_v = std::chrono::high_resolution_clock::now();
                    saxpy_gather_read_xidx_write_contig_y<T>(a, x_base, y_base, idx.data(), m);
                    auto end_v = std::chrono::high_resolution_clock::now();
                    double seconds_v = std::chrono::duration<double>(end_v - start_v).count();
                    double gflops_v = (seconds_v > 0.0) ? (flops_per_trial_g / (seconds_v * 1e9)) : 0.0;
                    double time_ms_v = seconds_v * 1e3;
                    std::cout << v_label << " | buf_elems = " << buf_elems << (!use_aligned_loads ? " (misaligned)" : "") << ":\n";
                    std::cout << "  Time: " << time_ms_v << " ms\n";
                    std::cout << "  GFLOP/s: " << gflops_v << "\n";
                    std::cout << "  y[0] = " << y_base[0] << "\n\n";
                    log_per_trial<T>("saxpy_benchmark.csv", v_label, dtype_name<T>(), buf_elems, t + 1, time_ms_v, gflops_v, y_base[0], use_aligned_loads, "gather-read-avx2", 0);
                }
            }
        }

        // Variant 3: scatter-write (read contig x[0..m-1] -> write y[idx])
        {
            std::string g_label = std::string(kernel_label) + " scatter-write";
            for (int t = 0; t < trials; ++t) {
                // reset scattered y
                for (size_t k = 0; k < m; ++k) y_base[idx[k]] = T(0.5);
                // warm-up
                saxpy_scatter_write_yidx_read_contig_x<T>(a, x_base, y_base, idx.data(), m);
                auto start = std::chrono::high_resolution_clock::now();
                saxpy_scatter_write_yidx_read_contig_x<T>(a, x_base, y_base, idx.data(), m);
                auto end = std::chrono::high_resolution_clock::now();
                double seconds = std::chrono::duration<double>(end - start).count();
                double gflops = (seconds > 0.0) ? (flops_per_trial_g / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;
                std::cout << g_label << " | buf_elems = " << buf_elems << (!use_aligned_loads ? " (misaligned)" : "") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  y[idx[0]] = " << y_base[idx[0]] << "\n\n";
                log_per_trial<T>("saxpy_benchmark.csv", g_label, dtype_name<T>(), buf_elems, t + 1, time_ms, gflops, y_base[idx[0]], use_aligned_loads, "scatter-write", 0);
            }
        }

        aligned_free_array(x_base);
        aligned_free_array(y_base);
        }
    }


int main() {
    std::ofstream clear("saxpy_benchmark.csv", std::ios::trunc);
    clear.close();
        // If fast_mode is set via env var FAST=1 or by later CLI patch, we could short-circuit.
        sweep_benchmark<float>("Scalar float32", saxpy_scalar<float>);
        sweep_benchmark<double>("Scalar float64", saxpy_scalar<double>);
        sweep_benchmark<float>("SSE2 float32", saxpy_sse2<float>);
        sweep_benchmark<double>("SSE2 float64", saxpy_sse2<double>);    
        sweep_benchmark<float>("AVX float32", saxpy_avx<float>);
        sweep_benchmark<double>("AVX float64", saxpy_avx<double>);
        sweep_benchmark<float>("AVX-512 float32", saxpy_avx512<float>);
        sweep_benchmark<double>("AVX-512 float64", saxpy_avx512<double>);   
        sweep_benchmark<int32_t>("Scalar int32", saxpy_scalar<int32_t>);
        sweep_benchmark<int32_t>("SSE2 int32", saxpy_sse2<int32_t>);
        sweep_benchmark<int32_t>("AVX int32", saxpy_avx<int32_t>);
        sweep_benchmark<int32_t>("AVX-512 int32", saxpy_avx512<int32_t>);
        sweep_stride_and_gather<float>("Scalar float32", saxpy_scalar<float>);
        sweep_stride_and_gather<double>("Scalar float64", saxpy_scalar<double>);
        sweep_stride_and_gather<float>("SSE2 float32", saxpy_sse2<float>);
        sweep_stride_and_gather<double>("SSE2 float64", saxpy_sse2<double>);    
        sweep_stride_and_gather<float>("AVX float32", saxpy_avx<float>);
        sweep_stride_and_gather<double>("AVX float64", saxpy_avx<double>);
        sweep_stride_and_gather<float>("AVX-512 float32", saxpy_avx512<float>);
        sweep_stride_and_gather<double>("AVX-512 float64", saxpy_avx512<double>);   
        sweep_stride_and_gather<int32_t>("Scalar int32", saxpy_scalar<int32_t>);
        sweep_stride_and_gather<int32_t>("SSE2 int32", saxpy_sse2<int32_t>);
        sweep_stride_and_gather<int32_t>("AVX int32", saxpy_avx<int32_t>);
        sweep_stride_and_gather<int32_t>("AVX-512 int32", saxpy_avx512<int32_t>);

}
