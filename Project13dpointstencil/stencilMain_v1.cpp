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

static inline __m256 load_ps256(const float* p) { return use_aligned_loads ? _mm256_load_ps(p) : _mm256_loadu_ps(p); }
static inline __m256d load_pd256(const double* p) { return use_aligned_loads ? _mm256_load_pd(p) : _mm256_loadu_pd(p); }

static inline __m512 load_ps512(const float* p) { return use_aligned_loads ? _mm512_load_ps(p) : _mm512_loadu_ps(p); }
static inline __m512d load_pd512(const double* p) { return use_aligned_loads ? _mm512_load_pd(p) : _mm512_loadu_pd(p); }
static inline __m128i load_si128(const int32_t* p) { return use_aligned_loads ? _mm_load_si128((const __m128i*)p) : _mm_loadu_si128((const __m128i*)p); }
static inline __m256i load_si256(const int32_t* p) { return use_aligned_loads ? _mm256_load_si256((const __m256i*)p) : _mm256_loadu_si256((const __m256i*)p); }
static inline __m512i load_si512(const int32_t* p) { return use_aligned_loads ? _mm512_load_si512((const void*)p) : _mm512_loadu_si512((const void*)p); }

template <typename T>
constexpr const char* dtype_name() {
    if (std::is_same<T, float>::value)    return "float32";
    if (std::is_same<T, double>::value)   return "float64";
    if (std::is_same<T, int32_t>::value)  return "int32";
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
                   const std::string &pattern) {
    std::ofstream f(filename, std::ios::app);
    if (!f) {
        std::cerr << "Failed to open CSV file: " << filename << "\n";
        return;
    }
    if (!csv_header_written) {
        f << "Kernel,Type,N_buf,Pattern,Trial,Time(ms),GFLOP/s,Result\n";
        csv_header_written = true;
    }
    f << kernel << "," << dtype << "," << n_buf << "," << pattern << ","
      << trial << "," << time_ms << "," << gflops << "," << result << "\n";
}

// Scalar implementation of 3-point stencil
template <typename T>
void stencil_scalar(const T* x, T* y, size_t n, T a, T b, T c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Main loop
    for (size_t i = 1; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

// SSE2 implementation
template <typename T>
void stencil_sse2(const T* x, T* y, size_t n, T a, T b, T c);

template <>
void stencil_sse2<float>(const float* x, float* y, size_t n, float a, float b, float c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Set up coefficient vectors
    __m128 va = _mm_set1_ps(a);
    __m128 vb = _mm_set1_ps(b);
    __m128 vc = _mm_set1_ps(c);
    
    // Main loop - process 4 elements at a time
    size_t i = 1;
    for (; i + 4 < n-1; i += 4) {
        __m128 prev = load_ps128(x + i - 1);  // Load previous elements
        __m128 curr = load_ps128(x + i);      // Load current elements
        __m128 next = load_ps128(x + i + 1);  // Load next elements
        
        // Compute stencil
        __m128 result = _mm_mul_ps(va, prev);
        result = _mm_add_ps(result, _mm_mul_ps(vb, curr));
        result = _mm_add_ps(result, _mm_mul_ps(vc, next));
        
        // Store result
        if (use_aligned_loads) {
            _mm_store_ps(y + i, result);
        } else {
            _mm_storeu_ps(y + i, result);
        }
    }
    
    // Handle remaining elements
    for (; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

template <>
void stencil_sse2<double>(const double* x, double* y, size_t n, double a, double b, double c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Set up coefficient vectors
    __m128d va = _mm_set1_pd(a);
    __m128d vb = _mm_set1_pd(b);
    __m128d vc = _mm_set1_pd(c);
    
    // Main loop - process 2 elements at a time
    size_t i = 1;
    for (; i + 2 < n-1; i += 2) {
        __m128d prev = load_pd128(x + i - 1);  // Load previous elements
        __m128d curr = load_pd128(x + i);      // Load current elements
        __m128d next = load_pd128(x + i + 1);  // Load next elements
        
        // Compute stencil
        __m128d result = _mm_mul_pd(va, prev);
        result = _mm_add_pd(result, _mm_mul_pd(vb, curr));
        result = _mm_add_pd(result, _mm_mul_pd(vc, next));
        
        // Store result
        if (use_aligned_loads) {
            _mm_store_pd(y + i, result);
        } else {
            _mm_storeu_pd(y + i, result);
        }
    }
    
    // Handle remaining elements
    for (; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

// AVX2 implementation
template <typename T>
void stencil_avx2(const T* x, T* y, size_t n, T a, T b, T c);

template <>
void stencil_avx2<float>(const float* x, float* y, size_t n, float a, float b, float c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Set up coefficient vectors
    __m256 va = _mm256_set1_ps(a);
    __m256 vb = _mm256_set1_ps(b);
    __m256 vc = _mm256_set1_ps(c);
    
    // Main loop - process 8 elements at a time
    size_t i = 1;
    for (; i + 8 < n-1; i += 8) {
        __m256 prev = load_ps256(x + i - 1);  // Load previous elements
        __m256 curr = load_ps256(x + i);      // Load current elements
        __m256 next = load_ps256(x + i + 1);  // Load next elements
        
        // Compute stencil using FMA
        __m256 result = _mm256_mul_ps(va, prev);
        result = _mm256_fmadd_ps(vb, curr, result);
        result = _mm256_fmadd_ps(vc, next, result);
        
        // Store result
        if (use_aligned_loads) {
            _mm256_store_ps(y + i, result);
        } else {
            _mm256_storeu_ps(y + i, result);
        }
    }
    
    // Handle remaining elements
    for (; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

template <>
void stencil_avx2<double>(const double* x, double* y, size_t n, double a, double b, double c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Set up coefficient vectors
    __m256d va = _mm256_set1_pd(a);
    __m256d vb = _mm256_set1_pd(b);
    __m256d vc = _mm256_set1_pd(c);
    
    // Main loop - process 4 elements at a time
    size_t i = 1;
    for (; i + 4 < n-1; i += 4) {
        __m256d prev = load_pd256(x + i - 1);  // Load previous elements
        __m256d curr = load_pd256(x + i);      // Load current elements
        __m256d next = load_pd256(x + i + 1);  // Load next elements
        
        // Compute stencil using FMA
        __m256d result = _mm256_mul_pd(va, prev);
        result = _mm256_fmadd_pd(vb, curr, result);
        result = _mm256_fmadd_pd(vc, next, result);
        
        // Store result
        if (use_aligned_loads) {
            _mm256_store_pd(y + i, result);
        } else {
            _mm256_storeu_pd(y + i, result);
        }
    }
    
    // Handle remaining elements
    for (; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

// AVX-512 implementation
template <typename T>
void stencil_avx512(const T* x, T* y, size_t n, T a, T b, T c);

template <>
void stencil_avx512<float>(const float* x, float* y, size_t n, float a, float b, float c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Set up coefficient vectors
    __m512 va = _mm512_set1_ps(a);
    __m512 vb = _mm512_set1_ps(b);
    __m512 vc = _mm512_set1_ps(c);
    
    // Main loop - process 16 elements at a time
    size_t i = 1;
    for (; i + 16 < n-1; i += 16) {
        __m512 prev = load_ps512(x + i - 1);  // Load previous elements
        __m512 curr = load_ps512(x + i);      // Load current elements
        __m512 next = load_ps512(x + i + 1);  // Load next elements
        
        // Compute stencil using FMA
        __m512 result = _mm512_mul_ps(va, prev);
        result = _mm512_fmadd_ps(vb, curr, result);
        result = _mm512_fmadd_ps(vc, next, result);
        
        // Store result
        if (use_aligned_loads) {
            _mm512_store_ps(y + i, result);
        } else {
            _mm512_storeu_ps(y + i, result);
        }
    }
    
    // Handle remaining elements with masking if needed
    if (i < n-1) {
        __mmask16 mask = (1ULL << (n - i - 1)) - 1;
        __m512 prev = _mm512_maskz_load_ps(mask, x + i - 1);
        __m512 curr = _mm512_maskz_load_ps(mask, x + i);
        __m512 next = _mm512_maskz_load_ps(mask, x + i + 1);
        
        __m512 result = _mm512_mul_ps(va, prev);
        result = _mm512_fmadd_ps(vb, curr, result);
        result = _mm512_fmadd_ps(vc, next, result);
        
        _mm512_mask_store_ps(y + i, mask, result);
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

template <>
void stencil_avx512<double>(const double* x, double* y, size_t n, double a, double b, double c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Set up coefficient vectors
    __m512d va = _mm512_set1_pd(a);
    __m512d vb = _mm512_set1_pd(b);
    __m512d vc = _mm512_set1_pd(c);
    
    // Main loop - process 8 elements at a time
    size_t i = 1;
    for (; i + 8 < n-1; i += 8) {
        __m512d prev = load_pd512(x + i - 1);  // Load previous elements
        __m512d curr = load_pd512(x + i);      // Load current elements
        __m512d next = load_pd512(x + i + 1);  // Load next elements
        
        // Compute stencil using FMA
        __m512d result = _mm512_mul_pd(va, prev);
        result = _mm512_fmadd_pd(vb, curr, result);
        result = _mm512_fmadd_pd(vc, next, result);
        
        // Store result
        if (use_aligned_loads) {
            _mm512_store_pd(y + i, result);
        } else {
            _mm512_storeu_pd(y + i, result);
        }
    }
    
    // Handle remaining elements with masking if needed
    if (i < n-1) {
        __mmask8 mask = (1ULL << (n - i - 1)) - 1;
        __m512d prev = _mm512_maskz_load_pd(mask, x + i - 1);
        __m512d curr = _mm512_maskz_load_pd(mask, x + i);
        __m512d next = _mm512_maskz_load_pd(mask, x + i + 1);
        
        __m512d result = _mm512_mul_pd(va, prev);
        result = _mm512_fmadd_pd(vb, curr, result);
        result = _mm512_fmadd_pd(vc, next, result);
        
        _mm512_mask_store_pd(y + i, mask, result);
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

// Helper function to check if stencil coefficients are safe for 32-bit arithmetic
inline bool check_stencil_coefficients(int32_t a, int32_t b, int32_t c) {
    // Check if coefficients sum to 1 (or -1) which guarantees no overflow
    // if input values are valid 32-bit integers
    return (a + b + c == 1) || (a + b + c == -1);
}

// Integer implementations with direct 32-bit arithmetic 
// Assumes coefficients are validated with check_stencil_coefficients()
template <>
void stencil_scalar<int32_t>(const int32_t* x, int32_t* y, size_t n, int32_t a, int32_t b, int32_t c) {
    // Validate input parameters
    if (x == nullptr || y == nullptr) {
        throw std::runtime_error("Input or output array is null");
    }
    if (n < 1) {
        throw std::runtime_error("Array size must be at least 1");
    }
    
    // Validate coefficients
    if (!check_stencil_coefficients(a, b, c)) {
        throw std::runtime_error("Stencil coefficients must sum to 1 or -1 to prevent overflow");
    }
    
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Main loop
    for (size_t i = 1; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

template <>
void stencil_sse2<int32_t>(const int32_t* x, int32_t* y, size_t n, int32_t a, int32_t b, int32_t c) {
    // Validate coefficients
    if (!check_stencil_coefficients(a, b, c)) {
        throw std::runtime_error("Stencil coefficients must sum to 1 or -1 to prevent overflow");
    }
    
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Main loop - process 4 elements at a time
    size_t i = 1;
    for (; i + 4 < n-1; i += 4) {
        __m128i prev = load_si128((const int32_t*)(x + i - 1));  // Load previous elements
        __m128i curr = load_si128((const int32_t*)(x + i));      // Load current elements
        __m128i next = load_si128((const int32_t*)(x + i + 1));  // Load next elements
        
        __m128i va = _mm_set1_epi32(a);
        __m128i vb = _mm_set1_epi32(b);
        __m128i vc = _mm_set1_epi32(c);
        
        // Compute stencil directly with 32-bit arithmetic
        __m128i result = _mm_mullo_epi32(prev, va);                 // a * prev
        result = _mm_add_epi32(result, _mm_mullo_epi32(curr, vb)); // + b * curr
        result = _mm_add_epi32(result, _mm_mullo_epi32(next, vc)); // + c * next
        
        // Store result directly since we're using 32-bit arithmetic
        
        // Store result
        if (use_aligned_loads) {
            _mm_store_si128((__m128i*)(y + i), result);
        } else {
            _mm_storeu_si128((__m128i*)(y + i), result);
        }
    }
    
    // Handle remaining elements with direct 32-bit arithmetic
    for (; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

template <>
void stencil_avx2<int32_t>(const int32_t* x, int32_t* y, size_t n, int32_t a, int32_t b, int32_t c) {
    // Validate coefficients
    if (!check_stencil_coefficients(a, b, c)) {
        throw std::runtime_error("Stencil coefficients must sum to 1 or -1 to prevent overflow");
    }

    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Main loop - process 8 elements at a time
    size_t i = 1;
    for (; i + 8 < n-1; i += 8) {
        __m256i prev = load_si256(x + i - 1);
        __m256i curr = load_si256(x + i);
        __m256i next = load_si256(x + i + 1);
        __m256i va = _mm256_set1_epi32(a);
        __m256i vb = _mm256_set1_epi32(b);
        __m256i vc = _mm256_set1_epi32(c);
        
        // Compute stencil directly with 32-bit arithmetic
        __m256i result = _mm256_mullo_epi32(prev, va);                 // a * prev
        result = _mm256_add_epi32(result, _mm256_mullo_epi32(curr, vb)); // + b * curr 
        result = _mm256_add_epi32(result, _mm256_mullo_epi32(next, vc)); // + c * next
        
        // Store results
        if (use_aligned_loads) {
            _mm256_store_si256((__m256i*)(y + i), result);
        } else {
            _mm256_storeu_si256((__m256i*)(y + i), result);
        }
    }
    
    // Handle remaining elements with direct 32-bit arithmetic
    for (; i < n-1; ++i) {
        y[i] = a * x[i-1] + b * x[i] + c * x[i+1];
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

template <>
void stencil_avx512<int32_t>(const int32_t* x, int32_t* y, size_t n, int32_t a, int32_t b, int32_t c) {
    // Handle first element separately (assume periodic boundary)
    y[0] = a * x[n-1] + b * x[0] + c * x[1];
    
    // Main loop - process 16 elements at a time
    size_t i = 1;
    for (; i + 16 < n-1; i += 16) {
        __m512i prev = load_si512(x + i - 1);
        __m512i curr = load_si512(x + i);
        __m512i next = load_si512(x + i + 1);
        
        __m512i va = _mm512_set1_epi32(a);
        __m512i vb = _mm512_set1_epi32(b);
        __m512i vc = _mm512_set1_epi32(c);
        
        // Compute stencil directly with 32-bit arithmetic
        __m512i result = _mm512_mullo_epi32(prev, va);                 // a * prev
        result = _mm512_add_epi32(result, _mm512_mullo_epi32(curr, vb)); // + b * curr
        result = _mm512_add_epi32(result, _mm512_mullo_epi32(next, vc)); // + c * next
        
        if (use_aligned_loads) {
            _mm512_store_si512(y + i, result);
        } else {
            _mm512_storeu_si512(y + i, result);
        }
    }
    
    // Handle remaining elements with masking if needed
    if (i < n-1) {
        __mmask16 mask = (1ULL << (n - i - 1)) - 1;
        __m512i prev = _mm512_maskz_load_epi32(mask, x + i - 1);
        __m512i curr = _mm512_maskz_load_epi32(mask, x + i);
        __m512i next = _mm512_maskz_load_epi32(mask, x + i + 1);
        
        __m512i va = _mm512_set1_epi32(a);
        __m512i vb = _mm512_set1_epi32(b);
        __m512i vc = _mm512_set1_epi32(c);
        
        __m512i result = _mm512_mullo_epi32(prev, va);                 // a * prev
        result = _mm512_add_epi32(result, _mm512_mullo_epi32(curr, vb)); // + b * curr
        result = _mm512_add_epi32(result, _mm512_mullo_epi32(next, vc)); // + c * next
        
        _mm512_mask_store_epi32(y + i, mask, result);
    }
    
    // Handle last element separately (assume periodic boundary)
    y[n-1] = a * x[n-2] + b * x[n-1] + c * x[0];
}

// Structure to hold array with its stride information
template <typename T>
struct StridedArray {
    T* data;           // Base pointer
    size_t stride;     // Stride between elements
    size_t total_size; // Total size including stride gaps
    
    StridedArray(size_t n, size_t s, size_t align = 64) : stride(s) {
        total_size = n * stride;
        data = aligned_alloc_array<T>(total_size, align);
    }
    
    ~StridedArray() {
        aligned_free_array(data);
    }
    
    // Get element at index i considering stride
    T& operator[](size_t i) { return data[i * stride]; }
    const T& operator[](size_t i) const { return data[i * stride]; }
    
    // Get pointer to element at index i
    T* ptr(size_t i) { return &data[i * stride]; }
    const T* ptr(size_t i) const { return &data[i * stride]; }
};

template <typename T>
void sweep_benchmark(const char* label, void (*func)(const T*, T*, size_t, T, T, T)) {
    std::vector<size_t> sizes = {
        1 << 10,  // 1 KB
        1 << 12,  // 4 KB
        1 << 14,  // 16 KB
        1 << 16,  // 64 KB
        1 << 18,  // 256 KB
        1 << 20,  // 1 MB
        1 << 22,  // 4 MB
        1 << 24,  // 16 MB
        1 << 26,  // 64 MB
        //1 << 28,  // 256 MB //breaks on stride 16
    };
    
    std::vector<size_t> strides = {1, 2, 4, 8, 16};

    const int trials = 10;
    const size_t ALIGN = 64;
    
    // Stencil coefficients - different for integer vs floating point
    T a, b, c;
    if constexpr (std::is_integral<T>::value) {
        // For integers, use exact values that sum to 1
        a = T(0); b = T(1); c = T(0);  // Simple integer coefficients that sum to 1
    } else {
        // For floating point, use fractional values
        a = T(0.2); b = T(0.6); c = T(0.2);
    }

    // Test different memory access patterns
    for (size_t stride : strides) {
        for (bool mis : {false, true}) {
            use_aligned_loads = !mis;
            std::string pattern = (stride == 1) ? "contiguous" : "stride_" + std::to_string(stride);
            if (mis) pattern += "_misaligned";
            
            for (size_t n : sizes) {
                // Allocate strided arrays with one extra element for misalignment
                StridedArray<T> strided_x(n + 1, stride, ALIGN);
                StridedArray<T> strided_y(n + 1, stride, ALIGN);
                
                if (!strided_x.data || !strided_y.data) {
                    std::cerr << "aligned_alloc failed for N=" << n << ", stride=" << stride << "\n";
                    continue;
                }

                // Initialize with interesting data pattern
                for (size_t i = 0; i < n + 1; ++i) {
                    strided_x[i] = T(1.0 + std::sin(i * 0.1));
                }

                const T* x = strided_x.data;
                T* y = strided_y.data;
                size_t effective_n = n;
                std::string lbl = label;
                if (mis) {
                    x = reinterpret_cast<const T*>(reinterpret_cast<const char*>(strided_x.data) + sizeof(T));
                    y = reinterpret_cast<T*>(reinterpret_cast<char*>(strided_y.data) + sizeof(T));
                    effective_n = (n > 0) ? n - 1 : 0;
                    lbl += " (misaligned)";
                }

                if (effective_n == 0) {
                    continue;
                }

                for (int t = 0; t < trials; ++t) {
                    // Warm-up
                    func(x, y, effective_n, a, b, c);

                    auto start = std::chrono::high_resolution_clock::now();
                    func(x, y, effective_n, a, b, c);
                    auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> elapsed = end - start;
                double seconds = elapsed.count();
                // 3 multiplications and 2 additions per element
                double flops = 5.0 * static_cast<double>(effective_n);
                double gflops = (seconds > 0.0) ? (flops / (seconds * 1e9)) : 0.0;
                double time_ms = seconds * 1e3;

                // Calculate a checksum of the result for verification
                T checksum = T(0);
                for (size_t i = 0; i < effective_n; ++i) {
                    checksum += y[i];
                }

                std::cout << lbl << " | N = " << effective_n << (mis ? " (misaligned)" : "") << ":\n";
                std::cout << "  Time: " << time_ms << " ms\n";
                std::cout << "  GFLOP/s: " << gflops << "\n";
                std::cout << "  Checksum = " << checksum << "\n\n";

                log_per_trial<T>("stencil_benchmark_v1.csv",
                                lbl,
                                dtype_name<T>(),
                                effective_n,
                                t + 1,
                                time_ms,
                                gflops,
                                checksum,
                                pattern);
                }
            }
        }
    }
}

int main() {
    // Clear the benchmark file
    std::ofstream f("stencil_benchmark_v1.csv", std::ios::trunc);
    f.close();
    csv_header_written = false;

    // Float32 benchmarks
    std::cout << "\n=== Float32 Benchmarks ===\n";
    sweep_benchmark("Scalar float32", stencil_scalar<float>);
    sweep_benchmark("SSE2 float32", stencil_sse2<float>);
    sweep_benchmark("AVX2 float32", stencil_avx2<float>);
    sweep_benchmark("AVX512 float32", stencil_avx512<float>);

    // Float64 benchmarks
    std::cout << "\n=== Float64 Benchmarks ===\n";
    sweep_benchmark("Scalar float64", stencil_scalar<double>);
    sweep_benchmark("SSE2 float64", stencil_sse2<double>);
    sweep_benchmark("AVX2 float64", stencil_avx2<double>);
    sweep_benchmark("AVX512 float64", stencil_avx512<double>);
    
    // Int32 benchmarks
    std::cout << "\n=== Int32 Benchmarks ===\n";
    sweep_benchmark("Scalar int32", stencil_scalar<int32_t>);
    sweep_benchmark("SSE2 int32", stencil_sse2<int32_t>);
    sweep_benchmark("AVX2 int32", stencil_avx2<int32_t>);
    sweep_benchmark("AVX512 int32", stencil_avx512<int32_t>);

    return 0;
}