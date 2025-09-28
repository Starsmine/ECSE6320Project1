// DotpMain_fixed.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <emmintrin.h>
#include <fstream>
#include <cstdlib>
#include <random>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cstdint>

#ifdef _WIN32
  #include <malloc.h>
#endif

static bool use_aligned_loads = true;
static bool csv_header_written = false;

// dtype helper
template <typename T>
constexpr const char* dtype_name() {
    if (std::is_same<T,float>::value)  return "float32";
    if (std::is_same<T,double>::value) return "float64";
    if (std::is_same<T,int32_t>::value) return "int32";
    return "unknown";
}

// CSV logger
template <typename T>
void log_per_trial(const std::string &filename,
                   const std::string &kernel,
                   const std::string &dtype,
                   size_t N_buf,
                   const std::string &pattern,
                   size_t stride,
                   int trial,
                   double time_ms,
                   double gflops,
                   T result)
{
    std::ofstream f(filename, std::ios::app);
    if (!f) { std::cerr<<"CSV open failed\n"; return; }
    if (!csv_header_written) {
        f << "Kernel,Type,N_buf,Pattern,Stride,Trial,Time(ms),GFLOP/s,Result\n";
        csv_header_written = true;
    }
    f << kernel << ',' << dtype << ',' << N_buf << ',' << pattern << ','
      << stride << ',' << trial << ',' << time_ms << ',' << gflops
      << ',' << result << "\n";
}

// aligned alloc/free
static void* aligned_malloc_bytes(size_t bytes, size_t align) {
#ifdef _WIN32
    return _aligned_malloc(bytes, align);
#else
    void* p=nullptr;
    if(posix_memalign(&p,align,bytes)) return nullptr;
    return p;
#endif
}
static void aligned_free_bytes(void* p){
    if(!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
template <typename T>
static T* aligned_alloc_array(size_t n, size_t align=64){
    return reinterpret_cast<T*>(aligned_malloc_bytes(n*sizeof(T), align));
}
template <typename T>
static void aligned_free_array(T* p){
    aligned_free_bytes(reinterpret_cast<void*>(p));
}

// load wrappers
static inline __m128  load_ps128(const float*  p){return use_aligned_loads?_mm_load_ps(p):_mm_loadu_ps(p);}
static inline __m128d load_pd128(const double* p){return use_aligned_loads?_mm_load_pd(p):_mm_loadu_pd(p);}
static inline __m256  load_ps256(const float*  p){return use_aligned_loads?_mm256_load_ps(p):_mm256_loadu_ps(p);}
static inline __m256d load_pd256(const double* p){return use_aligned_loads?_mm256_load_pd(p):_mm256_loadu_pd(p);}
static inline __m512  load_ps512(const float*  p){return use_aligned_loads?_mm512_load_ps(p):_mm512_loadu_ps(p);}
static inline __m512d load_pd512(const double* p){return use_aligned_loads?_mm512_load_pd(p):_mm512_loadu_pd(p);}

// contiguous reduction kernels
template<typename T>
void dotp_scalar(const T* x, const T* y, size_t n, T &out_sum){
    T s=0;
    for(size_t i=0;i<n;++i) s+= x[i]*y[i];
    out_sum=s;
}
template<> void dotp_scalar<int32_t>(const int32_t* x,const int32_t* y,size_t n,int32_t &out_sum){
    int32_t s=0; for(size_t i=0;i<n;++i)s+=x[i]*y[i]; out_sum=s;
}

template<typename T> void dotp_sse2(const T*,const T*,size_t,T&);
template<> void dotp_sse2<float>(const float* x,const float* y,size_t n,float &out_sum){
    __m128 acc=_mm_setzero_ps(); size_t i=0;
    for(;i+3<n;i+=4){
        __m128 xv=load_ps128(x+i), yv=load_ps128(y+i);
        acc=_mm_add_ps(acc,_mm_mul_ps(xv,yv));
    }
    float tmp[4]; _mm_storeu_ps(tmp,acc);
    float s=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    for(;i<n;++i) s+=x[i]*y[i];
    out_sum=s;
}
template<> void dotp_sse2<double>(const double* x,const double* y,size_t n,double &out_sum){
    __m128d acc=_mm_setzero_pd(); size_t i=0;
    for(;i+1<n;i+=2){
        __m128d xv=load_pd128(x+i), yv=load_pd128(y+i);
        acc=_mm_add_pd(acc,_mm_mul_pd(xv,yv));
    }
    double tmp[2]; _mm_storeu_pd(tmp,acc);
    double s=tmp[0]+tmp[1];
    for(;i<n;++i) s+=x[i]*y[i];
    out_sum=s;
}
template<> void dotp_sse2<int32_t>(const int32_t* x,const int32_t* y,size_t n,int32_t &out_sum){
    // fallback
    int32_t s=0; for(size_t i=0;i<n;++i)s+=x[i]*y[i]; out_sum=s;
}

template<typename T> void dotp_avx(const T*,const T*,size_t,T&);
template<> void dotp_avx<float>(const float* x,const float* y,size_t n,float &out_sum){
    __m256 acc=_mm256_setzero_ps(); size_t i=0;
    for(;i+7<n;i+=8){
        __m256 xv=load_ps256(x+i), yv=load_ps256(y+i);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(xv,yv));
    }
    float tmp[8]; _mm256_storeu_ps(tmp,acc);
    float s=0; for(int t=0;t<8;++t) s+=tmp[t];
    for(;i<n;++i) s+=x[i]*y[i];
    out_sum=s;
}
template<> void dotp_avx<double>(const double* x,const double* y,size_t n,double &out_sum){
    __m256d acc=_mm256_setzero_pd(); size_t i=0;
    for(;i+3<n;i+=4){
        __m256d xv=load_pd256(x+i), yv=load_pd256(y+i);
        acc=_mm256_add_pd(acc,_mm256_mul_pd(xv,yv));
    }
    double tmp[4]; _mm256_storeu_pd(tmp,acc);
    double s=0; for(int t=0;t<4;++t) s+=tmp[t];
    for(;i<n;++i) s+=x[i]*y[i];
    out_sum=s;
}
template<> void dotp_avx<int32_t>(const int32_t* x,const int32_t* y,size_t n,int32_t &out_sum){
    int32_t s=0; for(size_t i=0;i<n;++i) s+=x[i]*y[i]; out_sum=s;
}

template<typename T> void dotp_avx512(const T*,const T*,size_t,T&);
template<> void dotp_avx512<float>(const float* x,const float* y,size_t n,float &out_sum){
    __m512 acc=_mm512_setzero_ps(); size_t i=0;
    for(;i+15<n;i+=16){
        __m512 xv=load_ps512(x+i), yv=load_ps512(y+i);
        acc=_mm512_add_ps(acc,_mm512_mul_ps(xv,yv));
    }
    float tmp[16]; _mm512_storeu_ps(tmp,acc);
    float s=0; for(int t=0;t<16;++t) s+=tmp[t];
    for(;i<n;++i) s+=x[i]*y[i];
    out_sum=s;
}
template<> void dotp_avx512<double>(const double* x,const double* y,size_t n,double &out_sum){
    __m512d acc=_mm512_setzero_pd(); size_t i=0;
    for(;i+7<n;i+=8){
        __m512d xv=load_pd512(x+i), yv=load_pd512(y+i);
        acc=_mm512_add_pd(acc,_mm512_mul_pd(xv,yv));
    }
    double tmp[8]; _mm512_storeu_pd(tmp,acc);
    double s=0; for(int t=0;t<8;++t) s+=tmp[t];
    for(;i<n;++i) s+=x[i]*y[i];
    out_sum=s;
}
template<> void dotp_avx512<int32_t>(const int32_t* x,const int32_t* y,size_t n,int32_t &out_sum){
    int32_t s=0; for(size_t i=0;i<n;++i) s+=x[i]*y[i]; out_sum=s;
}

// stride & gather wrappers
template<typename T>
void dotp_strided_wrapper(void (*func)(const T*,const T*,size_t,T&),
                          const T* x,const T* y,size_t touched,size_t stride,T &out_sum){
    if(stride==1){ func(x,y,touched,out_sum); return; }
    T s=0;
    for(size_t k=0;k<touched;++k) s+= x[k*stride]*y[k*stride];
    out_sum=s;
}
template<typename T>
void dotp_gather_wrapper(const T* x,const T* y,const size_t* idx,size_t m,T &out_sum){
    T s=0;
    for(size_t k=0;k<m;++k) s += x[idx[k]]*y[idx[k]];
    out_sum=s;
}

// contiguous sweep
template<typename T>
void sweep_contiguous(const char* label, void (*func)(const T*,const T*,size_t,T&)){
    std::vector<size_t> sizes={1<<10,1<<12,1<<14,1<<16,1<<18,1<<20};
    const int trials=5; const size_t ALIGN=64;
    for(bool mis:{false,true}){
        use_aligned_loads = !mis;
        for(size_t n: sizes){
            T* base_x = aligned_alloc_array<T>(n+1,ALIGN);
            T* base_y = aligned_alloc_array<T>(n+1,ALIGN);
            if(!base_x||!base_y){ aligned_free_array(base_x); aligned_free_array(base_y); continue;}
            for(size_t i=0;i<n;++i){ base_x[i]=T(1); base_y[i]=T(2); }
            T* x=base_x; T* y=base_y; size_t len=n;
            if(mis){ x=(T*)((char*)base_x+sizeof(T)); y=(T*)((char*)base_y+sizeof(T)); len=n-1; }
            for(int t=0;t<trials;++t){
                T sum;
                func(x,y,len,sum); // warm-up
                auto st=std::chrono::high_resolution_clock::now();
                func(x,y,len,sum);
                auto ed=std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(ed-st).count();
                double gflops = (2.0*len)/(sec*1e9), ms=sec*1e3;
                log_per_trial<T>("dotp_benchmark.csv",label,dtype_name<T>(),
                                 len, mis?"misaligned":"contiguous",1,
                                 t+1,ms,gflops,sum);
            }
            aligned_free_array(base_x);
            aligned_free_array(base_y);
        }
    }
}

// stride & gather sweep
template<typename T>
void sweep_stride_and_gather(const char* label, void (*func)(const T*,const T*,size_t,T&)){
    std::vector<size_t> ms={1<<10,1<<12,1<<14,1<<16};
    std::vector<size_t> strides={1,2,4,8};
    const int trials=5; const size_t ALIGN=64;
    std::mt19937_64 rng(12345);

    for(bool mis:{false,true}){
        use_aligned_loads = !mis;
        for(size_t m: ms){
            size_t maxs=*std::max_element(strides.begin(),strides.end());
            size_t buf_e= m*maxs+1;
            T* base_x=aligned_alloc_array<T>(buf_e,ALIGN);
            T* base_y=aligned_alloc_array<T>(buf_e,ALIGN);
            if(!base_x||!base_y){ aligned_free_array(base_x);aligned_free_array(base_y);continue;}
            for(size_t i=0;i<buf_e;++i){ base_x[i]=T(1); base_y[i]=T(2); }
            T* x=base_x; T* y=base_y;
            if(mis){ x=(T*)((char*)base_x+sizeof(T)); y=(T*)((char*)base_y+sizeof(T)); }

            // strided
            for(size_t s:strides){
                size_t max_t = s?buf_e/s:0;
                size_t touched = std::min(m,max_t);
                if(!touched) continue;
                std::string lbl=std::string(label)+" stride="+std::to_string(s);
                for(int t=0;t<trials;++t){
                    T sum;
                    dotp_strided_wrapper<T>(func,x,y,touched,s,sum);
                    auto st=std::chrono::high_resolution_clock::now();
                    dotp_strided_wrapper<T>(func,x,y,touched,s,sum);
                    auto ed=std::chrono::high_resolution_clock::now();
                    double sec=std::chrono::duration<double>(ed-st).count();
                    double gflops=(2.0*touched)/(sec*1e9), ms=sec*1e3;
                    log_per_trial<T>("dotp_benchmark.csv",lbl,dtype_name<T>(),
                                     buf_e,"strided",s,t+1,ms,gflops,sum);
                }
            }

            // gather
            std::vector<size_t> idx(m);
            for(size_t k=0;k<m;++k) idx[k]=(k*maxs)%(buf_e-1);
            std::shuffle(idx.begin(),idx.end(),rng);
            for(int t=0;t<trials;++t){
                T sum;
                dotp_gather_wrapper<T>(x,y,idx.data(),m,sum);
                auto st=std::chrono::high_resolution_clock::now();
                dotp_gather_wrapper<T>(x,y,idx.data(),m,sum);
                auto ed=std::chrono::high_resolution_clock::now();
                double sec=std::chrono::duration<double>(ed-st).count();
                double gflops=(2.0*m)/(sec*1e9), ms=sec*1e3;
                log_per_trial<T>("dotp_benchmark.csv",label,dtype_name<T>(),
                                 buf_e,"gather",0,t+1,ms,gflops,sum);
            }

            aligned_free_array(base_x);
            aligned_free_array(base_y);
        }
    }
}

// main
int main(){
    std::ofstream clear("dotp_benchmark.csv",std::ios::trunc);
    clear.close();
    // contiguous
    sweep_contiguous<float>("Scalar float32",dotp_scalar<float>);
    sweep_contiguous<double>("Scalar float64",dotp_scalar<double>);
    sweep_contiguous<float>("SSE2 float32", dotp_sse2<float>);
    sweep_contiguous<double>("SSE2 float64", dotp_sse2<double>);
    sweep_contiguous<float>("AVX float32",   dotp_avx<float>);
    sweep_contiguous<double>("AVX float64",   dotp_avx<double>);
    sweep_contiguous<float>("AVX-512 f32",   dotp_avx512<float>);
    sweep_contiguous<double>("AVX-512 f64",  dotp_avx512<double>);
    // strided & gather
    sweep_stride_and_gather<float>("SSE2 float32", dotp_sse2<float>);
    sweep_stride_and_gather<double>("SSE2 float64", dotp_sse2<double>);
    sweep_stride_and_gather<float>("AVX float32",   dotp_avx<float>);
    sweep_stride_and_gather<double>("AVX float64",   dotp_avx<double>);
    sweep_stride_and_gather<float>("AVX-512 f32",   dotp_avx512<float>);
    sweep_stride_and_gather<double>("AVX-512 f64",  dotp_avx512<double>);
    return 0;
}