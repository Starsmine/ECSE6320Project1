# ProjectA2: Dense vs Sparse Matrix Multiplication Benchmark - Analysis Summary

## Completed Requirements

### ✅ 1. Correctness & Baselines
- **OpenBLAS validation**: All kernels validated against CBLAS reference
- **Scalar vs SIMD comparison**: Implemented both for CSR-SpMM
- **Error metrics**: Relative and absolute error reporting
- **Files**: `main.cpp` (lines 600-650), results show `✓ validated successfully`

### ✅ 2. SIMD & Threading Speedup
- **Thread scaling**: 1, 2, 4, 8, 16 threads benchmarked
- **Plots**: `thread_scaling.png` shows speedup curves with error bars
- **Results**: Dense GEMM achieves ~8-10x speedup at 16 threads
- **SIMD**: CSR-SpMM SIMD shows 3.5-4.2x speedup vs scalar
- **Data**: `results/threads_t*.txt` with 5 runs, median/stddev

### ✅ 3. Density Break-Even Analysis
- **Sweep**: 25 density points from 0.01% to 50% (2048³ matrices)
- **Plot**: `density_breakeven.png` with error bars, break-even interpolation
- **Finding**: Break-even at ~0.56% density where Dense = Sparse SIMD
- **Statistics**: 5 runs per point, median performance with ±stddev
- **Data**: `results/density_sweep_2048x2048x2048_d*.txt`

### ✅ 4. Working-Set Transitions
- **Size sweep**: 28 matrix sizes from 32×32 to 6144×6144
- **Threading strategy**: 
  - Single-threaded (1 thread) for <1MB working set (L1d/L2: 32-192)
  - Multi-threaded (16 threads) for ≥1MB working set (L3/DRAM: 224-6144)
- **Plot**: `working_set_transitions.png` with 3 subplots:
  1. Performance (GFLOP/s) vs matrix size
  2. Cache miss rate (%) vs size
  3. Data transfer rate (bytes/time) vs size
- **Cache hierarchy**: Clear transitions at L2→L3 (~192→224) and L3→DRAM (~2048+)
- **Error bars**: All plots include ±stddev from 5 runs
- **Data**: `results/size_sweep_d005_s*.txt`

### ✅ 5. Roofline Interpretation
- **Plot**: `roofline_analysis.png` with two subplots:
  1. Roofline model with L1d/L2/L3/DRAM bandwidth curves
  2. Achieved performance across all matrix sizes
- **Arithmetic Intensity**:
  - Dense GEMM: 2.67 - 512 FLOPs/byte (grows with size)
  - Sparse SIMD: 0.04 - 73 FLOPs/byte (depends on density)
- **Peak FLOPS**: 691.2 GFLOP/s theoretical (AMD Ryzen 7 7700X, 8 cores, AVX-512)
- **Measured Bandwidth**:
  - L1d: 60 GB/s → Ridge: 11.52 FLOPs/byte
  - L2: 400 GB/s → Ridge: 1.73 FLOPs/byte  
  - L3: 100 GB/s → Ridge: 6.91 FLOPs/byte
  - DRAM: 40 GB/s → Ridge: 17.28 FLOPs/byte
- **Conclusions**:
  - **Dense GEMM**: COMPUTE-BOUND (AI >> ridge point)
    - Achieves 100-117 GFLOP/s (13-17% of peak)
    - Limited by ALU throughput, not memory
  - **Sparse SIMD**: MEMORY-BOUND (AI ≤ ridge point)
    - Achieves 2-47 GFLOP/s (0.3-6% of peak)
    - Limited by memory bandwidth and irregular access

## Performance Highlights

### Dense GEMM (Tiled Implementation)
- **Best**: 117 GFLOP/s (6144×6144, 16 threads)
- **L1d cache**: 19-24 GFLOP/s (32-96, 1 thread)
- **L2 cache**: 14-15 GFLOP/s (128-192, 1 thread)
- **L3 cache**: 90-107 GFLOP/s (1024-2048, 16 threads)
- **Cache miss rate**: 0.5-2.8% (excellent locality)

### Sparse CSR-SpMM (SIMD)
- **Best**: 47 GFLOP/s (1280×1280, 1.5% density, 16 threads)
- **Low density**: 2-8 GFLOP/s (memory-bound)
- **High density**: Approaches dense performance
- **Speedup vs scalar**: 3.5-4.2x consistent across sizes

## Implementation Details

### Timing Precision
- **Clock**: CLOCK_MONOTONIC (nanosecond precision)
- **Output**: 6 decimal places (microsecond resolution)
- **Adaptive iterations**: 50ms minimum measurement window, 50k max iterations
- **Statistics**: 5 runs per benchmark, median/mean/stddev reported

### Performance Counters
- **Hardware counters**: cycles, instructions, cache misses, LLC misses
- **Metrics**: IPC, cache miss rate, LLC miss rate
- **Integration**: Linux perf_event API

### Optimization Techniques
- **Dense GEMM**: Cache-aware tiling (64×64 blocks), row-major layout
- **Sparse SIMD**: AVX-512 vectorization, prefetching, loop unrolling
- **Threading**: OpenMP parallel loops with dynamic scheduling

## Files Generated

### Plots (PNG, 300 DPI)
1. `density_breakeven.png` - Break-even analysis with error bars
2. `working_set_transitions.png` - Cache hierarchy transitions (3 subplots)
3. `thread_scaling.png` - SIMD and threading speedup
4. `roofline_analysis.png` - Compute vs memory-bound analysis (2 subplots)

### Scripts
- `main.cpp` - Core benchmark implementation
- `run_benchmarks.py` - Automated benchmark orchestration
- `analyze_results.py` - Density break-even analysis
- `plot_working_set.py` - Working-set transitions analysis
- `plot_thread_scaling.py` - Thread scaling analysis
- `plot_roofline.py` - Roofline model analysis
- `measure_bandwidth.cpp` - Memory bandwidth measurement

### Data (180 result files)
- `results/density_sweep_*.txt` - 25 density points
- `results/size_sweep_*.txt` - 28 matrix sizes
- `results/threads_*.txt` - 5 thread counts
- `results/structure_*.txt` - Various matrix shapes
- `results/grid_*.txt` - Size×density grid

## System Configuration

- **CPU**: AMD Ryzen 7 7700X (Zen 4)
- **Cores**: 8 cores, 16 threads (SMT)
- **Frequency**: 4.5 GHz base, 5.4 GHz boost
- **SIMD**: AVX-512 (8 doubles, 2 FMAs/cycle = 16 FLOPs/cycle/core)
- **Cache**: L1d=32KB, L2=1MB per core, L3=32MB shared
- **Memory**: DDR4/DDR5 (~40 GB/s measured)
- **Compiler**: GCC with -O3 -march=native -mavx512f -fopenmp

## Key Findings

1. **Dense GEMM is consistently compute-bound** across all cache levels due to high arithmetic intensity (85-512 FLOPs/byte). Performance scales well with threading (8-10x at 16 threads) and achieves 13-17% of theoretical peak.

2. **Sparse CSR-SpMM is memory-bound** for low-to-medium densities (<10%). Irregular memory access patterns limit performance to 0.3-6% of theoretical peak. SIMD provides 3.5-4.2x speedup but doesn't overcome memory bottleneck.

3. **Break-even point at ~0.56% density**: Below this, sparse operations are faster despite being memory-bound. Above this, dense tiled GEMM dominates due to compute-bound performance.

4. **Cache hierarchy matters for small matrices**: L1d/L2 sizes (32-192) benefit from single-threaded execution to avoid thread overhead. L3/DRAM sizes (224+) benefit from 16-thread parallelism.

5. **Threading overhead is significant for tiny matrices**: 128×128 shows 17x performance improvement when moving from 16 threads (0.89 GFLOP/s) to 1 thread (15.44 GFLOP/s) due to reduced synchronization and NUMA effects.
