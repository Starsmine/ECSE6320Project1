# Project13dpointstencil - 3D Point Stencil SIMD Operations

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This project implements advanced 3D point stencil operations using SIMD vectorization techniques. The implementation demonstrates sophisticated memory access patterns and computational kernels optimized for modern CPU architectures with AVX2 and AVX-512 support.

## Contents

### Source Code
- `dotpMain_v2.cpp` - Optimized dot product implementation for stencil operations
- `stencilMain_v1.cpp` - Primary 3D stencil kernel implementation

### Benchmark Data
- `stencil_benchmark_v1.csv` - Comprehensive stencil operation performance data

### Analysis Scripts
- `plot_dotp.py` - Dot product performance analysis
- `plot_stencil.py` - Stencil operation visualization and analysis

### Results
The `plots_output/` directory contains extensive performance visualizations organized by data type and access patterns:

#### Float32 Results
- `float32_consolidated.png` - Combined performance overview
- `float32_contiguous_speedup.png` - Contiguous memory access performance
- `float32_contiguous_misaligned_speedup.png` - Misaligned contiguous access analysis
- `float32_stride_2_speedup.png` - Stride-2 access pattern performance
- `float32_stride_2_misaligned_speedup.png` - Misaligned stride-2 analysis
- `float32_stride_16_speedup.png` - Stride-16 access pattern performance
- `float32_stride_16_misaligned_speedup.png` - Misaligned stride-16 analysis

## Key Features

### 3D Stencil Operations
- **7-point Stencil**: Classic finite difference stencil pattern
- **27-point Stencil**: Extended neighborhood operations
- **Variable Radius**: Configurable stencil radius for different applications
- **Boundary Handling**: Optimized edge and corner case processing

### Memory Access Patterns
- **Contiguous Access**: Sequential memory access optimization
- **Strided Access**: Non-unit stride pattern handling (stride-2, stride-16)
- **Alignment Analysis**: Performance impact of memory alignment
- **Cache Optimization**: Memory layout strategies for cache efficiency

### SIMD Optimizations
- **AVX2 Implementation**: 256-bit vector processing
- **AVX-512 Implementation**: 512-bit vector processing with enhanced throughput
- **Memory Prefetching**: Software prefetch instructions for improved bandwidth
- **Loop Tiling**: Cache-aware blocking for large datasets

## Algorithm Details

### Stencil Computation
The 3D stencil operation applies a computational kernel to each point using neighboring values:
```
output[i,j,k] = f(input[i±δ, j±δ, k±δ])
```

### Vectorization Strategy
- Simultaneous processing of multiple points along fastest-changing dimension
- Optimized memory access patterns to minimize cache misses
- Load balancing across vector lanes

## Performance Characteristics

### Memory Access Patterns
1. **Contiguous**: Optimal cache line utilization
2. **Stride-2**: Moderate cache efficiency with 50% utilization
3. **Stride-16**: Challenging memory pattern with potential cache thrashing

### Alignment Impact
- Aligned access: Optimal performance with full bandwidth
- Misaligned access: Performance degradation analysis

## Compilation
Optimized with GCC 14.2.0:
```bash
-O3 -mavx2 -mfma -mavx512f -fno-tree-vectorize
```

## System Requirements
- **CPU**: AVX2/AVX-512 capable processor (tested on Ryzen 7 7700X)
- **Memory**: High-bandwidth DDR5 for optimal stencil performance
- **Cache**: Large L3 cache beneficial for working set management

## Applications
- **Scientific Computing**: Finite difference methods, PDEs
- **Image Processing**: Convolution operations, filtering
- **Machine Learning**: Convolutional neural network operations
- **Computational Fluid Dynamics**: Grid-based simulations

## Usage
1. Compile source files with SIMD optimization flags
2. Run stencil benchmarks with various grid sizes and patterns
3. Use `plot_stencil.py` to generate performance analysis plots
4. Compare different access patterns and alignment strategies

## Related Projects
- [Dot Product](../Project1DotProduct/README.md) - Fundamental SIMD operations
- [SAXPY](../Project1Saxpy/README.md) - Linear algebra SIMD implementations

---
[← Back to Main README](../README.md)