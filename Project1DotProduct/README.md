# Project1DotProduct - SIMD Dot Product Optimization

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This project demonstrates the optimization of dot product calculations using SIMD (Single Instruction, Multiple Data) instructions. The implementation showcases performance improvements achieved through vectorization with AVX2 and AVX-512 instruction sets.

## Contents

### Source Code
- `dotpMain.cpp` - Original dot product implementation
- `dotpMain_v2.cpp` - Optimized version with enhanced SIMD utilization

### Benchmark Data
- `dotp_benchmark.csv` - Performance results from initial implementation
- `dotp_benchmark_v2.csv` - Results from optimized version

### Analysis Scripts
- `plot_dotp.py` - Python script for generating dot product performance plots
- `plot_saxpy.py` - Cross-reference plotting for SAXPY comparison

### Results
The `plots_output/` directory contains comprehensive performance visualizations showing:
- Speedup comparisons between scalar and vectorized implementations
- Performance scaling across different vector sizes
- Memory access pattern impact analysis
- SIMD instruction efficiency metrics

## Key Features

### SIMD Optimizations
- **AVX2 Implementation**: 256-bit vectors processing 8 single-precision floats simultaneously
- **AVX-512 Implementation**: 512-bit vectors processing 16 single-precision floats simultaneously
- **Memory Alignment**: Optimized data layout for maximum throughput
- **Loop Unrolling**: Reduced overhead and improved instruction pipeline utilization

### Performance Characteristics
- Theoretical speedup: 8x (AVX2) and 16x (AVX-512) over scalar implementation
- Actual measured performance gains varying by data size and access patterns
- Memory bandwidth considerations and cache optimization

## Compilation
Uses GCC 14.2.0 with optimization flags:
```bash
-O3 -mavx2 -mfma -mavx512f -fno-tree-vectorize
```

## System Requirements
- **CPU**: AVX2/AVX-512 capable processor (tested on Ryzen 7 7700X)
- **Memory**: DDR5 6000Mt/s for optimal performance
- **Compiler**: GCC 14.2.0 or compatible with SIMD support

## Benchmark Results
The implementation demonstrates significant performance improvements:
- Vector sizes from small (16 elements) to large (1M+ elements)
- Performance scaling analysis across different working set sizes
- Comparison of aligned vs. unaligned memory access patterns

## Usage
1. Compile the source code with appropriate SIMD flags
2. Run benchmarks across various vector sizes
3. Use `plot_dotp.py` to generate performance visualizations
4. Analyze results in the `plots_output/` directory

## Related Projects
- [SAXPY Implementation](../Project1Saxpy/README.md) - Related SIMD optimization project
- [3D Point Stencil](../Project13dpointstencil/README.md) - Advanced SIMD applications

---
[← Back to Main README](../README.md)