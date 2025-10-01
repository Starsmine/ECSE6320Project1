# Project1Saxpy - SIMD SAXPY Implementation

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This project implements the SAXPY operation (Single-precision A*X Plus Y) using SIMD vectorization techniques. SAXPY is a fundamental linear algebra operation that computes `Y = a*X + Y` where `a` is a scalar, and `X` and `Y` are vectors.

## Contents

## Compilation Requirements
- **Compiler**: GCC 14.2.0 with SIMD support
- **Flags**: `-O3 -mavx2 -mfma -mavx512f -fno-tree-vectorize`
- **Architecture**: x86-64 with AVX2/AVX-512 support

## System Configuration
- **CPU**: Ryzen 7 7700X (up to 5.5GHz)
- **Memory**: DDR5 6000Mt/s CL32 for optimal bandwidth
- **Cache**: 32KB L1d, 1MB L2, 32MB L3 per specifications
- **SMT State**: On

### Source Code
- `SaxpyMain.cpp` - Main SAXPY implementation with SIMD optimizations

### Benchmark Data
- `saxpy_benchmark.csv` - Comprehensive performance results

### Analysis Scripts
- `plot_saxpy.py` - Main plotting script for SAXPY performance analysis

### Results
The `plots_output/` directory contains extensive performance visualizations:


### Performance Analysis

#### Baseline vs Vectorized
![SAXPY Performance Results](./plots_output/saxpy_avg_gflops_base.png)
![SAXPY Performance Results](./plots_output/saxpy_speedup_base.png)


## Related Projects
- [Dot Product Implementation](../Project1DotProduct/README.md) - Complementary SIMD project
- [3D Point Stencil](../Project13dpointstencil/README.md) - Advanced SIMD applications

---
[← Back to Main README](../README.md)