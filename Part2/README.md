# Part2 - Cache Hierarchy and Memory Access Pattern Analysis

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This section contains comprehensive analysis of cache hierarchy behavior and memory access patterns using various benchmarking tools and methodologies.

## Contents

### Scripts
- `plot_results.py` - Python script for generating performance visualization plots
- `ZeroQue.ps1` - PowerShell script for zero-queue latency testing

### Results Data
The `results/` directory contains CSV files with benchmark data from multiple test runs:
- **Cache Hierarchy Analysis**: `cache_hierarchy_*` folders and combined results
- **Cache Miss Analysis**: `cache_miss_*` folders and combined data  
- **Granularity Sweep**: Testing different memory access granularities
- **Intensity Sweep**: Varying computational intensity measurements
- **Pattern Sweep**: Different memory access pattern evaluations
- **Read/Write Mix**: Mixed workload performance analysis
- **TLB Miss Analysis**: Translation Lookaside Buffer miss characterization

### Generated Plots
The `plots/` directory contains visualization of results:
- `granularity_sweep.png` - Memory access granularity performance
- `intensity_sweep.png` - Computational intensity vs performance
- `pattern_bandwidth_matrix.png` - Bandwidth matrix for different patterns
- `pattern_latency_matrix.png` - Latency matrix for access patterns
- `pattern_sweep_combined.png` - Combined pattern analysis
- `rw_mix.png` - Read/write mix performance comparison
- `wss_sweep.png` - Working set size sweep results
- `zero_queue_latencies.png` - Zero-queue latency measurements

## Key Findings

### Cache Hierarchy Performance
- L1 cache: 32 KiB per core, 8-way associative
- L2 cache: 1 MiB per core, 8-way associative  
- L3 cache: 32 MiB shared, 16-way associative

### Memory Access Patterns
Analysis covers various patterns including:
- Sequential access patterns
- Random access patterns
- Strided access patterns
- Mixed read/write workloads

## Usage
1. Run benchmark scripts to collect performance data
2. Use `plot_results.py` to generate visualizations
3. Analyze results in the `plots/` directory

## System Configuration
- **CPU**: Ryzen 7 7700X (up to 5.5GHz)
- **RAM**: DDR5 6000Mt/s CL32
- **Compiler**: GCC 14.2.0 with AVX2/AVX-512 optimizations

---
[← Back to Main README](../README.md)