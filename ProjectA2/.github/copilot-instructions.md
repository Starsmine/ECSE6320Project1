# Project A2: Dense vs Sparse Matrix Multiplication Benchmark

## Architecture Overview

This is a performance benchmarking project comparing dense and sparse matrix multiplication (GEMM) with SIMD optimizations (AVX2/AVX-512) and OpenMP multithreading. The codebase consists of:

- **Core C++ Benchmark** (`main.cpp`): Implements dense GEMM and sparse CSR-SpMM with explicit SIMD intrinsics and cache tiling
- **Python Orchestration** (`run_benchmarks.py`): Automates systematic parameter sweeps (density, size, structure, thread count)
- **Analysis Scripts** (`analyze_results.py`, `plot_thread_scaling.py`): Parse benchmark output and generate performance visualizations

**Key Design**: Results are written to timestamped `benchmark_results_*.txt` files, then moved/renamed into `results/` by Python scripts for systematic analysis.

## Build System & Dependencies

**Critical**: This project requires OpenBLAS for reference GEMM validation and AVX-512 CPU support.

```bash
# Build optimized version (uses -march=native, -O3, -mavx512f)
make all

# Build debug version (no optimizations, adds -g -DDEBUG)
make debug

# Run with Linux perf counters
make perf
```

**Makefile conventions**:
- AVX-512 is enabled by default (`-mavx512f`); switch to `-mavx2` for older CPUs
- OpenMP is mandatory (`-fopenmp` in both CXXFLAGS and LDFLAGS)
- Links against `-lopenblas` for validation baseline

## Matrix Format & Algorithms

### Data Structures

1. **DenseMatrix**: Row-major layout with `vector<double> data`, operator()(i,j) accessor
2. **CSRMatrix**: Compressed Sparse Row format with `values`, `col_indices`, `row_ptr` arrays

### Core Algorithms

- **`denseGEMM_tiled()`**: Cache-tiled dense GEMM with configurable tile size (default 64)
- **`csrSpMM_simd()`**: SIMD-optimized sparse multiply using AVX-512 intrinsics (`_mm512_fmadd_pd`)
- **`csrSpMM()`**: Scalar sparse baseline with explicit vectorization disabling (`__attribute__((optimize("no-tree-vectorize")))`)

**Important**: Both scalar and SIMD versions exist to measure pure SIMD speedup. SIMD code processes 8 columns at a time (AVX-512) or 4 columns (AVX2).

## Running Benchmarks

The C++ executable supports multiple modes via command-line arguments:

```bash
# Single custom experiment: m=1024, k=1024, n=1024, density=5%
./matrix_bench custom 1024 1024 1024 0.05

# Preset sweep modes
./matrix_bench density    # Sweep 0.1%-50% density at 1024³
./matrix_bench size       # Sweep 256-4096 size at 5% density
./matrix_bench structure  # Test square/tall/fat matrices
```

**Python automation** (recommended for comprehensive sweeps):

```bash
python3 run_benchmarks.py density_sweep --m 1024 --k 1024 --n 1024
python3 run_benchmarks.py size_sweep --density 0.05
python3 run_benchmarks.py grid_sweep  # Full size × density grid
```

Python scripts automatically rename and organize output files into `results/` with descriptive names like `density_sweep_1024x1024x1024_d0050.txt`.

## Output Format & Analysis

### Benchmark Output Structure

Each result file contains:
- Experiment parameters (m, k, n, density %)
- CSR statistics (nnz count, avg nnz per row)
- Arithmetic Intensity (FLOPs/byte) for dense and sparse
- Validation against OpenBLAS reference (relative/absolute error)
- Performance: Time (s), GFLOP/s, CPNZ (cycles per nonzero)

**Key metrics**:
- **GFLOP/s**: Standard throughput metric
- **CPNZ**: Cycles per nonzero (sparse-specific, assumes 2.4 GHz clock)
- **Speedup vs scalar**: SIMD speedup factor

### Analysis Workflow

```bash
# Parse density sweep and plot break-even curves
python3 analyze_results.py density_plot --pattern "density_sweep_1024*"

# Thread scaling analysis
python3 plot_thread_scaling.py --pattern "threads_t*"
```

**Parsing convention**: Scripts use regex patterns to extract structured data from human-readable benchmark output. See `BenchmarkAnalyzer.parse_result_file()` for field extraction patterns.

## Code Modification Guidelines

### Adding New Benchmark Modes

1. Add command-line mode in `main()` (see existing "density", "size", "structure" modes)
2. Add corresponding method to `BenchmarkRunner` class in Python script
3. Use consistent file naming: `<mode>_<params>.txt`

### SIMD Optimization

When modifying SIMD kernels:
- Always provide both `#ifdef __AVX512F__` and `#elif defined(__AVX2__)` paths
- Maintain scalar fallback for remainder columns: `for (; j < n; j++)`
- Use FMA intrinsics (`_mm512_fmadd_pd`) for best performance
- Test with `make perf` to measure cache behavior

### Validation

All new kernels must validate against `openblasGEMM()` with relative error < 1e-6. The validation workflow:
1. Generate reference with OpenBLAS on dense/sparse matrices
2. Compute relative error: `max_diff / (max_val + 1e-10)`
3. Check thresholds: < 1e-10 (excellent), < 1e-6 (acceptable), else warning

## Common Patterns

- **Logging macro**: `LOG(x)` outputs to both console and `g_logfile` stream
- **Benchmarking**: `benchmarkFunction()` takes lambda, runs 3 times, returns median
- **Thread control**: OpenMP parallelizes outer loops with `#pragma omp parallel for`
- **Result organization**: Python scripts use `Path.glob()` to batch-process result files

## Troubleshooting

- **Missing OpenBLAS**: Install `libopenblas-dev` (Ubuntu) or `openblas` (Arch/Fedora)
- **AVX-512 not available**: Change `CXXFLAGS += -mavx512f` to `-mavx2` in Makefile
- **Incorrect results**: Check OpenMP thread count with `omp_get_max_threads()` - validation uses single-threaded OpenBLAS reference
