# ECSE 6320 - Advanced Computer Systems Projects

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Starsmine/ECSE6320Project1)

This repository contains four comprehensive projects exploring modern computer architecture, performance analysis, and concurrent programming concepts.

## Projects

### [Project A1: Operating System and CPU Feature Performance Exploration](./ProjectA1/)
Explores performance impacts of OS and CPU microarchitectural features including:
- CPU Affinity and Thread Pinning
- Cache Prefetching (Hardware Prefetchers)
- Simultaneous Multithreading (SMT/Hyperthreading)
- Huge Pages (THP)
- Asynchronous I/O (io_uring with zero-copy)

**Key Results**: io_uring achieves 24× speedup (977 MB/s vs 66 MB/s sync I/O), cache prefetcher shows 3.4× performance gain for sequential access.

[📖 View Full Documentation](./ProjectA1/README.md)

---

### [Project A2: Dense vs Sparse Matrix Multiplication](./ProjectA2/)
Implements and benchmarks dense matrix multiplication (GEMM) using tiled/blocked algorithms versus sparse matrix multiplication (CSR-SpMM) with SIMD optimizations:
- Performance breakeven analysis (density: 8.5% scalar, 31% SIMD)
- Working-set transitions (cache hierarchy analysis)
- Arithmetic intensity and roofline modeling
- Thread scaling and perf counter analysis

**Key Results**: SIMD SpMM achieves 10.2× speedup. Break-even at 31% density (SIMD) and 8.5% (scalar).

[📖 View Full Documentation](./ProjectA2/README.md)

---

### [Project A3: Approximate Membership Filters](./ProjectA3/)
Comprehensive implementation and benchmarking of four filter types:
- Blocked Bloom Filter (best throughput: 62.8 Mops/s)
- XOR Filter (best space: 9.84 BPE, bit-packed)
- Cuckoo Filter (dynamic with deletes, 2× faster than Quotient at high load)
- Quotient Filter (dynamic with contiguous runs)

**Key Results**: XOR filter achieves 9.84 bits per entry. Bloom filter shows 3.19× thread scaling. SMT degradation observed at 16 threads.

[📖 View Full Documentation](./ProjectA3/README.md)

---

### [Project A4: Concurrent Data Structures and Memory Coherence](./ProjectA4/)
Thread-safe hash table with synchronization strategy comparison:
- Coarse-grained locking (single global mutex)
- Fine-grained locking (per-bucket locks, 10K mutexes)
- Cache coherence effects and false sharing analysis
- Amdahl's Law validation

**Key Results**: Fine-grained achieves 7.0× speedup (lookup, 16 threads). Coarse-grained shows negative scaling (0.34× at 16 threads due to contention).

[📖 View Full Documentation](./ProjectA4/README.md)

---

## System Configuration

All benchmarks were conducted on:

### Hardware
- **CPU**: AMD Ryzen 7 7700X (Zen 4 architecture)
  - 8 cores / 16 threads (SMT enabled)
  - Base Clock: 4.5 GHz, Boost Clock: 5.4 GHz
  - **Cache**: L1d: 32 KB × 8, L2: 1 MB × 8, L3: 32 MB (shared)
  - **AVX-512**: Double-pumped (2 cycles per 512-bit operation)
- **Memory**: DDR5-6000 (dual-channel, 48 GB)
- **Storage**: 
  - System: PNY CS3140 1TB NVMe SSD (PCIe Gen4)
  - Secondary: ADATA XPG GAMMIX S70 1TB NVMe SSD (PCIe Gen4)

### Software
- **OS**: Ubuntu 24.04.3 LTS (Noble Numbat)
- **Kernel**: 6.14.0-37-generic
- **Compiler**: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
- **Build Flags**: `-O3 -march=native -fopenmp` (project-specific variations)
- **Libraries**:
  - OpenBLAS (ProjectA2 validation)
  - liburing (ProjectA1 async I/O)
  - xxHash (ProjectA3 hashing)
  - 
## Repository Structure

```
FPA2/
├── README.md                    # This file
├── ProjectA1/                   # OS/CPU Feature Exploration
│   ├── main.cpp                 # 5 benchmark implementations
│   ├── plot_results.py          # Plotting script
│   ├── benchmark_results.txt    # Aggregated results
│   └── README.md                # Full documentation
├── ProjectA2/                   # Dense vs Sparse Matrix Multiplication
│   ├── main.cpp                 # GEMM + CSR-SpMM implementations
│   ├── run_benchmarks.py        # Automated benchmark suite
│   ├── analyze_results.py       # Analysis and plotting
│   ├── results/                 # 82 benchmark result files
│   └── README.md                # Full documentation
├── ProjectA3/                   # Approximate Membership Filters
│   ├── main.cpp                 # 4 filter implementations
│   ├── run_benchmarks.py        # Benchmark orchestration
│   ├── analyze_results.py       # Plotting script
│   ├── results/                 # 860 benchmark result files
│   └── README.md                # Full documentation
└── ProjectA4/                   # Concurrent Hash Tables
    ├── main.cpp                 # Coarse + fine-grained implementations
    ├── run_benchmarks.py        # Benchmark suite with perf counters
    ├── analyze_results.py       # Analysis and plotting
    ├── results/                 # 450 benchmark result files
    └── README.md                # Full documentation
```

## Building All Projects

Each project includes a `Makefile` for easy compilation:

```bash
# Build all projects
for project in ProjectA1 ProjectA2 ProjectA3 ProjectA4; do
    cd $project && make && cd ..
done

# Or build individually
cd ProjectA1 && make
cd ProjectA2 && make
cd ProjectA3 && make
cd ProjectA4 && make
```

## Running Benchmarks

Each project includes automated benchmark scripts:

```bash
# ProjectA1: Run all 5 benchmarks
cd ProjectA1 && ./os_features_bench

# ProjectA2: Full benchmark suite (82 configs, ~30 min)
cd ProjectA2 && python3 run_benchmarks.py

# ProjectA3: 4 experiments (860 configs, ~45 min)
cd ProjectA3 && python3 run_benchmarks.py

# ProjectA4: Thread scaling suite (450 configs, ~15 min)
cd ProjectA4 && python3 run_benchmarks.py
```

## Key Insights Across Projects

### Performance Optimization Themes

1. **Memory Access Patterns Matter** (A1, A2, A3)
   - Sequential access: 3.4× faster than random (prefetcher)
   - Cache blocking: Critical for dense GEMM performance
   - Bit-packing: 75% memory reduction in XOR filter

2. **Concurrency Challenges** (A3, A4)
   - Lock granularity: Fine-grained achieves 7× speedup vs coarse (negative scaling)
   - False sharing: Cache line bouncing degrades performance
   - SMT benefits diminish: 16 threads show degradation vs 8 cores

3. **Hardware-Software Co-design** (A1, A2, A4)
   - io_uring zero-copy: 24× speedup through kernel bypass
   - AVX-512 SIMD: 10.2× speedup in sparse matrix operations
   - Cache coherence: +11% miss rate at 16 threads (MESI protocol overhead)

4. **Algorithmic Trade-offs** (A2, A3)
   - Dense vs sparse: Break-even at 31% density (SIMD)
   - Space vs speed: XOR (9.84 BPE) vs Bloom (62.8 Mops/s)
   - Static vs dynamic: XOR fastest but immutable

## Dependencies

Common dependencies across projects:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    gcc g++ \
    libopenblas-dev \
    liburing-dev \
    libxxhash-dev \
    linux-tools-generic \
    python3 python3-pip

# Python packages
pip3 install matplotlib numpy scipy
```

## Performance Counter Tools

Using `perf` for hardware counter analysis:

```bash
# Enable perf for non-root users
sudo sysctl -w kernel.perf_event_paranoid=-1

# Example: Run with cache miss tracking
perf stat -e cycles,cache-misses,cache-references ./your_benchmark
```


## Contact

- GitHub: [@Starsmine](https://github.com/Starsmine)
- Repository: [ECSE6320Project1](https://github.com/Starsmine/ECSE6320Project1)

---

**Note**: All benchmarks include multiple runs with statistical analysis (mean ± stddev). Performance results are specific to the hardware/software configuration listed above and may vary on different systems.
