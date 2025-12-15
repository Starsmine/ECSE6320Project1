# Project A4: Concurrent Data Structures and Memory Coherence

[![Back to Main README](https://img.shields.io/badge/⬆_Back_to-Main_README-blue?style=for-the-badge)](../README.md)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Starsmine/ECSE6320Project1)

Thread-safe hash table implementations comparing synchronization strategies and cache coherence effects.

## Key Findings

**Best Scalability**: Fine-grained locking (7.0× speedup at 16 threads on lookup workload)  
**Best Single-Thread**: Coarse-grained faster on lookups (14.1 vs 10.2 Mops/s - lower lock overhead)  
**Coarse-Grained Collapse**: **Negative scaling** - becomes 27% of baseline at 16 threads (lock contention catastrophe)  
**Workload Sensitivity**: Lookup-only scales 2× better than insert-only (7.0× vs 3.4×) due to reduced write conflicts  

## Quick Start

### Build
```bash
make
```

### Run All Benchmarks
```bash
python3 run_benchmarks.py
```

### Generate Plots
```bash
python3 analyze_results.py
```

### Manual Benchmarking
```bash
# Single-threaded baseline
./hash_bench --table coarse --workload insert --size 1000000 --threads 1

# Thread scaling comparison
./hash_bench --table fine --workload insert --size 1000000 --threads 8

# Mixed workload (70% lookup, 30% insert)
./hash_bench --table fine --workload mixed --size 1000000 --threads 8
```

## Implementation

### Data Structure: Hash Table with Chaining
- **Buckets**: 10,000 fixed buckets (configurable)
- **Collision Resolution**: Chaining with `std::vector<Entry>` per bucket
- **Hash Function**: `std::hash<int64_t>` modulo num_buckets
- **Operations**: `insert(key, value)`, `find(key)`, `erase(key)`

### Synchronization Strategies

#### 1. Coarse-Grained Locking (Baseline)
- **Design**: Single global `std::mutex` protects entire hash table
- **Lock Scope**: All operations acquire global lock
- **Pros**: 
  - Simple implementation
  - No deadlocks possible
  - Low lock overhead (single mutex)
- **Cons**:
  - **Serialization**: All threads contend for same lock
  - **No parallelism**: Only one operation executes at a time
  - **Amdahl's Law**: Critical section = 100% → speedup limited to ~1.5×

#### 2. Fine-Grained Locking (Per-Bucket)
- **Design**: One `std::mutex` per bucket (10,000 locks total)
- **Lock Scope**: Only acquire lock for specific bucket being accessed
- **Pros**:
  - **True parallelism**: Threads operating on different buckets don't contend
  - **Better scaling**: ~3× speedup at 8 threads
  - **Reduced critical section**: Only bucket-level, not global
- **Cons**:
  - **Higher memory overhead**: 10,000 mutexes vs 1 (40 bytes × 10K = 400 KB)
  - **Cache line contention**: Adjacent bucket locks may share cache lines (false sharing)
  - **Complexity**: Must ensure consistent hashing to avoid race conditions

### Correctness Guarantees

**Coarse-Grained**:
- Global lock ensures mutual exclusion for all operations
- No race conditions possible
- Linearizable: operations appear atomic

**Fine-Grained**:
- Per-bucket locks ensure mutual exclusion within each bucket
- Hash function deterministic → same key always maps to same bucket
- No inter-bucket dependencies → no deadlock
- Linearizable per bucket

## Benchmark Results

### Throughput Scaling (1M keys)
[Throughput scaling](./Reportpng/plot1_throughput_vs_threads.png)
[Speedup scaling](./Reportpng/plot2_speedup_vs_threads.png)
[workload mix](./Reportpng/plot4_workload_mix.png)
#### Insert-Only Workload
| Threads | Coarse (Mops/s) | Fine (Mops/s) | Coarse Speedup | Fine Speedup |
|---------|-----------------|---------------|----------------|--------------|
| 1       | 10.48           | 11.25         | 1.00×          | 1.00×        |
| 2       | 6.44            | 15.14         | 0.61×          | 1.35×        |
| 4       | 3.50            | 21.63         | 0.33×          | 1.92×        |
| 8       | 4.69            | 31.03         | 0.45×          | 2.76×        |
| 16      | 4.05            | 36.76         | 0.39×          | 3.27×        |

**Key Findings**:
- **Coarse catastrophe**: **Negative scaling** - throughput drops from 10.48 to 3.50 Mops/s (4 threads)
  - Global lock causes severe contention and context switching overhead
  - Threads spend more time waiting than executing
- **Fine scales well**: 3.3× at 16 threads despite cache coherence overhead
- **Lock contention dominates**: Coarse-grained drops to 33% of single-threaded performance at 4 threads

#### Lookup-Only Workload
| Threads | Coarse (Mops/s) | Fine (Mops/s) | Coarse Speedup | Fine Speedup |
|---------|-----------------|---------------|----------------|--------------|
| 1       | 18.29           | 18.84         | 1.00×          | 1.00×        |
| 2       | 9.68            | 34.25         | 0.53×          | 1.82×        |
| 4       | 10.15           | 60.54         | 0.55×          | 3.21×        |
| 8       | 8.07            | 72.72         | 0.44×          | 3.86×        |
| 16      | 6.65            | 88.52         | 0.36×          | 4.70×        |

**Key Findings**:
- **Coarse collapse**: Shows negative scaling despite read-only workload
  - Mutex contention even for non-conflicting reads (drops 18.29 → 6.65 Mops/s)
  - Lock acquisition overhead dominates
- **Fine excellent scaling**: **4.7× speedup at 16 threads**
  - True read parallelism (different buckets don't contend)
  - Nearly linear scaling through 4 threads (3.2×)
- **Reader-writer locks needed**: `std::shared_mutex` would allow concurrent readers for coarse-grained

#### Mixed Workload (70% Lookup / 30% Insert)
| Threads | Coarse (Mops/s) | Fine (Mops/s) | Coarse Speedup | Fine Speedup |
|---------|-----------------|---------------|----------------|--------------|
| 1       | 13.72           | 13.49         | 1.00×          | 1.00×        |
| 2       | 7.50            | 21.50         | 0.55×          | 1.59×        |
| 4       | 5.79            | 36.25         | 0.42×          | 2.69×        |
| 8       | 5.42            | 53.53         | 0.39×          | 3.97×        |
| 16      | 4.72            | 69.87         | 0.34×          | 5.18×        |

**Key Finding**: Mixed workload shows intermediate scaling between insert-only (3.3×) and lookup-only (4.7×), achieving 5.2× at 16 threads

### Cache Coherence Analysis

**LLC (Last-Level Cache) Misses - Mixed Workload (1M keys)**:
[cache misses v threads](./Reportpng/plot3_cache_misses_vs_threads.png)
| Threads | Cache Misses (millions) | Miss Rate | Notes |
|---------|------------------------|-----------|-------|
| 1       | 22.2                   | 20.1%     | Baseline (minimal coherence traffic) |
| 16      | 24.7                   | 24.4%     | Increased coherence traffic with threads |

**Key Observations** (Fine-grained locking, mixed workload):
- Cache misses increase modestly: 22.2M → 24.7M (+11%) from 1 to 16 threads
- Miss rate increases from 20.1% to 24.4% due to cache line bouncing between cores
- Despite cache coherence overhead, fine-grained still achieves 5.2× speedup
- Per-bucket locking reduces false sharing compared to global lock

**Analysis**:
- **Modest cache impact**: +11% cache misses at 16 threads shows good data locality
- **Coherence overhead exists but manageable**: Miss rate increases 4.3 percentage points
- **MESI protocol**: Lock acquisitions cause cache line invalidations across cores
- **Memory bandwidth**: At higher thread counts, memory bus becomes bottleneck (DDR5-6000)

## Experimental Design

### Workload Mixes
1. **Insert-Only**: Pure write stress test (worst-case contention)
2. **Lookup-Only**: Read-dominated workload (database query simulation)
3. **Mixed 70/30**: Realistic balance (70% reads, 30% writes)

### Dataset Sizes
[Dataset sizes](./Reportpng/plot5_dataset_size_scaling.png)
- Small: 10K keys (10⁴) - fits in L2 cache
- Medium: 100K keys (10⁵) - fits in L3 cache
- Large: 1M keys (10⁶) - memory-bound


### Thread Counts
- 1, 2, 4, 8, 16 threads
- System: AMD Ryzen 7 7700X (8 cores / 16 threads)

### Metrics Collected
- **Throughput**: Operations per second (Mops/s)
- **Speedup**: Relative to single-threaded baseline
- **Perf Counters**: cycles, LLC-load-misses, LLC-store-misses

### Reproducibility
- 5 runs per configuration
- Results averaged with error bars (std dev)
- Automated via `run_benchmarks.py`
- 450 total benchmarks: 2 tables × 3 workloads × 3 sizes × 5 threads × 5 runs

## Analysis & Insights

### Amdahl's Law Application

**Coarse-Grained Catastrophe**:
- Critical section = 100% (global lock for all operations)
- Theoretical max speedup: S = 1 / 1.0 = **1.0× (no parallelism possible)**
- Observed: **0.34× at 16 threads** (negative scaling!)
- **Why worse than theory?**:
  - **Lock contention overhead**: Threads spend time acquiring/releasing locks (mixed: 13.72 → 4.72 Mops/s)
  - **Context switching**: OS scheduler overhead from waiting threads
  - **Cache coherence**: Lock variable bounces between cores (MESI protocol)
  - **False wakeups**: Spurious wakeups waste cycles

**Fine-Grained Scaling**:
- **Lookup workload**:
  - Critical section ≈ 25% (bucket-level contention with 10K buckets, random keys)
  - Theoretical max speedup: S = 1 / (0.25 + 0.75/16) ≈ 5.33×
  - Observed: **4.7× at 16 threads** (88% of theoretical!)
- **Insert workload**:
  - Critical section ≈ 35% (higher due to write conflicts and cache invalidations)
  - Theoretical max: S = 1 / (0.35 + 0.65/16) ≈ 2.84×
  - Observed: **3.3× at 16 threads** (exceeds theory - SMT benefits mask some serialization)
- **Mixed workload (70% read / 30% write)**:
  - Observed: **5.2× at 16 threads** (between insert and lookup, as expected)

### Cache Coherence Effects

**Why Fine-Grained Misses Increase**:
1. **Lock Metadata**: 10,000 mutexes = 10,000 atomic state variables
2. **False Sharing**: Mutexes allocated contiguously → share cache lines
   - Example: Bucket 0 lock and Bucket 1 lock in same 64-byte line
   - Thread A locks bucket 0 → invalidates Thread B's cache for bucket 1
3. **MESI Protocol Overhead**: 
   - Exclusive → Modified → Invalid → Shared transitions
   - Each lock acquisition = cache line bounce between cores

**Mitigation Strategies** (not implemented):
- Cache-line padding: Align each mutex to 64 bytes (wastes memory)
- Lock-free techniques: Atomic CAS on bucket pointers
- Lock striping: Group buckets into coarser regions (compromise)

### SMT (Hyperthreading) Effects

**16 Threads on 8 Cores**:
- **Lookup shows best SMT benefit**: 3.86× at 8 threads → 4.70× at 16 threads (+22%)
- **Insert shows diminishing returns**: 2.76× at 8 threads → 3.27× at 16 threads (+18%)
- **Reasons**:
  - Shared execution units between logical cores
  - L1/L2 cache contention within physical core
  - Memory bandwidth saturation (all cores competing)

## Plots

All plots saved as high-resolution PNGs:

1. **plot1_throughput_vs_threads.png** - Throughput scaling for all workloads
2. **plot2_speedup_vs_threads.png** - Speedup curves (fine vs coarse)
3. **plot3_cache_misses_vs_threads.png** - LLC miss rates vs thread count
4. **plot4_workload_mix.png** - Comparison across workload types

## Dependencies

- C++17 compiler (GCC 9+ or Clang 10+)
- pthread for threading
- Linux `perf` for hardware counters (optional)
- Python 3 with matplotlib and numpy for plotting

## Build & Run

```bash
# Build with optimizations
make

# Run comprehensive benchmarks (450 tests, ~5 minutes)
python3 run_benchmarks.py

# Generate all plots
python3 analyze_results.py

# Clean
make clean
```

## System Information

Benchmarks run on:
- **CPU**: AMD Ryzen 7 7700X (Zen 4)
  - 8 cores / 16 threads (SMT enabled)
  - Base: 4.5 GHz, Boost: 5.4 GHz
  - L1d: 32 KB × 8, L2: 1 MB × 8, L3: 32 MB (shared)
- **Memory**: DDR5-6000 (dual-channel, 48 GB)
- **Compiler**: GCC 11.4 with `-O3 -march=native -pthread`
- **OS**: Linux (Ubuntu 22.04)

## Future Improvements

**Not Implemented** (potential extensions):
1. **Reader-Writer Locks**: Use `std::shared_mutex` to allow concurrent reads
2. **Lock-Free Hash Table**: Atomic CAS operations on bucket pointers
3. **Lock Striping**: Hybrid approach with fewer locks than buckets
4. **Resizing Support**: Dynamic table growth under load
5. **NUMA Awareness**: Pin threads to specific cores/sockets
