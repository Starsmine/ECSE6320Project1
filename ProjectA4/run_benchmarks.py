#!/usr/bin/env python3
"""
Project A4: Benchmark Suite for Concurrent Hash Tables

Runs comprehensive experiments across:
- Thread counts: 1, 2, 4, 8, 16
- Workloads: lookup, insert, mixed
- Table types: coarse, fine
- Dataset sizes: 10^4, 10^5, 10^6

Collects performance counters using perf stat on Linux.
"""

import subprocess
import sys
import os
import re
from pathlib import Path

# Benchmark configuration
THREAD_COUNTS = [1, 2, 4, 8, 16]
WORKLOADS = ['lookup', 'insert', 'mixed']
TABLE_TYPES = ['coarse', 'fine']
DATASET_SIZES = [10000, 100000, 1000000]  # 10^4, 10^5, 10^6
NUM_RUNS = 5  # Multiple runs for error bars

# Fixed parameters
OPERATIONS_PER_TEST = 1000000
NUM_BUCKETS = 10000

def run_benchmark(table_type, workload, dataset_size, num_threads, run_id, use_perf=True):
    """Run a single benchmark and collect perf stats."""
    
    if use_perf:
        cmd = [
            'perf', 'stat',
            '-e', 'cycles,cache-misses,cache-references,L1-dcache-load-misses',
            './hash_bench',
            '--table', table_type,
            '--workload', workload,
            '--size', str(dataset_size),
            '--ops', str(OPERATIONS_PER_TEST),
            '--threads', str(num_threads),
            '--buckets', str(NUM_BUCKETS)
        ]
    else:
        cmd = [
            './hash_bench',
            '--table', table_type,
            '--workload', workload,
            '--size', str(dataset_size),
            '--ops', str(OPERATIONS_PER_TEST),
            '--threads', str(num_threads),
            '--buckets', str(NUM_BUCKETS)
        ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Timeout for {table_type}/{workload}/size={dataset_size}/threads={num_threads}")
        return None, None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None

def parse_benchmark_output(stdout, stderr):
    """Extract throughput and perf counters from output."""
    
    data = {}
    
    # Parse throughput from stdout
    throughput_match = re.search(r'Throughput:\s+([\d.]+)\s+Mops/s', stdout)
    if throughput_match:
        data['throughput'] = float(throughput_match.group(1))
    
    duration_match = re.search(r'Duration:\s+([\d.]+)\s+seconds', stdout)
    if duration_match:
        data['duration'] = float(duration_match.group(1))
    
    # Parse perf counters from stderr
    cycles_match = re.search(r'([\d,]+)\s+cycles', stderr)
    if cycles_match:
        data['cycles'] = int(cycles_match.group(1).replace(',', ''))
    
    cache_misses_match = re.search(r'([\d,]+)\s+cache-misses', stderr)
    if cache_misses_match:
        data['cache_misses'] = int(cache_misses_match.group(1).replace(',', ''))
    
    cache_refs_match = re.search(r'([\d,]+)\s+cache-references', stderr)
    if cache_refs_match:
        data['cache_references'] = int(cache_refs_match.group(1).replace(',', ''))
    
    l1_misses_match = re.search(r'([\d,]+)\s+L1-dcache-load-misses', stderr)
    if l1_misses_match:
        data['l1_dcache_load_misses'] = int(l1_misses_match.group(1).replace(',', ''))
    
    return data

def main():
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Project A4: Concurrent Hash Table Benchmark Suite")
    print("=" * 60)
    print(f"Thread counts: {THREAD_COUNTS}")
    print(f"Workloads: {WORKLOADS}")
    print(f"Table types: {TABLE_TYPES}")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print("=" * 60)
    print()
    
    # Check if executable exists
    if not os.path.exists('./hash_bench'):
        print("ERROR: hash_bench executable not found. Run 'make' first.")
        sys.exit(1)
    
    # Check if perf is available
    use_perf = True
    try:
        result = subprocess.run(['perf', '--version'], capture_output=True, check=True)
    except:
        use_perf = False
        print("WARNING: perf not available. Will run without performance counters.")
        print("On Linux, install with: sudo apt-get install linux-tools-common")
        print()
    
    total_experiments = len(TABLE_TYPES) * len(WORKLOADS) * len(DATASET_SIZES) * len(THREAD_COUNTS) * NUM_RUNS
    experiment_count = 0
    
    # Run experiments
    for table_type in TABLE_TYPES:
        for workload in WORKLOADS:
            for dataset_size in DATASET_SIZES:
                for num_threads in THREAD_COUNTS:
                    
                    print(f"Running: {table_type}/{workload}/size={dataset_size}/threads={num_threads}")
                    
                    # Multiple runs for error bars
                    for run_id in range(NUM_RUNS):
                        experiment_count += 1
                        print(f"  Run {run_id + 1}/{NUM_RUNS} ({experiment_count}/{total_experiments})...", end=' ')
                        
                        stdout, stderr = run_benchmark(
                            table_type, workload, dataset_size, num_threads, run_id, use_perf
                        )
                        
                        if stdout is None:
                            print("FAILED")
                            continue
                        
                        # Parse results
                        data = parse_benchmark_output(stdout, stderr)
                        
                        if 'throughput' in data:
                            print(f"✓ {data['throughput']:.2f} Mops/s")
                        else:
                            print("✓ (no throughput)")
                        
                        # Save results
                        output_filename = (
                            f"threads_t{num_threads:02d}_"
                            f"{table_type}_{workload}_"
                            f"s{dataset_size}_"
                            f"run{run_id}.txt"
                        )
                        output_path = results_dir / output_filename
                        
                        with open(output_path, 'w') as f:
                            f.write("=" * 60 + "\n")
                            f.write(f"Table: {table_type}\n")
                            f.write(f"Workload: {workload}\n")
                            f.write(f"Dataset size: {dataset_size}\n")
                            f.write(f"Threads: {num_threads}\n")
                            f.write(f"Run: {run_id}\n")
                            f.write("=" * 60 + "\n\n")
                            f.write("STDOUT:\n")
                            f.write(stdout)
                            f.write("\n\nSTDERR (perf stats):\n")
                            f.write(stderr)
                    
                    print()
    
    print("=" * 60)
    print("Benchmark suite complete!")
    print(f"Results saved to: {results_dir.absolute()}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  python3 analyze_results.py")

if __name__ == '__main__':
    main()
