#!/usr/bin/env python3
"""
Calculate statistics from multiple benchmark runs
"""
import re
import sys
import numpy as np

def parse_runs(filename="benchmark_results.txt"):
    """Parse multiple runs from file"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("Error: benchmark_results.txt not found")
        print("Run: for i in {1..5}; do ./os_features_bench --all; done > benchmark_results.txt")
        sys.exit(1)
    
    # Parse all runs
    data = {
        'affinity_nopin': [], 'affinity_pin': [],
        'seq': [], 'stride': [], 'random': [],
        'smt_single': [], 'smt_dual': [],
        'regular': [], 'huge': [],
        'sync': [], 'aio': [], 'uring': []
    }
    
    # Find all occurrences
    data['affinity_nopin'] = [int(x) for x in re.findall(r'Without CPU pinning:\s*(\d+)\s*ms', content)]
    data['affinity_pin'] = [int(x) for x in re.findall(r'With CPU pinning:\s*(\d+)\s*ms', content)]
    data['seq'] = [int(x) for x in re.findall(r'Sequential access:\s*(\d+)\s*ms', content)]
    data['stride'] = [int(x) for x in re.findall(r'Strided access:\s*(\d+)\s*ms', content)]
    data['random'] = [int(x) for x in re.findall(r'Random access:\s*(\d+)\s*ms', content)]
    data['smt_single'] = [int(x) for x in re.findall(r'Single thread \(no SMT\):\s*(\d+)\s*ms', content)]
    data['smt_dual'] = [int(x) for x in re.findall(r'Two threads \(with SMT\):\s*(\d+)\s*ms', content)]
    data['regular'] = [int(x) for x in re.findall(r'Regular pages \(4KB\):\s*(\d+)\s*ms', content)]
    data['huge'] = [int(x) for x in re.findall(r'Huge pages \(2MB\):\s*(\d+)\s*ms', content)]
    data['sync'] = [int(x) for x in re.findall(r'Synchronous I/O:\s*(\d+)\s*ms', content)]
    data['aio'] = [int(x) for x in re.findall(r'POSIX AIO:\s*(\d+)\s*ms', content)]
    data['uring'] = [int(x) for x in re.findall(r'io_uring:\s*(\d+)\s*ms', content)]
    
    return data

def print_stats(name, values):
    """Print mean ± stddev"""
    if not values:
        print(f"{name:30s}: No data")
        return
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    print(f"{name:30s}: {mean:6.1f} ± {std:4.1f} ms  (n={len(values)}, range={min(values)}-{max(values)})")

data = parse_runs()

print("\n" + "="*80)
print("BENCHMARK STATISTICS (5 runs)")
print("="*80)

print("\n1. CPU AFFINITY:")
print_stats("  Without pinning", data['affinity_nopin'])
print_stats("  With pinning", data['affinity_pin'])
if data['affinity_nopin'] and data['affinity_pin']:
    speedup = np.mean(data['affinity_nopin']) / np.mean(data['affinity_pin'])
    variance = np.std(data['affinity_pin'], ddof=1)
    print(f"  Speedup: {speedup:.2f}x (pinning {1/speedup:.2f}x slower)")
    print(f"  ⚠️  High variance in pinned: ±{variance:.1f}ms suggests CPU frequency scaling")

print("\n2. HARDWARE PREFETCHER:")
print_stats("  Sequential", data['seq'])
print_stats("  Strided", data['stride'])
print_stats("  Random", data['random'])
if data['seq'] and data['random']:
    speedup = np.mean(data['random']) / np.mean(data['seq'])
    print(f"  Prefetcher benefit: {speedup:.2f}x faster")

print("\n3. SMT INTERFERENCE:")
print_stats("  Single thread", data['smt_single'])
print_stats("  Dual SMT", data['smt_dual'])
if data['smt_single'] and data['smt_dual']:
    efficiency = 2 * np.mean(data['smt_single']) / np.mean(data['smt_dual'])
    overhead = 100 * (1 - efficiency/2)
    print(f"  Efficiency: {efficiency:.2f}x ({overhead:.1f}% overhead)")

print("\n4. HUGE PAGES:")
print_stats("  Regular pages", data['regular'])
print_stats("  Huge pages", data['huge'])
if data['regular'] and data['huge']:
    speedup = np.mean(data['regular']) / np.mean(data['huge'])
    print(f"  Speedup: {speedup:.2f}x (huge pages {1/speedup:.2f}x slower)")

print("\n5. ASYNC I/O:")
print_stats("  Sync I/O", data['sync'])
print_stats("  POSIX AIO", data['aio'])
print_stats("  io_uring", data['uring'])
if data['sync'] and data['aio'] and data['uring']:
    aio_speedup = np.mean(data['sync']) / np.mean(data['aio'])
    uring_speedup = np.mean(data['sync']) / np.mean(data['uring'])
    print(f"  POSIX AIO: {aio_speedup:.2f}x faster than sync")
    print(f"  io_uring:  {uring_speedup:.1f}x faster than sync (!)")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
print("• CPU Affinity: High variance suggests frequency scaling issues")
print("  → Fix CPU frequency: sudo cpupower frequency-set -g performance")
print("• Huge Pages: Verify usage during test:")
print("  → grep -i huge /proc/meminfo")
print("• Add perf stat: -e cycles,cache-misses,dTLB-load-misses")
print("="*80)
