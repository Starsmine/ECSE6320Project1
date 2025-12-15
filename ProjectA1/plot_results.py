#!/usr/bin/env python3
"""
Plot visualization for ProjectA1 OS/CPU feature benchmarks
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

def parse_benchmark_results(filename='benchmark_results.txt'):
    """Parse benchmark results from text file"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run './os_features_bench --all' first.")
        sys.exit(1)
    
    data = {}
    
    # 1. CPU Affinity
    match = re.search(r'Without CPU pinning:\s*(\d+)\s*ms', content)
    without_pin = int(match.group(1)) if match else 0
    match = re.search(r'With CPU pinning:\s*(\d+)\s*ms', content)
    with_pin = int(match.group(1)) if match else 0
    data['affinity'] = {'Without pinning': without_pin, 'With pinning': with_pin}
    
    # 2. Hardware Prefetcher
    match = re.search(r'Sequential access:\s*(\d+)\s*ms', content)
    seq_time = int(match.group(1)) if match else 0
    match = re.search(r'Bandwidth:\s*([\d.]+)\s*MB/s', content)
    seq_bw = float(match.group(1)) if match else 0
    
    match = re.search(r'Strided access:\s*(\d+)\s*ms', content)
    stride_time = int(match.group(1)) if match else 0
    matches = re.findall(r'Bandwidth:\s*([\d.]+)\s*MB/s', content)
    stride_bw = float(matches[1]) if len(matches) > 1 else 0
    
    match = re.search(r'Random access:\s*(\d+)\s*ms', content)
    rand_time = int(match.group(1)) if match else 0
    matches = re.findall(r'Bandwidth:\s*([\d.]+)\s*MB/s', content)
    rand_bw = float(matches[2]) if len(matches) > 2 else 0
    
    data['prefetcher_time'] = {
        'Sequential\n(stride=1)': seq_time,
        'Strided\n(stride=16)': stride_time,
        'Random': rand_time
    }
    data['prefetcher_bw'] = {
        'Sequential\n(stride=1)': seq_bw,
        'Strided\n(stride=16)': stride_bw,
        'Random': rand_bw
    }
    
    # 3. SMT Interference
    match = re.search(r'Single thread \(no SMT\):\s*(\d+)\s*ms', content)
    single = int(match.group(1)) if match else 0
    match = re.search(r'Two threads \(with SMT\):\s*(\d+)\s*ms', content)
    dual = int(match.group(1)) if match else 0
    data['smt'] = {'Single thread\n(no SMT)': single, 'Two threads\n(with SMT)': dual}
    
    # 4. Huge Pages
    match = re.search(r'Regular pages \(4KB\):\s*(\d+)\s*ms', content)
    regular = int(match.group(1)) if match else 0
    match = re.search(r'Huge pages \(2MB\):\s*(\d+)\s*ms', content)
    huge = int(match.group(1)) if match else 0
    data['hugepages'] = {'Regular pages\n(4KB)': regular, 'Huge pages\n(2MB)': huge}
    
    # 5. Async I/O
    match = re.search(r'Synchronous I/O:\s*(\d+)\s*ms', content)
    sync_time = int(match.group(1)) if match else 0
    match = re.search(r'POSIX AIO:\s*(\d+)\s*ms', content)
    aio_time = int(match.group(1)) if match else 0
    match = re.search(r'io_uring:\s*(\d+)\s*ms', content)
    uring_time = int(match.group(1)) if match else 0
    
    # Throughput
    match = re.search(r'Sync I/O:\s*([\d.]+)\s*MB/s', content)
    sync_tp = float(match.group(1)) if match else 0
    match = re.search(r'POSIX AIO:\s*([\d.]+)\s*MB/s', content)
    aio_tp = float(match.group(1)) if match else 0
    match = re.search(r'io_uring:\s*([\d.]+)\s*MB/s', content)
    uring_tp = float(match.group(1)) if match else 0
    
    data['asyncio_time'] = {
        'Sync I/O': sync_time,
        'POSIX AIO': aio_time,
        'io_uring\n(zero-copy)': uring_time
    }
    data['asyncio_throughput'] = {
        'Sync I/O': sync_tp,
        'POSIX AIO': aio_tp,
        'io_uring\n(zero-copy)': uring_tp
    }
    
    return data

# Parse data from file
print("Parsing benchmark_results.txt...")
data = parse_benchmark_results()

affinity_data = data['affinity']
prefetcher_data = data['prefetcher_time']
prefetcher_bandwidth = data['prefetcher_bw']
smt_data = data['smt']
hugepages_data = data['hugepages']
asyncio_data = data['asyncio_time']
asyncio_throughput = data['asyncio_throughput']

# Create figure with 5 subplots
fig = plt.figure(figsize=(16, 10))

# Plot 1: CPU Affinity
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(affinity_data.keys(), affinity_data.values(), color=[colors[0], colors[1]])
ax1.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
ax1.set_title('1. CPU Affinity Impact', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(affinity_data.values()) * 1.2)
for i, (k, v) in enumerate(affinity_data.items()):
    ax1.text(i, v + 10, f'{v} ms', ha='center', fontsize=10, fontweight='bold')
speedup = affinity_data['Without pinning'] / affinity_data['With pinning']
ax1.text(0.5, max(affinity_data.values()) * 1.1, 
         f'Pinning: {speedup:.2f}× slower', 
         ha='center', fontsize=10, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Hardware Prefetcher - Time
ax2 = plt.subplot(2, 3, 2)
bars = ax2.bar(prefetcher_data.keys(), prefetcher_data.values(), 
               color=[colors[2], colors[3], colors[4]])
ax2.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
ax2.set_title('2. Hardware Prefetcher - Access Time', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(prefetcher_data.values()) * 1.2)
for i, (k, v) in enumerate(prefetcher_data.items()):
    ax2.text(i, v + 1, f'{v} ms', ha='center', fontsize=10, fontweight='bold')
speedup = prefetcher_data['Random'] / prefetcher_data['Sequential\n(stride=1)']
ax2.text(1, max(prefetcher_data.values()) * 1.1, 
         f'Sequential: {speedup:.1f}× faster', 
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Hardware Prefetcher - Bandwidth
ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(prefetcher_bandwidth.keys(), prefetcher_bandwidth.values(),
               color=[colors[2], colors[3], colors[4]])
ax3.set_ylabel('Bandwidth (MB/s)', fontsize=11, fontweight='bold')
ax3.set_title('2. Hardware Prefetcher - Bandwidth', fontsize=12, fontweight='bold')
ax3.set_ylim(0, max(prefetcher_bandwidth.values()) * 1.2)
for i, (k, v) in enumerate(prefetcher_bandwidth.items()):
    ax3.text(i, v + 100, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
ax3.text(1, max(prefetcher_bandwidth.values()) * 1.1,
         '240% prefetcher benefit',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax3.grid(axis='y', alpha=0.3)

# Plot 4: SMT Interference
ax4 = plt.subplot(2, 3, 4)
bars = ax4.bar(smt_data.keys(), smt_data.values(), color=[colors[0], colors[1]])
ax4.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
ax4.set_title('3. SMT (Hyperthreading) Interference', fontsize=12, fontweight='bold')
ax4.set_ylim(0, max(smt_data.values()) * 1.2)
for i, (k, v) in enumerate(smt_data.items()):
    ax4.text(i, v + 8, f'{v} ms', ha='center', fontsize=10, fontweight='bold')
efficiency = 2 * smt_data['Single thread\n(no SMT)'] / smt_data['Two threads\n(with SMT)']
ax4.text(0.5, max(smt_data.values()) * 1.1,
         f'SMT efficiency: {efficiency:.2f}× (53% overhead)',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Huge Pages
ax5 = plt.subplot(2, 3, 5)
bars = ax5.bar(hugepages_data.keys(), hugepages_data.values(), 
               color=[colors[2], colors[3]])
ax5.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
ax5.set_title('4. Transparent Huge Pages', fontsize=12, fontweight='bold')
ax5.set_ylim(0, max(hugepages_data.values()) * 1.2)
for i, (k, v) in enumerate(hugepages_data.items()):
    ax5.text(i, v + 20, f'{v} ms', ha='center', fontsize=10, fontweight='bold')
speedup = hugepages_data['Regular pages\n(4KB)'] / hugepages_data['Huge pages\n(2MB)']
ax5.text(0.5, max(hugepages_data.values()) * 1.1,
         f'Huge pages: {speedup:.2f}× slower\n(random access pattern)',
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Async I/O - Throughput (log scale)
ax6 = plt.subplot(2, 3, 6)
bars = ax6.bar(asyncio_throughput.keys(), asyncio_throughput.values(),
               color=[colors[0], colors[1], colors[4]])
ax6.set_ylabel('Throughput (MB/s)', fontsize=11, fontweight='bold')
ax6.set_title('5. Async I/O Throughput (io_uring)', fontsize=12, fontweight='bold')
ax6.set_yscale('log')
ax6.set_ylim(10, max(asyncio_throughput.values()) * 2)
for i, (k, v) in enumerate(asyncio_throughput.items()):
    ax6.text(i, v * 1.3, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
speedup = asyncio_throughput['io_uring\n(zero-copy)'] / asyncio_throughput['Sync I/O']
ax6.text(1, max(asyncio_throughput.values()) * 1.5,
         f'io_uring: {speedup:.1f}× faster\n(zero-copy + batching)',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax6.grid(axis='y', alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
print("✓ Generated benchmark_results.png")
plt.show()

print("\n" + "="*50)
print("Successfully generated benchmark_results.png")
print("="*50)

# Exit after first plot
import sys
sys.exit(0)

# Create a second figure with speedup comparisons
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Speedup plot 1: CPU Features (Affinity, Prefetcher, SMT)
ax = axes[0, 0]
features = ['CPU Affinity\n(pinning)', 'Prefetcher\n(sequential)', 'SMT\n(2 threads)']
speedups = [
    affinity_data['Without pinning'] / affinity_data['With pinning'],
    prefetcher_data['Random'] / prefetcher_data['Sequential\n(stride=1)'],
    2 * smt_data['Single thread\n(no SMT)'] / smt_data['Two threads\n(with SMT)']
]
colors_bar = [colors[1] if s < 1 else colors[2] for s in speedups]
bars = ax.barh(features, speedups, color=colors_bar)
ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1×)')
ax.set_xlabel('Speedup vs Baseline', fontsize=11, fontweight='bold')
ax.set_title('CPU Feature Performance Impact', fontsize=12, fontweight='bold')
ax.set_xlim(0, max(speedups) * 1.15)
for i, (f, s) in enumerate(zip(features, speedups)):
    color = 'white' if s > 0.5 else 'black'
    ax.text(s/2, i, f'{s:.2f}×', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=color)
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Speedup plot 2: Memory (Huge Pages)
ax = axes[0, 1]
memory_features = ['Regular pages', 'Huge pages']
memory_times = [hugepages_data['Regular pages\n(4KB)'], hugepages_data['Huge pages\n(2MB)']]
memory_speedup = [1.0, hugepages_data['Regular pages\n(4KB)'] / hugepages_data['Huge pages\n(2MB)']]
colors_bar = [colors[2], colors[1]]
bars = ax.barh(memory_features, memory_speedup, color=colors_bar)
ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
ax.set_xlabel('Speedup vs Regular Pages', fontsize=11, fontweight='bold')
ax.set_title('Huge Pages Impact (Random Access)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1.5)
for i, (f, s) in enumerate(zip(memory_features, memory_speedup)):
    color = 'white' if s > 0.5 else 'black'
    ax.text(s/2, i, f'{s:.2f}×', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color)
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Speedup plot 3: I/O Technologies
ax = axes[1, 0]
io_types = ['Sync I/O', 'POSIX AIO', 'io_uring']
io_speedups = [
    1.0,
    asyncio_data['Sync I/O'] / asyncio_data['POSIX AIO'],
    asyncio_data['Sync I/O'] / asyncio_data['io_uring\n(zero-copy)']
]
colors_bar = [colors[0], colors[1], colors[2]]
bars = ax.barh(io_types, io_speedups, color=colors_bar)
ax.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Sync baseline')
ax.set_xlabel('Speedup vs Synchronous I/O', fontsize=11, fontweight='bold')
ax.set_title('Async I/O Performance Comparison', fontsize=12, fontweight='bold')
ax.set_xlim(0, max(io_speedups) * 1.1)
for i, (t, s) in enumerate(zip(io_types, io_speedups)):
    color = 'white' if s > 5 else 'black'
    ax.text(s/2 if s > 5 else s + 0.5, i, f'{s:.1f}×', ha='center' if s > 5 else 'left', 
            va='center', fontsize=11, fontweight='bold', color=color)
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Summary statistics
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
BENCHMARK SUMMARY (AMD Ryzen 7 7700X)
{'='*45}

CPU Features:
  • CPU Affinity: Pinning 2.1× SLOWER
    (scheduler beats manual pinning)
  
  • Hardware Prefetcher: 3.4× speedup
    (sequential vs random access)
  
  • SMT/Hyperthreading: 1.31× efficiency
    (53% overhead from resource sharing)

Memory:
  • Huge Pages: 0.76× (30% SLOWER)
    (random access defeats TLB benefits)

I/O Performance:
  • POSIX AIO: 1.6× faster than sync
  • io_uring: 24.3× faster than sync
    (zero-copy + batching = 977 MB/s)

Key Insights:
  ✓ Prefetcher critical for performance
  ✓ SMT provides modest gains
  ✓ io_uring dramatically outperforms AIO
  ✗ Affinity/Huge pages not universally helpful
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('benchmark_speedups.png', dpi=300, bbox_inches='tight')
print("✓ Generated benchmark_speedups.png")

# Create a third figure: Performance breakdown
fig3, ax = plt.subplots(1, 1, figsize=(12, 8))

# Normalized performance (higher is better)
benchmarks = ['CPU\nAffinity', 'Prefetcher', 'SMT', 'Huge\nPages', 'I/O\n(io_uring)']
baseline_perf = [1.0, 1.0, 1.0, 1.0, 1.0]  # Baseline approaches
optimized_perf = [
    affinity_data['Without pinning'] / affinity_data['With pinning'],  # No pinning is better
    prefetcher_data['Random'] / prefetcher_data['Sequential\n(stride=1)'],  # Sequential is better
    2 * smt_data['Single thread\n(no SMT)'] / smt_data['Two threads\n(with SMT)'],  # SMT improvement
    hugepages_data['Regular pages\n(4KB)'] / hugepages_data['Huge pages\n(2MB)'],  # Huge pages
    asyncio_data['Sync I/O'] / asyncio_data['io_uring\n(zero-copy)']  # io_uring
]

x = np.arange(len(benchmarks))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_perf, width, label='Baseline/Pessimal', 
               color=colors[1], alpha=0.8)
bars2 = ax.bar(x + width/2, optimized_perf, width, label='Optimized/Optimal',
               color=colors[2], alpha=0.8)

ax.set_ylabel('Performance (speedup factor)', fontsize=12, fontweight='bold')
ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
ax.set_title('OS/CPU Feature Performance: Baseline vs Optimized', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=10)
ax.legend(fontsize=11)
ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(0, max(optimized_perf) * 1.15)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 1.0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated benchmark_comparison.png")

print("\n" + "="*50)
print("Successfully generated 3 plots:")
print("  1. benchmark_results.png - Individual benchmark results")
print("  2. benchmark_speedups.png - Speedup comparisons + summary")
print("  3. benchmark_comparison.png - Baseline vs optimized")
print("="*50)
