#!/usr/bin/env python3
"""
Project A4: Analysis and Visualization

Generates 4 plots:
1. Throughput vs Threads (line chart, both table types)
2. Speedup vs Threads (line chart with ideal line)
3. Cache Misses vs Threads (line chart, LLC load/store misses)
4. Workload Mix Throughput (bar chart comparing workloads)

All plots include error bars from multiple runs.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def parse_result_file(filepath):
    """Parse a single result file."""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {}
    
    # Parse metadata
    table_match = re.search(r'Table:\s+(\w+)', content)
    workload_match = re.search(r'Workload:\s+(\w+)', content)
    size_match = re.search(r'Dataset size:\s+(\d+)', content)
    threads_match = re.search(r'Threads:\s+(\d+)', content)
    run_match = re.search(r'Run:\s+(\d+)', content)
    
    if table_match:
        data['table'] = table_match.group(1)
    if workload_match:
        data['workload'] = workload_match.group(1)
    if size_match:
        data['size'] = int(size_match.group(1))
    if threads_match:
        data['threads'] = int(threads_match.group(1))
    if run_match:
        data['run'] = int(run_match.group(1))
    
    # Parse throughput
    throughput_match = re.search(r'Throughput:\s+([\d.]+)\s+Mops/s', content)
    if throughput_match:
        data['throughput'] = float(throughput_match.group(1))
    
    # Parse duration
    duration_match = re.search(r'Duration:\s+([\d.]+)\s+seconds', content)
    if duration_match:
        data['duration'] = float(duration_match.group(1))
    
    # Parse perf counters
    cycles_match = re.search(r'([\d,]+)\s+cycles', content)
    if cycles_match:
        data['cycles'] = int(cycles_match.group(1).replace(',', ''))
    
    cache_misses_match = re.search(r'([\d,]+)\s+cache-misses', content)
    if cache_misses_match:
        data['cache_misses'] = int(cache_misses_match.group(1).replace(',', ''))
    
    cache_refs_match = re.search(r'([\d,]+)\s+cache-references', content)
    if cache_refs_match:
        data['cache_references'] = int(cache_refs_match.group(1).replace(',', ''))
    
    l1_misses_match = re.search(r'([\d,]+)\s+L1-dcache-load-misses', content)
    if l1_misses_match:
        data['l1_dcache_load_misses'] = int(l1_misses_match.group(1).replace(',', ''))
    
    return data

def load_all_results(results_dir='results'):
    """Load all result files."""
    
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"ERROR: Results directory '{results_dir}' not found")
        return results
    
    for filepath in sorted(results_path.glob('threads_*.txt')):
        data = parse_result_file(filepath)
        if data:
            results.append(data)
    
    print(f"Loaded {len(results)} result files")
    return results

def aggregate_results(results):
    """Aggregate multiple runs into mean and std."""
    
    # Group by (table, workload, size, threads)
    grouped = defaultdict(list)
    
    for result in results:
        key = (
            result.get('table'),
            result.get('workload'),
            result.get('size'),
            result.get('threads')
        )
        grouped[key].append(result)
    
    # Compute statistics
    aggregated = []
    
    for key, runs in grouped.items():
        table, workload, size, threads = key
        
        agg = {
            'table': table,
            'workload': workload,
            'size': size,
            'threads': threads,
            'num_runs': len(runs)
        }
        
        # Throughput
        throughputs = [r['throughput'] for r in runs if 'throughput' in r]
        if throughputs:
            agg['throughput_mean'] = np.mean(throughputs)
            agg['throughput_std'] = np.std(throughputs, ddof=1) if len(throughputs) > 1 else 0
        
        # Cycles
        cycles = [r['cycles'] for r in runs if 'cycles' in r]
        if cycles:
            agg['cycles_mean'] = np.mean(cycles)
            agg['cycles_std'] = np.std(cycles, ddof=1) if len(cycles) > 1 else 0
        
        # Cache misses
        cache_misses = [r['cache_misses'] for r in runs if 'cache_misses' in r]
        if cache_misses:
            agg['cache_misses_mean'] = np.mean(cache_misses)
            agg['cache_misses_std'] = np.std(cache_misses, ddof=1) if len(cache_misses) > 1 else 0
        
        # Cache references
        cache_refs = [r['cache_references'] for r in runs if 'cache_references' in r]
        if cache_refs:
            agg['cache_references_mean'] = np.mean(cache_refs)
            agg['cache_references_std'] = np.std(cache_refs, ddof=1) if len(cache_refs) > 1 else 0
        
        # L1 dcache load misses
        l1_misses = [r['l1_dcache_load_misses'] for r in runs if 'l1_dcache_load_misses' in r]
        if l1_misses:
            agg['l1_dcache_load_misses_mean'] = np.mean(l1_misses)
            agg['l1_dcache_load_misses_std'] = np.std(l1_misses, ddof=1) if len(l1_misses) > 1 else 0
        
        aggregated.append(agg)
    
    return aggregated

def plot_throughput_vs_threads(aggregated, dataset_size=100000):
    """Plot 1: Throughput vs Threads for both table types."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Throughput vs Thread Count (Dataset Size: {dataset_size})', 
                 fontsize=14, fontweight='bold')
    
    workloads = ['lookup', 'insert', 'mixed']
    colors = {'coarse': 'red', 'fine': 'blue'}
    markers = {'coarse': 'o', 'fine': 's'}
    
    for idx, workload in enumerate(workloads):
        ax = axes[idx]
        
        for table_type in ['coarse', 'fine']:
            # Filter data
            data = [r for r in aggregated 
                   if r['table'] == table_type 
                   and r['workload'] == workload
                   and r['size'] == dataset_size
                   and 'throughput_mean' in r]
            
            if not data:
                continue
            
            data = sorted(data, key=lambda x: x['threads'])
            
            threads = [r['threads'] for r in data]
            throughput_mean = [r['throughput_mean'] for r in data]
            throughput_std = [r['throughput_std'] for r in data]
            
            ax.errorbar(threads, throughput_mean, yerr=throughput_std,
                       label=table_type.capitalize(),
                       marker=markers[table_type], markersize=8,
                       color=colors[table_type], linewidth=2,
                       capsize=5, capthick=2)
        
        ax.set_xlabel('Number of Threads', fontsize=11)
        ax.set_ylabel('Throughput (Mops/s)', fontsize=11)
        ax.set_title(f'{workload.capitalize()} Workload', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks([1, 2, 4, 8, 16])
    
    plt.tight_layout()
    plt.savefig('plot1_throughput_vs_threads.png', dpi=300, bbox_inches='tight')
    print("Saved: plot1_throughput_vs_threads.png")

def plot_speedup_vs_threads(aggregated, dataset_size=100000, workload='mixed'):
    """Plot 2: Speedup vs Threads with ideal line."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {'coarse': 'red', 'fine': 'blue'}
    markers = {'coarse': 'o', 'fine': 's'}
    
    for table_type in ['coarse', 'fine']:
        # Filter data
        data = [r for r in aggregated 
               if r['table'] == table_type 
               and r['workload'] == workload
               and r['size'] == dataset_size
               and 'throughput_mean' in r]
        
        if not data:
            continue
        
        data = sorted(data, key=lambda x: x['threads'])
        
        # Get baseline (1 thread)
        baseline = next((r for r in data if r['threads'] == 1), None)
        if not baseline:
            continue
        
        baseline_throughput = baseline['throughput_mean']
        
        threads = [r['threads'] for r in data]
        throughput_mean = [r['throughput_mean'] for r in data]
        throughput_std = [r['throughput_std'] for r in data]
        
        # Compute speedup
        speedup = [t / baseline_throughput for t in throughput_mean]
        
        # Error propagation: speedup_std = (throughput_std / baseline_throughput)
        speedup_std = [t_std / baseline_throughput for t_std in throughput_std]
        
        ax.errorbar(threads, speedup, yerr=speedup_std,
                   label=f'{table_type.capitalize()} (actual)',
                   marker=markers[table_type], markersize=8,
                   color=colors[table_type], linewidth=2,
                   capsize=5, capthick=2)
    
    # Ideal speedup line
    max_threads = max([r['threads'] for r in aggregated])
    ideal_threads = range(1, max_threads + 1)
    ideal_speedup = list(ideal_threads)
    ax.plot(ideal_threads, ideal_speedup, 'k--', linewidth=2, label='Ideal (linear)')
    
    ax.set_xlabel('Number of Threads', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(f'Speedup vs Thread Count ({workload.capitalize()} Workload, Size={dataset_size})', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks([1, 2, 4, 8, 16])
    
    plt.tight_layout()
    plt.savefig('plot2_speedup_vs_threads.png', dpi=300, bbox_inches='tight')
    print("Saved: plot2_speedup_vs_threads.png")

def plot_cache_misses_vs_threads(aggregated, dataset_size=100000, workload='mixed'):
    """Plot 3: Cache Misses vs Threads."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Cache Performance vs Thread Count ({workload.capitalize()}, Size={dataset_size})', 
                fontsize=14, fontweight='bold')
    
    colors = {'coarse': 'red', 'fine': 'blue'}
    markers = {'coarse': 'o', 'fine': 's'}
    
    # Plot cache misses (last-level cache)
    ax = axes[0]
    for table_type in ['coarse', 'fine']:
        data = [r for r in aggregated 
               if r['table'] == table_type 
               and r['workload'] == workload
               and r['size'] == dataset_size
               and 'cache_misses_mean' in r]
        
        if not data:
            continue
        
        data = sorted(data, key=lambda x: x['threads'])
        
        threads = [r['threads'] for r in data]
        cache_misses_mean = [r['cache_misses_mean'] / 1e6 for r in data]  # Convert to millions
        cache_misses_std = [r['cache_misses_std'] / 1e6 for r in data]
        
        ax.errorbar(threads, cache_misses_mean, yerr=cache_misses_std,
                   label=table_type.capitalize(),
                   marker=markers[table_type], markersize=8,
                   color=colors[table_type], linewidth=2,
                   capsize=5, capthick=2)
    
    ax.set_xlabel('Number of Threads', fontsize=11)
    ax.set_ylabel('Cache Misses (millions)', fontsize=11)
    ax.set_title('Last-Level Cache Misses', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks([1, 2, 4, 8, 16])
    
    # Plot L1 dcache load misses
    ax = axes[1]
    for table_type in ['coarse', 'fine']:
        data = [r for r in aggregated 
               if r['table'] == table_type 
               and r['workload'] == workload
               and r['size'] == dataset_size
               and 'l1_dcache_load_misses_mean' in r]
        
        if not data:
            continue
        
        data = sorted(data, key=lambda x: x['threads'])
        
        threads = [r['threads'] for r in data]
        l1_misses_mean = [r['l1_dcache_load_misses_mean'] / 1e6 for r in data]
        l1_misses_std = [r['l1_dcache_load_misses_std'] / 1e6 for r in data]
        
        ax.errorbar(threads, l1_misses_mean, yerr=l1_misses_std,
                   label=table_type.capitalize(),
                   marker=markers[table_type], markersize=8,
                   color=colors[table_type], linewidth=2,
                   capsize=5, capthick=2)
    
    ax.set_xlabel('Number of Threads', fontsize=11)
    ax.set_ylabel('L1 Load Misses (millions)', fontsize=11)
    ax.set_title('L1 Data Cache Load Misses', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks([1, 2, 4, 8, 16])
    
    plt.tight_layout()
    plt.savefig('plot3_cache_misses_vs_threads.png', dpi=300, bbox_inches='tight')
    print("Saved: plot3_cache_misses_vs_threads.png")

def plot_workload_mix_throughput(aggregated, dataset_size=100000, num_threads=8):
    """Plot 4: Bar chart comparing workload throughputs."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    workloads = ['lookup', 'insert', 'mixed']
    table_types = ['coarse', 'fine']
    
    x = np.arange(len(workloads))
    width = 0.35
    
    for idx, table_type in enumerate(table_types):
        throughputs = []
        errors = []
        
        for workload in workloads:
            data = [r for r in aggregated 
                   if r['table'] == table_type 
                   and r['workload'] == workload
                   and r['size'] == dataset_size
                   and r['threads'] == num_threads
                   and 'throughput_mean' in r]
            
            if data:
                throughputs.append(data[0]['throughput_mean'])
                errors.append(data[0]['throughput_std'])
            else:
                throughputs.append(0)
                errors.append(0)
        
        offset = width * (idx - 0.5)
        color = 'red' if table_type == 'coarse' else 'blue'
        
        ax.bar(x + offset, throughputs, width, yerr=errors,
              label=table_type.capitalize(), color=color, alpha=0.8,
              capsize=5)
    
    ax.set_xlabel('Workload Type', fontsize=12)
    ax.set_ylabel('Throughput (Mops/s)', fontsize=12)
    ax.set_title(f'Workload Comparison (Threads={num_threads}, Size={dataset_size})', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([w.capitalize() for w in workloads])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plot4_workload_mix.png', dpi=300, bbox_inches='tight')
    print("Saved: plot4_workload_mix.png")

def plot_dataset_size_scaling(aggregated, num_threads=8, workload='mixed'):
    """Plot 5: Throughput vs Dataset Size."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'coarse': 'red', 'fine': 'blue'}
    markers = {'coarse': 'o', 'fine': 's'}
    
    dataset_sizes = [10000, 100000, 1000000]
    
    for table_type in ['coarse', 'fine']:
        throughputs_mean = []
        throughputs_std = []
        
        for size in dataset_sizes:
            data = [r for r in aggregated 
                   if r['table'] == table_type 
                   and r['workload'] == workload
                   and r['size'] == size
                   and r['threads'] == num_threads
                   and 'throughput_mean' in r]
            
            if data:
                throughputs_mean.append(data[0]['throughput_mean'])
                throughputs_std.append(data[0]['throughput_std'])
            else:
                throughputs_mean.append(0)
                throughputs_std.append(0)
        
        ax.errorbar(dataset_sizes, throughputs_mean, yerr=throughputs_std,
                   label=table_type.capitalize(),
                   marker=markers[table_type], markersize=10,
                   color=colors[table_type], linewidth=2,
                   capsize=5, capthick=2)
    
    ax.set_xlabel('Dataset Size (number of keys)', fontsize=12)
    ax.set_ylabel('Throughput (Mops/s)', fontsize=12)
    ax.set_title(f'Performance vs Dataset Size ({workload.capitalize()}, Threads={num_threads})', 
                fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    # Format x-axis labels
    ax.set_xticks(dataset_sizes)
    ax.set_xticklabels(['10⁴\n(10K)', '10⁵\n(100K)', '10⁶\n(1M)'])
    
    plt.tight_layout()
    plt.savefig('plot5_dataset_size_scaling.png', dpi=300, bbox_inches='tight')
    print("Saved: plot5_dataset_size_scaling.png")

def main():
    print("=" * 60)
    print("Project A4: Analyzing Benchmark Results")
    print("=" * 60)
    print()
    
    # Load results
    results = load_all_results()
    if not results:
        print("No results found. Run benchmarks first:")
        print("  python3 run_benchmarks.py")
        return
    
    # Aggregate multiple runs
    print("Aggregating results...")
    aggregated = aggregate_results(results)
    print(f"Aggregated into {len(aggregated)} unique configurations")
    print()
    
    # Generate plots
    print("Generating plots...")
    print()
    
    # Use dataset_size=100000 (10^5) for main analysis
    dataset_size = 100000
    
    plot_throughput_vs_threads(aggregated, dataset_size=dataset_size)
    plot_speedup_vs_threads(aggregated, dataset_size=dataset_size, workload='mixed')
    plot_cache_misses_vs_threads(aggregated, dataset_size=dataset_size, workload='mixed')
    plot_workload_mix_throughput(aggregated, dataset_size=dataset_size, num_threads=8)
    plot_dataset_size_scaling(aggregated, num_threads=8, workload='mixed')
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("Generated plots:")
    print("  1. plot1_throughput_vs_threads.png")
    print("  2. plot2_speedup_vs_threads.png")
    print("  3. plot3_cache_misses_vs_threads.png")
    print("  4. plot4_workload_mix.png")
    print("  5. plot5_dataset_size_scaling.png")
    print("=" * 60)

if __name__ == '__main__':
    main()
