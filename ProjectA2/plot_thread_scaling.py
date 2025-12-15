#!/usr/bin/env python3
"""
Thread scaling analysis and visualization
Shows speedup curves and efficiency metrics across thread counts
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

class ThreadScalingAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
    
    def parse_thread_result(self, filepath):
        """Parse a single thread scaling result file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Extract thread count from filename (threads_t##_...)
        thread_match = re.search(r'threads_t(\d+)_', filepath.name)
        if not thread_match:
            return None
        
        threads = int(thread_match.group(1))
        
        # Extract experiment parameters
        exp_match = re.search(r'Experiment: m=(\d+), k=(\d+), n=(\d+), density=([\d.]+)%', content)
        if not exp_match:
            return None
        
        result = {
            'threads': threads,
            'm': int(exp_match.group(1)),
            'k': int(exp_match.group(2)),
            'n': int(exp_match.group(3)),
            'density': float(exp_match.group(4)) / 100.0,
        }
        
        # Dense GEMM - updated to handle ±stddev and additional perf fields
        dense_match = re.search(
            r'Dense GEMM \(tiled\):\s*\n\s*Time:\s*([\d.]+)\s*s(?:\s*\(±([\d.]+)\s*s\))?\s*\n\s*GFLOP/s:\s*([\d.]+)',
            content
        )
        if dense_match:
            result['dense_time'] = float(dense_match.group(1))
            result['dense_time_stddev'] = float(dense_match.group(2)) if dense_match.group(2) else 0
            result['dense_gflops'] = float(dense_match.group(3))
            # Estimate GFLOP/s stddev from time stddev
            if result['dense_time'] > 0 and result['dense_time_stddev'] > 0:
                result['dense_gflops_stddev'] = result['dense_gflops'] * (result['dense_time_stddev'] / result['dense_time'])
            else:
                result['dense_gflops_stddev'] = 0
        
        # CSR-SpMM (scalar) - updated to handle ±stddev and additional perf fields
        sparse_scalar_match = re.search(
            r'CSR-SpMM \(scalar\):\s*\n\s*Time:\s*([\d.]+)\s*s(?:\s*\(±([\d.]+)\s*s\))?\s*\n\s*GFLOP/s:\s*([\d.]+)\s*\n\s*CPNZ:\s*([\d.]+)',
            content
        )
        if sparse_scalar_match:
            result['sparse_scalar_time'] = float(sparse_scalar_match.group(1))
            result['sparse_scalar_time_stddev'] = float(sparse_scalar_match.group(2)) if sparse_scalar_match.group(2) else 0
            result['sparse_scalar_gflops'] = float(sparse_scalar_match.group(3))
            result['sparse_scalar_cpnz'] = float(sparse_scalar_match.group(4))
            if result['sparse_scalar_time'] > 0 and result['sparse_scalar_time_stddev'] > 0:
                result['sparse_scalar_gflops_stddev'] = result['sparse_scalar_gflops'] * (result['sparse_scalar_time_stddev'] / result['sparse_scalar_time'])
            else:
                result['sparse_scalar_gflops_stddev'] = 0
        
        # CSR-SpMM (SIMD) - updated to handle ±stddev and additional perf fields
        sparse_simd_match = re.search(
            r'CSR-SpMM \(SIMD\):\s*\n\s*Time:\s*([\d.]+)\s*s(?:\s*\(±([\d.]+)\s*s\))?\s*\n\s*GFLOP/s:\s*([\d.]+)\s*\n\s*CPNZ:\s*([\d.]+)',
            content
        )
        if sparse_simd_match:
            result['sparse_simd_time'] = float(sparse_simd_match.group(1))
            result['sparse_simd_time_stddev'] = float(sparse_simd_match.group(2)) if sparse_simd_match.group(2) else 0
            result['sparse_simd_gflops'] = float(sparse_simd_match.group(3))
            result['sparse_simd_cpnz'] = float(sparse_simd_match.group(4))
            if result['sparse_simd_time'] > 0 and result['sparse_simd_time_stddev'] > 0:
                result['sparse_simd_gflops_stddev'] = result['sparse_simd_gflops'] * (result['sparse_simd_time_stddev'] / result['sparse_simd_time'])
            else:
                result['sparse_simd_gflops_stddev'] = 0
        
        return result
    
    def load_thread_results(self, pattern="threads_t*"):
        """Load all thread scaling results"""
        files = sorted(self.results_dir.glob(pattern))
        
        if not files:
            print(f"No files matching pattern '{pattern}' found in {self.results_dir}")
            return []
        
        results = []
        for filepath in files:
            result = self.parse_thread_result(filepath)
            if result:
                results.append(result)
        
        # Sort by thread count
        results.sort(key=lambda x: x['threads'])
        return results
    
    def plot_thread_scaling(self, results, output_file="thread_scaling.png"):
        """Plot thread scaling analysis"""
        if not results:
            print("No results to plot!")
            return
        
        threads = [r['threads'] for r in results]
        
        # Get baseline (1 thread) values
        baseline = results[0]
        baseline_dense_time = baseline.get('dense_time', 1)
        baseline_scalar_time = baseline.get('sparse_scalar_time', 1)
        baseline_simd_time = baseline.get('sparse_simd_time', 1)
        
        # Calculate speedups (relative to 1 thread)
        dense_speedup = [baseline_dense_time / r.get('dense_time', 1) for r in results]
        scalar_speedup = [baseline_scalar_time / r.get('sparse_scalar_time', 1) for r in results]
        simd_speedup = [baseline_simd_time / r.get('sparse_simd_time', 1) for r in results]
        
        # Calculate efficiency (speedup / threads)
        dense_efficiency = [s / t * 100 for s, t in zip(dense_speedup, threads)]
        scalar_efficiency = [s / t * 100 for s, t in zip(scalar_speedup, threads)]
        simd_efficiency = [s / t * 100 for s, t in zip(simd_speedup, threads)]
        
        # Get GFLOP/s with error bars
        dense_gflops = [r.get('dense_gflops', 0) for r in results]
        dense_gflops_err = [r.get('dense_gflops_stddev', 0) for r in results]
        scalar_gflops = [r.get('sparse_scalar_gflops', 0) for r in results]
        scalar_gflops_err = [r.get('sparse_scalar_gflops_stddev', 0) for r in results]
        simd_gflops = [r.get('sparse_simd_gflops', 0) for r in results]
        simd_gflops_err = [r.get('sparse_simd_gflops_stddev', 0) for r in results]
        
        # Get CPNZ
        scalar_cpnz = [r.get('sparse_scalar_cpnz', 0) for r in results]
        simd_cpnz = [r.get('sparse_simd_cpnz', 0) for r in results]
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Speedup vs Thread Count
        ax1.plot(threads, dense_speedup, 'o-', linewidth=2, markersize=8, 
                label='Dense GEMM', color='#2E86AB')
        ax1.plot(threads, scalar_speedup, 's-', linewidth=2, markersize=8,
                label='Sparse (scalar)', color='#A23B72')
        ax1.plot(threads, simd_speedup, '^-', linewidth=2, markersize=8,
                label='Sparse (SIMD)', color='#F18F01')
        
        # Ideal linear speedup
        ax1.plot(threads, threads, 'k--', linewidth=1.5, alpha=0.5, label='Ideal (linear)')
        
        ax1.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speedup vs 1 Thread', fontsize=12, fontweight='bold')
        ax1.set_title(f'Thread Scaling: Speedup (Matrix: {results[0]["m"]}×{results[0]["k"]}×{results[0]["n"]}, '
                     f'Density: {results[0]["density"]*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log', base=2)
        
        # Plot 2: Parallel Efficiency
        ax2.plot(threads, dense_efficiency, 'o-', linewidth=2, markersize=8,
                label='Dense GEMM', color='#2E86AB')
        ax2.plot(threads, scalar_efficiency, 's-', linewidth=2, markersize=8,
                label='Sparse (scalar)', color='#A23B72')
        ax2.plot(threads, simd_efficiency, '^-', linewidth=2, markersize=8,
                label='Sparse (SIMD)', color='#F18F01')
        ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='100% Efficient')
        ax2.axhline(y=80, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='80% Threshold')
        
        ax2.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Parallel Efficiency (Speedup / Threads)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Dynamically set y-limit based on max efficiency (with some headroom)
        max_efficiency = max(max(dense_efficiency), max(scalar_efficiency), max(simd_efficiency))
        ax2.set_ylim([0, min(max(110, max_efficiency * 1.1), 200)])
        
        # Plot 3: GFLOP/s vs Thread Count (with error bars)
        ax3.errorbar(threads, dense_gflops, yerr=dense_gflops_err, fmt='o-', linewidth=2, markersize=8,
                    capsize=4, capthick=1.5, label='Dense GEMM', color='#2E86AB')
        ax3.errorbar(threads, scalar_gflops, yerr=scalar_gflops_err, fmt='s-', linewidth=2, markersize=8,
                    capsize=4, capthick=1.5, label='Sparse (scalar)', color='#A23B72')
        ax3.errorbar(threads, simd_gflops, yerr=simd_gflops_err, fmt='^-', linewidth=2, markersize=8,
                    capsize=4, capthick=1.5, label='Sparse (SIMD)', color='#F18F01')
        
        ax3.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Performance (GFLOP/s)', fontsize=12, fontweight='bold')
        ax3.set_title('Absolute Performance vs Thread Count', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Plot 4: Cycles Per Non-Zero (CPNZ) for Sparse
        ax4.plot(threads, scalar_cpnz, 's-', linewidth=2, markersize=8,
                label='Sparse (scalar)', color='#A23B72')
        ax4.plot(threads, simd_cpnz, '^-', linewidth=2, markersize=8,
                label='Sparse (SIMD)', color='#F18F01')
        
        ax4.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cycles Per Non-Zero (CPNZ)', fontsize=12, fontweight='bold')
        ax4.set_title('Sparse Matrix: Cycles Per Non-Zero', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
        plt.close()
        
        # Print analysis
        self._print_thread_analysis(results)
    
    def _print_thread_analysis(self, results):
        """Print detailed thread scaling analysis"""
        print("\n" + "="*70)
        print("THREAD SCALING ANALYSIS")
        print("="*70)
        
        baseline = results[0]
        final = results[-1]
        
        print(f"\nConfiguration:")
        print(f"  Matrix size: {baseline['m']}×{baseline['k']}×{baseline['n']}")
        print(f"  Density: {baseline['density']*100:.2f}%")
        print(f"  Thread range: {baseline['threads']} to {final['threads']}")
        
        print(f"\n{'Threads':<10} {'Dense':>15} {'Sparse(scalar)':>18} {'Sparse(SIMD)':>18}")
        print(f"{'':10} {'Time':>7} {'GFLOP/s':>7} {'Time':>7} {'CPNZ':>10} {'Time':>7} {'CPNZ':>10}")
        print("-"*70)
        
        for r in results:
            print(f"{r['threads']:<10} "
                  f"{r.get('dense_time', 0)*1000:>6.2f}ms {r.get('dense_gflops', 0):>7.2f} "
                  f"{r.get('sparse_scalar_time', 0)*1000:>6.2f}ms {r.get('sparse_scalar_cpnz', 0):>10.1f} "
                  f"{r.get('sparse_simd_time', 0)*1000:>6.2f}ms {r.get('sparse_simd_cpnz', 0):>10.1f}")
        
        print("\n" + "-"*70)
        print("SPEEDUP SUMMARY (vs 1 thread)")
        print("-"*70)
        
        baseline_dense = baseline.get('dense_time', 1)
        baseline_scalar = baseline.get('sparse_scalar_time', 1)
        baseline_simd = baseline.get('sparse_simd_time', 1)
        
        final_dense_speedup = baseline_dense / final.get('dense_time', 1)
        final_scalar_speedup = baseline_scalar / final.get('sparse_scalar_time', 1)
        final_simd_speedup = baseline_simd / final.get('sparse_simd_time', 1)
        
        print(f"Dense GEMM:")
        print(f"  {final['threads']} threads: {final_dense_speedup:.2f}x speedup")
        print(f"  Efficiency: {final_dense_speedup / final['threads'] * 100:.1f}%")
        
        print(f"\nSparse (scalar):")
        print(f"  {final['threads']} threads: {final_scalar_speedup:.2f}x speedup")
        print(f"  Efficiency: {final_scalar_speedup / final['threads'] * 100:.1f}%")
        
        print(f"\nSparse (SIMD):")
        print(f"  {final['threads']} threads: {final_simd_speedup:.2f}x speedup")
        print(f"  Efficiency: {final_simd_speedup / final['threads'] * 100:.1f}%")
        
        print("\n" + "-"*70)
        print("SIMD SPEEDUP (vs scalar, same thread count)")
        print("-"*70)
        
        for r in results:
            scalar_time = r.get('sparse_scalar_time', 1)
            simd_time = r.get('sparse_simd_time', 1)
            simd_advantage = scalar_time / simd_time
            print(f"  {r['threads']:2d} threads: {simd_advantage:.2f}x")
        
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        print("""
Strong Scaling (efficiency > 80%):
  - Good parallelization with minimal overhead
  - Memory bandwidth not yet saturated
  - Load balancing effective

Weak Scaling (efficiency 50-80%):
  - Reasonable parallelization
  - Some contention or synchronization overhead
  - May be approaching memory bandwidth limits

Poor Scaling (efficiency < 50%):
  - Memory-bound or synchronization-limited
  - Overhead dominates for sparse operations
  - Amdahl's Law limiting speedup

CPNZ trends:
  - Decreasing CPNZ = better per-element efficiency with threads
  - Increasing CPNZ = overhead or contention growing
        """)
        print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot thread scaling results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing benchmark results')
    parser.add_argument('--pattern', default='threads_t*',
                       help='File pattern to match for thread scaling')
    parser.add_argument('--output', default='thread_scaling.png',
                       help='Output filename for plot')
    
    args = parser.parse_args()
    
    analyzer = ThreadScalingAnalyzer(args.results_dir)
    results = analyzer.load_thread_results(args.pattern)
    
    if not results:
        print(f"Error: No results found matching pattern '{args.pattern}' in {args.results_dir}")
        print("\nMake sure you've run the thread scaling benchmark first:")
        print("  python run_benchmarks.py --threads")
        return 1
    
    print(f"Loaded {len(results)} thread scaling results")
    print(f"Thread counts: {[r['threads'] for r in results]}")
    
    analyzer.plot_thread_scaling(results, args.output)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
