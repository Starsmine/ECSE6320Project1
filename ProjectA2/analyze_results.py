#!/usr/bin/env python3
"""
Analysis and plotting script for Dense vs Sparse Matrix Multiplication benchmarks
Generates density break-even plots and other visualizations
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

class BenchmarkAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.data = defaultdict(list)
        
    def parse_result_file(self, filepath):
        """Parse a single benchmark result file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract experiment parameters
        exp_match = re.search(r'Experiment: m=(\d+), k=(\d+), n=(\d+), density=([\d.]+)%', content)
        if not exp_match:
            return None
        
        m, k, n = int(exp_match.group(1)), int(exp_match.group(2)), int(exp_match.group(3))
        density = float(exp_match.group(4)) / 100.0
        
        # Extract performance metrics
        result = {
            'm': m, 'k': k, 'n': n,
            'density': density,
            'nnz': 0,
            'avg_nnz_per_row': 0,
            'ai_dense': 0,
            'ai_sparse': 0,
        }
        
        # CSR non-zeros
        nnz_match = re.search(r'CSR non-zeros: (\d+)', content)
        if nnz_match:
            result['nnz'] = int(nnz_match.group(1))
        
        # Avg non-zeros per row
        avg_nnz_match = re.search(r'Avg non-zeros per row: ([\d.]+)', content)
        if avg_nnz_match:
            result['avg_nnz_per_row'] = float(avg_nnz_match.group(1))
        
        # Arithmetic intensity
        ai_dense_match = re.search(r'Arithmetic Intensity - Dense: ([\d.]+)', content)
        ai_sparse_match = re.search(r'Arithmetic Intensity - Sparse: ([\d.]+)', content)
        if ai_dense_match:
            result['ai_dense'] = float(ai_dense_match.group(1))
        if ai_sparse_match:
            result['ai_sparse'] = float(ai_sparse_match.group(1))
        
        # OpenBLAS GEMM
        openblas_match = re.search(r'OpenBLAS GEMM: ([\d.]+) s, ([\d.]+) GFLOP/s', content)
        if openblas_match:
            result['openblas_time'] = float(openblas_match.group(1))
            result['openblas_gflops'] = float(openblas_match.group(2))
        
        # Dense GEMM (tiled) - with optional error bars
        dense_match = re.search(r'Dense GEMM \(tiled\):\s+Time: ([\d.]+) s(?: \(±([\d.]+) s\))?\s+GFLOP/s: ([\d.]+)', content, re.MULTILINE)
        if dense_match:
            result['dense_time'] = float(dense_match.group(1))
            result['dense_stddev'] = float(dense_match.group(2)) if dense_match.group(2) else 0.0
            result['dense_gflops'] = float(dense_match.group(3))
        
        # CSR-SpMM (scalar) - with optional error bars
        sparse_scalar_match = re.search(r'CSR-SpMM \(scalar\):\s+Time: ([\d.]+) s(?: \(±([\d.]+) s\))?\s+GFLOP/s: ([\d.]+)\s+CPNZ: ([\d.]+)', content, re.MULTILINE)
        if sparse_scalar_match:
            result['sparse_scalar_time'] = float(sparse_scalar_match.group(1))
            result['sparse_scalar_stddev'] = float(sparse_scalar_match.group(2)) if sparse_scalar_match.group(2) else 0.0
            result['sparse_scalar_gflops'] = float(sparse_scalar_match.group(3))
            result['sparse_scalar_cpnz'] = float(sparse_scalar_match.group(4))
        
        # CSR-SpMM (SIMD) - with optional error bars
        sparse_simd_match = re.search(r'CSR-SpMM \(SIMD\):\s+Time:\s+([\d.]+)\s+s(?:\s+\(±([\d.]+)\s+s\))?\s+GFLOP/s:\s+([\d.]+)\s+CPNZ:\s+([\d.]+)\s+cycles/nonzero\s+Speedup vs scalar:\s+([\d.]+)x', content, re.MULTILINE | re.DOTALL)
        if sparse_simd_match:
            result['sparse_simd_time'] = float(sparse_simd_match.group(1))
            result['sparse_simd_stddev'] = float(sparse_simd_match.group(2)) if sparse_simd_match.group(2) else 0.0
            result['sparse_simd_gflops'] = float(sparse_simd_match.group(3))
            result['sparse_simd_cpnz'] = float(sparse_simd_match.group(4))
            result['simd_speedup'] = float(sparse_simd_match.group(5))
        
        return result
    
    def load_density_sweep(self, pattern="density_sweep_*"):
        """Load all density sweep results for a given matrix size"""
        files = sorted(self.results_dir.glob(pattern))
        
        if not files:
            print(f"No files matching pattern '{pattern}' found in {self.results_dir}")
            return []
        
        results = []
        for filepath in files:
            result = self.parse_result_file(filepath)
            if result:
                results.append(result)
        
        # Sort by density
        results.sort(key=lambda x: x['density'])
        return results
    
    def plot_density_breakeven(self, results, output_file="density_breakeven.png"):
        """
        Plot density break-even analysis
        Shows Dense GEMM vs Sparse (scalar and SIMD) performance across densities
        """
        if not results:
            print("No results to plot!")
            return
        
        densities = [r['density'] * 100 for r in results]  # Convert to percentage
        dense_gflops = [r.get('dense_gflops', 0) for r in results]
        sparse_scalar_gflops = [r.get('sparse_scalar_gflops', 0) for r in results]
        sparse_simd_gflops = [r.get('sparse_simd_gflops', 0) for r in results]
        
        # IMPORTANT: For break-even, compare RUNTIME not GFLOP/s
        # Sparse does fewer operations, so lower GFLOP/s can still mean faster execution
        dense_times = [r.get('dense_time', 0) * 1000 for r in results]  # Convert to ms
        sparse_scalar_times = [r.get('sparse_scalar_time', 0) * 1000 for r in results]
        sparse_simd_times = [r.get('sparse_simd_time', 0) * 1000 for r in results]
        
        # Error bars (stddev converted to ms)
        dense_errors = [r.get('dense_stddev', 0) * 1000 for r in results]
        sparse_scalar_errors = [r.get('sparse_scalar_stddev', 0) * 1000 for r in results]
        sparse_simd_errors = [r.get('sparse_simd_stddev', 0) * 1000 for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Runtime vs Density (PRIMARY - for break-even analysis)
        ax1.errorbar(densities, dense_times, yerr=dense_errors, fmt='o-', linewidth=2, 
                    markersize=8, capsize=4, label='Dense GEMM (tiled)', color='#2E86AB')
        ax1.errorbar(densities, sparse_scalar_times, yerr=sparse_scalar_errors, fmt='s-', 
                    linewidth=2, markersize=8, capsize=4, label='Sparse CSR-SpMM (scalar)', 
                    color='#A23B72')
        ax1.errorbar(densities, sparse_simd_times, yerr=sparse_simd_errors, fmt='^-', 
                    linewidth=2, markersize=8, capsize=4, label='Sparse CSR-SpMM (SIMD)', 
                    color='#F18F01')
        
        # Find and mark break-even points (where sparse TIME becomes HIGHER than dense)
        breakeven_scalar = self._find_breakeven(densities, sparse_scalar_times, dense_times)
        breakeven_simd = self._find_breakeven(densities, sparse_simd_times, dense_times)
        
        if breakeven_scalar:
            ax1.axvline(x=breakeven_scalar, color='#A23B72', linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(breakeven_scalar, max(max(dense_times), max(sparse_scalar_times)) * 0.9, 
                    f'Break-even (scalar)\n{breakeven_scalar:.2f}%',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if breakeven_simd:
            ax1.axvline(x=breakeven_simd, color='#F18F01', linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(breakeven_simd, max(max(dense_times), max(sparse_simd_times)) * 0.7, 
                    f'Break-even (SIMD)\n{breakeven_simd:.2f}%',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Sparsity Density (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Runtime: Density Break-Even Analysis (Matrix: {results[0]["m"]}×{results[0]["k"]}×{results[0]["n"]})',
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Add shaded regions
        if breakeven_simd:
            ax1.axvspan(densities[0], breakeven_simd, alpha=0.1, color='green', 
                       label='Sparse (SIMD) wins')
        
        # Plot 2: GFLOP/s vs Density (SECONDARY - for efficiency analysis)
        ax2.plot(densities, dense_gflops, 'o-', linewidth=2, markersize=8, 
                label='Dense GEMM (tiled)', color='#2E86AB')
        ax2.plot(densities, sparse_scalar_gflops, 's-', linewidth=2, markersize=8,
                label='Sparse CSR-SpMM (scalar)', color='#A23B72')
        ax2.plot(densities, sparse_simd_gflops, '^-', linewidth=2, markersize=8,
                label='Sparse CSR-SpMM (SIMD)', color='#F18F01')
        
        if breakeven_scalar:
            ax2.axvline(x=breakeven_scalar, color='#A23B72', linestyle='--', alpha=0.7, linewidth=2)
        if breakeven_simd:
            ax2.axvline(x=breakeven_simd, color='#F18F01', linestyle='--', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Sparsity Density (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Performance (GFLOP/s)', fontsize=12, fontweight='bold')
        ax2.set_title('GFLOP/s: Computational Efficiency (Note: Sparse does fewer ops)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
        plt.close()
        
        # Print analysis
        self._print_breakeven_analysis(results, breakeven_scalar, breakeven_simd)
    
    def _find_breakeven(self, densities, sparse_times, dense_times):
        """
        Find density where sparse TIME crosses above dense TIME (sparse becomes slower).
        At break-even: sparse_time == dense_time
        Below break-even: sparse_time < dense_time (sparse wins)
        Above break-even: sparse_time > dense_time (dense wins)
        """
        for i in range(1, len(densities)):
            if dense_times[i] > 0 and sparse_times[i] > 0:
                # Check if we crossed the break-even point
                if sparse_times[i-1] < dense_times[i-1] and sparse_times[i] > dense_times[i]:
                    # Interpolate to find exact crossing point
                    return self._interpolate_breakeven(
                        densities[i-1], sparse_times[i-1], dense_times[i-1],
                        densities[i], sparse_times[i], dense_times[i]
                    )
        
        # No crossing found - check edge cases
        if len(sparse_times) > 0 and len(dense_times) > 0:
            # If sparse is always slower, return None
            if all(s > d for s, d in zip(sparse_times, dense_times) if s > 0 and d > 0):
                return None
            # If sparse is always faster, return the highest density tested
            if all(s < d for s, d in zip(sparse_times, dense_times) if s > 0 and d > 0):
                return densities[-1]
        
        return None
    
    def _interpolate_breakeven(self, d1, sparse1, dense1, d2, sparse2, dense2):
        """
        Linear interpolation to find exact break-even point.
        Find density where sparse_time(d) == dense_time(d)
        """
        # Linear functions: sparse(d) = sparse1 + (sparse2-sparse1)/(d2-d1) * (d-d1)
        #                   dense(d) = dense1 + (dense2-dense1)/(d2-d1) * (d-d1)
        # At break-even: sparse(d) = dense(d)
        
        sparse_slope = (sparse2 - sparse1) / (d2 - d1) if d2 != d1 else 0
        dense_slope = (dense2 - dense1) / (d2 - d1) if d2 != d1 else 0
        
        # If slopes are equal, lines are parallel - use midpoint
        if abs(sparse_slope - dense_slope) < 1e-10:
            return (d1 + d2) / 2
        
        # Solve: sparse1 + sparse_slope*(d-d1) = dense1 + dense_slope*(d-d1)
        # sparse1 - dense1 = (dense_slope - sparse_slope)*(d-d1)
        # d = d1 + (sparse1 - dense1) / (dense_slope - sparse_slope)
        
        d_breakeven = d1 + (dense1 - sparse1) / (sparse_slope - dense_slope)
        
        # Clamp to interval [d1, d2]
        d_breakeven = max(d1, min(d2, d_breakeven))
        
        return d_breakeven
    
    def _print_breakeven_analysis(self, results, breakeven_scalar, breakeven_simd):
        """Print detailed break-even analysis"""
        print("\n" + "="*70)
        print("DENSITY BREAK-EVEN ANALYSIS")
        print("="*70)
        
        if breakeven_scalar:
            print(f"\n✓ Break-even (Scalar): {breakeven_scalar:.2f}%")
            print(f"  Below this density, sparse (scalar) is faster than dense")
        else:
            print(f"\n✗ Break-even (Scalar): Not found in tested range")
            print(f"  Dense GEMM remains faster across all tested densities")
        
        if breakeven_simd:
            print(f"\n✓ Break-even (SIMD): {breakeven_simd:.2f}%")
            print(f"  Below this density, sparse (SIMD) is faster than dense")
        else:
            print(f"\n✗ Break-even (SIMD): Not found in tested range")
        
        # Arithmetic intensity analysis
        print("\n" + "-"*70)
        print("ARITHMETIC INTENSITY ANALYSIS")
        print("-"*70)
        
        for i, r in enumerate(results):
            if i % max(1, len(results) // 5) == 0:  # Print ~5 samples
                print(f"\nDensity: {r['density']*100:.2f}%")
                print(f"  AI (Dense):  {r.get('ai_dense', 0):.2f} FLOPs/byte")
                print(f"  AI (Sparse): {r.get('ai_sparse', 0):.2f} FLOPs/byte")
                print(f"  Sparse/Dense ratio: {r.get('ai_sparse', 0) / max(r.get('ai_dense', 1), 0.01):.3f}")
        
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        print("""
Dense GEMM:
  - High arithmetic intensity (40-100 FLOPs/byte)
  - Compute-bound for small matrices in cache
  - Memory-bound when exceeding cache capacity
  - Benefits from cache tiling and SIMD vectorization

Sparse CSR-SpMM:
  - Lower arithmetic intensity (1-30 FLOPs/byte, depends on density)
  - Memory-bound due to irregular access patterns
  - SIMD speedup limited by memory bandwidth
  - Benefits most when density is low (<5%)

Break-even occurs when:
  - Sparse operations skip enough zeros to compensate for irregular memory access
  - Lower density → fewer operations but more cache-friendly
  - SIMD helps but doesn't change fundamental memory-bound nature
        """)
        print("="*70 + "\n")
    
    def plot_arithmetic_intensity(self, results, output_file="arithmetic_intensity.png"):
        """Plot arithmetic intensity vs density"""
        if not results:
            return
        
        densities = [r['density'] * 100 for r in results]
        ai_dense = [r.get('ai_dense', 0) for r in results]
        ai_sparse = [r.get('ai_sparse', 0) for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(densities, ai_dense, 'o-', linewidth=2, markersize=8,
               label='Dense GEMM', color='#2E86AB')
        ax.plot(densities, ai_sparse, 's-', linewidth=2, markersize=8,
               label='Sparse CSR-SpMM', color='#F18F01')
        
        # Add memory bandwidth ceiling (example: 10 FLOPs/byte for typical DRAM)
        # ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Typical DRAM Bandwidth Ceiling')
        
        ax.set_xlabel('Sparsity Density (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12, fontweight='bold')
        ax.set_title('Arithmetic Intensity vs Sparsity Density', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot matrix multiplication benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing benchmark results')
    parser.add_argument('--pattern', default='density_sweep_*',
                       help='File pattern to match for density sweep')
    parser.add_argument('--output', default='density_breakeven.png',
                       help='Output filename for plot')
    parser.add_argument('--ai-plot', action='store_true',
                       help='Also generate arithmetic intensity plot')
    parser.add_argument('--all', action='store_true',
                       help='Generate all analysis plots (density, working-set, thread scaling)')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_dir)
    results = analyzer.load_density_sweep(args.pattern)
    
    if not results:
        print(f"Error: No results found matching pattern '{args.pattern}' in {args.results_dir}")
        print("\nMake sure you've run the density sweep first:")
        print("  python run_benchmarks.py --density")
        return 1
    
    print(f"Loaded {len(results)} benchmark results")
    print(f"Matrix size: {results[0]['m']}×{results[0]['k']}×{results[0]['n']}")
    print(f"Density range: {results[0]['density']*100:.2f}% to {results[-1]['density']*100:.2f}%")
    
    # Generate main plot
    analyzer.plot_density_breakeven(results, args.output)
    
    # Generate AI plot if requested
    if args.ai_plot:
        ai_output = args.output.replace('.png', '_ai.png')
        analyzer.plot_arithmetic_intensity(results, ai_output)
    
    # Generate all plots if --all is specified
    if args.all:
        import subprocess
        import sys
        
        print("\n" + "="*70)
        print("GENERATING ALL ANALYSIS PLOTS")
        print("="*70)
        
        # Working-set transitions plot
        print("\n1. Working-Set Transitions...")
        try:
            result = subprocess.run([sys.executable, 'plot_working_set.py'], 
                                   capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate working-set plot: {e}")
            print(e.stderr)
        
        # Thread scaling plot
        print("\n2. Thread Scaling...")
        try:
            result = subprocess.run([sys.executable, 'plot_thread_scaling.py'], 
                                   capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate thread scaling plot: {e}")
            print(e.stderr)
        
        # Roofline analysis
        print("\n3. Roofline Model Analysis...")
        try:
            result = subprocess.run([sys.executable, 'plot_roofline.py'], 
                                   capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate thread scaling plot: {e}")
            print(e.stderr)
        
        print("\n" + "="*70)
        print("ALL PLOTS GENERATED")
        print("="*70)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
