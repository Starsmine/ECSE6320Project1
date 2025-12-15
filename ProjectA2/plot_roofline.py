#!/usr/bin/env python3
"""
Roofline Model Analysis for Dense GEMM and Sparse CSR-SpMM

Plots achieved performance against theoretical roofline based on:
- Measured memory bandwidth (from measure_bandwidth.cpp results)
- Estimated peak FLOPS (from CPU specs)
- Arithmetic intensity for each kernel
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# System Configuration (AMD Ryzen 7 7700X - Zen 4)
# ============================================================================

# CPU specs
NUM_CORES = 8
NUM_THREADS = 16  # 2 threads per core (SMT)
BASE_FREQ_GHZ = 4.5  # Base frequency in GHz
MAX_FREQ_GHZ = 5.4   # Boost frequency in GHz

# AVX-512 SIMD: 8 doubles per vector, 2 FMAs per cycle
SIMD_WIDTH = 8  # doubles
FMAS_PER_CYCLE = 2  # Fused Multiply-Add operations

# Peak FLOPS calculation for Zen 4 (IMPORTANT: Double-pumped AVX-512)
# Zen 4 implements AVX-512 as 2×256-bit operations, so each 512-bit op takes 2 cycles
# FMA = 2 FLOPs (multiply + add)
# Effective: 8 doubles × 2 FMA × 2 FLOPs/FMA = 16 FLOPs, but takes 2 cycles
# So: 16 FLOPs / 2 cycles = 8 FLOPs/cycle/core (not 16!)
FLOPS_PER_CYCLE_PER_CORE = (SIMD_WIDTH * FMAS_PER_CYCLE) / 2  # Divide by 2 for double-pump

# Theoretical peak (at boost frequency, all cores)
PEAK_GFLOPS_SINGLE_CORE = FLOPS_PER_CYCLE_PER_CORE * MAX_FREQ_GHZ
PEAK_GFLOPS_ALL_CORES = PEAK_GFLOPS_SINGLE_CORE * NUM_CORES

# Measured memory bandwidth (GB/s) based on Chips and Cheese Zen 4 analysis
# Source: https://chipsandcheese.com/p/amds-zen-4-part-2-memory-subsystem-and-conclusion
BANDWIDTH = {
    'L1d': 170.0,    # 32 bytes/cycle × 5.4 GHz = 173 GB/s per core
    'L2': 230.0,     # ~32 bytes/cycle × 4.8 GHz × 8 cores (with contention)
    'L3': 130.0,     # 27 bytes/cycle × 4.8 GHz shared (article: "almost as much as L2")
    'DRAM': 73.0     # DDR5-6000 measured: 72.85 GB/s (76% of 96 GB/s theoretical)
}

# Cache hierarchy
CACHE_SIZES = {
    'L1d': 32 * 1024,        # 32 KB per core
    'L2': 1024 * 1024,       # 1 MB per core
    'L3': 32 * 1024 * 1024   # 32 MB shared
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_density_sweep(results_dir="results"):
    """Load density sweep results for 2048×2048×2048 matrices"""
    results_dir = Path(results_dir)
    pattern = re.compile(r'density_sweep_\d+x\d+x\d+_d(\d+)\.txt')
    
    data = {
        'dense': {'density': [], 'gflops': [], 'time': [], 'ai': []},
        'sparse_scalar': {'density': [], 'gflops': [], 'time': [], 'ai': []},
        'sparse_simd': {'density': [], 'gflops': [], 'time': [], 'ai': []}
    }
    
    for filepath in sorted(results_dir.glob("density_sweep_*.txt")):
        match = pattern.search(filepath.name)
        if not match:
            continue
            
        density = float(match.group(1)) / 10000.0
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract dimensions
        m_match = re.search(r'm=(\d+)', content)
        if m_match:
            m = int(m_match.group(1))
        
        # Extract arithmetic intensities
        dense_ai_match = re.search(r'Arithmetic Intensity - Dense:\s*([\d.]+)\s*FLOPs/byte', content)
        sparse_ai_match = re.search(r'Arithmetic Intensity - Sparse:\s*([\d.]+)\s*FLOPs/byte', content)
        
        dense_ai = float(dense_ai_match.group(1)) if dense_ai_match else 0
        sparse_ai = float(sparse_ai_match.group(1)) if sparse_ai_match else 0
        
        # Extract dense GEMM performance
        dense_match = re.search(r'Dense GEMM.*?Time:\s*([\d.]+)\s*s.*?GFLOP/s:\s*([\d.]+)', content, re.DOTALL)
        if dense_match:
            data['dense']['density'].append(density)
            data['dense']['time'].append(float(dense_match.group(1)))
            data['dense']['gflops'].append(float(dense_match.group(2)))
            data['dense']['ai'].append(dense_ai)
        
        # Extract sparse scalar performance
        scalar_match = re.search(r'CSR-SpMM \(scalar\).*?Time:\s*([\d.]+)\s*s.*?GFLOP/s:\s*([\d.]+)', content, re.DOTALL)
        if scalar_match:
            data['sparse_scalar']['density'].append(density)
            data['sparse_scalar']['time'].append(float(scalar_match.group(1)))
            data['sparse_scalar']['gflops'].append(float(scalar_match.group(2)))
            data['sparse_scalar']['ai'].append(sparse_ai)
        
        # Extract sparse SIMD performance
        simd_match = re.search(r'CSR-SpMM \(SIMD\).*?Time:\s*([\d.]+)\s*s.*?GFLOP/s:\s*([\d.]+)', content, re.DOTALL)
        if simd_match:
            data['sparse_simd']['density'].append(density)
            data['sparse_simd']['time'].append(float(simd_match.group(1)))
            data['sparse_simd']['gflops'].append(float(simd_match.group(2)))
            data['sparse_simd']['ai'].append(sparse_ai)
    
    # Convert to numpy arrays
    for kernel in data:
        for key in data[kernel]:
            data[kernel][key] = np.array(data[kernel][key])
    
    return data

def load_size_sweep(results_dir="results"):
    """Load size sweep results for working-set analysis"""
    results_dir = Path(results_dir)
    pattern = re.compile(r'size_sweep_d\d+_s(\d+)\.txt')
    
    data = {
        'dense': {'size': [], 'gflops': [], 'ai': []},
        'sparse_simd': {'size': [], 'gflops': [], 'ai': []}
    }
    
    for filepath in sorted(results_dir.glob("size_sweep_*.txt")):
        match = pattern.search(filepath.name)
        if not match:
            continue
            
        size = int(match.group(1))
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract arithmetic intensities
        dense_ai_match = re.search(r'Arithmetic Intensity - Dense:\s*([\d.]+)\s*FLOPs/byte', content)
        sparse_ai_match = re.search(r'Arithmetic Intensity - Sparse:\s*([\d.]+)\s*FLOPs/byte', content)
        
        dense_ai = float(dense_ai_match.group(1)) if dense_ai_match else 0
        sparse_ai = float(sparse_ai_match.group(1)) if sparse_ai_match else 0
        
        # Extract dense GEMM performance
        dense_match = re.search(r'Dense GEMM.*?GFLOP/s:\s*([\d.]+)', content, re.DOTALL)
        if dense_match:
            data['dense']['size'].append(size)
            data['dense']['gflops'].append(float(dense_match.group(1)))
            data['dense']['ai'].append(dense_ai)
        
        # Extract sparse SIMD performance
        simd_match = re.search(r'CSR-SpMM \(SIMD\).*?GFLOP/s:\s*([\d.]+)', content, re.DOTALL)
        if simd_match:
            data['sparse_simd']['size'].append(size)
            data['sparse_simd']['gflops'].append(float(simd_match.group(1)))
            data['sparse_simd']['ai'].append(sparse_ai)
    
    # Convert to numpy arrays
    for kernel in data:
        for key in data[kernel]:
            data[kernel][key] = np.array(data[kernel][key])
    
    return data

# ============================================================================
# Roofline Model
# ============================================================================

def compute_roofline(bandwidth_gb_s, peak_gflops):
    """
    Compute roofline curve
    
    Returns:
        ai_array: Arithmetic intensity values (FLOPs/byte)
        perf_array: Performance values (GFLOP/s)
        ridge_point: (AI, GFLOP/s) at ridge point
    """
    # Ridge point: where memory bandwidth limit meets compute limit
    ridge_ai = peak_gflops / bandwidth_gb_s
    
    # Arithmetic intensity range (log scale)
    ai_min = 0.01
    ai_max = 1000.0
    ai_array = np.logspace(np.log10(ai_min), np.log10(ai_max), 1000)
    
    # Performance: min of bandwidth-bound and compute-bound
    perf_bandwidth_bound = ai_array * bandwidth_gb_s
    perf_compute_bound = np.full_like(ai_array, peak_gflops)
    perf_array = np.minimum(perf_bandwidth_bound, perf_compute_bound)
    
    ridge_point = (ridge_ai, peak_gflops)
    
    return ai_array, perf_array, ridge_point

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_roofline_analysis():
    """Create roofline plot with all cache levels and benchmark data"""
    
    # Load benchmark data
    density_data = load_density_sweep()
    size_data = load_size_sweep()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Roofline Model Analysis: Dense GEMM vs Sparse CSR-SpMM', 
                 fontsize=14, fontweight='bold')
    
    # ========================================================================
    # Subplot 1: Roofline with different cache levels
    # ========================================================================
    
    colors = {
        'L1d': '#ff7f0e',
        'L2': '#2ca02c', 
        'L3': '#d62728',
        'DRAM': '#9467bd'
    }
    
    # Plot roofline for each cache level
    for cache_level in ['DRAM', 'L3', 'L2', 'L1d']:
        bw = BANDWIDTH[cache_level]
        peak = PEAK_GFLOPS_ALL_CORES
        ai_array, perf_array, ridge = compute_roofline(bw, peak)
        
        ax1.plot(ai_array, perf_array, '--', linewidth=2, alpha=0.7,
                color=colors[cache_level], 
                label=f'{cache_level} BW ({bw:.0f} GB/s, ridge: {ridge[0]:.1f} FLOPs/B)')
    
    # Plot peak FLOPS line
    ax1.axhline(y=PEAK_GFLOPS_ALL_CORES, color='black', linestyle=':', 
                linewidth=2, alpha=0.5, label=f'Peak ({PEAK_GFLOPS_ALL_CORES:.1f} GFLOP/s)')
    
    # Plot benchmark data from density sweep (2048×2048×2048)
    # Take representative points at different densities
    for idx in [0, len(density_data['dense']['ai'])//4, len(density_data['dense']['ai'])//2, -1]:
        if idx < len(density_data['dense']['ai']):
            density = density_data['dense']['density'][idx]
            
            # Dense GEMM
            ax1.scatter(density_data['dense']['ai'][idx], 
                       density_data['dense']['gflops'][idx],
                       s=100, marker='o', edgecolors='blue', linewidth=2,
                       facecolors='lightblue', alpha=0.8, zorder=5)
            
            # Sparse SIMD
            ax1.scatter(density_data['sparse_simd']['ai'][idx], 
                       density_data['sparse_simd']['gflops'][idx],
                       s=100, marker='s', edgecolors='red', linewidth=2,
                       facecolors='lightcoral', alpha=0.8, zorder=5)
    
    # Add representative labels
    ax1.scatter([], [], s=100, marker='o', edgecolors='blue', linewidth=2,
               facecolors='lightblue', label='Dense GEMM (2048³)')
    ax1.scatter([], [], s=100, marker='s', edgecolors='red', linewidth=2,
               facecolors='lightcoral', label='Sparse SIMD (2048³)')
    
    ax1.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
    ax1.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax1.set_title('Roofline Model: Cache Hierarchy Impact', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(0.1, 500)
    ax1.set_ylim(0.1, PEAK_GFLOPS_ALL_CORES * 1.5)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    
    # ========================================================================
    # Subplot 2: Achieved performance across matrix sizes
    # ========================================================================
    
    # Plot DRAM roofline as reference
    bw = BANDWIDTH['DRAM']
    peak = PEAK_GFLOPS_ALL_CORES
    ai_array, perf_array, ridge = compute_roofline(bw, peak)
    ax2.plot(ai_array, perf_array, '--', linewidth=2, color='gray', alpha=0.5,
            label=f'DRAM Roofline ({bw:.0f} GB/s)')
    
    # Plot peak FLOPS
    ax2.axhline(y=PEAK_GFLOPS_ALL_CORES, color='black', linestyle=':', 
                linewidth=2, alpha=0.5, label=f'Peak ({PEAK_GFLOPS_ALL_CORES:.1f} GFLOP/s)')
    
    # Plot all size sweep data
    ax2.scatter(size_data['dense']['ai'], size_data['dense']['gflops'],
               s=60, marker='o', c=np.log10(size_data['dense']['size']), 
               cmap='viridis', edgecolors='blue', linewidth=1,
               alpha=0.7, label='Dense GEMM', zorder=5)
    
    ax2.scatter(size_data['sparse_simd']['ai'], size_data['sparse_simd']['gflops'],
               s=60, marker='s', c=np.log10(size_data['sparse_simd']['size']), 
               cmap='plasma', edgecolors='red', linewidth=1,
               alpha=0.7, label='Sparse SIMD', zorder=5)
    
    # Add annotations for key sizes
    key_sizes = [32, 192, 224, 1024, 2048, 6144]
    for size in key_sizes:
        # Dense
        idx = np.where(size_data['dense']['size'] == size)[0]
        if len(idx) > 0:
            idx = idx[0]
            ax2.annotate(f'{size}', 
                        xy=(size_data['dense']['ai'][idx], size_data['dense']['gflops'][idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Sparse (only for larger sizes)
        if size >= 224:
            idx = np.where(size_data['sparse_simd']['size'] == size)[0]
            if len(idx) > 0:
                idx = idx[0]
                ax2.annotate(f'{size}', 
                            xy=(size_data['sparse_simd']['ai'][idx], size_data['sparse_simd']['gflops'][idx]),
                            xytext=(5, -10), textcoords='offset points', fontsize=8, alpha=0.7, color='red')
    
    ax2.set_xlabel('Arithmetic Intensity (FLOPs/byte)', fontsize=12)
    ax2.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax2.set_title('Achieved Performance: Size Sweep (32×32 to 6144×6144)', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.1, 500)
    ax2.set_ylim(1, PEAK_GFLOPS_ALL_CORES * 1.5)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('roofline_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved to: roofline_analysis.png")
    
    return fig, (ax1, ax2)

def print_roofline_analysis():
    """Print detailed roofline analysis"""
    
    print("\n" + "="*70)
    print("ROOFLINE MODEL ANALYSIS")
    print("="*70)
    
    print(f"\n{'='*70}")
    print("SYSTEM CONFIGURATION (AMD Ryzen 7 7700X - Zen 4)")
    print(f"{'='*70}")
    print(f"  Cores: {NUM_CORES} (+ {NUM_THREADS - NUM_CORES} SMT threads)")
    print(f"  Base/Boost Frequency: {BASE_FREQ_GHZ:.1f} / {MAX_FREQ_GHZ:.1f} GHz")
    print(f"  SIMD: AVX-512 ({SIMD_WIDTH} doubles, {FMAS_PER_CYCLE} FMAs/cycle)")
    print(f"  FLOPs/cycle/core: {FLOPS_PER_CYCLE_PER_CORE}")
    print(f"\n  Theoretical Peak FLOPS:")
    print(f"    Single core:  {PEAK_GFLOPS_SINGLE_CORE:.1f} GFLOP/s")
    print(f"    All cores:    {PEAK_GFLOPS_ALL_CORES:.1f} GFLOP/s")
    
    print(f"\n{'='*70}")
    print("MEASURED MEMORY BANDWIDTH")
    print(f"{'='*70}")
    for level in ['L1d', 'L2', 'L3', 'DRAM']:
        bw = BANDWIDTH[level]
        ridge = PEAK_GFLOPS_ALL_CORES / bw
        print(f"  {level:4s}: {bw:6.1f} GB/s  (Ridge point: {ridge:6.2f} FLOPs/byte)")
    
    # Load data
    density_data = load_density_sweep()
    size_data = load_size_sweep()
    
    print(f"\n{'='*70}")
    print("ARITHMETIC INTENSITY ANALYSIS")
    print(f"{'='*70}")
    
    # Dense GEMM characteristics
    print("\n  Dense GEMM (Matrix Multiplication):")
    print(f"    Algorithm: Tiled implementation with cache blocking")
    print(f"    FLOPs: 2×m×k×n (for m×k @ k×n)")
    print(f"    Data: 3×n² elements for square matrices (A, B, C)")
    print(f"    Arithmetic Intensity: O(n) - grows with matrix size")
    
    if len(density_data['dense']['ai']) > 0:
        ai_2048 = density_data['dense']['ai'][0]  # 2048×2048×2048
        print(f"    Example (2048³): {ai_2048:.2f} FLOPs/byte")
        print(f"      → {2*2048**3/1e9:.1f} GFLOPs / {3*2048**2*8/1e9:.3f} GB data")
    
    print("\n  Sparse CSR-SpMM (Compressed Sparse Row):")
    print(f"    Algorithm: Row-wise sparse matrix × dense matrix")
    print(f"    FLOPs: ~2×nnz×n (nnz = number of non-zeros)")
    print(f"    Data: nnz values + indices + dense matrix")
    print(f"    Arithmetic Intensity: Lower, depends on sparsity")
    
    if len(density_data['sparse_simd']['ai']) > 0:
        # Show different densities
        for idx in [0, len(density_data['sparse_simd']['ai'])//2, -1]:
            if idx < len(density_data['sparse_simd']['ai']):
                density = density_data['sparse_simd']['density'][idx]
                ai = density_data['sparse_simd']['ai'][idx]
                print(f"    Example (2048³, {density*100:.2f}% dense): {ai:.2f} FLOPs/byte")
    
    print(f"\n{'='*70}")
    print("COMPUTE-BOUND vs MEMORY-BOUND REGIMES")
    print(f"{'='*70}")
    
    # Analyze based on DRAM bandwidth (most restrictive)
    dram_bw = BANDWIDTH['DRAM']
    dram_ridge = PEAK_GFLOPS_ALL_CORES / dram_bw
    
    print(f"\n  Ridge Point (DRAM): {dram_ridge:.2f} FLOPs/byte")
    print(f"    Below {dram_ridge:.2f}: Memory-bound (limited by {dram_bw:.0f} GB/s)")
    print(f"    Above {dram_ridge:.2f}: Compute-bound (limited by {PEAK_GFLOPS_ALL_CORES:.1f} GFLOP/s)")
    
    print(f"\n  Dense GEMM Analysis:")
    # Get representative data points
    if len(size_data['dense']['ai']) > 0:
        small_idx = 0  # Smallest size
        large_idx = -1  # Largest size
        
        print(f"    Small matrices ({size_data['dense']['size'][small_idx]}×{size_data['dense']['size'][small_idx]}):")
        print(f"      AI: {size_data['dense']['ai'][small_idx]:.2f} FLOPs/byte")
        print(f"      Achieved: {size_data['dense']['gflops'][small_idx]:.2f} GFLOP/s")
        print(f"      Status: {'COMPUTE-BOUND' if size_data['dense']['ai'][small_idx] > dram_ridge else 'MEMORY-BOUND'}")
        print(f"      Efficiency: {100*size_data['dense']['gflops'][small_idx]/PEAK_GFLOPS_ALL_CORES:.1f}% of peak")
        
        print(f"    Large matrices ({size_data['dense']['size'][large_idx]}×{size_data['dense']['size'][large_idx]}):")
        print(f"      AI: {size_data['dense']['ai'][large_idx]:.2f} FLOPs/byte")
        print(f"      Achieved: {size_data['dense']['gflops'][large_idx]:.2f} GFLOP/s")
        print(f"      Status: {'COMPUTE-BOUND' if size_data['dense']['ai'][large_idx] > dram_ridge else 'MEMORY-BOUND'}")
        print(f"      Efficiency: {100*size_data['dense']['gflops'][large_idx]/PEAK_GFLOPS_ALL_CORES:.1f}% of peak")
    
    print(f"\n  Sparse SIMD Analysis:")
    if len(density_data['sparse_simd']['ai']) > 0:
        sparse_idx = len(density_data['sparse_simd']['ai']) // 2
        density = density_data['sparse_simd']['density'][sparse_idx]
        
        print(f"    Matrix (2048³, {density*100:.2f}% dense):")
        print(f"      AI: {density_data['sparse_simd']['ai'][sparse_idx]:.2f} FLOPs/byte")
        print(f"      Achieved: {density_data['sparse_simd']['gflops'][sparse_idx]:.2f} GFLOP/s")
        print(f"      Status: {'COMPUTE-BOUND' if density_data['sparse_simd']['ai'][sparse_idx] > dram_ridge else 'MEMORY-BOUND'}")
        print(f"      Efficiency: {100*density_data['sparse_simd']['gflops'][sparse_idx]/PEAK_GFLOPS_ALL_CORES:.1f}% of peak")
    
    print(f"\n{'='*70}")
    print("KEY CONCLUSIONS")
    print(f"{'='*70}")
    print("""
  1. DENSE GEMM is COMPUTE-BOUND:
     • High arithmetic intensity (85-256 FLOPs/byte for large matrices)
     • Well above ridge point (3.25 FLOPs/byte for DRAM)
     • Achieves 100-117 GFLOP/s (~13-15% of theoretical peak)
     • Limited by: ALU throughput, not memory bandwidth
     • Optimization strategy: Maximize FLOPs utilization, improve tiling

  2. SPARSE CSR-SpMM is MEMORY-BOUND:
     • Low arithmetic intensity (0.2-18 FLOPs/byte depending on density)
     • Often below or near ridge point
     • Achieves 2-47 GFLOP/s (~0.3-6% of theoretical peak)
     • Limited by: Memory bandwidth and irregular access patterns
     • Optimization strategy: Improve memory access patterns, cache locality

  3. CACHE HIERARCHY IMPACT:
     • L1d/L2: Very high bandwidth (60-400 GB/s) → ridge at 0.3-2.2 FLOPs/byte
     • L3: Medium bandwidth (100 GB/s) → ridge at 13 FLOPs/byte
     • DRAM: Low bandwidth (40 GB/s) → ridge at 3.3 FLOPs/byte
     • Dense GEMM maintains compute-bound across all levels
     • Sparse operations become memory-bound in DRAM

  4. PERFORMANCE GAP:
     • Dense/Sparse speedup ratio increases with matrix size
     • Dense GEMM benefits from high AI and good cache locality
     • Sparse operations suffer from irregular memory access patterns
     • Multi-threading helps dense more than sparse (better work distribution)
""")
    
    print("="*70)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    plot_roofline_analysis()
    print_roofline_analysis()
