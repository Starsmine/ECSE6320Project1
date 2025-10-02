#!/usr/bin/env python3
"""
Roofline Model Plot for Dot Product Benchmarks
==============================================

This script generates roofline plots for dot product performance analysis.
The roofline model helps visualize the performance limits imposed by
computational throughput and memory bandwidth.

System Specifications:
- CPU: Ryzen 7 7700X (up to 5.5GHz)
- DRAM Bandwidth: 60 GB/s
- Peak FLOPS: Calculated based on SIMD width and frequency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# System specifications
DRAM_BANDWIDTH_GBS = 60.0  # GB/s
CPU_FREQ_GHZ = 5.5  # Base frequency in GHz
CORES = 1  # Single core testing

# Peak theoretical FLOPS (GFLOP/s) for SINGLE CORE
# With FMA (2 ops per instruction: multiply + add)
PEAK_SCALAR_GFLOPS_FMA = CPU_FREQ_GHZ * CORES * 2  # 2 FMA ops per cycle per core
PEAK_SSE2_GFLOPS_FMA = CPU_FREQ_GHZ * CORES * 4 * 2  # 4 float32 per SSE2, 2 ops (FMA)
PEAK_AVX_GFLOPS_FMA = CPU_FREQ_GHZ * CORES * 8 * 2   # 8 float32 per AVX, 2 ops (FMA)
PEAK_AVX512_GFLOPS_FMA = CPU_FREQ_GHZ * CORES * 16 * 2  # 16 float32 per AVX-512, 2 ops (FMA)

# Without FMA (1 op per instruction: either multiply OR add)
PEAK_SCALAR_GFLOPS_NO_FMA = CPU_FREQ_GHZ * CORES * 1  # 1 op per cycle per core
PEAK_SSE2_GFLOPS_NO_FMA = CPU_FREQ_GHZ * CORES * 4 * 1  # 4 float32 per SSE2, 1 op
PEAK_AVX_GFLOPS_NO_FMA = CPU_FREQ_GHZ * CORES * 8 * 1   # 8 float32 per AVX, 1 op
PEAK_AVX512_GFLOPS_NO_FMA = CPU_FREQ_GHZ * CORES * 16 * 1  # 16 float32 per AVX-512, 1 op

def calculate_arithmetic_intensity(kernel_type):
    """
    Calculate arithmetic intensity for dot product operation.
    Dot Product: sum(A[i] * B[i])
    
    Operations: 2 FLOP per element (1 multiply + 1 add to accumulator)
    Memory traffic: 
    - Read A: 4 bytes (float32) or 8 bytes (float64)
    - Read B: 4 bytes (float32) or 8 bytes (float64)  
    Total: 8 bytes (float32) or 16 bytes (float64)
    
    Arithmetic Intensity = FLOP / Bytes
    """
    if 'float32' in kernel_type or 'int32' in kernel_type:
        return 2.0 / 8.0  # 2 FLOP / 8 bytes = 0.25 FLOP/byte
    elif 'float64' in kernel_type:
        return 2.0 / 16.0  # 2 FLOP / 16 bytes = 0.125 FLOP/byte
    else:
        return 2.0 / 8.0  # Default to float32

def get_peak_performance(kernel_type, use_fma=True):
    """Get theoretical peak performance for different kernel types."""
    if use_fma:
        if 'AVX-512' in kernel_type:
            return PEAK_AVX512_GFLOPS_FMA
        elif 'AVX' in kernel_type:
            return PEAK_AVX_GFLOPS_FMA
        elif 'SSE2' in kernel_type:
            return PEAK_SSE2_GFLOPS_FMA
        else:  # Scalar
            return PEAK_SCALAR_GFLOPS_FMA
    else:
        if 'AVX-512' in kernel_type:
            return PEAK_AVX512_GFLOPS_NO_FMA
        elif 'AVX' in kernel_type:
            return PEAK_AVX_GFLOPS_NO_FMA
        elif 'SSE2' in kernel_type:
            return PEAK_SSE2_GFLOPS_NO_FMA
        else:  # Scalar
            return PEAK_SCALAR_GFLOPS_NO_FMA

def create_roofline_plot(df, output_dir='plots_output', title_suffix='', use_fma=True):
    """Create roofline plot from benchmark data."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Select the appropriate peak values based on FMA usage
    if use_fma:
        peak_scalar = PEAK_SCALAR_GFLOPS_FMA
        peak_sse2 = PEAK_SSE2_GFLOPS_FMA
        peak_avx = PEAK_AVX_GFLOPS_FMA
        peak_avx512 = PEAK_AVX512_GFLOPS_FMA
        fma_suffix = " (with FMA)"
    else:
        peak_scalar = PEAK_SCALAR_GFLOPS_NO_FMA
        peak_sse2 = PEAK_SSE2_GFLOPS_NO_FMA
        peak_avx = PEAK_AVX_GFLOPS_NO_FMA
        peak_avx512 = PEAK_AVX512_GFLOPS_NO_FMA
        fma_suffix = " (without FMA)"
    
    # Calculate arithmetic intensity for each kernel type
    kernel_types = df['Kernel'].unique()
    
    # Group data by kernel and calculate average performance
    performance_data = []
    
    for kernel in kernel_types:
        kernel_data = df[df['Kernel'] == kernel]
        avg_gflops = kernel_data['GFLOP/s'].mean()
        arithmetic_intensity = calculate_arithmetic_intensity(kernel)
        peak_perf = get_peak_performance(kernel, use_fma)
        
        performance_data.append({
            'Kernel': kernel,
            'Arithmetic_Intensity': arithmetic_intensity,
            'Performance_GFLOPS': avg_gflops,
            'Peak_Performance': peak_perf,
            'Efficiency': avg_gflops / peak_perf * 100
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Define roofline boundaries
    ai_range = np.logspace(-3, 2, 1000)  # Arithmetic intensity from 0.001 to 100 FLOP/byte
    
    # Memory bandwidth roof (DRAM bandwidth in GFLOP/s)
    memory_roof = DRAM_BANDWIDTH_GBS * ai_range
    
    # Compute roofs for different instruction types
    scalar_roof = np.full_like(ai_range, peak_scalar)
    sse2_roof = np.full_like(ai_range, peak_sse2)
    avx_roof = np.full_like(ai_range, peak_avx)
    avx512_roof = np.full_like(ai_range, peak_avx512)
    
    # Plot rooflines
    plt.loglog(ai_range, memory_roof, 'k-', linewidth=2, label=f'Memory Bandwidth ({DRAM_BANDWIDTH_GBS} GB/s)')
    plt.loglog(ai_range, scalar_roof, '--', color='red', linewidth=2, label=f'Scalar Peak ({peak_scalar:.0f} GFLOP/s){fma_suffix}')
    plt.loglog(ai_range, sse2_roof, '--', color='orange', linewidth=2, label=f'SSE2 Peak ({peak_sse2:.0f} GFLOP/s){fma_suffix}')
    plt.loglog(ai_range, avx_roof, '--', color='blue', linewidth=2, label=f'AVX Peak ({peak_avx:.0f} GFLOP/s){fma_suffix}')
    plt.loglog(ai_range, avx512_roof, '--', color='purple', linewidth=2, label=f'AVX-512 Peak ({peak_avx512:.0f} GFLOP/s){fma_suffix}')
    
    # Color mapping for different kernel types
    color_map = {
        'Scalar': 'red',
        'SSE2': 'orange', 
        'AVX': 'blue',
        'AVX-512': 'purple'
    }
    
    # Plot measured performance points
    for _, row in perf_df.iterrows():
        kernel_name = row['Kernel']
        
        # Determine color based on instruction set
        color = 'gray'  # default
        for inst_set, inst_color in color_map.items():
            if inst_set in kernel_name:
                color = inst_color
                break
        
        # Different markers for different data types and access patterns
        marker = 'o'  # default
        if 'stride' in kernel_name.lower():
            marker = 's'  # square for strided access
        elif 'float64' in kernel_name:
            marker = 'D'  # diamond for float64
        elif 'int32' in kernel_name:
            marker = 'h'  # hexagon for int32
            
        plt.loglog(row['Arithmetic_Intensity'], row['Performance_GFLOPS'], 
                  marker=marker, color=color, markersize=8, alpha=0.7,
                  label=kernel_name if len(kernel_name) < 20 else kernel_name[:17] + '...')
    
    # Customize plot
    plt.xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=14)
    plt.ylabel('Performance (GFLOP/s)', fontsize=14)
    plt.title(f'Roofline Model for Dot Product Benchmarks{title_suffix}{fma_suffix}\n'
              f'Ryzen 7 7700X @ {CPU_FREQ_GHZ}GHz (Single Core), {DRAM_BANDWIDTH_GBS} GB/s DRAM', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(0.01, 10)
    plt.ylim(1, peak_avx512 * 1.2)
    
    # Legend with limited entries (avoid overcrowding)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Keep only roofline labels and a few representative kernel labels
    roof_handles = handles[:5]  # First 5 are the rooflines
    roof_labels = labels[:5]
    
    plt.legend(roof_handles, roof_labels, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    fma_file_suffix = "_with_fma" if use_fma else "_without_fma"
    output_path = Path(output_dir) / f'dotp_roofline_model{title_suffix.lower().replace(" ", "_")}{fma_file_suffix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Roofline plot saved to: {output_path}")
    
    return perf_df

def create_detailed_roofline_by_type(df, output_dir='plots_output', use_fma=True):
    """Create separate roofline plots for different data types and access patterns."""
    
    # Float32 contiguous access patterns
    float32_contiguous = df[
        (df['Kernel'].str.contains('float32')) & 
        (df['Pattern'] == 'contiguous')
    ]
    
    if not float32_contiguous.empty:
        create_roofline_plot(float32_contiguous, output_dir, ' - Float32 Contiguous', use_fma)
    
    # Strided access patterns (if available)
    if 'Stride' in df.columns:
        strided_data = df[df['Stride'] > 1]
        if not strided_data.empty:
            create_roofline_plot(strided_data, output_dir, ' - Strided Access', use_fma)

def print_performance_summary(perf_df, use_fma=True):
    """Print a summary of performance efficiency."""
    print("\n" + "="*80)
    print("DOT PRODUCT ROOFLINE ANALYSIS SUMMARY")
    print("="*80)
    
    # Select the appropriate peak values based on FMA usage
    if use_fma:
        peak_scalar = PEAK_SCALAR_GFLOPS_FMA
        peak_sse2 = PEAK_SSE2_GFLOPS_FMA
        peak_avx = PEAK_AVX_GFLOPS_FMA
        peak_avx512 = PEAK_AVX512_GFLOPS_FMA
        fma_note = " (with FMA)"
    else:
        peak_scalar = PEAK_SCALAR_GFLOPS_NO_FMA
        peak_sse2 = PEAK_SSE2_GFLOPS_NO_FMA
        peak_avx = PEAK_AVX_GFLOPS_NO_FMA
        peak_avx512 = PEAK_AVX512_GFLOPS_NO_FMA
        fma_note = " (without FMA)"
    
    print(f"\nSystem Configuration:")
    print(f"  CPU: Ryzen 7 7700X @ {CPU_FREQ_GHZ} GHz (Single Core)")
    print(f"  DRAM Bandwidth: {DRAM_BANDWIDTH_GBS} GB/s")
    print(f"  Test Configuration: Single-threaded")
    
    print(f"\nTheoretical Peak Performance{fma_note}:")
    print(f"  Scalar:   {peak_scalar:8.0f} GFLOP/s")
    print(f"  SSE2:     {peak_sse2:8.0f} GFLOP/s")
    print(f"  AVX:      {peak_avx:8.0f} GFLOP/s")
    print(f"  AVX-512:  {peak_avx512:8.0f} GFLOP/s")
    
    print(f"\nArithmetic Intensity:")
    print(f"  Dot Product float32: {calculate_arithmetic_intensity('float32'):.3f} FLOP/Byte")
    print(f"  Dot Product float64: {calculate_arithmetic_intensity('float64'):.3f} FLOP/Byte")
    
    print(f"\nTop Performing Kernels:")
    top_kernels = perf_df.nlargest(10, 'Performance_GFLOPS')
    for _, row in top_kernels.iterrows():
        print(f"  {row['Kernel']:<30}: {row['Performance_GFLOPS']:8.2f} GFLOP/s ({row['Efficiency']:5.1f}% efficiency)")
    
    print(f"\nMemory vs Compute Bound Analysis:")
    ai_threshold = DRAM_BANDWIDTH_GBS / peak_avx512
    print(f"  Ridge point (AVX-512): {ai_threshold:.3f} FLOP/Byte")
    
    memory_bound = perf_df[perf_df['Arithmetic_Intensity'] < ai_threshold]
    compute_bound = perf_df[perf_df['Arithmetic_Intensity'] >= ai_threshold]
    
    print(f"  Memory-bound kernels: {len(memory_bound)}")
    print(f"  Compute-bound kernels: {len(compute_bound)}")

def main():
    parser = argparse.ArgumentParser(description='Generate roofline plots for dot product benchmarks')
    parser.add_argument('--input', '-i', default='dotp_benchmark_v2.csv', 
                       help='Input CSV file with benchmark data')
    parser.add_argument('--output', '-o', default='plots_output',
                       help='Output directory for plots')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Generate detailed plots by access pattern')
    parser.add_argument('--no-fma', action='store_true',
                       help='Generate plots without FMA (since you mentioned not using FMA)')
    
    args = parser.parse_args()
    
    try:
        # Load benchmark data
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} benchmark records from {args.input}")
        
        # Create roofline plots for both FMA and non-FMA cases
        if args.no_fma:
            # Generate only non-FMA plots
            print("\n" + "="*60)
            print("GENERATING ROOFLINE PLOTS WITHOUT FMA")
            print("="*60)
            perf_df = create_roofline_plot(df, args.output, use_fma=False)
            
            if args.detailed:
                create_detailed_roofline_by_type(df, args.output, use_fma=False)
            
            print_performance_summary(perf_df, use_fma=False)
        else:
            # Generate both FMA and non-FMA plots for comparison
            print("\n" + "="*60)
            print("GENERATING ROOFLINE PLOTS WITH FMA")
            print("="*60)
            perf_df_fma = create_roofline_plot(df, args.output, use_fma=True)
            
            print("\n" + "="*60)
            print("GENERATING ROOFLINE PLOTS WITHOUT FMA")
            print("="*60)
            perf_df_no_fma = create_roofline_plot(df, args.output, use_fma=False)
            
            # Create detailed plots if requested
            if args.detailed:
                create_detailed_roofline_by_type(df, args.output, use_fma=True)
                create_detailed_roofline_by_type(df, args.output, use_fma=False)
            
            # Print performance summary for both cases
            print_performance_summary(perf_df_fma, use_fma=True)
            print_performance_summary(perf_df_no_fma, use_fma=False)
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input}'")
        print("Make sure you're running this script from the Project1DotProduct directory.")
        print("Available files: dotp_benchmark.csv, dotp_benchmark_v2.csv")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()