#!/usr/bin/env python3
"""
plot_stencil.py

Generates mean GFLOP/s and speedup plots for 3-point stencil benchmarks,
separated by data type and alignment.
"""

import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parameters ----------
DEFAULT_CSV = "stencil_benchmark_v1.csv"
OUT_DIR = "plots_output"
TRIM_ZERO = True
USE_MEDIAN = False
ERROR_KIND = "std"  # "std" or "sem"
BYTE_WIDTH = {"float32": 4, "float64": 8, "int32": 4}
CACHE_LEVELS = {"L1 (~32KB)": 32, "L2 (~1024KB)": 1024, "LLC (~32768KB)": 32768}
COLORS = plt.cm.tab10.colors

# ---------- Helpers ----------
def detect_vector_isa(kernel_name: str) -> str:
    k = (kernel_name or "").lower()
    if "avx512" in k or "avx-512" in k:
        return "avx512"
    if "avx2" in k:
        return "avx2"
    if "avx" in k:
        return "avx"
    if "sse2" in k or "sse" in k:
        return "sse2"
    if "scalar" in k:
        return "scalar"
    return "unknown"

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_fig(fig, path, dpi=300):
    safe_mkdir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def normalize_pattern(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    s = re.sub(r'^["\']|["\']$', '', s).lower().strip()
    return s

# ---------- Plot routines ----------
def plot_consolidated(df_type, out_file, dtype):
    """Create a consolidated plot showing all kernels and strides."""
    if df_type.empty:
        return
        
    # Create a grid of subplots, one for each kernel type
    kernels = sorted(df_type["Kernel"].unique())
    n_kernels = len(kernels)
    fig, axes = plt.subplots(n_kernels, 1, figsize=(12, 4*n_kernels), sharex=True)
    if n_kernels == 1:
        axes = [axes]
    
    # Define line styles for different patterns
    patterns = df_type["Pattern"].unique()
    styles = {
        "contiguous": ("-", "o"),
        "contiguous_misaligned": ("--", "o"),
        "stride_2": ("-", "s"),
        "stride_2_misaligned": ("--", "s"),
        "stride_4": ("-", "^"),
        "stride_4_misaligned": ("--", "^"),
        "stride_8": ("-", "D"),
        "stride_8_misaligned": ("--", "D"),
        "stride_16": ("-", "v"),
        "stride_16_misaligned": ("--", "v"),
    }
    
    for ax_idx, kernel in enumerate(kernels):
        ax = axes[ax_idx]
        kernel_data = df_type[df_type["Kernel"] == kernel]
        
        ymax = 0
        for pattern in patterns:
            subset = kernel_data[kernel_data["Pattern"] == pattern]
            if subset.empty:
                continue
                
            # Calculate statistics for this subset
            grouped = subset.groupby("Size_KB")
            if USE_MEDIAN:
                central = grouped["GFLOP/s"].median()
                err = grouped["GFLOP/s"].std()
            else:
                central = grouped["GFLOP/s"].mean()
                err = grouped["GFLOP/s"].std() if ERROR_KIND == "std" else grouped["GFLOP/s"].sem()
                
            # Convert to DataFrame and sort
            stats_df = pd.DataFrame({
                "Size_KB": central.index,
                "central": central.values,
                "err": err.values
            }).sort_values("Size_KB")
            
            linestyle, marker = styles.get(pattern, ("-", "o"))
            
            ax.errorbar(
                stats_df["Size_KB"],
                stats_df["central"],
                yerr=stats_df["err"],
                label=pattern,
                marker=marker,
                linestyle=linestyle,
                capsize=3,
                linewidth=1
            )
            ymax = max(ymax, stats_df["central"].max() * 1.1)
            
        # Add cache level indicators
        for label, size_kb in CACHE_LEVELS.items():
            ax.axvline(x=size_kb, color="gray", linestyle=":", linewidth=1)
            ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")
            
        ax.set_xscale("log", base=2)
        ax.set_ylabel("GFLOP/s")
        ax.set_title(f"{kernel} Performance - {dtype}")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
    axes[-1].set_xlabel("Array Size (KB)")
    fig.tight_layout()
    save_fig(fig, out_file)

def plot_mean(df_summary, out_file, title_suffix):
    """Legacy plot function kept for compatibility"""
    if df_summary.empty:
        return
    kernels = sorted(df_summary["Kernel"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, kernel in enumerate(kernels):
        subset = df_summary[df_summary["Kernel"] == kernel]
        if subset.empty:
            continue
        subset = subset.sort_values("Size_KB")
        ax.errorbar(
            subset["Size_KB"],
            subset["central"],
            yerr=subset["err"],
            label=kernel,
            marker="o",
            linestyle="-",
            color=COLORS[i % len(COLORS)],
            capsize=3,
            linewidth=1
        )
    
    ymax = df_summary["central"].max() * 1.1
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
        ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"3-Point Stencil Performance{title_suffix}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_file)

def plot_speedup(df_summary, out_file, title_suffix, baseline_pattern="Scalar"):
    if df_summary.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    kernels = sorted(df_summary["Kernel"].unique())
    
    # Find baseline performance for each size
    baseline_df = df_summary[df_summary["Kernel"].str.contains(baseline_pattern)]
    if baseline_df.empty:
        print(f"Warning: No baseline kernel found matching '{baseline_pattern}'")
        return
    
    for i, kernel in enumerate(kernels):
        if baseline_pattern in kernel:
            continue
            
        subset = df_summary[df_summary["Kernel"] == kernel].sort_values("Size_KB")
        if subset.empty:
            continue
            
        # Match baseline values to current kernel's sizes
        baseline_perf = []
        baseline_err = []
        for size in subset["Size_KB"]:
            base = baseline_df[baseline_df["Size_KB"] == size]
            if not base.empty:
                baseline_perf.append(base["central"].iloc[0])
                baseline_err.append(base["err"].iloc[0])
            else:
                print(f"Warning: No baseline found for size {size}")
                baseline_perf.append(np.nan)
                baseline_err.append(np.nan)
        
        baseline_perf = np.array(baseline_perf)
        baseline_err = np.array(baseline_err)
        speedup = subset["central"] / baseline_perf
        
        # Error propagation for division
        rel_err = np.sqrt(
            (subset["err"] / subset["central"])**2 + 
            (baseline_err / baseline_perf)**2
        )
        err = speedup * rel_err
        
        ax.errorbar(
            subset["Size_KB"],
            speedup,
            yerr=err,
            label=kernel,
            marker="o",
            linestyle="-",
            color=COLORS[i % len(COLORS)],
            capsize=3,
            linewidth=1
        )
    
    ymax = df_summary["central"].max() / baseline_df["central"].min() * 1.1
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
        ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")
    
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel(f"Speedup over {baseline_pattern}")
    ax.set_title(f"3-Point Stencil Speedup{title_suffix}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_file)

def main():
    # Read and preprocess data
    df = pd.read_csv(DEFAULT_CSV)
    
    # Convert array sizes to KB
    df["Size_KB"] = df["N_buf"] * df["Type"].map(BYTE_WIDTH) / 1024
    
    # Create consolidated plots by data type
    for dtype in df["Type"].unique():
        df_type = df[df["Type"] == dtype]
        
        # Create a consolidated plot for this data type
        plot_consolidated(df_type, f"{OUT_DIR}/{dtype.lower()}_consolidated.png", dtype)
        
        # Group by pattern, size and kernel for speedup plots
        patterns = df_type["Pattern"].unique()
        for pattern in patterns:
            df_subset = df_type[df_type["Pattern"] == pattern]
            
            # Group by size and kernel, calculate statistics
            grouped = df_subset.groupby(["Size_KB", "Kernel"])
            if USE_MEDIAN:
                central = grouped["GFLOP/s"].median()
                err = grouped["GFLOP/s"].std()
            else:
                central = grouped["GFLOP/s"].mean()
                err = grouped["GFLOP/s"].std() if ERROR_KIND == "std" else grouped["GFLOP/s"].sem()
            
            df_summary = pd.DataFrame({
                "central": central,
                "err": err
            }).reset_index()
            
            # Generate speedup plot for this pattern
            pattern_suffix = f" - {pattern}"
            title_suffix = f" - {dtype}{pattern_suffix}"
            prefix = f"{dtype.lower()}_{pattern}"
            plot_speedup(df_summary, f"{OUT_DIR}/{prefix}_speedup.png", title_suffix)

if __name__ == "__main__":
    main()