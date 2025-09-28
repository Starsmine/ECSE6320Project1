#!/usr/bin/env python3
"""
plot_dotp.py

Generates mean GFLOP/s and speedup plots for dot product benchmarks,
separated by data type and access pattern.
"""

import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parameters ----------
DEFAULT_CSV = "dotp_benchmark_v2.csv"
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
def plot_mean(df_summary, out_file, title_suffix):
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
    ax.set_title(f"Dot Product Performance{title_suffix}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_file)

def plot_speedup(df_summary, out_file, title_suffix, baseline_pattern="Scalar"):
    if df_summary.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    kernels = sorted(df_summary["Kernel"].unique())
    
    # Get the baseline performance for each size
    baseline_data = df_summary[df_summary["Kernel"].str.contains(baseline_pattern, case=False)].copy()
    # Prefer the aligned baseline (exclude any "(misaligned)" variants)
    baseline_data = baseline_data[~baseline_data["Kernel"].str.contains(r"\(misaligned\)$")]
    if baseline_data.empty:
        print(f"Warning: No baseline kernel found matching pattern '{baseline_pattern}'")
        return
    
    baseline_data = baseline_data.sort_values("Size_KB")
    baseline_perf = baseline_data.set_index("Size_KB")["central"]
    
    for i, kernel in enumerate(kernels):
        if baseline_pattern.lower() in kernel.lower():
            continue
            
        subset = df_summary[df_summary["Kernel"] == kernel].copy()
        if subset.empty:
            continue
            
        subset = subset.sort_values("Size_KB")
        # Calculate speedup only for matching sizes and reindex baseline to the same order
        common_sizes = np.intersect1d(subset["Size_KB"].values, baseline_data["Size_KB"].values)
        if common_sizes.size == 0:
            continue
        subset = subset[subset["Size_KB"].isin(common_sizes)].copy()
        subset = subset.sort_values("Size_KB")
        baseline_vals = baseline_perf.reindex(subset["Size_KB"]).values
        speedup = subset["central"].values / baseline_vals
        
        ax.plot(
            subset["Size_KB"],
            speedup,
            label=kernel,
            marker="o",
            linestyle="-",
            color=COLORS[i % len(COLORS)],
            linewidth=1
        )
    
    ymax = ax.get_ylim()[1] * 1.1
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
        ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("Speedup vs Scalar")
    ax.set_title(f"Dot Product Speedup{title_suffix}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_file)

def plot_stride_comparison(df_summary, out_file, title_suffix):
    if df_summary.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    kernels = sorted(df_summary["Kernel"].unique())
    strides = sorted(df_summary["Stride"].unique())
    
    for i, kernel in enumerate(kernels):
        for stride in strides:
            subset = df_summary[(df_summary["Kernel"] == kernel) & 
                              (df_summary["Stride"] == stride)]
            if subset.empty:
                continue
                
            subset = subset.sort_values("Size_KB")
            label = f"{kernel} (stride={stride})"
            linestyle = "-" if stride == 1 else "--"
            
            ax.plot(
                subset["Size_KB"],
                subset["central"],
                label=label,
                marker="o" if stride == 1 else "s",
                linestyle=linestyle,
                color=COLORS[i % len(COLORS)],
                alpha=0.7 if stride > 1 else 1.0,
                linewidth=1
            )
    
    ymax = df_summary["central"].max() * 1.1
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
        ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")

    ax.set_xscale("log", base=2)  # Keep x-axis logarithmic (base-2) for memory sizes
    # Y-axis is now linear
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Dot Product Stride Performance{title_suffix}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    # Ensure y-axis starts at 0 for better comparison
    ax.set_ylim(bottom=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    fig.tight_layout()
    save_fig(fig, out_file)

def plot_aligned_vs_misaligned(df_summary, out_file, title_suffix):
    """Compare aligned vs misaligned runs for the same kernel base name."""
    if df_summary.empty:
        return

    # Normalize kernel names by removing the misaligned suffix when present
    df = df_summary.copy()
    df["BaseKernel"] = df["Kernel"].str.replace(r" \(misaligned\)$", "", regex=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    base_kernels = sorted(df["BaseKernel"].unique())

    for i, base in enumerate(base_kernels):
        aligned = df[(df["BaseKernel"] == base) & (~df["Kernel"].str.contains(r"\(misaligned\)$"))]
        misaligned = df[(df["BaseKernel"] == base) & (df["Kernel"].str.contains(r"\(misaligned\)$"))]
        if aligned.empty and misaligned.empty:
            continue

        if not aligned.empty:
            aligned = aligned.sort_values("Size_KB")
            ax.plot(aligned["Size_KB"], aligned["central"], label=f"{base} (aligned)",
                    marker="o", linestyle="-", color=COLORS[(2*i) % len(COLORS)])
        if not misaligned.empty:
            misaligned = misaligned.sort_values("Size_KB")
            ax.plot(misaligned["Size_KB"], misaligned["central"], label=f"{base} (misaligned)",
                    marker="x", linestyle="--", color=COLORS[(2*i+1) % len(COLORS)])

    ymax = df["central"].max() * 1.1
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
        ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Array Size (KB)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Aligned vs Misaligned{title_suffix}")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    save_fig(fig, out_file)

def main():
    # Read and preprocess data
    df = pd.read_csv(DEFAULT_CSV)
    
    # Calculate array size in KB (already in N_buf column)
    df["Size_KB"] = df["N_buf"] * df["Type"].map(BYTE_WIDTH) / 1024
    
    # Group by relevant columns and compute statistics
    group_cols = ["Kernel", "Type", "N_buf", "Pattern", "Stride"]
    
    if USE_MEDIAN:
        df_summary = df.groupby(group_cols).agg({
            "GFLOP/s": ["median", "std"]
        }).reset_index()
        df_summary.columns = [*group_cols, "central", "err"]
    else:
        df_summary = df.groupby(group_cols).agg({
            "GFLOP/s": ["mean", "std"]
        }).reset_index()
        df_summary.columns = [*group_cols, "central", "err"]
        
    if ERROR_KIND == "sem":
        n_trials = df.groupby(group_cols).size()
        df_summary["err"] = df_summary["err"] / np.sqrt(n_trials)
    
    # Add size in KB
    df_summary["Size_KB"] = df_summary["N_buf"] * df_summary["Type"].map(BYTE_WIDTH) / 1024
    
    # Generate plots for different data types and patterns
    for dtype in df_summary["Type"].unique():
        # Filter by data type
        dtype_data = df_summary[df_summary["Type"] == dtype]
        
        # Contiguous access pattern plots
        contiguous_data = dtype_data[dtype_data["Pattern"] == "contiguous"]
        plot_mean(
            contiguous_data,
            os.path.join(OUT_DIR, f"dotp_performance_{dtype}.png"),
            f" ({dtype})"
        )
        plot_speedup(
            contiguous_data,
            os.path.join(OUT_DIR, f"dotp_speedup_{dtype}.png"),
            f" ({dtype})"
        )
        # Aligned vs Misaligned comparison for contiguous kernels
        plot_aligned_vs_misaligned(
            contiguous_data,
            os.path.join(OUT_DIR, f"dotp_aligned_comparison_{dtype}.png"),
            f" ({dtype})"
        )
        
        # Strided access pattern plots
        strided_data = dtype_data[dtype_data["Pattern"] == "strided"]
        plot_stride_comparison(
            strided_data,
            os.path.join(OUT_DIR, f"dotp_stride_comparison_{dtype}.png"),
            f" ({dtype})"
        )

if __name__ == "__main__":
    main()