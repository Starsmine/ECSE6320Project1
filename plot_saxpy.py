#!/usr/bin/env python3
"""
saxpy_plot_split_with_alignment.py

Generates mean GFLOP/s and speedup plots split by kernel group.
Produces additional stride plots split by data type and vector ISA.
Adds aligned-vs-misaligned ratio plots for base kernels (contiguous pattern).
"""

import os
import re
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parameters ----------
DEFAULT_CSV = "saxpy_benchmark.csv"
OUT_DIR = "plots_output"
TRIM_ZERO = True
USE_MEDIAN = False
ERROR_KIND = "std"  # "std" or "sem"
BYTE_WIDTH = {"float32": 4, "float64": 8, "int32": 4}
CACHE_LEVELS = {"L1 (~32KB)": 32, "L2 (~1024KB)": 1024, "LLC (~32768KB)": 32768}
COLORS = plt.cm.tab10.colors

# ---------- Helpers ----------
def classify_kernel(name: str) -> str:
    ln = (name or "").lower()
    if "gather" in ln:
        return "gather"
    if "stride" in ln:
        return "stride"
    return "base"

def detect_vector_isa(kernel_name: str) -> str:
    k = (kernel_name or "").lower()
    if "avx512" in k or "avx-512" in k or "avx512f" in k:
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
    if re.search(r'contig', s):
        return "contiguous"
    if re.search(r'stride', s):
        return "stride"
    if re.search(r'gather', s):
        return "gather"
    return s

# ---------- Plot routines (unchanged behavior) ----------
def plot_mean(df_summary, out_file, title_suffix):
    if df_summary.empty:
        return
    kernels = sorted(df_summary["Kernel"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, kernel in enumerate(kernels):
        for aligned in [1, 0]:
            subset = df_summary[(df_summary["Kernel"] == kernel) & (df_summary["Aligned"] == aligned)]
            if subset.empty:
                continue
            subset = subset.sort_values("Size_KB")
            label = f"{kernel}" + ("" if aligned == 1 else " (misaligned)")
            linestyle = "-" if aligned == 1 else "--"
            ax.errorbar(
                subset["Size_KB"],
                subset["central"],
                yerr=subset["err"],
                label=label,
                marker="o",
                linestyle=linestyle,
                color=COLORS[i % len(COLORS)],
                capsize=3,
                linewidth=1
            )
    ymax = df_summary["central"].max() * 1.1
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
        ax.text(size_kb * 1.02, ymax * 0.9, label, rotation=90, color="gray", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Working-set size KB")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"SAXPY mean GFLOP/s vs working-set size {title_suffix}")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    plt.tight_layout()
    save_fig(fig, out_file)

def plot_speedup(df_speedup, out_file, title_suffix):
    if df_speedup.empty:
        return
    kernels = sorted(df_speedup["Kernel"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, kernel in enumerate(kernels):
        for aligned in [1, 0]:
            subset = df_speedup[(df_speedup["Kernel"] == kernel) & (df_speedup["Aligned"] == aligned)]
            if subset.empty:
                continue
            subset = subset.sort_values("Size_KB")
            label = f"{kernel}" + ("" if aligned == 1 else " (misaligned)")
            linestyle = "-" if aligned == 1 else "--"
            ax.plot(subset["Size_KB"], subset["speedup"], marker="o", linestyle=linestyle,
                    color=COLORS[i % len(COLORS)], label=label)
    for label, size_kb in CACHE_LEVELS.items():
        ax.axvline(x=size_kb, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Working-set size KB")
    ax.set_ylabel("Speedup over scalar")
    ax.set_title(f"SIMD speedup vs scalar {title_suffix}")
    ax.axhline(1.0, color="k", linewidth=0.6, linestyle=":")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)
    plt.tight_layout()
    save_fig(fig, out_file)

def plot_stride_by_type_vector(stride_df, out_dir):
    if stride_df.empty:
        return
    speedup_dir = os.path.join(out_dir, "stride_by_type_vec")
    safe_mkdir(speedup_dir)
    df = stride_df.copy()
    df["Vector"] = df["Kernel"].map(detect_vector_isa)
    for dtype in sorted(df["Type"].unique()):
        for vec in sorted(df["Vector"].unique()):
            subset = df[(df["Type"] == dtype) & (df["Vector"] == vec)]
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            kernels = sorted(subset["Kernel"].unique())
            for i, kernel in enumerate(kernels):
                for aligned in [1, 0]:
                    s = subset[(subset["Kernel"] == kernel) & (subset["Aligned"] == aligned)]
                    if s.empty:
                        continue
                    s = s.sort_values("Size_KB")
                    label = f"{kernel}" + ("" if aligned == 1 else " (misaligned)")
                    linestyle = "-" if aligned == 1 else "--"
                    ax.plot(s["Size_KB"], s["speedup"], marker="o", linestyle=linestyle,
                            color=COLORS[i % len(COLORS)], label=label)
            for lbl, kb in CACHE_LEVELS.items():
                ax.axvline(x=kb, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("Working-set size KB")
            ax.set_ylabel("Speedup over scalar")
            ax.set_title(f"Stride speedup — Type {dtype} ISA {vec}")
            ax.axhline(1.0, color="k", linestyle=":", linewidth=0.6)
            ax.grid(True, which="both", linestyle="--", linewidth=0.4)
            ax.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            safe_vec = vec.replace("/", "_")
            out_file = os.path.join(speedup_dir, f"saxpy_speedup_stride_{dtype}_{safe_vec}.png")
            save_fig(fig, out_file)

def plot_align_vs_misalign_for_base_contiguous_v2(summary_df, out_dir, verbose=True, max_n_diff=1):
    """
    Pair aligned and misaligned rows for Pattern==contiguous, with:
      - kernel-base normalization (canonical AVX/AVX512/SSE names)
      - exact join on (Kernel_base, Type, N)
      - fuzzy join accepting abs(N_aligned - N_misaligned) <= max_n_diff
    Produces per-Type PNGs in out_dir/align_vs_misalign_contiguous_v2 and prints diagnostics.
    """
    df = summary_df.copy()

    # --- filter to contiguous pattern ---
    if "Pattern" in df.columns:
        df = df[df["Pattern"].astype(str).str.contains("contiguous", case=False, na=False)].copy()
    else:
        df = df[df["Kernel"].astype(str).str.contains("contiguous", case=False, na=False)].copy()

    if df.empty:
        if verbose:
            print("No rows match pattern = contiguous; nothing to plot.")
        return

    # --- ensure Aligned is 0/1 ---
    if "Aligned" not in df.columns:
        if verbose:
            print("Missing 'Aligned' column; aborting.")
        return
    df["Aligned"] = df["Aligned"].apply(lambda x: 1 if str(x).strip().lower() in ("1","true","t","yes","y") else 0)

    # --- required columns check ---
    for col in ("Kernel", "Type", "N", "Size_KB", "central", "err"):
        if col not in df.columns:
            if verbose:
                print(f"Missing column {col}; aborting.")
            return

    if verbose:
        print("Filtered to contiguous: rows =", len(df), "aligned/misaligned counts =", df.groupby("Aligned").size().to_dict())

    # --- improved kernel-base normalization ---
    def normalize_kernel_base(name):
        if pd.isna(name):
            return name
        s = str(name).lower()
        # normalize common ISA notations
        s = re.sub(r'\b(avx[\s\-]*512|avx512f|avx-512)\b', 'avx512', s)
        s = re.sub(r'\b(avx2)\b', 'avx2', s)
        s = re.sub(r'\b(avx)\b', 'avx', s)
        s = re.sub(r'\b(sse2|sse)\b', 'sse2', s)
        # remove scalar token since it is a kernel family but not part of base name matching
        s = re.sub(r'\bscalar\b', 'scalar', s)
        # strip alignment markers and parenthetical tags
        s = re.sub(r'[\[\(\{].*?[\]\)\}]', ' ', s)
        s = re.sub(r'(?i)\b(mis-?aligned|misaligned|misalign|unaligned|unalign|aligned)\b', ' ', s)
        s = re.sub(r'[_\-\s]+', ' ', s).strip()
        return s

    df["Kernel_base"] = df["Kernel"].map(normalize_kernel_base)

    # --- split aligned / misaligned frames as dataframes with useful columns ---
    aligned_df = df[df["Aligned"] == 1][["Kernel_base","Type","N","Size_KB","central","err"]].copy().reset_index(drop=True)
    misaligned_df = df[df["Aligned"] == 0][["Kernel_base","Type","N","Size_KB","central","err"]].copy().reset_index(drop=True)
    aligned_df.columns = ["Kernel_base","Type","N","Size_KB","central_aligned","err_aligned"]
    misaligned_df.columns = ["Kernel_base","Type","N","Size_KB","central_misaligned","err_misaligned"]

    # --- exact matching on Kernel_base, Type, N ---
    merged_exact = aligned_df.merge(misaligned_df, on=["Kernel_base","Type","N"], suffixes=("_a","_m"))
    # record keys matched
    exact_keys = set(zip(merged_exact["Kernel_base"], merged_exact["Type"], merged_exact["N"]))

    # --- find remaining unmatched aligned and misaligned rows ---
    def make_key(r):
        return (r["Kernel_base"], r["Type"], r["N"])
    aligned_remaining = aligned_df[~aligned_df.apply(make_key, axis=1).isin(exact_keys)].copy().reset_index(drop=True)
    misaligned_remaining = misaligned_df[~misaligned_df.apply(make_key, axis=1).isin(exact_keys)].copy().reset_index(drop=True)

    # --- fuzzy matching by Kernel_base and Type with abs(N diff) <= max_n_diff ---
    # Build lookup of misaligned candidates grouped by (Kernel_base, Type)
    mis_map = {}
    for idx, row in misaligned_remaining.iterrows():
        key = (row["Kernel_base"], row["Type"])
        mis_map.setdefault(key, []).append((idx, int(row["N"]), float(row["Size_KB"])))

    fuzzy_matches = []
    used_mis_idxs = set()
    for a_idx, a_row in aligned_remaining.iterrows():
        key = (a_row["Kernel_base"], a_row["Type"])
        if key not in mis_map:
            continue
        # pick candidate with minimal abs(N diff)
        best = None
        best_diff = None
        best_mis_idx = None
        aN = int(a_row["N"])
        for (mis_idx, misN, misSizeKB) in mis_map[key]:
            if mis_idx in used_mis_idxs:
                continue
            diff = abs(aN - misN)
            if best is None or diff < best_diff:
                best = (mis_idx, misN, misSizeKB)
                best_diff = diff
        if best is not None and best_diff is not None and best_diff <= max_n_diff:
            mis_row = misaligned_remaining.loc[best[0]]
            # build matched pair from aligned_remaining row and mis_row
            pair = {
                "Kernel_base": a_row["Kernel_base"],
                "Type": a_row["Type"],
                "N": a_row["N"],
                "Size_KB": a_row["Size_KB"],
                "central_aligned": a_row["central_aligned"],
                "err_aligned": a_row["err_aligned"],
                "central_misaligned": mis_row["central_misaligned"],
                "err_misaligned": mis_row["err_misaligned"],
                "N_misaligned": mis_row["N"],
                "Size_KB_misaligned": mis_row["Size_KB"]
            }
            fuzzy_matches.append(pair)
            used_mis_idxs.add(best[0])

    # turn fuzzy matches into DataFrame
    merged_fuzzy = pd.DataFrame(fuzzy_matches)

    # --- combine exact and fuzzy matches ---
    if not merged_exact.empty:
        # normalize merged_exact column names to match merged_fuzzy
        ex = merged_exact.rename(columns={
            "N":"N",
            "Size_KB_a":"Size_KB",
            "central_aligned":"central_aligned",
            "err_aligned":"err_aligned",
            "Size_KB_m":"Size_KB_misaligned",
            "central_misaligned":"central_misaligned",
            "err_misaligned":"err_misaligned"
        })
        # ensure columns names consistent with merged_fuzzy for concat
        ex = ex[["Kernel_base","Type","N","Size_KB","central_aligned","err_aligned","central_misaligned","err_misaligned"]]
    else:
        ex = pd.DataFrame(columns=["Kernel_base","Type","N","Size_KB","central_aligned","err_aligned","central_misaligned","err_misaligned"])

    if not merged_fuzzy.empty:
        # align columns (drop N_misaligned/Size_KB_misaligned from fuzzy for consistency)
        fuzzy = merged_fuzzy[["Kernel_base","Type","N","Size_KB","central_aligned","err_aligned","central_misaligned","err_misaligned"]]
    else:
        fuzzy = pd.DataFrame(columns=ex.columns)

    merged_all = pd.concat([ex, fuzzy], ignore_index=True, sort=False)

    # diagnostics
    n_exact = len(ex)
    n_fuzzy = len(fuzzy)
    n_total_pairs = len(merged_all)
    if verbose:
        print(f"Exact pairs: {n_exact}, fuzzy pairs: {n_fuzzy}, total paired: {n_total_pairs}")
        # show a few unmatched examples if any
        # compute unmatched aligned keys
        paired_keys = set(zip(merged_all["Kernel_base"], merged_all["Type"], merged_all["N"]))
        aligned_keys = set(zip(aligned_df["Kernel_base"], aligned_df["Type"], aligned_df["N"]))
        mis_keys = set(zip(misaligned_df["Kernel_base"], misaligned_df["Type"], misaligned_df["N"]))
        only_aligned = list(aligned_keys - paired_keys)[:10]
        only_misaligned = list(mis_keys - paired_keys)[:10]
        if only_aligned:
            print("Examples present only as aligned (first 10):", only_aligned)
        if only_misaligned:
            print("Examples present only as misaligned (first 10):", only_misaligned)

    if merged_all.empty:
        if verbose:
            print("No matched pairs found after fuzzy matching; nothing to plot.")
        return

    # --- compute ratio and propagate error ---
    merged_all["err_aligned"] = merged_all["err_aligned"].fillna(0.0)
    merged_all["err_misaligned"] = merged_all["err_misaligned"].fillna(0.0)

    def compute_ratio_err(row):
        cA = row["central_aligned"]
        cM = row["central_misaligned"]
        eA = row["err_aligned"]
        eM = row["err_misaligned"]
        if pd.isna(cA) or pd.isna(cM) or cA == 0 or cM == 0:
            return (np.nan, np.nan)
        r = cA / cM
        rel_sq = 0.0
        if cA != 0:
            rel_sq += (eA / cA) ** 2
        if cM != 0:
            rel_sq += (eM / cM) ** 2
        err_r = abs(r) * np.sqrt(rel_sq) if rel_sq > 0 else 0.0
        return (r, err_r)

    merged_all[["align_speedup","err_align"]] = merged_all.apply(lambda r: pd.Series(compute_ratio_err(r)), axis=1)
    merged_all = merged_all[~merged_all["align_speedup"].isna()].copy()

    if merged_all.empty:
        if verbose:
            print("All computed ratios are NaN after checks; nothing to plot.")
        return

    # --- plotting per Type ---
    align_dir = os.path.join(out_dir, "align_vs_misalign_contiguous_v2")
    safe_mkdir(align_dir)

    for dtype in sorted(merged_all["Type"].unique()):
        sub = merged_all[merged_all["Type"] == dtype].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        kernels = sorted(sub["Kernel_base"].unique())
        for i, kb in enumerate(kernels):
            s = sub[sub["Kernel_base"] == kb].sort_values("Size_KB")
            if s.empty:
                continue
            ax.errorbar(
                s["Size_KB"], s["align_speedup"], yerr=s["err_align"],
                marker="o", linestyle="-", color=plt.cm.tab10.colors[i % 10],
                capsize=3, linewidth=1, label=kb
            )
        for lbl, kb_val in CACHE_LEVELS.items():
            ax.axvline(x=kb_val, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Working-set size KB")
        ax.set_ylabel("Aligned / Misaligned")
        ax.set_title(f"Aligned vs Misaligned (contiguous) — Type {dtype}")
        ax.axhline(1.0, color="k", linestyle=":", linewidth=0.6)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        out_file = os.path.join(align_dir, f"saxpy_align_vs_misalign_contiguous_{dtype}.png")
        save_fig(fig, out_file)
        if verbose:
            print("Wrote", out_file)

    if verbose:
        print("Done. Pairs written:", n_total_pairs, " (exact:", n_exact, "fuzzy:", n_fuzzy, ")")

# ---------- Main processing ----------
def main(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if TRIM_ZERO and "GFLOP/s" in df.columns:
        df = df[df["GFLOP/s"] > 0]

    # normalize Pattern and ensure column preserved through grouping
    if "Pattern" in df.columns:
        df["Pattern"] = df["Pattern"].map(normalize_pattern)
    else:
        df["Pattern"] = ""

    df["Size_KB"] = df["N"] * df["Type"].map(BYTE_WIDTH) / 1024.0

    group_cols = ["Kernel", "Type", "N", "Aligned", "Size_KB", "Pattern"]
    agg_funcs = {"GFLOP/s": ["mean", "std", "median", "count"]}
    summary = df.groupby(group_cols).agg(agg_funcs)
    summary.columns = ["_".join(c).strip() for c in summary.columns.values]
    summary = summary.reset_index()

    summary["central"] = summary["GFLOP/s_median"] if USE_MEDIAN else summary["GFLOP/s_mean"]
    if ERROR_KIND == "sem":
        summary["err"] = summary["GFLOP/s_std"] / np.sqrt(summary["GFLOP/s_count"].replace(0, np.nan))
    else:
        summary["err"] = summary["GFLOP/s_std"]

    summary = summary[summary["GFLOP/s_count"] > 0].copy()
    summary["Group"] = summary["Kernel"].map(classify_kernel)

    # Mean plots per group
    safe_mkdir(out_dir)
    mean_base = os.path.join(out_dir, "saxpy_avg_gflops_base.png")
    mean_stride = os.path.join(out_dir, "saxpy_avg_gflops_stride.png")
    mean_gather = os.path.join(out_dir, "saxpy_avg_gflops_gather.png")
    plot_mean(summary[summary["Group"] == "base"], mean_base, "(base kernels)")
    plot_mean(summary[summary["Group"] == "stride"], mean_stride, "(stride kernels)")
    plot_mean(summary[summary["Group"] == "gather"], mean_gather, "(gather kernels)")

    # Compute speedup vs scalar baseline
    scalar_mask = summary["Kernel"].str.contains("Scalar", case=False)
    if not scalar_mask.any():
        print("Warning: no scalar baseline found. Speedup plots will be skipped.")
    else:
        baseline = summary[scalar_mask].set_index(["Type", "N", "Aligned"])[["central"]].rename(columns={"central": "scalar_central"})
        merged = summary.merge(baseline, how="left", left_on=["Type", "N", "Aligned"], right_index=True)
        merged = merged[~merged["scalar_central"].isna()].copy()
        merged["speedup"] = merged["central"] / merged["scalar_central"]
        speedup_df = merged[~merged["Kernel"].str.contains("Scalar", case=False)].copy()
        speedup_df["Group"] = speedup_df["Kernel"].map(classify_kernel)

        # Speedup plots per group
        sp_base = os.path.join(out_dir, "saxpy_speedup_base.png")
        sp_stride = os.path.join(out_dir, "saxpy_speedup_stride.png")
        sp_gather = os.path.join(out_dir, "saxpy_speedup_gather.png")
        plot_speedup(speedup_df[speedup_df["Group"] == "base"], sp_base, "(base kernels)")
        plot_speedup(speedup_df[speedup_df["Group"] == "stride"], sp_stride, "(stride kernels)")
        plot_speedup(speedup_df[speedup_df["Group"] == "gather"], sp_gather, "(gather kernels)")

        # Per-type and per-vector ISA stride speedup plots
        plot_stride_by_type_vector(speedup_df[speedup_df["Group"] == "stride"], out_dir)
        print("Saved stride comparison plots to", out_dir)

        # Aligned vs Misaligned for base kernels (contiguous only)
        plot_align_vs_misalign_for_base_contiguous_v2(summary, out_dir)
        print("Saved speedup and alignment comparison plots to", out_dir)

    print("Saved mean plots to", out_dir)
    print("Done")

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Generate mean GFLOP/s and speedup plots split by kernel groups.
            Produces additional stride plots split by data type and vector ISA.
            Produces aligned-vs-misaligned ratio plots for base kernels (contiguous).
            """)
    )
    p.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Input CSV file with benchmark trials")
    p.add_argument("--out", type=str, default=OUT_DIR, help="Output directory for PNG plots")
    args = p.parse_args()
    main(args.csv, args.out)