#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_latency_stats(operation_data):
    if operation_data['total_ios'] == 0:
        return None
    
    latency_stats = {
        'min': operation_data['lat_ns']['min'] / 1000,  # Convert to μs
        'max': operation_data['lat_ns']['max'] / 1000,
        'mean': operation_data['lat_ns']['mean'] / 1000,
    }
    
    if 'percentile' in operation_data['lat_ns']:
        percentiles = operation_data['lat_ns']['percentile']
        latency_stats.update({
            'p50': percentiles.get('50.000000', 0) / 1000,
            'p95': percentiles.get('95.000000', 0) / 1000,
            'p99': percentiles.get('99.000000', 0) / 1000,
            'p99.9': percentiles.get('99.900000', 0) / 1000
        })
    
    return latency_stats

def extract_job_metrics(job):
    """Extract metrics from a single FIO job"""
    metrics = {
        'job_name': job['jobname'],
        'rw_type': job['job options'].get('rw', 'unknown'),
        'bs': job['job options'].get('bs', 'unknown'),
        'iodepth': int(job['job options'].get('iodepth', 1)),
        'numjobs': int(job['job options'].get('numjobs', 1))
    }
    
    # Read metrics
    read_data = job['read']
    metrics.update({
        'read_iops': read_data['iops'],
        'read_bw_mb': read_data['bw'] / 1024,  # Convert to MB/s
    })
    
    read_lat = get_latency_stats(read_data)
    if read_lat:
        metrics.update({f'read_{k}': v for k, v in read_lat.items()})
    
    # Write metrics
    write_data = job['write']
    metrics.update({
        'write_iops': write_data['iops'],
        'write_bw_mb': write_data['bw'] / 1024,  # Convert to MB/s
    })
    
    write_lat = get_latency_stats(write_data)
    if write_lat:
        metrics.update({f'write_{k}': v for k, v in write_lat.items()})
    
    return metrics

def plot_baseline_table(df):
    """Create baseline performance table plot"""
    plt.figure(figsize=(14, 6))  # Make it wider for the additional bandwidth column
    plt.axis('off')
    
    # Filter for QD=1 baseline tests
    baseline_df = df[df['iodepth'] == 1].copy()
    
    # Create table data
    table_data = []
    for _, row in baseline_df.iterrows():
        if '4k' in row['job_name'].lower():
            pattern = '4KiB Random'
        else:
            pattern = '128KiB Sequential'
            
        if 'read' in row['job_name'].lower():
            op = 'Read'
            iops = row['read_iops']
            bw_mb = row['read_bw_mb']
            lat_mean = row['read_mean']
            lat_p95 = row['read_p95']
            lat_p99 = row['read_p99']
        else:
            op = 'Write'
            iops = row['write_iops']
            bw_mb = row['write_bw_mb']
            lat_mean = row['write_mean']
            lat_p95 = row['write_p95']
            lat_p99 = row['write_p99']
            
        table_data.append([
            f"{pattern} {op}",
            f"{iops:.2f}",
            f"{bw_mb:.2f}",
            f"{lat_mean:.2f}",
            f"{lat_p95:.2f}",
            f"{lat_p99:.2f}"
        ])
    
    plt.table(
        cellText=table_data,
        colLabels=['Test', 'IOPS', 'BW (MB/s)', 'Mean Lat (μs)', 'P95 Lat (μs)', 'P99 Lat (μs)'],
        loc='center',
        cellLoc='center'
    )
    plt.title('Zero-Queue (QD=1) Baseline Performance')
    plt.tight_layout()
    plt.savefig('baseline_table.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_blocksize_sweep(df, access_pattern):
    """Plot blocksize sweep results"""
    pattern_df = df[df['job_name'].str.contains(access_pattern, case=False)].copy()
    
    if pattern_df.empty:
        print(f"Warning: No data found for pattern '{access_pattern}'")
        return
    
    # Convert block size strings to numeric values
    def parse_bs(bs):
        if isinstance(bs, str):
            if 'k' in bs.lower():
                return float(bs.lower().replace('k', ''))
            elif 'm' in bs.lower():
                return float(bs.lower().replace('m', '')) * 1024
        return float(bs)
    
    pattern_df['bs_kb'] = pattern_df['bs'].apply(parse_bs)
    pattern_df = pattern_df.sort_values('bs_kb')
    
    # Create figure with appropriate size and margins
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    # Plot IOPS/MB/s ratio on the left axis for better comparison
    
    # Plot IOPS/MB/s ratio with solid lines
    if (pattern_df['read_bw_mb'] > 0).any() and (pattern_df['read_iops'] > 0).any():
        read_ratio = pattern_df['read_iops'] / pattern_df['read_bw_mb']
        ax1.plot(range(len(pattern_df)), read_ratio, 'b-o', label='Read IOPS/MB/s')
    if (pattern_df['write_bw_mb'] > 0).any() and (pattern_df['write_iops'] > 0).any():
        write_ratio = pattern_df['write_iops'] / pattern_df['write_bw_mb']
        ax1.plot(range(len(pattern_df)), write_ratio, 'r-o', label='Write IOPS/MB/s')
    
    ax1.set_ylabel('IOPS per MB/s')
    
    # Plot latency with error checking
    if (pattern_df['read_mean'] > 0).any():
        ax2.plot(range(len(pattern_df)), pattern_df['read_mean'], 'b--^', label='Read Latency')
    if (pattern_df['write_mean'] > 0).any():
        ax2.plot(range(len(pattern_df)), pattern_df['write_mean'], 'r--^', label='Write Latency')
    ax2.set_ylabel('Latency (μs)')
    
    # Set x-axis labels
    bs_labels = [f'{int(x)}K' for x in pattern_df['bs_kb']]
    ax1.set_xticks(range(len(pattern_df)))
    ax1.set_xticklabels(bs_labels, rotation=45)
    ax1.set_xlabel('Block Size')
    
    # Customize y-axis scaling and ticks
    if ax1.get_lines():
        data = np.concatenate([line.get_ydata() for line in ax1.get_lines()])
        if np.all(data > 0) and len(data) > 0:
            # Use log scale to handle the different scales of IOPS and MB/s
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
    
    if ax2.get_lines():
        data = np.concatenate([line.get_ydata() for line in ax2.get_lines()])
        if np.all(data > 0) and len(data) > 0:
            ax2.set_yscale('log')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.15, 1.0))
    
    plt.title(f'{access_pattern.title()} Access Pattern: Performance vs Block Size')
    
    # Adjust layout with specific margins
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig(f'blocksize_sweep_{access_pattern.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_rw_mix(df):
    """Plot read/write mix results"""
    mix_df = df[df['job_name'].str.contains('mix', case=False)].copy()
    mix_df['rw_ratio'] = mix_df['job_name'].str.extract(r'(\d+)').astype(int)
    mix_df = mix_df.sort_values('rw_ratio')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot IOPS
    ax1.plot(mix_df['rw_ratio'], mix_df['read_iops'], 'b-', label='Read IOPS')
    ax1.plot(mix_df['rw_ratio'], mix_df['write_iops'], 'r-', label='Write IOPS')
    ax1.set_xlabel('Read Percentage')
    ax1.set_ylabel('IOPS')
    ax1.legend()
    ax1.grid(True)
    
    # Plot latency
    ax2.plot(mix_df['rw_ratio'], mix_df['read_mean'], 'b-', label='Read Latency')
    ax2.plot(mix_df['rw_ratio'], mix_df['write_mean'], 'r-', label='Write Latency')
    ax2.set_xlabel('Read Percentage')
    ax2.set_ylabel('Latency (μs)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Read/Write Mix Effects')
    plt.tight_layout()
    plt.savefig('rw_mix.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_qd_sweep(df, pattern='qd'):
    """Plot queue depth sweep results"""
    qd_df = df[df['job_name'].str.contains(pattern, case=False)].copy()
    
    if qd_df.empty:
        print(f"Warning: No data found for QD pattern '{pattern}'")
        return
        
    qd_df = qd_df.sort_values('iodepth')
    
    plt.figure(figsize=(10, 6))
    
    # Filter out rows with zero or negative values for plotting
    valid_read = qd_df[(qd_df['read_mean'] > 0) & (qd_df['read_iops'] > 0)]
    valid_write = qd_df[(qd_df['write_mean'] > 0) & (qd_df['write_iops'] > 0)]
    
    # Plot throughput vs latency only if we have valid data
    if not valid_read.empty:
        plt.errorbar(valid_read['read_mean'], valid_read['read_iops'], 
                    yerr=valid_read['read_iops']*0.05,  # Assuming 5% error
                    fmt='bo-', label='Read')
        
        for i, row in valid_read.iterrows():
            plt.annotate(f'QD={row["iodepth"]}', 
                        (row['read_mean'], row['read_iops']),
                        xytext=(5, 5), textcoords='offset points')
    
    if not valid_write.empty:
        plt.errorbar(valid_write['write_mean'], valid_write['write_iops'],
                    yerr=valid_write['write_iops']*0.05,
                    fmt='ro-', label='Write')
    
    plt.xlabel('Latency (μs)')
    plt.ylabel('IOPS')
    
    # Only use log scale if we have positive data
    if not valid_read.empty or not valid_write.empty:
        all_latency = []
        all_iops = []
        if not valid_read.empty:
            all_latency.extend(valid_read['read_mean'])
            all_iops.extend(valid_read['read_iops'])
        if not valid_write.empty:
            all_latency.extend(valid_write['write_mean'])
            all_iops.extend(valid_write['write_iops'])
        
        if all(x > 0 for x in all_latency):
            plt.xscale('log')
        if all(x > 0 for x in all_iops):
            plt.yscale('log')
    
    plt.grid(True)
    plt.legend()
    plt.title('Queue Depth vs Performance Trade-off')
    
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig('qd_sweep.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_tail_latency(df, pattern=''):
    """Plot tail latency analysis"""
    if pattern:
        tail_df = df[df['job_name'].str.contains(pattern, case=False)].copy()
    else:
        tail_df = df.copy()  # Use all data if no pattern specified
    
    if tail_df.empty:
        print(f"Warning: No data found for tail latency pattern '{pattern}'")
        return
    
    percentiles = ['p50', 'p95', 'p99', 'p99.9']
    percentile_labels = ['Median\n(50th)', '95th\nPercentile', '99th\nPercentile', 'Tail\n(99.9th)']
    
    plt.figure(figsize=(12, 6))  # Make it wider to accommodate longer labels
    
    plotted_any = False
    for op in ['read', 'write']:
        lat_data = []
        valid_percentiles = []
        valid_labels = []
        
        # Check which percentile columns actually exist and have valid data
        for i, p in enumerate(percentiles):
            col_name = f'{op}_{p}'
            if col_name in tail_df.columns and not tail_df[col_name].isna().all():
                # Get the mean value across all jobs for this percentile
                mean_val = tail_df[col_name].mean()
                if mean_val > 0:  # Only include positive values
                    lat_data.append(mean_val)
                    valid_percentiles.append(p)
                    valid_labels.append(percentile_labels[i])
        
        # Only plot if we have valid data
        if lat_data and valid_percentiles:
            plt.plot(range(len(valid_percentiles)), lat_data, 
                    'o-', label=f'{op.capitalize()} Latency', markersize=8, linewidth=2)
            plotted_any = True
    
    if not plotted_any:
        print(f"Warning: No valid latency percentile data found for '{pattern}' pattern")
        plt.close()
        return
    
    plt.xticks(range(len(valid_labels)), valid_labels)
    plt.xlabel('Latency Distribution Points')
    plt.ylabel('Latency (μs)')
    
    # Only use log scale if all data is positive
    if plt.gca().get_lines():
        all_data = []
        for line in plt.gca().get_lines():
            all_data.extend(line.get_ydata())
        if all(x > 0 for x in all_data):
            plt.yscale('log')
    
    plt.grid(True)
    plt.legend()
    plt.title('Tail Latency Analysis')
    
    plt.subplots_adjust(right=0.85, bottom=0.15)
    plt.savefig('tail_latency.png', bbox_inches='tight', dpi=300)
    plt.close()

def parse_fio_json(file_path):
    """Parse FIO JSON output and create all required plots"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract all job metrics into a DataFrame
    all_metrics = []
    for job in data['jobs']:
        metrics = extract_job_metrics(job)
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    
    # Create output directory for plots
    output_dir = Path(file_path).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)
    
    # Determine which plots to generate based on the data available
    file_name = Path(file_path).stem
    
    if 'baseline' in file_name.lower():
        plot_baseline_table(df)
    elif 'blocksize' in file_name.lower():
        if 'random' in file_name.lower():
            plot_blocksize_sweep(df, 'rand')  # Match job names like "4k-random"
        elif 'sequential' in file_name.lower():
            plot_blocksize_sweep(df, 'seq')   # Match job names like "4k-seq"
    elif 'rw_mix' in file_name.lower():
        plot_rw_mix(df)
    elif 'qd_sweep' in file_name.lower():
        # For QD sweep, match jobs that start with 'qd' 
        plot_qd_sweep(df, 'qd')
    elif 'tail_latency' in file_name.lower():
        # For tail latency tests, look for any job patterns
        plot_tail_latency(df, '')
    
    # Print summary statistics
    print(f"\nSummary Statistics for {file_name}:")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Only show columns that exist
    summary_cols = ['job_name']
    for col in ['read_iops', 'write_iops', 'read_mean', 'write_mean']:
        if col in df.columns:
            summary_cols.append(col)
    
    if len(summary_cols) > 1:
        print(df[summary_cols].to_string())
    else:
        print("No standard performance metrics found in data")

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_results.py <path_to_json_file>")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    if not json_file.exists():
        print(f"Error: File {json_file} does not exist")
        sys.exit(1)
    
    parse_fio_json(json_file)

if __name__ == "__main__":
    main()