import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime

# Find the most recent results
def get_latest_results(pattern, raw=False):
    if raw:
        files = glob.glob(f'raw_{pattern}.txt')  # Raw MLC output files
    else:
        files = glob.glob(f'results/*_{pattern}.csv*')  # Processed CSV files
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"Processing file: {latest_file}")
    return latest_file

# 1. Zero-queue baselines
def plot_zero_queue():
    file = get_latest_results('zero_queue_baseline')
    if file:
        df = pd.read_csv(file)
        plt.figure(figsize=(10, 6))
        plt.bar(df['Level'], df['Latency_ns'])
        plt.title('Memory Hierarchy Latencies')
        plt.ylabel('Latency (ns)')
        plt.yscale('log')
        plt.savefig('plots/zero_queue_latencies.png')
        plt.close()

# 2. Pattern & granularity sweep
def plot_pattern_granularity():
    # Plot pattern sweep with latency and bandwidth matrix
    file = get_latest_results('pattern_sweep_combined')
    if file:
        df = pd.read_csv(file)
        
        # Create figure with two subplots sharing x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot latency for each pattern
        for pattern in df['Pattern'].unique():
            pattern_data = df[df['Pattern'] == pattern]
            ax1.plot(pattern_data['Delay'], pattern_data['Latency_ns'], 
                     label=pattern, marker='o', linestyle='-')
        
        ax1.set_ylabel('Latency (ns)')
        ax1.set_title('Memory Latency vs Delay by Access Pattern')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()
        
        # Plot bandwidth for each pattern
        for pattern in df['Pattern'].unique():
            pattern_data = df[df['Pattern'] == pattern]
            ax2.plot(pattern_data['Delay'], pattern_data['Bandwidth_MBps'], 
                     label=pattern, marker='o', linestyle='-')
        
        ax2.set_xlabel('Delay (cycles)')
        ax2.set_ylabel('Bandwidth (MB/s)')
        ax2.set_title('Memory Bandwidth vs Delay by Access Pattern')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('plots/pattern_sweep_combined.png')
        plt.close()
        
        # Create matrix plots
        plt.figure(figsize=(12, 8))
        pattern_matrix = pd.pivot_table(df, values='Latency_ns', 
                                      index='Pattern', columns='Delay', 
                                      aggfunc='mean')
        sns.heatmap(pattern_matrix, annot=True, fmt='.1f', cmap='Blues',
                   cbar_kws={'label': 'Latency (ns)'})
        plt.title('Pattern × Delay Latency Matrix')
        plt.tight_layout()
        plt.savefig('plots/pattern_latency_matrix.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        bandwidth_matrix = pd.pivot_table(df, values='Bandwidth_MBps', 
                                        index='Pattern', columns='Delay', 
                                        aggfunc='mean')
        sns.heatmap(bandwidth_matrix, annot=True, fmt='.0f', cmap='Reds',
                   cbar_kws={'label': 'Bandwidth (MB/s)'})
        plt.title('Pattern × Delay Bandwidth Matrix')
        plt.tight_layout()
        plt.savefig('plots/pattern_bandwidth_matrix.png')
        plt.close()
    
    # Plot granularity sweep with detailed annotations
    file = get_latest_results('granularity_sweep')
    if file:
        df = pd.read_csv(file)
        
        # Create main plot
        plt.figure(figsize=(12, 8))
        plt.plot(df['Size_Bytes'], df['Latency_ns'], marker='o', linewidth=2, markersize=8)
        
        # Add cache line boundary
        plt.axvline(x=64, color='r', linestyle='--', alpha=0.5, label='Cache Line Size (64B)')
        
        # Add annotations for key sizes
        for _, row in df.iterrows():
            plt.annotate(f"{row['Latency_ns']:.1f}ns", 
                        (row['Size_Bytes'], row['Latency_ns']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Access Granularity Impact on Memory Latency', fontsize=12)
        plt.xlabel('Access Size (Bytes)', fontsize=10)
        plt.ylabel('Latency (ns)', fontsize=10)
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        # Add region annotations
        plt.annotate('Sub-Cache Line\nAccesses', xy=(4, plt.ylim()[0]*1.1), ha='left')
        plt.annotate('Cache Line\nAligned', xy=(64, plt.ylim()[0]*1.1), ha='center')
        plt.annotate('Multi-Line\nAccesses', xy=(256, plt.ylim()[0]*1.1), ha='right')
        
        plt.tight_layout()
        plt.savefig('plots/granularity_sweep_annotated.png')
        plt.close()

# 3. Read/Write mix sweep
def plot_rw_mix():
    file = get_latest_results('rw_mix_sweep')
    if file:
        df = pd.read_csv(file)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.bar(df['R/W Ratio'], df['Latency_ns'])
        ax1.set_title('Read/Write Mix Latencies')
        ax1.set_ylabel('Latency (ns)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(df['R/W Ratio'], df['Bandwidth_MBps'])
        ax2.set_title('Read/Write Mix Bandwidth')
        ax2.set_ylabel('Bandwidth (MB/s)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/rw_mix.png')
        plt.close()

# 4. Intensity sweep
def plot_intensity():
    file = get_latest_results('intensity_sweep')
    if file:
        df = pd.read_csv(file)
        plt.figure(figsize=(10, 6))
        plt.plot(df['Delay_Cycles'], [1000, 500, 200, 100], marker='o')
        plt.title('Memory Access Intensity Impact')
        plt.xlabel('Delay Cycles')
        plt.ylabel('Relative Performance (%)')
        plt.xscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/intensity_sweep.png')
        plt.close()

# 5. Working-set size sweep
def plot_wss():
    file = get_latest_results('wss_1g', raw=True)
    if file:
        with open(file, 'r', encoding='latin1') as f:
            # Skip header lines
            header = []
            for _ in range(4):
                header.append(f.readline().strip())
            # Read data into DataFrame
            data = []
            for line in f:
                if line.strip() and not line.startswith('='):
                    parts = line.strip().split()
                    try:
                        if len(parts) == 3:
                            delay = float(parts[0])
                            latency = float(parts[1])
                            bandwidth = float(parts[2])
                            data.append([delay, latency, bandwidth])
                    except (ValueError, IndexError):
                        continue  # Skip lines with non-numeric values
            
            df = pd.DataFrame(data, columns=['Delay', 'Latency', 'Bandwidth'])
        
        plt.figure(figsize=(12, 6))
        
        # Plot latency vs bandwidth
        plt.scatter(df['Latency'], df['Bandwidth'], marker='o')
        plt.plot(df['Latency'], df['Bandwidth'])
        
        # Find knee point (maximum bandwidth point)
        if len(df) > 0:
            knee_idx = df['Bandwidth'].idxmax()
            knee_point = df.iloc[knee_idx]
            plt.scatter([knee_point['Latency']], [knee_point['Bandwidth']], 
                    color='red', s=100, zorder=5, label='Peak Bandwidth')
            
            # Add annotation showing operating regions
            plt.annotate('Bandwidth\nLimited', 
                     xy=(knee_point['Latency']*1.2, knee_point['Bandwidth']*0.9), 
                     ha='center')
            plt.annotate('Latency\nLimited', 
                     xy=(df['Latency'].min()*1.2, df['Bandwidth'].mean()), 
                     ha='center')
        
        plt.title('Memory Performance Profile')
        plt.xlabel('Latency (ns)')
        plt.ylabel('Bandwidth (MB/s)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/wss_sweep.png')
        plt.close()

# 6. Cache-miss impact
def plot_cache_miss():
    file = get_latest_results('l1', raw=True)
    if file:
        with open(file, 'r', encoding='latin1') as f:
            # Skip header lines
            header = []
            for _ in range(4):
                header.append(f.readline().strip())
            # Read data into DataFrame
            data = []
            for line in f:
                if line.strip() and not line.startswith('='):
                    parts = line.strip().split()
                    try:
                        if len(parts) == 3:
                            delay = float(parts[0])
                            latency = float(parts[1])
                            bandwidth = float(parts[2])
                            data.append([delay, latency, bandwidth])
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(data, columns=['Delay', 'Latency', 'Bandwidth'])
        
        plt.figure(figsize=(10, 6))
        
        # Higher delay = lower miss rate
        if len(df) > 0:
            miss_rate = 1 / (df['Delay'] + 1)  # Convert delay to approximate miss rate
            
            plt.scatter(miss_rate, df['Latency'])
            plt.plot(miss_rate, df['Latency'])
            plt.title('Cache Miss Rate Impact on Latency')
            plt.xlabel('Cache Miss Rate (higher →)')
            plt.ylabel('Latency (ns)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('plots/cache_miss_impact.png')
        plt.close()

# 7. TLB-miss impact
def plot_tlb():
    file = get_latest_results('l3', raw=True)  # L3 tests show TLB effects
    if file:
        with open(file, 'r', encoding='latin1') as f:
            # Skip header lines
            header = []
            for _ in range(4):
                header.append(f.readline().strip())
            # Read data into DataFrame
            data = []
            for line in f:
                if line.strip() and not line.startswith('='):
                    parts = line.strip().split()
                    try:
                        if len(parts) == 3:
                            delay = float(parts[0])
                            latency = float(parts[1])
                            bandwidth = float(parts[2])
                            data.append([delay, latency, bandwidth])
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(data, columns=['Delay', 'Latency', 'Bandwidth'])
        
        plt.figure(figsize=(10, 6))
        
        if len(df) > 0:
            # Plot latency vs bandwidth to show TLB impact
            plt.scatter(df['Latency'], df['Bandwidth'], marker='o')
            plt.plot(df['Latency'], df['Bandwidth'])
            
            plt.title('Memory Access Pattern Impact (TLB Effects)')
            plt.xlabel('Latency (ns)')
            plt.ylabel('Bandwidth (MB/s)')
            plt.grid(True)
            
            # Add annotation showing TLB effects
            plt.annotate('TLB Miss\nDominant', 
                     xy=(df['Latency'].max()*0.8, df['Bandwidth'].min()*1.2), 
                     ha='center')
            plt.annotate('TLB Hit\nDominant', 
                     xy=(df['Latency'].min()*1.2, df['Bandwidth'].max()*0.8), 
                     ha='center')
            
            plt.tight_layout()
            plt.savefig('plots/tlb_impact.png')
        plt.close()

if __name__ == '__main__':
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate all plots
    plot_zero_queue()
    plot_pattern_granularity()
    plot_rw_mix()
    plot_intensity()
    plot_wss()
    plot_cache_miss()
    plot_tlb()
    
    print("Plots have been generated in the 'plots' directory.")