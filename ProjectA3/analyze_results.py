#!/usr/bin/env python3
"""
Project A3: Analysis and Plotting
Generates all four required plots from benchmark results
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

class FilterAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.filter_colors = {
            'bloom': '#2E86AB',
            'xor': '#A23B72',
            'cuckoo': '#F18F01',
            'quotient': '#06A77D'
        }
        self.filter_markers = {
            'bloom': 'o',
            'xor': 's',
            'cuckoo': '^',
            'quotient': 'D'
        }
    
    def parse_result_file(self, filepath):
        """Parse a single benchmark result file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().replace('\r\n', '\n').replace('\r', '\n')
        
        result = {}
        
        # Extract filter name
        for filt in ['Blocked Bloom', 'XOR', 'Cuckoo', 'Quotient']:
            if f'{filt} Filter:' in content:
                result['filter'] = filt.replace('Blocked ', '').lower()
                break
        
        # Extract metrics
        patterns = {
            'size': r'Set size:\s*(\d+)',
            'target_fpr': r'Target FPR:\s*([\d.]+)%',
            'measured_fpr': r'Measured FPR:\s*([\d.]+)%',
            'bpe': r'Bits per entry:\s*([\d.]+)',
            'throughput': r'Throughput:\s*([\d.]+)\s*Mops/s',
            'p50': r'Latency p50:\s*([\d.]+)\s*ns',
            'p95': r'Latency p95:\s*([\d.]+)\s*ns',
            'p99': r'Latency p99:\s*([\d.]+)\s*ns',
            'load_factor': r'Load factor:\s*([\d.]+)%',
            'insertion_failures': r'Insertion failures:\s*(\d+)',
            'stash_size': r'Stash size:\s*(\d+)',
            'threads': r'Threads:\s*(\d+)',
            'negative_rate': r'Negative rate:\s*([\d.]+)%',
            'avg_kicks': r'Average kicks per insertion:\s*([\d.]+)',
            'avg_probes': r'Average probes per query:\s*([\d.]+)',
            'stash_hit_rate': r'Stash hit rate:\s*([\d.]+)%',
            'dynamic_throughput': r'Dynamic Operations Throughput:\s*([\d.]+)\s*Mops/s',
            'insert_count': r'Insert operations:\s*(\d+)',
            'delete_count': r'Delete operations:\s*(\d+)',
            'insert_failure_rate': r'Insert failure rate:\s*([\d.]+)%'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                result[key] = float(match.group(1))
        
        # Extract cluster length histogram (Quotient filter)
        if 'Cluster length histogram:' in content:
            result['cluster_histogram'] = {}
            # Match lines like "    Length 1: 528511 (48.0465%)"
            for match in re.finditer(r'Length (\d+):\s*(\d+)', content):
                length = int(match.group(1))
                count = int(match.group(2))
                result['cluster_histogram'][length] = count
        
        return result if result else None
    
    def plot1_space_vs_accuracy(self, output_file="plot1_space_vs_accuracy.png"):
        """Plot 1: Space efficiency and FPR accuracy"""
        print("Generating Plot 1: Space vs Accuracy...")
        
        files = sorted(self.results_dir.glob("space_accuracy_*.txt"))
        if not files:
            print(f"  No files found matching 'space_accuracy_*.txt'")
            return
        
        # Organize data by filter and FPR
        data = {}
        for filepath in files:
            result = self.parse_result_file(filepath)
            if result and 'filter' in result:
                filt = result['filter']
                if filt not in data:
                    data[filt] = {}
                
                if 'target_fpr' in result:
                    target_fpr = result['target_fpr'] / 100.0  # Convert % to decimal
                    if target_fpr not in data[filt]:
                        data[filt][target_fpr] = {'bpe': [], 'achieved_fpr': []}
                    
                    if 'bpe' in result:
                        data[filt][target_fpr]['bpe'].append(result['bpe'])
                    if 'measured_fpr' in result:
                        data[filt][target_fpr]['achieved_fpr'].append(result['measured_fpr'] / 100.0)
        
        if not data:
            print("  No data found!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Bits Per Entry vs Target FPR
        for filter_type in ['bloom', 'xor', 'cuckoo', 'quotient']:
            if filter_type in data:
                fprs = sorted(data[filter_type].keys())
                # Average BPE and std error for each FPR
                bpes = []
                bpe_errors = []
                for fpr in fprs:
                    if data[filter_type][fpr]['bpe']:
                        bpe_values = data[filter_type][fpr]['bpe']
                        bpes.append(np.mean(bpe_values))
                        bpe_errors.append(np.std(bpe_values, ddof=1) if len(bpe_values) > 1 else 0)
                
                if bpes:
                    ax1.errorbar([f*100 for f in fprs], bpes, yerr=bpe_errors,
                            marker=self.filter_markers.get(filter_type, 'o'),
                            color=self.filter_colors.get(filter_type, 'gray'),
                            linewidth=2, markersize=8, capsize=5,
                            label=filter_type.capitalize())
        
        # Add theoretical Bloom filter line
        theory_fprs = np.logspace(-3, -1, 50)  # 0.1% to 10%
        theory_bpe = -np.log2(theory_fprs) * 1.44  # m/n = -ln(ε) / ln(2)
        ax1.plot(theory_fprs * 100, theory_bpe, '--', 
                color='gray', linewidth=2, label='Bloom (theory)', alpha=0.7)
        
        ax1.set_xlabel('Target FPR (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Bits Per Entry', fontsize=12, fontweight='bold')
        ax1.set_title('Space Efficiency: Bits Per Entry vs Target FPR', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Right plot: Achieved FPR vs Target FPR
        for filter_type in ['bloom', 'xor', 'cuckoo', 'quotient']:
            if filter_type in data:
                fprs = sorted(data[filter_type].keys())
                # Average achieved FPR and std error for each target FPR
                achieved = []
                achieved_errors = []
                for fpr in fprs:
                    if data[filter_type][fpr]['achieved_fpr']:
                        fpr_values = data[filter_type][fpr]['achieved_fpr']
                        achieved.append(np.mean(fpr_values))
                        achieved_errors.append(np.std(fpr_values, ddof=1) if len(fpr_values) > 1 else 0)
                
                if achieved:
                    ax2.errorbar([f*100 for f in fprs], [a*100 for a in achieved], yerr=[e*100 for e in achieved_errors],
                            marker=self.filter_markers.get(filter_type, 'o'),
                            color=self.filter_colors.get(filter_type, 'gray'),
                            linewidth=2, markersize=8, capsize=5,
                            label=filter_type.capitalize())
        
        # Add perfect accuracy line (diagonal)
        perfect_range = [0.1, 0.5, 1, 5, 10]
        ax2.plot(perfect_range, perfect_range, '--', color='gray', linewidth=2, 
                label='Perfect (achieved = target)', alpha=0.7)
        
        ax2.set_xlabel('Target FPR (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Achieved FPR (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy: Achieved vs Target FPR', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def plot2_throughput_vs_negative_rate(self, output_file="plot2_throughput_latency.png"):
        """
        Plot 2: Lookup Throughput & Tail Latency vs Negative Rate
        """
        print("Generating Plot 2: Throughput & Latency vs Negative Rate...")
        
        files = sorted(self.results_dir.glob("throughput_*.txt"))
        if not files:
            print(f"  No files found matching 'throughput_*.txt'")
            return
        
        # Group data by FPR, filter, and negative rate for multiple runs
        raw_data = {}
        for filepath in files:
            result = self.parse_result_file(filepath)
            if result and 'filter' in result and 'negative_rate' in result:
                filt = result['filter']
                neg_rate = result['negative_rate']
                target_fpr = result.get('target_fpr', 1.0)  # Default to 1% if not found
                
                if target_fpr not in raw_data:
                    raw_data[target_fpr] = {}
                if filt not in raw_data[target_fpr]:
                    raw_data[target_fpr][filt] = {}
                if neg_rate not in raw_data[target_fpr][filt]:
                    raw_data[target_fpr][filt][neg_rate] = {'throughput': [], 'p50': [], 'p95': [], 'p99': []}
                
                raw_data[target_fpr][filt][neg_rate]['throughput'].append(result.get('throughput', 0))
                raw_data[target_fpr][filt][neg_rate]['p50'].append(result.get('p50', 0))
                raw_data[target_fpr][filt][neg_rate]['p95'].append(result.get('p95', 0))
                raw_data[target_fpr][filt][neg_rate]['p99'].append(result.get('p99', 0))
        
        # Compute means and std errors
        data = {}
        for fpr in raw_data:
            data[fpr] = {}
            for filt in raw_data[fpr]:
                data[fpr][filt] = {'neg_rate': [], 'throughput': [], 'throughput_err': [],
                                   'p50': [], 'p50_err': [], 'p95': [], 'p95_err': [], 
                                   'p99': [], 'p99_err': []}
                for neg_rate in sorted(raw_data[fpr][filt].keys()):
                    data[fpr][filt]['neg_rate'].append(neg_rate)
                    for metric in ['throughput', 'p50', 'p95', 'p99']:
                        values = raw_data[fpr][filt][neg_rate][metric]
                        data[fpr][filt][metric].append(np.mean(values))
                        data[fpr][filt][f'{metric}_err'].append(np.std(values, ddof=1) if len(values) > 1 else 0)
        
        if not data:
            print("  No data found!")
            return
        
        # Create 3×2 subplot grid (3 FPRs × 2 metrics)
        fprs = sorted(data.keys(), reverse=True)  # [5.0, 1.0, 0.1]
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        for row, fpr in enumerate(fprs):
            ax_throughput = axes[row, 0]
            ax_latency = axes[row, 1]
            
            # Throughput subplot
            for filt in sorted(data[fpr].keys()):
                values = data[fpr][filt]
                if values['neg_rate'] and values['throughput']:
                    ax_throughput.errorbar(values['neg_rate'], values['throughput'], 
                            yerr=values['throughput_err'],
                            marker=self.filter_markers.get(filt, 'o'),
                            color=self.filter_colors.get(filt, 'gray'),
                            linewidth=2, markersize=8, capsize=5,
                            label=filt.capitalize())
            
            ax_throughput.set_xlabel('Negative Query Rate (%)', fontsize=11, fontweight='bold')
            ax_throughput.set_ylabel('Throughput (Mops/s)', fontsize=11, fontweight='bold')
            ax_throughput.set_title(f'Throughput @ {fpr:.1f}% FPR', fontsize=12, fontweight='bold')
            ax_throughput.legend(fontsize=10)
            ax_throughput.grid(True, alpha=0.3)
            
            # Latency subplot
            for filt in sorted(data[fpr].keys()):
                values = data[fpr][filt]
                if values['neg_rate'] and values['p99']:
                    color = self.filter_colors.get(filt, 'gray')
                    marker = self.filter_markers.get(filt, 'o')
                    
                    # p50 with solid line (thinner)
                    ax_latency.errorbar(values['neg_rate'], values['p50'], yerr=values['p50_err'],
                            marker=marker, color=color,
                            linewidth=1.5, markersize=6, capsize=4, alpha=0.9,
                            label=f'{filt.capitalize()} p50')
                    
                    # p95 with solid line
                    ax_latency.errorbar(values['neg_rate'], values['p95'], yerr=values['p95_err'],
                            marker=marker, color=color,
                            linewidth=2, markersize=7, capsize=5,
                            label=f'{filt.capitalize()} p95')
                    
                    # p99 with dashed line
                    ax_latency.errorbar(values['neg_rate'], values['p99'], yerr=values['p99_err'],
                            marker=marker, color=color,
                            linewidth=2, markersize=5, linestyle='--', capsize=4, alpha=0.7,
                            label=f'{filt.capitalize()} p99')
            
            ax_latency.set_xlabel('Negative Query Rate (%)', fontsize=11, fontweight='bold')
            ax_latency.set_ylabel('Latency (ns)', fontsize=11, fontweight='bold')
            ax_latency.set_title(f'Latency (p50/p95/p99) @ {fpr:.1f}% FPR', fontsize=12, fontweight='bold')
            ax_latency.legend(fontsize=8, ncol=3)
            ax_latency.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def plot3_dynamic_load_factor(self, output_file="plot3_dynamic_ops.png"):
        """
        Plot 3: Insert/Delete Performance vs Load Factor (Dynamic Filters)
        """
        print("Generating Plot 3: Dynamic Operations...")
        
        files = sorted(self.results_dir.glob("dynamic_*.txt"))
        if not files:
            print(f"  No files found matching 'dynamic_*.txt'")
            return
        
        # Group data by filter and load factor for multiple runs
        raw_data = {'cuckoo': {}, 'quotient': {}}
        
        for filepath in files:
            result = self.parse_result_file(filepath)
            if result and 'filter' in result and 'load_factor' in result:
                filt = result['filter']
                if filt in raw_data:
                    lf = result['load_factor']
                    if lf not in raw_data[filt]:
                        raw_data[filt][lf] = {'throughput': [], 'failures': [], 
                                              'avg_kicks': [], 'avg_probes': [], 'clusters': []}
                    
                    # Use dynamic throughput if available, otherwise regular throughput
                    throughput = result.get('dynamic_throughput', result.get('throughput', 0))
                    raw_data[filt][lf]['throughput'].append(throughput)
                    
                    if filt == 'cuckoo':
                        # Use insert_failure_rate if available, otherwise calculate from insertion_failures
                        failure_rate = result.get('insert_failure_rate', 0)
                        if failure_rate == 0 and result.get('insertion_failures', 0) > 0:
                            # Calculate rate from absolute failures
                            insert_count = result.get('insert_count', result.get('size', 1))
                            failure_rate = (result.get('insertion_failures', 0) / insert_count) * 100
                        raw_data[filt][lf]['failures'].append(failure_rate)
                        raw_data[filt][lf]['avg_kicks'].append(result.get('avg_kicks', 0))
                        raw_data[filt][lf].setdefault('stash_hit_rate', []).append(result.get('stash_hit_rate', 0))
                    
                    if filt == 'quotient':
                        raw_data[filt][lf]['avg_probes'].append(result.get('avg_probes', 0))
                        if 'cluster_histogram' in result:
                            raw_data[filt][lf]['clusters'].append(result['cluster_histogram'])
        
        # Compute means and std errors
        data = {'cuckoo': {'lf': [], 'throughput': [], 'throughput_err': [], 
                          'failures': [], 'failures_err': [], 'avg_kicks': [], 'avg_kicks_err': [],
                          'stash_hit_rate': [], 'stash_hit_rate_err': []},
                'quotient': {'lf': [], 'throughput': [], 'throughput_err': [], 
                           'avg_probes': [], 'avg_probes_err': [], 'clusters': {}}}
        
        for filt in raw_data:
            for lf in sorted(raw_data[filt].keys()):
                data[filt]['lf'].append(lf)
                
                # Throughput
                tput = raw_data[filt][lf]['throughput']
                data[filt]['throughput'].append(np.mean(tput))
                data[filt]['throughput_err'].append(np.std(tput, ddof=1) if len(tput) > 1 else 0)
                
                if filt == 'cuckoo':
                    fails = raw_data[filt][lf]['failures']
                    kicks = raw_data[filt][lf]['avg_kicks']
                    stash = raw_data[filt][lf].get('stash_hit_rate', [0])
                    data[filt]['failures'].append(np.mean(fails))
                    data[filt]['failures_err'].append(np.std(fails, ddof=1) if len(fails) > 1 else 0)
                    data[filt]['avg_kicks'].append(np.mean(kicks))
                    data[filt]['avg_kicks_err'].append(np.std(kicks, ddof=1) if len(kicks) > 1 else 0)
                    data[filt]['stash_hit_rate'].append(np.mean(stash))
                    data[filt]['stash_hit_rate_err'].append(np.std(stash, ddof=1) if len(stash) > 1 else 0)
                
                if filt == 'quotient':
                    probes = raw_data[filt][lf]['avg_probes']
                    data[filt]['avg_probes'].append(np.mean(probes))
                    data[filt]['avg_probes_err'].append(np.std(probes, ddof=1) if len(probes) > 1 else 0)
                    # Use first run's cluster histogram (they should be similar)
                    if raw_data[filt][lf]['clusters']:
                        data[filt]['clusters'][lf] = raw_data[filt][lf]['clusters'][0]
        
        # Create plot with 2x3 subplots
        fig = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        
        # Subplot 1: Throughput vs Load Factor
        for filt in ['cuckoo', 'quotient']:
            if data[filt]['lf'] and data[filt]['throughput']:
                ax1.errorbar(data[filt]['lf'], data[filt]['throughput'], 
                        yerr=data[filt]['throughput_err'],
                        marker=self.filter_markers.get(filt, 'o'),
                        color=self.filter_colors.get(filt, 'gray'),
                        linewidth=2, markersize=8, capsize=5,
                        label=filt.capitalize())
        
        ax1.set_xlabel('Load Factor (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Throughput (Mops/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Throughput vs Load Factor', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Average probes/kicks per operation (dual y-axes)
        ax2_right = ax2.twinx()  # Create secondary y-axis
        
        # Plot Quotient probes on left y-axis
        if data['quotient']['lf'] and data['quotient']['avg_probes']:
            line1 = ax2.errorbar(data['quotient']['lf'], data['quotient']['avg_probes'],
                    yerr=data['quotient']['avg_probes_err'],
                    marker=self.filter_markers['quotient'],
                    color=self.filter_colors['quotient'],
                    linewidth=2, markersize=8, capsize=5,
                    label='Quotient (probes/query)')
            ax2.set_ylabel('Avg Probes per Query (Quotient)', fontsize=12, fontweight='bold', 
                          color=self.filter_colors['quotient'])
            ax2.tick_params(axis='y', labelcolor=self.filter_colors['quotient'])
        
        # Plot Cuckoo kicks on right y-axis
        if data['cuckoo']['lf'] and data['cuckoo']['avg_kicks']:
            line2 = ax2_right.errorbar(data['cuckoo']['lf'], data['cuckoo']['avg_kicks'],
                    yerr=data['cuckoo']['avg_kicks_err'],
                    marker=self.filter_markers['cuckoo'],
                    color=self.filter_colors['cuckoo'],
                    linewidth=2, markersize=8, capsize=5,
                    label='Cuckoo (kicks/insert)')
            ax2_right.set_ylabel('Avg Kicks per Insertion (Cuckoo)', fontsize=12, fontweight='bold',
                                color=self.filter_colors['cuckoo'])
            ax2_right.tick_params(axis='y', labelcolor=self.filter_colors['cuckoo'])
        
        ax2.set_xlabel('Load Factor (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Probe/Kick Statistics', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Cuckoo insertion failure rate vs load factor
        if data['cuckoo']['lf'] and data['cuckoo']['failures']:
            has_failures = any(f > 0 for f in data['cuckoo']['failures'])
            if has_failures:
                ax3.errorbar(data['cuckoo']['lf'], data['cuckoo']['failures'],
                        yerr=data['cuckoo']['failures_err'],
                        marker=self.filter_markers['cuckoo'],
                        color=self.filter_colors['cuckoo'],
                        linewidth=2, markersize=8, capsize=5)
                # Use linear scale with percentage formatting
                ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
            else:
                ax3.bar(data['cuckoo']['lf'], [1] * len(data['cuckoo']['lf']),
                       color=self.filter_colors['cuckoo'], alpha=0.3)
                ax3.set_ylim([0, 2])
                ax3.text(0.5, 0.5, 'No insertion failures\n(all items placed successfully)', 
                        ha='center', va='center', fontsize=11, transform=ax3.transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_xlabel('Load Factor (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Insertion Failure Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Cuckoo: Insertion Failure Rate', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Cuckoo stash hit rate vs load factor
        if data['cuckoo']['lf'] and data['cuckoo']['stash_hit_rate']:
            has_stash_hits = any(s > 0 for s in data['cuckoo']['stash_hit_rate'])
            if has_stash_hits:
                ax4.errorbar(data['cuckoo']['lf'], data['cuckoo']['stash_hit_rate'],
                        yerr=data['cuckoo']['stash_hit_rate_err'],
                        marker=self.filter_markers['cuckoo'],
                        color=self.filter_colors['cuckoo'],
                        linewidth=2, markersize=8, capsize=5)
            else:
                ax4.bar(data['cuckoo']['lf'], [0.1] * len(data['cuckoo']['lf']),
                       color=self.filter_colors['cuckoo'], alpha=0.3)
                ax4.set_ylim([0, 1])
                ax4.text(0.5, 0.5, 'No stash usage\n(all items in main buckets)', 
                        ha='center', va='center', fontsize=11, transform=ax4.transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.set_xlabel('Load Factor (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Stash Hit Rate (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Cuckoo: Stash Hit Rate', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Quotient cluster length histogram at different load factors
        if data['quotient']['clusters']:
            # Show histogram for 3 load factors: low, medium, high
            load_factors = sorted(data['quotient']['clusters'].keys())
            if len(load_factors) >= 3:
                selected_lfs = [load_factors[0], load_factors[len(load_factors)//2], load_factors[-1]]
            else:
                selected_lfs = load_factors
            
            # Use different colors for different load factors
            colors = ['#2ca02c', '#ff7f0e', '#d62728']  # green, orange, red
            for i, lf in enumerate(selected_lfs):
                cluster_hist = data['quotient']['clusters'][lf]
                lengths = sorted(cluster_hist.keys())
                counts = [cluster_hist[l] for l in lengths]
                ax5.bar([l + i*0.25 for l in lengths], counts, alpha=0.6, 
                       color=colors[i % len(colors)],
                       label=f'{lf:.0f}% load', width=0.25)
            
            ax5.set_xlabel('Cluster Length (probes)', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax5.set_title('Quotient: Cluster Length Histogram', fontsize=13, fontweight='bold')
            ax5.legend(fontsize=11)
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.set_yscale('log')
        else:
            ax5.text(0.5, 0.5, 'No cluster data available', 
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def plot4_thread_scaling(self, output_file="plot4_thread_scaling.png"):
        """
        Plot 4: Thread Scaling
        """
        print("Generating Plot 4: Thread Scaling...")
        
        files = sorted(self.results_dir.glob("threads_*.txt"))
        if not files:
            print(f"  No files found matching 'threads_*.txt'")
            return
        
        # Group data by filter and thread count for multiple runs
        raw_data = {}
        for filepath in files:
            result = self.parse_result_file(filepath)
            if result and 'filter' in result and 'threads' in result:
                filt = result['filter']
                thread_count = result['threads']
                
                if filt not in raw_data:
                    raw_data[filt] = {}
                if thread_count not in raw_data[filt]:
                    raw_data[filt][thread_count] = []
                
                if 'throughput' in result:
                    raw_data[filt][thread_count].append(result['throughput'])
        
        # Compute means and std errors
        data = {}
        for filt in raw_data:
            data[filt] = {'threads': [], 'throughput': [], 'throughput_err': []}
            for thread_count in sorted(raw_data[filt].keys()):
                tput = raw_data[filt][thread_count]
                data[filt]['threads'].append(thread_count)
                data[filt]['throughput'].append(np.mean(tput))
                data[filt]['throughput_err'].append(np.std(tput, ddof=1) if len(tput) > 1 else 0)
        
        if not data:
            print("  No data found!")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate speedups (relative to 1 thread)
        for filt, values in sorted(data.items()):
            if values['threads'] and values['throughput']:
                threads = values['threads']
                throughputs = values['throughput']
                throughput_errs = values['throughput_err']
                
                # Absolute throughput with error bars
                ax1.errorbar(threads, throughputs, yerr=throughput_errs,
                        marker=self.filter_markers.get(filt, 'o'),
                        color=self.filter_colors.get(filt, 'gray'),
                        linewidth=2, markersize=8, capsize=5,
                        label=filt.capitalize())
                
                # Speedup relative to 1 thread
                if threads[0] == 1:
                    baseline = throughputs[0]
                    baseline_err = throughput_errs[0]
                    speedups = [t / baseline for t in throughputs]
                    # Error propagation for division: err(a/b) = |a/b| * sqrt((err_a/a)^2 + (err_b/b)^2)
                    speedup_errs = []
                    for t, t_err in zip(throughputs, throughput_errs):
                        if baseline > 0 and t > 0:
                            rel_err = np.sqrt((t_err/t)**2 + (baseline_err/baseline)**2) if t_err > 0 or baseline_err > 0 else 0
                            speedup_errs.append(abs(t/baseline) * rel_err)
                        else:
                            speedup_errs.append(0)
                    
                    ax2.errorbar(threads, speedups, yerr=speedup_errs,
                            marker=self.filter_markers.get(filt, 'o'),
                            color=self.filter_colors.get(filt, 'gray'),
                            linewidth=2, markersize=8, capsize=5,
                            label=filt.capitalize())
        
        # Ideal linear scaling
        max_threads = int(max([max(d['threads']) for d in data.values() if d['threads']]))
        ideal_threads = list(range(1, max_threads + 1))
        ax2.plot(ideal_threads, ideal_threads, 'k--', linewidth=1.5, 
                alpha=0.5, label='Ideal (linear)')
        
        ax1.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Throughput (Mops/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Absolute Throughput', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        ax2.set_xlabel('Thread Count', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Speedup vs 1 Thread', fontsize=12, fontweight='bold')
        ax2.set_title('Thread Scaling Efficiency', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all four required plots"""
        print("\n" + "="*70)
        print("GENERATING ALL PLOTS")
        print("="*70 + "\n")
        
        self.plot1_space_vs_accuracy()
        self.plot2_throughput_vs_negative_rate()
        self.plot3_dynamic_load_factor()
        self.plot4_thread_scaling()
        
        print("\n" + "="*70)
        print("All plots generated successfully!")
        print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot Project A3 filter benchmarks"
    )
    
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing benchmark results')
    parser.add_argument('--plot', type=int, choices=[1, 2, 3, 4],
                       help='Generate specific plot (1-4), or all if not specified')
    
    args = parser.parse_args()
    
    analyzer = FilterAnalyzer(args.results_dir)
    
    if args.plot == 1:
        analyzer.plot1_space_vs_accuracy()
    elif args.plot == 2:
        analyzer.plot2_throughput_vs_negative_rate()
    elif args.plot == 3:
        analyzer.plot3_dynamic_load_factor()
    elif args.plot == 4:
        analyzer.plot4_thread_scaling()
    else:
        analyzer.generate_all_plots()

if __name__ == "__main__":
    main()
