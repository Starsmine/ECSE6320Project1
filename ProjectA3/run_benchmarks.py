#!/usr/bin/env python3
"""
Project A3: Benchmark Runner
Orchestrates all four required experiments and saves results for plotting
"""

import subprocess
import os
import time
from pathlib import Path

class FilterBenchRunner:
    def __init__(self, executable="./filter_bench", results_dir="results"):
        self.executable = executable
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def run_command(self, args, output_file=None):
        """Run benchmark command and optionally save output"""
        cmd = [self.executable] + args
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(result.stderr)
            print(f"  → Saved to {output_path}")
        
        return result.stdout
    
    def experiment1_space_vs_accuracy(self):
        """
        Experiment 1: Space vs Accuracy
        For each structure at three target FPRs (5%, 1%, 0.1%),
        measure bits per entry and achieved FPR
        Run 5 times for error bars
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: Space vs Accuracy")
        print("="*70 + "\n")
        
        fprs = [0.05, 0.01, 0.001]
        sizes = [1000000, 5000000]
        filters = ["bloom", "xor", "cuckoo", "quotient"]
        num_runs = 5
        
        for size in sizes:
            for fpr in fprs:
                for filt in filters:
                    for run in range(num_runs):
                        output_file = f"space_accuracy_{filt}_n{size//1000000}M_fpr{int(fpr*10000):04d}_run{run}.txt"
                        args = [
                            "--filter", filt,
                            "--size", str(size),
                            "--fpr", str(fpr),
                            "--threads", "1"
                        ]
                        self.run_command(args, output_file)
                        time.sleep(0.1)
    
    def experiment2_throughput_and_latency(self):
        """
        Experiment 2: Lookup Throughput & Tail Latency
        Queries/sec vs negative-lookup share (0% → 90%)
        Report p50/p95/p99 latency for each structure and FPR
        Run 5 times for error bars
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: Lookup Throughput & Tail Latency")
        print("="*70 + "\n")
        
        negative_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        fprs = [0.05, 0.01, 0.001]  # Test multiple FPRs (affects k hash functions)
        size = 1000000
        filters = ["bloom", "xor", "cuckoo", "quotient"]
        num_runs = 5
        
        for fpr in fprs:
            for neg_rate in negative_rates:
                for filt in filters:
                    for run in range(num_runs):
                        output_file = f"throughput_{filt}_fpr{int(fpr*10000):04d}_neg{int(neg_rate*100):02d}_run{run}.txt"
                        args = [
                            "--filter", filt,
                            "--size", str(size),
                            "--fpr", str(fpr),
                            "--negative", str(neg_rate),
                            "--threads", "1",
                            "--workload", "readonly"
                        ]
                        self.run_command(args, output_file)
                        time.sleep(0.1)
    
    def experiment3_dynamic_ops(self):
        """
        Experiment 3: Insert/Delete Throughput (Dynamic Filters Only)
        Cuckoo & Quotient: ops/s across load factors
        Record insertion failure rate and probe lengths
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: Insert/Delete Throughput (Dynamic)")
        print("="*70 + "\n")
        
        # Load factor sweep for Cuckoo and Quotient
        # Strategy: Insert varying amounts into fixed-capacity filter
        # Note: Higher load factors cause exponentially longer probe sequences
        load_factors_cuckoo = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98]
        load_factors_quotient = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # Cap at 90% (95% too slow)
        base_capacity = 500000  # Reduced capacity for faster benchmarking
        fpr = 0.01
        num_runs = 5
        
        # Run Cuckoo benchmarks
        for lf in load_factors_cuckoo:
            num_items = int(base_capacity * lf)
            for run in range(num_runs):
                output_file = f"dynamic_cuckoo_lf{int(lf*100):02d}_run{run}.txt"
                args = [
                    "--filter", "cuckoo",
                    "--size", str(num_items),  # Number of items to insert
                    "--capacity", str(base_capacity),  # Override auto-capacity
                    "--fpr", str(fpr),
                    "--workload", "balanced",  # 50% insert, 50% delete
                    "--threads", "1"
                ]
                self.run_command(args, output_file)
                time.sleep(0.1)
        
        # Run Quotient benchmarks (with lower max load factor)
        for lf in load_factors_quotient:
            num_items = int(base_capacity * lf)
            for run in range(num_runs):
                output_file = f"dynamic_quotient_lf{int(lf*100):02d}_run{run}.txt"
                args = [
                    "--filter", "quotient",
                    "--size", str(num_items),
                    "--capacity", str(base_capacity),  # Override auto-capacity
                    "--fpr", str(fpr),
                    "--workload", "balanced",  # 50% insert, 50% delete
                    "--threads", "1"
                ]
                self.run_command(args, output_file)
                time.sleep(0.1)
    
    def experiment4_thread_scaling(self):
        """
        Experiment 4: Thread Scaling
        Throughput vs threads for read-mostly and balanced workloads
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Thread Scaling")
        print("="*70 + "\n")
        
        thread_counts = [1, 2, 4, 8, 16]
        size = 5000000
        fpr = 0.01
        filters = ["bloom", "xor", "cuckoo", "quotient"]
        workloads = ["readonly", "readmostly", "balanced"]
        num_runs = 5
        
        for workload in workloads:
            for threads in thread_counts:
                for filt in filters:
                    for run in range(num_runs):
                        output_file = f"threads_{filt}_t{threads:02d}_{workload}_run{run}.txt"
                        args = [
                            "--filter", filt,
                            "--size", str(size),
                            "--fpr", str(fpr),
                            "--threads", str(threads),
                            "--workload", workload,
                            "--negative", "0.5"
                        ]
                        self.run_command(args, output_file)
                        time.sleep(0.1)
    
    def run_all_experiments(self):
        """Run all four required experiments"""
        print("\n" + "="*70)
        print("PROJECT A3: FILTER BENCHMARK SUITE")
        print("Running all required experiments...")
        print("="*70)
        
        start_time = time.time()
        
        self.experiment1_space_vs_accuracy()
        self.experiment2_throughput_and_latency()
        self.experiment3_dynamic_ops()
        self.experiment4_thread_scaling()
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f"All experiments completed in {elapsed:.1f} seconds")
        print(f"Results saved to: {self.results_dir}")
        print("="*70 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Project A3 filter benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--executable', default='./filter_bench',
                       help='Path to filter_bench executable')
    parser.add_argument('--results-dir', default='results',
                       help='Directory to save results')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3, 4],
                       help='Run specific experiment (1-4), or all if not specified')
    
    args = parser.parse_args()
    
    runner = FilterBenchRunner(args.executable, args.results_dir)
    
    if args.experiment == 1:
        runner.experiment1_space_vs_accuracy()
    elif args.experiment == 2:
        runner.experiment2_throughput_and_latency()
    elif args.experiment == 3:
        runner.experiment3_dynamic_ops()
    elif args.experiment == 4:
        runner.experiment4_thread_scaling()
    else:
        runner.run_all_experiments()

if __name__ == "__main__":
    main()
