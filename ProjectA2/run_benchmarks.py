#!/usr/bin/env python3
"""
Benchmark runner script for Dense vs Sparse Matrix Multiplication
Generates comprehensive sweeps and organizes results
"""

import subprocess
import sys
import os
import time
import shutil
from pathlib import Path
import argparse

class BenchmarkRunner:
    def __init__(self, executable="./matrix_bench"):
        # Handle Windows filesystem mount - copy to /tmp for execution
        if not os.access(executable, os.X_OK):
            print(f"Warning: {executable} not executable (likely Windows filesystem)")
            print("Copying to /tmp for execution...")
            tmp_exec = f"/tmp/matrix_bench_{os.getpid()}"
            shutil.copy2(executable, tmp_exec)
            os.chmod(tmp_exec, 0o755)
            self.executable = tmp_exec
            self.cleanup_tmp = True
        else:
            self.executable = executable
            self.cleanup_tmp = False
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_command(self, args, output_name=None, threads=None):
        """Run benchmark with given arguments
        
        Args:
            args: Command line arguments
            output_name: Name for the output file
            threads: Number of OpenMP threads (None = default, 1 = single-threaded)
        """
        cmd = [self.executable] + args
        env = os.environ.copy()
        
        if threads is not None:
            env['OMP_NUM_THREADS'] = str(threads)
            print(f"\n{'='*70}")
            print(f"Running (OMP_NUM_THREADS={threads}): {' '.join(cmd)}")
            print(f"{'='*70}")
        else:
            print(f"\n{'='*70}")
            print(f"Running: {' '.join(cmd)}")
            print(f"{'='*70}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            print(result.stdout)
            
            # Find and move the results file
            if output_name:
                self._organize_results(output_name)
                
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark: {e}")
            print(f"STDERR: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"Error: Executable '{self.executable}' not found!")
            print("Please compile the project first with 'make'")
            return False
    
    def _organize_results(self, custom_name):
        """Move and rename the latest benchmark results file"""
        # Find the most recent benchmark_results file
        result_files = sorted(Path(".").glob("benchmark_results_*.txt"))
        if result_files:
            latest = result_files[-1]
            new_path = self.results_dir / f"{custom_name}.txt"
            latest.rename(new_path)
            print(f"\n✓ Results saved to: {new_path}")
            
            # Clean up any remaining old benchmark_results files
            for old_file in Path(".").glob("benchmark_results_*.txt"):
                try:
                    old_file.unlink()
                except Exception:
                    pass  # Ignore errors during cleanup
    
    def density_sweep(self, m=2048, k=2048, n=2048):
        """Sweep over density: 0.01% to 50% with twice as many points"""
        print(f"\n=== DENSITY SWEEP: {m}×{k}×{n} ===")
        # Doubled the number of density points for finer granularity
        densities = [
            0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.00075,
            0.001, 0.0015, 0.002, 0.003, 0.005, 0.0075,
            0.01, 0.015, 0.02, 0.03, 0.05, 0.075,
            0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
        ]
        
        for density in densities:
            self.run_command(
                ["custom", str(m), str(k), str(n), str(density)],
                f"density_sweep_{m}x{k}x{n}_d{int(density*10000):05d}"
            )
            time.sleep(0.5)  # Small delay between runs
    
    def size_sweep(self, density=0.05):
        """Sweep over matrix sizes (square matrices) - expanded for cache hierarchy analysis
        
        Zen 4 cache hierarchy (Ryzen 7 7700X):
        - L1d: 32 KB per core → ~64×64 matrix
        - L2:  1 MB per core  → ~192×192 matrix (working set: 0.84 MB)
        - L3:  32 MB shared   → ~1024×1024 matrix (working set: 24 MB)
        - DRAM: Beyond L3 cache size
        
        Uses single thread for L1d/L2 (<1MB) to eliminate overhead, multi-threaded for L3/DRAM (≥1MB).
        """
        print(f"\n=== SIZE SWEEP: density={density*100}% ===")
        
        # Expanded sweep with focus on cache transitions
        # L1d and L2 regions: single-threaded to isolate cache effects (working set < 1 MB)
        l1_l2_sizes = [
            # L1d region (32 KB per core)
            32, 48, 64, 80, 96,
            # L1d → L2 transition
            128, 160, 192,
        ]
        
        # L3 and DRAM regions: multi-threaded for better performance (working set ≥ 1 MB)
        l3_dram_sizes = [
            # L2 → L3 transition and L3 region
            224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
            # L3 → DRAM transition
            1280, 1536, 1792, 2048,
            # DRAM region
            2560, 3072, 3584, 4096, 5120, 6144
        ]
        
        print(f"Running L1d/L2 sizes (single-threaded, <1MB): {l1_l2_sizes[0]}-{l1_l2_sizes[-1]}")
        for size in l1_l2_sizes:
            self.run_command(
                ["custom", str(size), str(size), str(size), str(density)],
                f"size_sweep_d{int(density*100):03d}_s{size:04d}",
                threads=1  # Single-threaded for L1d/L2 cache analysis
            )
            time.sleep(0.5)
        
        print(f"\nRunning L3/DRAM sizes (multi-threaded, ≥1MB): {l3_dram_sizes[0]}-{l3_dram_sizes[-1]}")
        for size in l3_dram_sizes:
            self.run_command(
                ["custom", str(size), str(size), str(size), str(density)],
                f"size_sweep_d{int(density*100):03d}_s{size:04d}"
                # No threads parameter = use default multi-threading
            )
            time.sleep(0.5)
    
    def structure_sweep(self, base_size=1024, density=0.05):
        """Sweep over different matrix structures"""
        print(f"\n=== STRUCTURE SWEEP: base={base_size}, density={density*100}% ===")
        
        structures = [
            ("square", base_size, base_size, base_size),
            ("tall_2x", base_size * 2, base_size, base_size // 2),
            ("tall_4x", base_size * 4, base_size, base_size // 4),
            ("tall_8x", base_size * 8, base_size, base_size // 8),
            ("fat_2x", base_size // 2, base_size, base_size * 2),
            ("fat_4x", base_size // 4, base_size, base_size * 4),
        ]
        
        for name, m, k, n in structures:
            self.run_command(
                ["custom", str(m), str(k), str(n), str(density)],
                f"structure_{name}_b{base_size}_d{int(density*100):03d}"
            )
            time.sleep(0.5)
    
    def grid_sweep(self, sizes=None, densities=None):
        """Full grid sweep: sizes × densities"""
        if sizes is None:
            sizes = [256, 512, 1024, 2048]
        if densities is None:
            densities = [0.001, 0.01, 0.05, 0.10, 0.20]
        
        print(f"\n=== GRID SWEEP: {len(sizes)} sizes × {len(densities)} densities ===")
        total = len(sizes) * len(densities)
        current = 0
        
        for size in sizes:
            for density in densities:
                current += 1
                print(f"\n[{current}/{total}] Testing {size}³ @ {density*100}%")
                self.run_command(
                    ["custom", str(size), str(size), str(size), str(density)],
                    f"grid_s{size:04d}_d{int(density*1000):04d}"
                )
                time.sleep(0.5)
    
    def thread_scaling(self, m=1024, k=1024, n=1024, density=0.05):
        """Test performance across different thread counts"""
        print(f"\n=== THREAD SCALING: {m}×{k}×{n}, density={density*100}% ===")
        
        import multiprocessing
        max_threads = multiprocessing.cpu_count()
        thread_counts = [1, 2, 4, 8, 16, 32, 64]
        thread_counts = [t for t in thread_counts if t <= max_threads]
        
        for threads in thread_counts:
            # Set OMP_NUM_THREADS environment variable
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(threads)
            
            print(f"\nTesting with {threads} threads...")
            cmd = [self.executable, "custom", str(m), str(k), str(n), str(density)]
            
            try:
                result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
                print(result.stdout)
                self._organize_results(f"threads_t{threads:02d}_s{m}_d{int(density*100):03d}")
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
            
            time.sleep(0.5)
    
    def custom_sweep(self, config_file):
        """Run custom sweep from configuration file"""
        print(f"\n=== CUSTOM SWEEP from {config_file} ===")
        
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        m, k, n, density = parts[:4]
                        name = parts[4] if len(parts) > 4 else f"custom_{m}x{k}x{n}_d{density}"
                        
                        self.run_command(
                            ["custom", m, k, n, density],
                            name
                        )
                        time.sleep(0.5)
        except FileNotFoundError:
            print(f"Error: Config file '{config_file}' not found!")

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive matrix multiplication benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --density          # Density sweep (1024³)
  python run_benchmarks.py --size             # Size sweep (5% density)
  python run_benchmarks.py --structure        # Structure sweep
  python run_benchmarks.py --grid             # Full grid sweep
  python run_benchmarks.py --threads          # Thread scaling analysis
  python run_benchmarks.py --all              # Run all sweeps
  python run_benchmarks.py --custom config.txt # Custom from file
  
  # Custom density sweep
  python run_benchmarks.py --density --m 2048 --k 2048 --n 512
  
  # Custom size sweep  
  python run_benchmarks.py --size --density 0.10
        """
    )
    
    parser.add_argument('--density', action='store_true', 
                        help='Run density sweep')
    parser.add_argument('--size', action='store_true',
                        help='Run size sweep')
    parser.add_argument('--structure', action='store_true',
                        help='Run structure sweep')
    parser.add_argument('--grid', action='store_true',
                        help='Run full grid sweep')
    parser.add_argument('--threads', action='store_true',
                        help='Run thread scaling analysis')
    parser.add_argument('--all', action='store_true',
                        help='Run all sweeps')
    parser.add_argument('--custom', metavar='FILE',
                        help='Run custom sweep from file')
    
    # Parameters for customizing sweeps
    parser.add_argument('--m', type=int, default=2048,
                        help='Matrix dimension m (rows of A)')
    parser.add_argument('--k', type=int, default=2048,
                        help='Matrix dimension k (cols of A, rows of B)')
    parser.add_argument('--n', type=int, default=2048,
                        help='Matrix dimension n (cols of B)')
    parser.add_argument('--density-val', type=float, default=0.05, dest='density_val',
                        help='Density value (0.0 to 1.0)')
    parser.add_argument('--base-size', type=int, default=2048,
                        help='Base size for structure sweep')
    parser.add_argument('--executable', default='./matrix_bench',
                        help='Path to benchmark executable')
    
    args = parser.parse_args()
    
    # Check if executable exists
    if not Path(args.executable).exists():
        print(f"Error: Executable '{args.executable}' not found!")
        print("Please compile the project first with 'make'")
        return 1
    
    runner = BenchmarkRunner(args.executable)
    
    # If no flags, show help
    if not any([args.density, args.size, args.structure, args.grid, 
                args.threads, args.all, args.custom]):
        parser.print_help()
        return 0
    
    # Run requested sweeps
    if args.all:
        print("\n" + "="*70)
        print("RUNNING ALL BENCHMARK SWEEPS")
        print("="*70)
        runner.density_sweep()
        runner.size_sweep()
        runner.structure_sweep()
        runner.grid_sweep()
        runner.thread_scaling()
    else:
        if args.density:
            runner.density_sweep(args.m, args.k, args.n)
        
        if args.size:
            runner.size_sweep(args.density_val)
        
        if args.structure:
            runner.structure_sweep(args.base_size, args.density_val)
        
        if args.grid:
            runner.grid_sweep()
        
        if args.threads:
            runner.thread_scaling(args.m, args.k, args.n, args.density_val)
        
        if args.custom:
            runner.custom_sweep(args.custom)
    
    print("\n" + "="*70)
    print("ALL BENCHMARKS COMPLETE!")
    print(f"Results saved in: {runner.results_dir}")
    print("="*70)
    
    # Cleanup any remaining benchmark_results files
    for old_file in Path(".").glob("benchmark_results_*.txt"):
        try:
            old_file.unlink()
            print(f"Cleaned up: {old_file}")
        except Exception:
            pass
    
    # Cleanup temporary executable if created
    if runner.cleanup_tmp and os.path.exists(runner.executable):
        os.remove(runner.executable)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
