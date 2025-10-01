# Part3 - SSD Performance Profiling and I/O Benchmarking

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This section focuses on comprehensive SSD performance analysis using FIO (Flexible I/O Tester) and custom benchmarking methodologies. The project explores various I/O patterns, queue depths, and block sizes to characterize storage performance.

## Contents

### Documentation
- `Class-wide projects-1-1.pdf` - Project specifications and requirements
- `Class-wide projects-1-1.txt` - Text version of project documentation
- `output.json` - Benchmark results in JSON format
- `test.fio` - FIO configuration file for testing

### Project Structure
The `project3/` subdirectory contains:
- [**Detailed Project README**](./project3/README.md) - Complete setup and usage instructions
- **Configuration Files** (`configs/`) - FIO job configurations for different test scenarios
- **Results Data** (`results_*/`) - Raw benchmark data and processed results
- **Scripts** (`scripts/`) - Automation scripts for running benchmark suites
- **Plots** (`plots/`) - Generated performance visualizations

## Key Test Categories

### 1. Zero-Queue Baseline Tests
- 4 KiB random reads/writes (QD=1)
- 128 KiB sequential reads/writes (QD=1)
- Establishes baseline latency characteristics

### 2. Block Size Sweep
- Sizes: 4K, 16K, 32K, 64K, 128K, 256K
- Both random and sequential access patterns
- Bandwidth and IOPS characterization

### 3. Read/Write Mix Analysis
- 100% Read workloads
- 100% Write workloads  
- 70/30 Read/Write mixed workloads
- 50/50 Read/Write balanced workloads

### 4. Queue Depth Scaling
- QD progression: 1→2→4→8→16→32→64→128→256
- Parallelism impact on throughput and latency

## System Configuration
- **Storage**: XPG Gammix S70 Blade 1TB (Micron TLC B47R 512Gb)
- **OS**: Windows 11 (primary), WSL (experimental)
- **Test Environment**: Dedicated G: drive for isolated testing

## Safety Features
- Uses dedicated test files instead of raw device access
- Configurable test file sizes to prevent data loss
- Automated cleanup procedures

## Quick Start
1. Review the [detailed project README](./project3/README.md) for complete setup instructions
2. Install FIO and configure test environment
3. Run baseline tests before comprehensive benchmarking
4. Analyze results using provided visualization tools

---
[← Back to Main README](../README.md) | [Detailed Setup Guide →](./project3/README.md)