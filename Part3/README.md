# Part3 - SSD Performance Profiling and I/O Benchmarking

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This Bench does not work as I wanted it to. Windows version for me was bugged so I switched to using WSL, but then FUSE spoiled all results. Both issues took quite a few hours to figure out so my next step would be to install a full linux distro and go from there but no time. 

## Contents

### Project Structure
The `project3/scripts` subdirectory contains:
Run_all_tests.sh - runs all tests

The `project3/config` subdirectory contains:
Configuration for all tests

## How the drive is supposed to perform

## Results from WSL testing spoiled by fuse. 
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
- **OS**:  WSL 
- **Test Environment**: Dedicated NTFS G: drive for isolated testing


## Quick Start
2. Install FIO and configure test environment
3. Run baseline tests before comprehensive benchmarking
4. Analyze results using provided visualization tools

---
[← Back to Main README](../README.md) | [Detailed Setup Guide →](./project3/README.md)