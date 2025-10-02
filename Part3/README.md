# Part3 - SSD Performance Profiling and I/O Benchmarking

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview
This Bench does not work as I wanted it to. Windows version for me was bugged way to hard so I switched to using WSL where I thought it worked until I noticed the results coming in, FUSE spoiled all results. Both issues took quite a 3-4 hours to figure out. So my next step would be to install a full linux distro and go from there but no time. 

## Contents

### Project Structure
The `project3/scripts` subdirectory contains:
Run_all_tests.sh - runs all tests

The `project3/config` subdirectory contains:
Configuration for all tests

## How the drive is supposed to perform
![Crystal Disk mark](/Part3/Screenshot%202025-10-01%20145841.png)
![Crystal Disk mark](/Part3/Screenshot%202025-10-01%20145847.png)
![Crystal Disk mark](/Part3/Screenshot%202025-10-01%20145852.png)


## Results from WSL testing spoiled by fuse. 
### 1. Zero-Queue Baseline Tests
![Spoiled results](/Part3/project3/results_20251001_211920/plots/baseline_table.png)
As you can see The latency is in the >160 microseconds when it should be in the 50s. for read or 15 for write as I have a DRAM and SLC write buffer. 
IOPS are also 10 times lower then they are on the actual drive.


### 2. Block Size Sweep
![Spoiled results](/Part3/project3/results_20251001_211920/plots/blocksize_sweep_seq.png)
![Spoiled results](/Part3/project3/results_20251001_211920/plots/blocksize_sweep_rand.png)
The larger the block size the fewer IOPS you need to get the capped bandwidth that I have in FUSE. I dont know what cross over is that is being asked of me to discuss.

### 3. Read/Write Mix Analysis
Incomplete due to spending time on all other bugs. 
![Spoiled results](/Part3/project3/results_20251001_211920/plots/rw_mix.png)
with these two data poitns as read percentage went up, both read and write latency went up
as write percentage went up, but read and write latency dropped. I found this odd, but I need more data, and have the data be good. 

### 4. Queue Depth Scaling
Analysis was scuffed/not done. Too much bad data in my sweep to make sense of it. Needs to be re looked at. Everything was getting around 600 IOPS. 

### 3. Tail Latency
![Spoiled results](/Part3/project3/results_20251001_211920/plots/tail_latency.png)
50%, 95%, 99%, and 99.9% latency on the x axis

To me having the .1% only be 40% higher then the top 5% which was only 10% higher then the mean meant very consistent results. 

## System Configuration
- **Storage**: XPG Gammix S70 Blade 1TB (Micron TLC B47R 512Gb)
- **OS**:  WSL 
- **Test Environment**: Dedicated NTFS G: drive for isolated testing


## Quick Start
2. Install FIO and configure test environment
3. Run baseline tests before comprehensive benchmarking
4. Analyze results; each json has to be manually put in to the parse_results, batch processing not added. 

---
[← Back to Main README](../README.md) | [Detailed Setup Guide →](./project3/README.md)