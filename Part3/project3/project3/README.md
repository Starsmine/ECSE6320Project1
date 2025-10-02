# SSD Performance Profiling Project Setup

## Prerequisites Installation

1. Install FIO (Flexible I/O Tester):
   - Download the latest Windows binary from: https://bsdio.com/fio/
   - Extract the zip file to a local directory
   - Add the FIO directory to your system PATH

2. Verify Drive G: Setup
   ```powershell
   # Check drive information
   Get-Volume -DriveLetter G
   # Check available space
   Get-PSDrive G
   ```

3. Create a dedicated test file (safer than raw device testing):
   ```powershell
   # Create a 20GB test file (adjust size as needed)
   fsutil file createnew G:\fio_testfile 21474836480
   ```

## Project Structure
- /project3
  - /configs       # FIO job configuration files
  - /results      # Test results and raw data
  - /scripts      # PowerShell scripts for automation
  - /plots        # Generated plots and visualizations

## Initial Tests to Run (In Order)

1. Zero-queue baselines:
   - 4 KiB random reads/writes (QD=1)
   - 128 KiB sequential reads/writes (QD=1)

2. Block-size sweep:
   - Sizes: 4K, 16K, 32K, 64K, 128K, 256K
   - Both random and sequential patterns

3. Read/Write mix tests:
   - 100% Read
   - 100% Write
   - 70/30 Read/Write
   - 50/50 Read/Write

4. Queue depth scaling:
   - QD progression: 1→2→4→8→16→32→64→128→256

## Safety Notes
- Always use a dedicated test file
- Monitor drive temperature
- Run tests multiple times for statistical validity
- Document all system settings