# Command line parameters
param(
    [int]$Iterations = 3,  # Number of times to run each test
    [switch]$SkipPlot = $false  # Skip running plot script after tests
)

########################
# Configuration Section
########################

# Path to mlc.exe (adjust if not in PATH)
$mlc = ".\mlc.exe"

# CPU core to pin latency measurements to
$core = 0

# Get current CPU clock speed in GHz (reads live frequency, not base)
$ClockSpeedGHz = 5.5 #This code grabs the clock before the core turbos (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty CurrentClockSpeed) / 1000.0
#Write-Host "Detected current CPU speed: $ClockSpeedGHz GHz"

# Buffer sizes for cache levels (adjust to your CPU's L1/L2/L3 sizes)
$l1_buf = "32k"     # fits in L1 cache
$l2_buf = "512k"    # fits in L2 cache
$l3_buf = "8m"      # fits in L3 cache
$dram_buf = "200m"  # forces DRAM

# Set up results directory
$resultsDir = ".\results"

# Clean up old results
if (Test-Path $resultsDir) {
    Write-Host "`nCleaning up old results..."
    Get-ChildItem -Path $resultsDir -Recurse | ForEach-Object {
        try {
            Remove-Item $_.FullName -Force -Recurse
            Write-Host "Removed $($_.FullName)"
        } catch {
            Write-Warning "Could not remove $($_.FullName): $_"
        }
    }
} else {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
    Write-Host "Created new results directory"
}

# Save timestamp for this run
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

# Print run configuration
Write-Host "`n=== Configuration ==="
Write-Host "Number of iterations: $Iterations"
Write-Host "CPU Core: $core"
Write-Host "Clock Speed: $ClockSpeedGHz GHz"

########################
# Helper Functions
########################

# Function to convert nanoseconds to CPU cycles
function ConvertTo-Cycles($ns, $GHz) {
    if ($ns -is [double]) {
        return [math]::Round($ns * $GHz, 2)
    }
    return "N/A"
}

# Function to parse MLC data from output
function Parse-MLC-Data($output) {
    $data = @()
    $inDataSection = $false
    $headerPattern = "^(?:Inject|Delay)\s+(?:Latency|ns)\s+(?:Bandwidth|MB/sec)"
    
    if ($null -eq $output) {
        Write-Warning "No output provided to Parse-MLC-Data"
        return $data
    }
    
    foreach ($line in $output) {
        if ($null -eq $line) { continue }
        
        # Check for data section start
        if ($line -match $headerPattern) {
            $inDataSection = $true
            continue
        }
        # Skip separator line
        if ($line -match "^=+$") { continue }
        
        if ($inDataSection -and $line.Trim()) {
            # Skip header-like lines
            if ($line -match "(?:Delay|Latency|Bandwidth)") { continue }
            
            # Split and clean up the line, handling multiple spaces
            $parts = $line.Trim() -split '\s+'
            if ($parts.Count -ge 3) {
                try {
                    # Try to parse delay (removing any leading zeros)
                    $delayStr = $parts[0] -replace '^0+', ''
                    if ([string]::IsNullOrWhiteSpace($delayStr)) { $delayStr = "0" }
                    
                    # Clean up and parse values
                    $latencyStr = $parts[1] -replace '[^0-9\.]', ''
                    $bwStr = $parts[2] -replace '[^0-9\.]', ''
                    
                    if ($delayStr -match '^\d+$' -and 
                        $latencyStr -match '^\d*\.?\d+$' -and 
                        $bwStr -match '^\d*\.?\d+$') {
                        $data += [PSCustomObject]@{
                            Delay = [int]$delayStr
                            Latency_ns = [double]$latencyStr
                            Bandwidth_MBps = [double]$bwStr
                        }
                    }
                } catch {
                    Write-Warning "Failed to parse line: $line"
                    Write-Warning "Error: $_"
                    continue
                }
            }
        }
    }
    return $data
}

# Function to parse single latency measurement from mlc output
function ConvertFrom-MLCData {
    param(
        [Parameter(Mandatory=$true)]
        [string[]]$output
    )
    
    $results = @()
    $dataSection = $false
    $headerFound = $false
    
    foreach ($line in $output) {
        # Look for the start of data section
        if ($line -match "^(Inject|Delay)\s+Latency\s+Bandwidth") {
            $dataSection = $true
            $headerFound = $true
            continue
        }
        
        if ($dataSection -and $headerFound) {
            if ($line -match '^\s*$') {
                $dataSection = $false  # End of data section
                continue
            }
            
            # Parse data lines with flexible whitespace
            if ($line -match '^\s*(\d+)\s+([\d\.]+)\s+([\d\.]+)') {
                $results += [PSCustomObject]@{
                    'Delay_Cycles' = [int]$matches[1]
                    'Latency_ns' = [double]$matches[2]
                    'Bandwidth_MBps' = [double]$matches[3]
                }
            }
        }
    }
    
    return $results
}

function Parse-Single-Latency($output) {
    if ($null -eq $output) {
        Write-Warning "No output provided to Parse-Single-Latency"
        return 0
    }
    
    try {
        foreach ($line in $output) {
            if ($null -eq $line) { continue }
            
            # Look for latency in different formats
            if ($line -match "Average Latency\s*:\s*([\d\.]+)\s*ns") {
                return [double]$matches[1]
            }
            if ($line -match "\(\s*([\d\.]+)\s*ns\)") {
                return [double]$matches[1]
            }
            if ($line -match "([\d\.]+)\s*ns") {
                return [double]$matches[1]
            }
            if ($line -match "Latency\s*:\s*([\d\.]+)(?:\s*ns)?") {
                return [double]$matches[1]
            }
        }
    } catch {
        Write-Warning "Error parsing latency: $_"
    }
    
    Write-Warning "No latency value found in output"
    return 0
}

# Helper function to create and get experiment directory
function Get-ExperimentDir($experimentName, $iteration) {
    $dir = Join-Path $resultsDir "${experimentName}_${iteration}"
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    return $dir
}

# Helper function to sanitize names for filenames
function Get-SafeFilename($name) {
    return $name -replace '[\\/:*?"<>|]', '_'
}

# Function to save results to CSV
function Save-ResultsToCSV($results, $filename, $iteration = 0) {
    # Add iteration number to filename if doing multiple runs
    $iterationSuffix = if ($Iterations -gt 1) { "_iter$iteration" } else { "" }
    $results | Export-Csv -Path "$resultsDir\${timestamp}_$filename$iterationSuffix.csv" -NoTypeInformation
}

# Function to convert size strings to bytes
function Convert-Size-To-Bytes($size) {
    $units = @{
        'k' = 1024
        'm' = 1024 * 1024
        'g' = 1024 * 1024 * 1024
    }
    
    $match = $size -match '(\d+)([kmg])'
    if ($match) {
        $value = [int]$matches[1]
        $unit = $matches[2].ToLower()
        return $value * $units[$unit]
    }
    return $size
}

########################
# Start of Experiments
########################

# Run cache hierarchy measurements
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running Cache Hierarchy Tests (Iteration $iter/$Iterations) ===`n"
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "cache_hierarchy" $iter
    
    $l1_out   = & $mlc --idle_latency ("-b$($l1_buf)") ("-c$($core)") | Tee-Object -FilePath (Join-Path $expDir "raw_l1.txt")
    $l2_out   = & $mlc --idle_latency ("-b$($l2_buf)") ("-c$($core)") | Tee-Object -FilePath (Join-Path $expDir "raw_l2.txt")
    $l3_out   = & $mlc --idle_latency ("-b$($l3_buf)") ("-c$($core)") | Tee-Object -FilePath (Join-Path $expDir "raw_l3.txt")
    $dram_out = & $mlc --idle_latency ("-b$($dram_buf)") ("-c$($core)") | Tee-Object -FilePath (Join-Path $expDir "raw_dram.txt")

    # Print and save cache hierarchy results
    Write-Host "`n=== Cache Hierarchy Results (Iteration $iter/$Iterations) ==="
    $cache_results = @(
        [PSCustomObject]@{ Level="L1";   Size=$l1_buf;   Latency_ns=(Parse-Single-Latency $l1_out);   Latency_cycles=(ConvertTo-Cycles (Parse-Single-Latency $l1_out) $ClockSpeedGHz) }
        [PSCustomObject]@{ Level="L2";   Size=$l2_buf;   Latency_ns=(Parse-Single-Latency $l2_out);   Latency_cycles=(ConvertTo-Cycles (Parse-Single-Latency $l2_out) $ClockSpeedGHz) }
        [PSCustomObject]@{ Level="L3";   Size=$l3_buf;   Latency_ns=(Parse-Single-Latency $l3_out);   Latency_cycles=(ConvertTo-Cycles (Parse-Single-Latency $l3_out) $ClockSpeedGHz) }
        [PSCustomObject]@{ Level="DRAM"; Size=$dram_buf; Latency_ns=(Parse-Single-Latency $dram_out); Latency_cycles=(ConvertTo-Cycles (Parse-Single-Latency $dram_out) $ClockSpeedGHz) }
    )

    # Save detailed data for analysis
    $cache_results | Export-Csv -Path (Join-Path $expDir "summary.csv") -NoTypeInformation
    
    # Process and save detailed data if available
    foreach ($level in @("l1", "l2", "l3", "dram")) {
        $out = Get-Variable -Name "${level}_out" -ValueOnly
        $detailedData = Parse-MLC-Data $out
        if ($detailedData.Count -gt 0) {
            $detailedData | Export-Csv -Path (Join-Path $expDir "${level}_detail.csv") -NoTypeInformation
        }
    }
    
    # Append to combined results
    $cache_results | Select-Object @{Name='Iteration';Expression={$iter}}, * |
        Export-Csv -Path "$resultsDir\${timestamp}_cache_hierarchy_combined.csv" -NoTypeInformation -Append
    
    $cache_results | Format-Table -AutoSize
}

# Run pattern sweep test
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running Pattern Sweep Tests (Iteration $iter/$Iterations) ===`n"
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "pattern_sweep" $iter
    $combinedData = @()
    
    # List of patterns to test (extend this list for more patterns)
    $patterns = @(
        @{name="Sequential"; args=@("--loaded_latency")},
        @{name="Random"; args=@("--loaded_latency", "-r")},
        @{name="Small-Range"; args=@("--loaded_latency", "-r", "-D64")},      # 64-byte range
        @{name="Medium-Range"; args=@("--loaded_latency", "-r", "-D4096")},   # 4KB range (default)
        @{name="Large-Range"; args=@("--loaded_latency", "-r", "-D65536")}    # 64KB range
    )
    
    foreach ($pattern in $patterns) {
        $patternName = Get-SafeFilename $pattern.Name
        $output = & $mlc $pattern.Cmd ("-c$core") | Tee-Object -FilePath (Join-Path $expDir "raw_${patternName}.txt")
        
        # Parse data
        $data = Parse-MLC-Data $output
        if ($data.Count -gt 0) {
            # Add pattern name and save detailed data
            $data | Add-Member -NotePropertyName 'Pattern' -NotePropertyValue $pattern.Name
            $data | Add-Member -NotePropertyName 'Iteration' -NotePropertyValue $iter
            
            # Add to combined data and save individual results
            $combinedData += $data
            $data | Export-Csv -Path (Join-Path $expDir "${patternName}_detail.csv") -NoTypeInformation
        }
        
        # Get peak values
        $peakBandwidth = ($data | Measure-Object -Property Bandwidth_MBps -Maximum).Maximum
        $latencyAtPeak = ($data | Where-Object { $_.Bandwidth_MBps -eq $peakBandwidth } | Select-Object -First 1).Latency_ns
        
        # Print summary for this pattern
        Write-Host "`n$($pattern.Name):"
        Write-Host "Peak Bandwidth: $peakBandwidth MB/s"
        Write-Host "Latency at Peak: $latencyAtPeak ns"
    }
    
    # Save combined data for this iteration
    $combinedData | Export-Csv -Path "$resultsDir\${timestamp}_pattern_sweep_combined.csv" -NoTypeInformation -Append
    
    # Save summary
    $summary = $patterns | ForEach-Object {
        $patternData = $combinedData | Where-Object { $_.Pattern -eq $_.Name }
        $peakBW = ($patternData | Measure-Object -Property Bandwidth_MBps -Maximum).Maximum
        $latencyAtPeak = ($patternData | Where-Object { $_.Bandwidth_MBps -eq $peakBW } | Select-Object -First 1).Latency_ns
        
        [PSCustomObject]@{
            Pattern = $_.Name
            Peak_Bandwidth_MBps = $peakBW
            Latency_At_Peak_ns = $latencyAtPeak
            Iteration = $iter
        }
    }
    $summary | Export-Csv -Path "$resultsDir\${timestamp}_pattern_sweep_summary.csv" -NoTypeInformation -Append
    
    # Display summary table
    Write-Host "`n=== Pattern Sweep Summary (Iteration $iter/$Iterations) ==="
    $summary | Format-Table -AutoSize
}

# Run granularity sweep experiment
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running Granularity Sweep Tests (Iteration $iter/$Iterations) ===`n"
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "granularity" $iter
    
    # Test different data sizes (in bytes)
    $sizes = @(
        @{size="4"; name="4B"},
        @{size="8"; name="8B"},
        @{size="16"; name="16B"},
        @{size="32"; name="32B"},
        @{size="64"; name="64B"},            # Cache line size
        @{size="128"; name="128B"},
        @{size="256"; name="256B"}
    )
    
    $gran_results = @()
    foreach ($size in $sizes) {
        Write-Host "Testing $($size.name) granularity..."
        $outputFile = Join-Path $expDir "raw_gran_$($size.name).txt"
        
        # Use idle_latency with data size parameter
        $out = & $mlc --idle_latency ("-b$($dram_buf)") ("-c$($core)") "-l$($size.size)" | 
            Tee-Object -FilePath $outputFile
        
        $lat = Parse-Single-Latency $out
        
        $result = [PSCustomObject]@{
            Size_Bytes = [int]$size.size
            Latency_ns = $lat
            Latency_cycles = (ConvertTo-Cycles $lat $ClockSpeedGHz)
        }
        $gran_results += $result
        
        # Save individual result
        $result | Export-Csv -Path (Join-Path $expDir "gran_$($size.name)_detail.csv") -NoTypeInformation
    }
    
    # Save summary for this iteration
    $gran_results | Export-Csv -Path (Join-Path $expDir "summary.csv") -NoTypeInformation
    
    # Append to combined results
    $gran_results | Select-Object @{Name='Iteration';Expression={$iter}}, * |
        Export-Csv -Path "$resultsDir\${timestamp}_granularity_sweep_combined.csv" -NoTypeInformation -Append
    
    Write-Host "`n=== Granularity Sweep Results (Iteration $iter/$Iterations) ==="
    $gran_results | Format-Table -AutoSize
}

# Run Read/Write Mix sweep experiment
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running Read/Write Mix Sweep Tests (Iteration $iter/$Iterations) ===`n"
    $rw_patterns = @(
        @{name="100% Read"; args=@("--peak_injection_bandwidth")},           # Default is read-only
        @{name="75% Read/ 25% Write"; args=@("--peak_injection_bandwidth", "-W3")},    #3:1 ratio
        @{name="70% Read / 30% Write"; args=@("--peak_injection_bandwidth", "-W7")}, # 7:3 ratio
        @{name="50% Read / 50% Write"; args=@("--peak_injection_bandwidth", "-W5")}  # 1:1 ratio
    )
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "rw_mix" $iter
    
    $rw_results = @()
    foreach ($pattern in $rw_patterns) {
        Write-Host "Testing $($pattern.name) ratio..."
        $arguments = $pattern.args + @("-b$($dram_buf)", "-c$($core)")
        $safeName = Get-SafeFilename $pattern.name
        
        $outputFile = Join-Path $expDir "raw_rw_${safeName}.txt"
        $out = & $mlc $arguments | Tee-Object -FilePath $outputFile
        
        # Initialize defaults
        $lat = 0
        $bw = 0
        
        if ($null -ne $out) {
            $lat = Parse-Single-Latency $out
            
            # More robust bandwidth parsing
            try {
                $bwMatch = $out | Where-Object { $_ -match "(?:bandwidth|MB/sec)\s*:\s*([\d\.]+)" }
                if ($null -ne $bwMatch) {
                    $bw = [double]$matches[1]
                } else {
                    Write-Warning "No bandwidth value found in output"
                }
            } catch {
                Write-Warning "Error parsing bandwidth: $_"
            }
        } else {
            Write-Warning "No output received from MLC command"
        }
        
        $result = [PSCustomObject]@{
            'R/W Ratio' = $pattern.name
            Latency_ns = $lat
            Bandwidth_MBps = $bw
            Latency_cycles = (ConvertTo-Cycles $lat $ClockSpeedGHz)
        }
        $rw_results += $result
        
        # Save individual result
        $result | Export-Csv -Path (Join-Path $expDir "rw_${safeName}_detail.csv") -NoTypeInformation
    }
    
    # Save summary for this iteration
    $rw_results | Export-Csv -Path (Join-Path $expDir "summary.csv") -NoTypeInformation
    
    # Append to combined results
    $rw_results | Select-Object @{Name='Iteration';Expression={$iter}}, * |
        Export-Csv -Path "$resultsDir\${timestamp}_rw_mix_sweep_combined.csv" -NoTypeInformation -Append
    
    Write-Host "`n=== Read/Write Mix Sweep Results (Iteration $iter/$Iterations) ==="
    $rw_results | Format-Table -AutoSize
}

# Run intensity sweep experiment
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running Intensity Sweep Tests (Iteration $iter/$Iterations) ===`n"
    $intensities = @(
        @{name="Maximum"; args=@("--loaded_latency", "-d0")},         # No delay = maximum intensity
        @{name="Very High"; args=@("--loaded_latency", "-d10")},      # Very low delay
        @{name="High"; args=@("--loaded_latency", "-d100")},          # Low delay
        @{name="Medium High"; args=@("--loaded_latency", "-d400")},   # Medium-low delay
        @{name="Medium"; args=@("--loaded_latency", "-d1000")},       # Medium delay
        @{name="Medium Low"; args=@("--loaded_latency", "-d2000")},   # Medium-high delay
        @{name="Low"; args=@("--loaded_latency", "-d5000")},          # High delay
        @{name="Very Low"; args=@("--loaded_latency", "-d10000")}     # Very high delay = minimum intensity
    )
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "intensity" $iter
    
    $intensity_results = @()
    foreach ($intensity in $intensities) {
        Write-Host "Testing $($intensity.name) intensity..."
        # Add -W5 for 50/50 read/write mix to get better loaded latency curve
        $arguments = $intensity.args + @("-b$($dram_buf)", "-c$($core)", "-W5")
        $safeName = Get-SafeFilename $intensity.name
        
        $outputFile = Join-Path $expDir "raw_intensity_${safeName}.txt"
        $out = & $mlc $arguments | Tee-Object -FilePath $outputFile
        
        # Save raw output first
        if ($null -ne $out) {
            Set-Content -Path $outputFile -Value $out
        }
        
        # Initialize defaults
        $lat = 0
        $bw = 0
        $detailedData = @()
        
        if ($null -ne $out) {
            # Parse the detailed data first
            $detailedData = ConvertFrom-MLCData $out
            
            if ($detailedData.Count -gt 0) {
                # Get maximum bandwidth and corresponding latency
                $maxBw = ($detailedData | Measure-Object -Property Bandwidth_MBps -Maximum).Maximum
                $latAtMaxBw = ($detailedData | Where-Object { $_.Bandwidth_MBps -eq $maxBw } | Select-Object -First 1).Latency_ns
                
                $bw = $maxBw
                $lat = $latAtMaxBw
            } else {
                # Fallback to single value parsing
                $lat = Parse-Single-Latency $out
                
                # More robust bandwidth parsing
                try {
                    $bwMatch = $out | Where-Object { $_ -match "(?:bandwidth|MB/sec)\s*:\s*([\d\.]+)" }
                    if ($null -ne $bwMatch) {
                        $bw = [double]$matches[1]
                    } else {
                        Write-Warning "No bandwidth value found in output"
                    }
                } catch {
                    Write-Warning "Error parsing bandwidth: $_"
                }
            }
        } else {
            Write-Warning "No output received from MLC command for $($intensity.name)"
            Write-Warning "Command: mlc $($arguments -join ' ')"
        }
        
        # Extract delay value from arguments
        $delayValue = [int]($intensity.args[-1] -replace "^-d")
        
        # Create result object
        $result = [PSCustomObject]@{
            'Intensity' = $intensity.name
            'Delay_Cycles' = $delayValue
            'Latency_ns' = $lat
            'Bandwidth_MBps' = $bw
            'Latency_cycles' = (ConvertTo-Cycles $lat $ClockSpeedGHz)
        }
        $intensity_results += $result
        
        # Save individual result
        $result | Export-Csv -Path (Join-Path $expDir "intensity_${safeName}_detail.csv") -NoTypeInformation
    }
    
    # Save summary for this iteration
    $intensity_results | Export-Csv -Path (Join-Path $expDir "summary.csv") -NoTypeInformation
    
    # Append to combined results
    $intensity_results | Select-Object @{Name='Iteration';Expression={$iter}}, * |
        Export-Csv -Path "$resultsDir\${timestamp}_intensity_sweep_combined.csv" -NoTypeInformation -Append
    
    Write-Host "`n=== Intensity Sweep Results (Iteration $iter/$Iterations) ==="
    $intensity_results | Format-Table -AutoSize
}

# Run cache-miss impact experiment
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running Cache Miss Impact Tests (Iteration $iter/$Iterations) ===`n"
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "cache_miss" $iter
    
    # Test different stride sizes to control cache miss rates
    $strides = @(
        @{name="Sequential (Low Miss)"; args=@("--loaded_latency", "-S1")},      # Sequential access
        @{name="64B Stride"; args=@("--loaded_latency", "-S64")},                # One cache line skip
        @{name="128B Stride"; args=@("--loaded_latency", "-S128")},              # Two cache lines skip
        @{name="4KB Stride"; args=@("--loaded_latency", "-S4096")},              # Page size stride
        @{name="Random (High Miss)"; args=@("--loaded_latency", "-r")}           # Random access
    )
    
    $miss_results = @()
    foreach ($stride in $strides) {
        Write-Host "Testing $($stride.name) access pattern..."
        # Use small buffer to keep light kernel footprint
        $arguments = $stride.args + @("-b2m", "-c$($core)")
        $safeName = Get-SafeFilename $stride.name
        
        $outputFile = Join-Path $expDir "raw_stride_${safeName}.txt"
        $out = & $mlc $arguments | Tee-Object -FilePath $outputFile
        
        $lat = Parse-Single-Latency $out
        $bw = ($out | Where-Object { $_ -match "(?:bandwidth|MB/sec)\s*:\s*([\d\.]+)" } | ForEach-Object { [double]$matches[1] })[0]
        
        $result = [PSCustomObject]@{
            'Access_Pattern' = $stride.name
            'Stride_Size' = if ($stride.name -eq "Random (High Miss)") { "Random" } else { ($stride.args[-1] -replace "^-S") + "B" }
            Latency_ns = $lat
            Bandwidth_MBps = $bw
            Latency_cycles = (ConvertTo-Cycles $lat $ClockSpeedGHz)
        }
        $miss_results += $result
        
        # Save individual result
        $result | Export-Csv -Path (Join-Path $expDir "stride_${safeName}_detail.csv") -NoTypeInformation
    }
    
    # Save summary for this iteration
    $miss_results | Export-Csv -Path (Join-Path $expDir "summary.csv") -NoTypeInformation
    
    # Append to combined results
    $miss_results | Select-Object @{Name='Iteration';Expression={$iter}}, * |
        Export-Csv -Path "$resultsDir\${timestamp}_cache_miss_combined.csv" -NoTypeInformation -Append
    
    Write-Host "`n=== Cache Miss Impact Results (Iteration $iter/$Iterations) ==="
    $miss_results | Format-Table -AutoSize
}

# Run TLB-miss impact experiment
for ($iter = 1; $iter -le $Iterations; $iter++) {
    Write-Host "`n=== Running TLB Miss Impact Tests (Iteration $iter/$Iterations) ===`n"
    
    # Create experiment directory
    $expDir = Get-ExperimentDir "tlb_miss" $iter
    
    # Test different access patterns that stress TLB
    $pageTests = @(
        # Small sequential range - low TLB pressure
        @{name="Small Sequential"; args=@("--loaded_latency", "-T4K")},          # 4KB total range
        # Small random range - moderate TLB pressure within few pages
        @{name="Small Random"; args=@("--loaded_latency", "-T4K", "-r")},        # Random within 4KB
        # Medium sequential range - moderate TLB pressure
        @{name="Medium Sequential"; args=@("--loaded_latency", "-T2M")},         # 2MB range
        # Medium random range - high TLB pressure
        @{name="Medium Random"; args=@("--loaded_latency", "-T2M", "-r")},       # Random within 2MB
        # Large sequential range - high TLB pressure
        @{name="Large Sequential"; args=@("--loaded_latency", "-T1G")},          # 1GB range
        # Large random range - maximum TLB pressure
        @{name="Large Random"; args=@("--loaded_latency", "-T1G", "-r")}         # Random within 1GB
    )
    
    $tlb_results = @()
    foreach ($test in $pageTests) {
        Write-Host "Testing $($test.name) page access..."
        # Use large buffer to ensure TLB pressure
        $arguments = $test.args + @("-b1g", "-c$($core)")
        $safeName = Get-SafeFilename $test.name
        
        $outputFile = Join-Path $expDir "raw_page_${safeName}.txt"
        $out = & $mlc $arguments | Tee-Object -FilePath $outputFile
        
        # Initialize defaults
        $lat = 0
        $bw = 0
        
        if ($null -ne $out) {
            $lat = Parse-Single-Latency $out
            
            # More robust bandwidth parsing
            try {
                $bwMatch = $out | Where-Object { $_ -match "(?:bandwidth|MB/sec)\s*:\s*([\d\.]+)" }
                if ($null -ne $bwMatch) {
                    $bw = [double]$matches[1]
                } else {
                    Write-Warning "No bandwidth value found in output"
                }
            } catch {
                Write-Warning "Error parsing bandwidth: $_"
            }
        } else {
            Write-Warning "No output received from MLC command for $($test.name)"
            Write-Warning "Command: mlc $($arguments -join ' ')"
        }
        
        # Extract test range size
        $rangeSize = ($test.args[1] -replace "^-T")
        
        $result = [PSCustomObject]@{
            'Test_Config' = $test.name
            'Range_Size' = $rangeSize
            'Access_Type' = if ($test.args.Count -gt 2) { "Random" } else { "Sequential" }
            Latency_ns = $lat
            Bandwidth_MBps = $bw
            Latency_cycles = (ConvertTo-Cycles $lat $ClockSpeedGHz)
        }
        $tlb_results += $result
        
        # Save individual result
        $result | Export-Csv -Path (Join-Path $expDir "page_${safeName}_detail.csv") -NoTypeInformation
    }
    
    # Save summary for this iteration
    $tlb_results | Export-Csv -Path (Join-Path $expDir "summary.csv") -NoTypeInformation
    
    # Append to combined results
    $tlb_results | Select-Object @{Name='Iteration';Expression={$iter}}, * |
        Export-Csv -Path "$resultsDir\${timestamp}_tlb_miss_combined.csv" -NoTypeInformation -Append
    
    Write-Host "`n=== TLB Miss Impact Results (Iteration $iter/$Iterations) ==="
    $tlb_results | Format-Table -AutoSize
}