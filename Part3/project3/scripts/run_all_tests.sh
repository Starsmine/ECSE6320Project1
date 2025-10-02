#!/bin/bash

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

# Check if G: drive is mounted
if [ ! -d "/mnt/g" ]; then
    echo "Error: G: drive is not mounted at /mnt/g"
    exit 1
fi

echo "Cleaning up previous test files and results..."
rm -rf "${PROJECT_DIR}"/results_*

# Create results directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_DIR}/results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Function to run fio test and save results
run_fio_test() {
    local config_file=$1
    local output_name=$2
    echo "Running test: ${output_name}"
    fio "${PROJECT_DIR}/configs/${config_file}" --output="${RESULTS_DIR}/${output_name}.json" --output-format=json
    sleep 5  # Cool-down period between tests
}

# 1. Zero-queue baseline tests
echo "=== Running zero-queue baseline tests ==="
run_fio_test "baseline_tests.fio" "baseline_tests"

# 2. Block-size sweep tests
echo "=== Running block-size sweep tests ==="
run_fio_test "blocksize_sweep_random.fio" "blocksize_sweep_random"
run_fio_test "blocksize_sweep_sequential.fio" "blocksize_sweep_sequential"

# 3. Read/Write mix tests
echo "=== Running read/write mix tests ==="
run_fio_test "rw_mix_tests.fio" "rw_mix_tests"

# 4. Queue depth sweep tests
echo "=== Running queue depth sweep tests ==="
run_fio_test "qd_sweep_4k.fio" "qd_sweep_4k"
run_fio_test "qd_sweep_128k.fio" "qd_sweep_128k"

# 5. Tail latency tests
echo "=== Running tail latency tests ==="
run_fio_test "tail_latency_tests.fio" "tail_latency_tests"

echo "All tests completed. Results saved in ${RESULTS_DIR}"