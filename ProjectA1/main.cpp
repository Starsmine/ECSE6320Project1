#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <aio.h>
#include <liburing.h>
#include <sys/uio.h>

using namespace std;
using namespace std::chrono;

// Forward declarations for benchmark functions
void benchmark_cpu_affinity();
void benchmark_cache_prefetcher();
void benchmark_smt_interference();
void benchmark_huge_pages();
void benchmark_async_io();

void print_system_info() {
    cout << "\n======================================================================\n";
    cout << "System Configuration\n";
    cout << "======================================================================\n";
    
    // CPU info
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    cout << "Available CPUs: " << num_cpus << endl;
    
    // NUMA info
    if (numa_available() >= 0) {
        cout << "NUMA nodes: " << numa_num_configured_nodes() << endl;
        cout << "CPUs per node: " << numa_num_configured_cpus() / numa_num_configured_nodes() << endl;
    } else {
        cout << "NUMA: Not available\n";
    }
    
    // Page size
    long page_size = sysconf(_SC_PAGESIZE);
    cout << "Page size: " << (page_size / 1024) << " KB\n";
    
    cout << "======================================================================\n\n";
}

void print_usage() {
    cout << "Usage: ./os_features_bench [option]\n";
    cout << "Options:\n";
    cout << "  --all             Run all benchmarks\n";
    cout << "  --affinity        CPU affinity benchmark\n";
    cout << "  --prefetcher      Cache prefetcher benchmark\n";
    cout << "  --smt             SMT interference benchmark\n";
    cout << "  --hugepages       Huge pages benchmark\n";
    cout << "  --asyncio         Asynchronous I/O benchmark\n";
    cout << "  --help            Show this help message\n";
}

// Benchmark 1: CPU Affinity
void benchmark_cpu_affinity() {
    cout << "\n======================================================================\n";
    cout << "BENCHMARK 1: CPU AFFINITY\n";
    cout << "======================================================================\n";
    cout << "Testing performance impact of thread migration vs pinned threads...\n\n";
    
    const size_t ITERATIONS = 1000000000;
    const int NUM_THREADS = 4;
    
    // Test 1: No affinity (allow migration)
    auto start = high_resolution_clock::now();
    
    vector<thread> threads;
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([ITERATIONS]() {
            volatile long sum = 0;
            for (size_t i = 0; i < ITERATIONS; i++) {
                sum += i;
            }
        });
    }
    for (auto& th : threads) th.join();
    
    auto end = high_resolution_clock::now();
    auto duration_no_pin = duration_cast<milliseconds>(end - start).count();
    
    cout << "Without CPU pinning: " << duration_no_pin << " ms\n";
    
    // Test 2: With affinity (pinned threads)
    start = high_resolution_clock::now();
    
    threads.clear();
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([ITERATIONS, t]() {
            // Pin to specific CPU
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(t, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            
            volatile long sum = 0;
            for (size_t i = 0; i < ITERATIONS; i++) {
                sum += i;
            }
        });
    }
    for (auto& th : threads) th.join();
    
    end = high_resolution_clock::now();
    auto duration_pinned = duration_cast<milliseconds>(end - start).count();
    
    cout << "With CPU pinning:    " << duration_pinned << " ms\n";
    cout << "Speedup:             " << (double)duration_no_pin / duration_pinned << "x\n";
}

// Benchmark 2: Hardware Prefetcher
void benchmark_cache_prefetcher() {
    cout << "\n======================================================================\n";
    cout << "BENCHMARK 2: HARDWARE PREFETCHER EFFECTIVENESS\n";
    cout << "======================================================================\n";
    cout << "Testing sequential vs random access patterns to measure prefetcher impact...\n\n";
    
    const size_t SIZE = 64 * 1024 * 1024; // 64 MB (exceeds L3 cache)
    const size_t ITERATIONS = 10000000;
    const size_t CACHE_LINE_SIZE = 64; // bytes
    
    // Allocate array
    int* data = new int[SIZE / sizeof(int)];
    for (size_t i = 0; i < SIZE / sizeof(int); i++) {
        data[i] = i;
    }
    
    // Test 1: Sequential access (prefetcher friendly)
    cout << "Test 1: Sequential access (stride = 1 cache line)...\n";
    auto start = high_resolution_clock::now();
    volatile long sum = 0;
    for (size_t i = 0; i < ITERATIONS; i++) {
        size_t idx = (i * (CACHE_LINE_SIZE / sizeof(int))) % (SIZE / sizeof(int));
        sum += data[idx];
    }
    auto end = high_resolution_clock::now();
    auto sequential_time = duration_cast<milliseconds>(end - start).count();
    
    cout << "Sequential access: " << sequential_time << " ms\n";
    cout << "Bandwidth: " << (ITERATIONS * sizeof(int) / 1024.0 / 1024.0) / (sequential_time / 1000.0) << " MB/s\n\n";
    
    // Test 2: Strided access (moderate prefetcher effectiveness)
    cout << "Test 2: Strided access (stride = 16 cache lines)...\n";
    start = high_resolution_clock::now();
    sum = 0;
    for (size_t i = 0; i < ITERATIONS; i++) {
        size_t idx = (i * 16 * (CACHE_LINE_SIZE / sizeof(int))) % (SIZE / sizeof(int));
        sum += data[idx];
    }
    end = high_resolution_clock::now();
    auto strided_time = duration_cast<milliseconds>(end - start).count();
    
    cout << "Strided access:    " << strided_time << " ms\n";
    cout << "Bandwidth: " << (ITERATIONS * sizeof(int) / 1024.0 / 1024.0) / (strided_time / 1000.0) << " MB/s\n";
    cout << "Slowdown vs sequential: " << (double)strided_time / sequential_time << "x\n\n";
    
    // Test 3: Random access (defeats prefetcher)
    cout << "Test 3: Random access (defeats prefetcher)...\n";
    
    // Pre-generate random indices
    size_t* indices = new size_t[ITERATIONS];
    for (size_t i = 0; i < ITERATIONS; i++) {
        indices[i] = (rand() * 37) % (SIZE / sizeof(int));
    }
    
    start = high_resolution_clock::now();
    sum = 0;
    for (size_t i = 0; i < ITERATIONS; i++) {
        sum += data[indices[i]];
    }
    end = high_resolution_clock::now();
    auto random_time = duration_cast<milliseconds>(end - start).count();
    
    cout << "Random access:     " << random_time << " ms\n";
    cout << "Bandwidth: " << (ITERATIONS * sizeof(int) / 1024.0 / 1024.0) / (random_time / 1000.0) << " MB/s\n";
    cout << "Slowdown vs sequential: " << (double)random_time / sequential_time << "x\n\n";
    
    cout << "Analysis:\n";
    cout << "  Prefetcher benefit: " << ((double)random_time / sequential_time - 1) * 100 << "% performance gain\n";
    cout << "  Sequential is " << (double)random_time / sequential_time << "x faster than random\n";
    
    delete[] data;
    delete[] indices;
}

// Benchmark 3: SMT Interference
void benchmark_smt_interference() {
    cout << "\n======================================================================\n";
    cout << "BENCHMARK 3: SMT (HYPERTHREADING) INTERFERENCE\n";
    cout << "======================================================================\n";
    cout << "Testing performance impact of sibling threads on same physical core...\n\n";
    
    const size_t ITERATIONS = 500000000;
    
    // Test 1: Single thread on physical core 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    auto start = high_resolution_clock::now();
    volatile long sum = 0;
    for (size_t i = 0; i < ITERATIONS; i++) {
        sum += i * i;
    }
    auto end = high_resolution_clock::now();
    auto single_time = duration_cast<milliseconds>(end - start).count();
    
    cout << "Single thread (no SMT):      " << single_time << " ms\n";
    
    // Test 2: Two threads on sibling cores (assuming CPU 0 and 8 are siblings on 16-core system)
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    int sibling_cpu = num_cpus / 2; // Typical sibling mapping
    
    start = high_resolution_clock::now();
    
    thread t1([ITERATIONS]() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        
        volatile long sum = 0;
        for (size_t i = 0; i < ITERATIONS; i++) {
            sum += i * i;
        }
    });
    
    thread t2([ITERATIONS, sibling_cpu]() {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(sibling_cpu, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        
        volatile long sum = 0;
        for (size_t i = 0; i < ITERATIONS; i++) {
            sum += i * i;
        }
    });
    
    t1.join();
    t2.join();
    
    end = high_resolution_clock::now();
    auto smt_time = duration_cast<milliseconds>(end - start).count();
    
    cout << "Two threads (with SMT):      " << smt_time << " ms\n";
    cout << "Efficiency:                  " << (double)(2 * single_time) / smt_time << "x\n";
    cout << "SMT overhead:                " << ((double)smt_time / single_time - 1) * 100 << "%\n";
}

// Benchmark 4: Huge Pages
void benchmark_huge_pages() {
    cout << "\n======================================================================\n";
    cout << "BENCHMARK 4: TRANSPARENT HUGE PAGES\n";
    cout << "======================================================================\n";
    cout << "Testing TLB miss reduction with huge pages...\n\n";
    
    const size_t SIZE = 1024 * 1024 * 1024; // 1 GB
    const size_t ITERATIONS = 100000000;
    
    // Test with regular pages
    char* data_regular = (char*)mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(data_regular, 1, SIZE);
    
    auto start = high_resolution_clock::now();
    volatile long sum = 0;
    for (size_t i = 0; i < ITERATIONS; i++) {
        sum += data_regular[(i * 4096) % SIZE]; // Jump by page size
    }
    auto end = high_resolution_clock::now();
    auto regular_time = duration_cast<milliseconds>(end - start).count();
    
    munmap(data_regular, SIZE);
    
    cout << "Regular pages (4KB):  " << regular_time << " ms\n";
    
    // Test with huge pages
    char* data_huge = (char*)mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    
    if (data_huge == MAP_FAILED) {
        cout << "Huge pages allocation failed. Try: sudo sysctl -w vm.nr_hugepages=1000\n";
        return;
    }
    
    memset(data_huge, 1, SIZE);
    
    start = high_resolution_clock::now();
    sum = 0;
    for (size_t i = 0; i < ITERATIONS; i++) {
        sum += data_huge[(i * 4096) % SIZE];
    }
    end = high_resolution_clock::now();
    auto huge_time = duration_cast<milliseconds>(end - start).count();
    
    munmap(data_huge, SIZE);
    
    cout << "Huge pages (2MB):     " << huge_time << " ms\n";
    cout << "Speedup:              " << (double)regular_time / huge_time << "x\n";
}

// Benchmark 5: Asynchronous I/O (io_uring vs POSIX AIO)
void benchmark_async_io() {
    cout << "\n======================================================================\n";
    cout << "BENCHMARK 5: ASYNCHRONOUS I/O (io_uring + zero-copy)\n";
    cout << "======================================================================\n";
    cout << "Comparing sync I/O, POSIX AIO, and io_uring with registered buffers...\n\n";
    
    const char* filename = "/tmp/bench_io_test.dat";
    const size_t FILE_SIZE = 256 * 1024 * 1024; // 256 MB
    const size_t IO_BLOCK_SIZE = 4096; // 4 KB blocks
    const int NUM_OPS = 1000;
    
    // Create test file
    int fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) {
        cout << "Failed to create test file\n";
        return;
    }
    
    // Fill with data
    char* buffer = new char[FILE_SIZE];
    memset(buffer, 0xAB, FILE_SIZE);
    write(fd, buffer, FILE_SIZE);
    close(fd);
    
    // Test 1: Synchronous I/O
    fd = open(filename, O_RDONLY | O_DIRECT);
    char* read_buf = (char*)aligned_alloc(IO_BLOCK_SIZE, IO_BLOCK_SIZE * NUM_OPS);
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < NUM_OPS; i++) {
        pread(fd, read_buf + i * IO_BLOCK_SIZE, IO_BLOCK_SIZE, (i * IO_BLOCK_SIZE * 256) % FILE_SIZE);
    }
    auto end = high_resolution_clock::now();
    auto sync_time = duration_cast<milliseconds>(end - start).count();
    close(fd);
    
    cout << "Synchronous I/O:  " << sync_time << " ms (" << NUM_OPS << " operations)\n";
    
    // Test 2: Asynchronous I/O
    fd = open(filename, O_RDONLY | O_DIRECT);
    
    struct aiocb* aio_requests = new struct aiocb[NUM_OPS];
    memset(aio_requests, 0, sizeof(struct aiocb) * NUM_OPS);
    
    start = high_resolution_clock::now();
    
    // Submit all async requests
    for (int i = 0; i < NUM_OPS; i++) {
        aio_requests[i].aio_fildes = fd;
        aio_requests[i].aio_buf = read_buf + i * IO_BLOCK_SIZE;
        aio_requests[i].aio_nbytes = IO_BLOCK_SIZE;
        aio_requests[i].aio_offset = (i * IO_BLOCK_SIZE * 256) % FILE_SIZE;
        aio_read(&aio_requests[i]);
    }
    
    // Wait for all to complete
    for (int i = 0; i < NUM_OPS; i++) {
        while (aio_error(&aio_requests[i]) == EINPROGRESS) {
            usleep(10);
        }
    }
    
    end = high_resolution_clock::now();
    auto posix_aio_time = duration_cast<milliseconds>(end - start).count();
    
    close(fd);
    delete[] aio_requests;
    
    cout << "POSIX AIO:        " << posix_aio_time << " ms (" << NUM_OPS << " operations)\n";
    
    // Test 3: io_uring Asynchronous I/O with registered buffers (zero-copy)
    fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        cout << "Failed to open file for io_uring test\n";
        free(read_buf);
        delete[] buffer;
        unlink(filename);
        return;
    }
    
    const int QUEUE_DEPTH = 32; // Submit up to 32 operations at once
    
    struct io_uring ring;
    int ret = io_uring_queue_init(QUEUE_DEPTH, &ring, 0);
    if (ret < 0) {
        cout << "Failed to initialize io_uring: " << strerror(-ret) << "\n";
        close(fd);
        free(read_buf);
        delete[] buffer;
        unlink(filename);
        return;
    }
    
    // Register buffers for zero-copy I/O
    struct iovec iov[NUM_OPS];
    for (int i = 0; i < NUM_OPS; i++) {
        iov[i].iov_base = read_buf + i * IO_BLOCK_SIZE;
        iov[i].iov_len = IO_BLOCK_SIZE;
    }
    int reg_ret = io_uring_register_buffers(&ring, iov, NUM_OPS);
    bool zero_copy = (reg_ret == 0);
    if (reg_ret < 0) {
        cout << "Note: Buffer registration failed (" << strerror(-reg_ret) << "), using standard async\n";
    }
    
    start = high_resolution_clock::now();
    
    int submitted = 0;
    int completed = 0;
    
    // Submit operations in batches
    while (submitted < NUM_OPS || completed < NUM_OPS) {
        // Submit up to QUEUE_DEPTH operations
        while (submitted < NUM_OPS && (submitted - completed) < QUEUE_DEPTH) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            
            off_t offset = (submitted * IO_BLOCK_SIZE * 256) % FILE_SIZE;
            
            if (zero_copy) {
                // Use registered buffer (zero-copy)
                io_uring_prep_read_fixed(sqe, fd, read_buf + submitted * IO_BLOCK_SIZE, 
                                        IO_BLOCK_SIZE, offset, submitted);
            } else {
                // Fallback to regular async read
                io_uring_prep_read(sqe, fd, read_buf + submitted * IO_BLOCK_SIZE, 
                                  IO_BLOCK_SIZE, offset);
            }
            
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)submitted);
            submitted++;
        }
        
        // Submit all prepared operations
        int to_submit = io_uring_submit(&ring);
        if (to_submit < 0) {
            cout << "io_uring_submit failed: " << strerror(-to_submit) << "\n";
            break;
        }
        
        // Wait for at least one completion
        if (completed < NUM_OPS) {
            struct io_uring_cqe *cqe;
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                cout << "io_uring_wait_cqe failed: " << strerror(-ret) << "\n";
                break;
            }
            
            // Process all available completions
            unsigned head;
            unsigned count = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                if (cqe->res < 0) {
                    cout << "I/O error: " << strerror(-cqe->res) << "\n";
                }
                completed++;
                count++;
            }
            io_uring_cq_advance(&ring, count);
        }
    }
    
    end = high_resolution_clock::now();
    auto uring_time = duration_cast<milliseconds>(end - start).count();
    
    // Cleanup
    if (zero_copy) {
        io_uring_unregister_buffers(&ring);
    }
    io_uring_queue_exit(&ring);
    close(fd);
    free(read_buf);
    delete[] buffer;
    unlink(filename);
    
    cout << "io_uring:         " << uring_time << " ms (" << NUM_OPS << " operations, depth=" << QUEUE_DEPTH;
    if (zero_copy) cout << ", zero-copy";
    cout << ")\n";
    cout << "\nSpeedup vs sync:\n";
    cout << "  POSIX AIO:      " << (double)sync_time / posix_aio_time << "x\n";
    cout << "  io_uring:       " << (double)sync_time / uring_time << "x\n";
    cout << "\nThroughput:\n";
    cout << "  Sync I/O:       " << (NUM_OPS * IO_BLOCK_SIZE / 1024.0 / 1024.0) / (sync_time / 1000.0) << " MB/s\n";
    cout << "  POSIX AIO:      " << (NUM_OPS * IO_BLOCK_SIZE / 1024.0 / 1024.0) / (posix_aio_time / 1000.0) << " MB/s\n";
    cout << "  io_uring:       " << (NUM_OPS * IO_BLOCK_SIZE / 1024.0 / 1024.0) / (uring_time / 1000.0) << " MB/s\n";
}

int main(int argc, char* argv[]) {
    print_system_info();
    
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    string arg = argv[1];
    
    if (arg == "--help") {
        print_usage();
    } else if (arg == "--all") {
        benchmark_cpu_affinity();
        benchmark_cache_prefetcher();
        benchmark_smt_interference();
        benchmark_huge_pages();
        benchmark_async_io();
    } else if (arg == "--affinity") {
        benchmark_cpu_affinity();
    } else if (arg == "--prefetcher") {
        benchmark_cache_prefetcher();
    } else if (arg == "--smt") {
        benchmark_smt_interference();
    } else if (arg == "--hugepages") {
        benchmark_huge_pages();
    } else if (arg == "--asyncio") {
        benchmark_async_io();
    } else {
        cout << "Unknown option: " << arg << endl;
        print_usage();
        return 1;
    }
    
    return 0;
}
