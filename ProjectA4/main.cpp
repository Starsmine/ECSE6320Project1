/*
 * Project A4: Concurrent Data Structures and Memory Coherence
 * 
 * Thread-safe hash table with multiple synchronization strategies:
 * 1. Coarse-grained locking (single global mutex)
 * 2. Fine-grained locking (per-bucket locks)
 * 
 * Supports: insert(key, value), find(key), erase(key)
 */

#include <iostream>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <random>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iomanip>

//=============================================================================
// Common Types
//=============================================================================

using Key = int64_t;
using Value = int64_t;

struct Entry {
    Key key;
    Value value;
    bool occupied;
    
    Entry() : key(0), value(0), occupied(false) {}
    Entry(Key k, Value v) : key(k), value(v), occupied(true) {}
};

//=============================================================================
// Base Hash Table Interface
//=============================================================================

class HashTable {
public:
    virtual ~HashTable() = default;
    
    virtual bool insert(Key key, Value value) = 0;
    virtual bool find(Key key, Value* value) = 0;
    virtual bool erase(Key key) = 0;
    virtual size_t size() const = 0;
    virtual std::string name() const = 0;
};

//=============================================================================
// Coarse-Grained Hash Table (Single Global Lock)
//=============================================================================

class CoarseHashTable : public HashTable {
private:
    std::vector<std::vector<Entry>> buckets_;
    mutable std::mutex global_mutex_;
    std::atomic<size_t> size_;
    size_t num_buckets_;
    
    size_t hash(Key key) const {
        // Simple hash function
        return std::hash<Key>{}(key) % num_buckets_;
    }
    
public:
    CoarseHashTable(size_t num_buckets) 
        : buckets_(num_buckets), size_(0), num_buckets_(num_buckets) {
    }
    
    bool insert(Key key, Value value) override {
        std::lock_guard<std::mutex> lock(global_mutex_);
        
        size_t bucket_idx = hash(key);
        auto& bucket = buckets_[bucket_idx];
        
        // Check if key already exists
        for (auto& entry : bucket) {
            if (entry.occupied && entry.key == key) {
                entry.value = value;  // Update existing
                return true;
            }
        }
        
        // Insert new entry
        bucket.emplace_back(key, value);
        size_++;
        return true;
    }
    
    bool find(Key key, Value* value) override {
        std::lock_guard<std::mutex> lock(global_mutex_);
        
        size_t bucket_idx = hash(key);
        const auto& bucket = buckets_[bucket_idx];
        
        for (const auto& entry : bucket) {
            if (entry.occupied && entry.key == key) {
                if (value) *value = entry.value;
                return true;
            }
        }
        return false;
    }
    
    bool erase(Key key) override {
        std::lock_guard<std::mutex> lock(global_mutex_);
        
        size_t bucket_idx = hash(key);
        auto& bucket = buckets_[bucket_idx];
        
        for (auto& entry : bucket) {
            if (entry.occupied && entry.key == key) {
                entry.occupied = false;
                size_--;
                return true;
            }
        }
        return false;
    }
    
    size_t size() const override {
        return size_.load();
    }
    
    std::string name() const override {
        return "CoarseGrained";
    }
};

//=============================================================================
// Fine-Grained Hash Table (Per-Bucket Locks)
//=============================================================================

class FineHashTable : public HashTable {
private:
    std::vector<std::vector<Entry>> buckets_;
    std::vector<std::mutex> bucket_mutexes_;
    std::atomic<size_t> size_;
    size_t num_buckets_;
    
    size_t hash(Key key) const {
        return std::hash<Key>{}(key) % num_buckets_;
    }
    
public:
    FineHashTable(size_t num_buckets) 
        : buckets_(num_buckets), 
          bucket_mutexes_(num_buckets),
          size_(0), 
          num_buckets_(num_buckets) {
    }
    
    bool insert(Key key, Value value) override {
        size_t bucket_idx = hash(key);
        std::lock_guard<std::mutex> lock(bucket_mutexes_[bucket_idx]);
        
        auto& bucket = buckets_[bucket_idx];
        
        // Check if key already exists
        for (auto& entry : bucket) {
            if (entry.occupied && entry.key == key) {
                entry.value = value;  // Update existing
                return true;
            }
        }
        
        // Insert new entry
        bucket.emplace_back(key, value);
        size_++;
        return true;
    }
    
    bool find(Key key, Value* value) override {
        size_t bucket_idx = hash(key);
        std::lock_guard<std::mutex> lock(bucket_mutexes_[bucket_idx]);
        
        const auto& bucket = buckets_[bucket_idx];
        
        for (const auto& entry : bucket) {
            if (entry.occupied && entry.key == key) {
                if (value) *value = entry.value;
                return true;
            }
        }
        return false;
    }
    
    bool erase(Key key) override {
        size_t bucket_idx = hash(key);
        std::lock_guard<std::mutex> lock(bucket_mutexes_[bucket_idx]);
        
        auto& bucket = buckets_[bucket_idx];
        
        for (auto& entry : bucket) {
            if (entry.occupied && entry.key == key) {
                entry.occupied = false;
                size_--;
                return true;
            }
        }
        return false;
    }
    
    size_t size() const override {
        return size_.load();
    }
    
    std::string name() const override {
        return "FineGrained";
    }
};

//=============================================================================
// Benchmark Infrastructure
//=============================================================================

struct BenchmarkResult {
    double throughput_mops;
    double duration_sec;
    size_t total_ops;
    size_t num_threads;
};

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_seconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
};

void worker_thread(HashTable* table, 
                   const std::vector<Key>& keys,
                   size_t start_idx,
                   size_t end_idx,
                   const std::string& workload,
                   std::atomic<size_t>& ops_completed) {
    std::mt19937_64 rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = start_idx; i < end_idx; i++) {
        Key key = keys[i % keys.size()];
        Value value = key;
        
        if (workload == "lookup") {
            Value result;
            table->find(key, &result);
        } else if (workload == "insert") {
            table->insert(key, value);
        } else if (workload == "mixed") {
            // 70% lookups, 30% inserts
            if (dist(rng) < 0.7) {
                Value result;
                table->find(key, &result);
            } else {
                table->insert(key, value);
            }
        }
        
        ops_completed++;
    }
}

BenchmarkResult run_benchmark(HashTable* table,
                               const std::vector<Key>& keys,
                               size_t total_ops,
                               int num_threads,
                               const std::string& workload) {
    std::atomic<size_t> ops_completed{0};
    std::vector<std::thread> threads;
    
    size_t ops_per_thread = total_ops / num_threads;
    
    Timer timer;
    
    for (int t = 0; t < num_threads; t++) {
        size_t start_idx = t * ops_per_thread;
        size_t end_idx = (t == num_threads - 1) ? total_ops : (t + 1) * ops_per_thread;
        
        threads.emplace_back(worker_thread, table, std::ref(keys),
                           start_idx, end_idx, workload, std::ref(ops_completed));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    double elapsed = timer.elapsed_seconds();
    
    BenchmarkResult result;
    result.throughput_mops = ops_completed.load() / elapsed / 1e6;
    result.duration_sec = elapsed;
    result.total_ops = ops_completed.load();
    result.num_threads = num_threads;
    
    return result;
}

//=============================================================================
// Main
//=============================================================================

void print_usage() {
    std::cout << "Usage: hash_bench [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --table <coarse|fine>          Hash table type (default: coarse)\n";
    std::cout << "  --workload <lookup|insert|mixed> Workload type (default: mixed)\n";
    std::cout << "  --size <n>                     Number of keys (default: 100000)\n";
    std::cout << "  --ops <n>                      Total operations (default: 1000000)\n";
    std::cout << "  --threads <n>                  Number of threads (default: 4)\n";
    std::cout << "  --buckets <n>                  Number of buckets (default: 10000)\n";
    std::cout << "  --help                         Show this message\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string table_type = "coarse";
    std::string workload = "mixed";
    size_t num_keys = 100000;
    size_t total_ops = 1000000;
    int num_threads = 4;
    size_t num_buckets = 10000;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--table" && i + 1 < argc) {
            table_type = argv[++i];
        } else if (arg == "--workload" && i + 1 < argc) {
            workload = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            num_keys = std::stoull(argv[++i]);
        } else if (arg == "--ops" && i + 1 < argc) {
            total_ops = std::stoull(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--buckets" && i + 1 < argc) {
            num_buckets = std::stoull(argv[++i]);
        }
    }
    
    // Generate keys
    std::vector<Key> keys(num_keys);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < num_keys; i++) {
        keys[i] = rng();
    }
    
    // Create hash table
    HashTable* table = nullptr;
    if (table_type == "coarse") {
        table = new CoarseHashTable(num_buckets);
    } else if (table_type == "fine") {
        table = new FineHashTable(num_buckets);
    } else {
        std::cerr << "Unknown table type: " << table_type << "\n";
        return 1;
    }
    
    // Pre-populate with some data for lookup workloads
    if (workload == "lookup" || workload == "mixed") {
        for (size_t i = 0; i < num_keys / 2; i++) {
            table->insert(keys[i], keys[i]);
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Concurrent Hash Table Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Table type: " << table->name() << "\n";
    std::cout << "Workload: " << workload << "\n";
    std::cout << "Num keys: " << num_keys << "\n";
    std::cout << "Total ops: " << total_ops << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Buckets: " << num_buckets << "\n";
    std::cout << "========================================\n\n";
    
    // Run benchmark
    auto result = run_benchmark(table, keys, total_ops, num_threads, workload);
    
    std::cout << "Results:\n";
    std::cout << "  Duration: " << std::fixed << std::setprecision(3) 
              << result.duration_sec << " seconds\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << result.throughput_mops << " Mops/s\n";
    std::cout << "  Operations completed: " << result.total_ops << "\n";
    std::cout << "  Final table size: " << table->size() << "\n\n";
    
    delete table;
    return 0;
}
