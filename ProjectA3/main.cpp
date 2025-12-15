/*
 * Project A3: Approximate Membership Filters
 * XOR Filter, Cuckoo Filter, Quotient Filter, Blocked Bloom Filter
 * 
 * Implements all four filter types with common API for fair comparisons
 * Benchmarks: space efficiency, throughput, tail latency, thread scaling
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <omp.h>
#include <time.h>

// Use xxHash for fast non-cryptographic hashing
#include <xxhash.h>

// Global log file
std::ofstream g_logfile;

#define LOG(x) do { std::cout << x; if (g_logfile.is_open()) g_logfile << x; } while(0)

//=============================================================================
// Utility Functions
//=============================================================================

// High-precision timer using CLOCK_MONOTONIC (nanosecond precision)
inline double get_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

class Timer {
    double start_;
public:
    Timer() : start_(get_timestamp()) {}
    
    double elapsed_seconds() const {
        return get_timestamp() - start_;
    }
    
    uint64_t elapsed_nanoseconds() const {
        return static_cast<uint64_t>((get_timestamp() - start_) * 1e9);
    }
    
    void reset() { start_ = get_timestamp(); }
};

// Calculate percentile from sorted vector
template<typename T>
T percentile(std::vector<T>& data, double p) {
    if (data.empty()) return T{};
    std::sort(data.begin(), data.end());
    size_t idx = static_cast<size_t>(p * data.size());
    if (idx >= data.size()) idx = data.size() - 1;
    return data[idx];
}

// Hash functions using xxHash with different seeds
inline uint64_t hash1(uint64_t key) {
    return XXH64(&key, sizeof(key), 0);
}

inline uint64_t hash2(uint64_t key) {
    return XXH64(&key, sizeof(key), 1);
}

inline uint64_t hash3(uint64_t key) {
    return XXH64(&key, sizeof(key), 2);
}

// Extract fingerprint from hash
inline uint32_t fingerprint(uint64_t hash, int bits) {
    return static_cast<uint32_t>(hash & ((1ULL << bits) - 1));
}

//=============================================================================
// Filter Interface
//=============================================================================

class Filter {
public:
    virtual ~Filter() = default;
    
    // Core operations
    virtual bool insert(uint64_t key) = 0;
    virtual bool query(uint64_t key) const = 0;
    virtual bool remove(uint64_t key) = 0;
    
    // Statistics
    virtual size_t size_bytes() const = 0;
    virtual double bits_per_entry() const = 0;
    virtual std::string name() const = 0;
    virtual void print_stats() const = 0;
    virtual double load_factor() const { return 0.0; }
    virtual size_t get_num_items() const { return 0; }
    virtual size_t insertion_failures() const { return 0; }
    virtual double avg_probe_length() const { return 0.0; }
    
    // Helper for testing
    double measure_fpr(const std::vector<uint64_t>& negative_set) const {
        size_t false_positives = 0;
        for (uint64_t key : negative_set) {
            if (query(key)) {
                false_positives++;
            }
        }
        return static_cast<double>(false_positives) / negative_set.size();
    }
};

//=============================================================================
// Blocked Bloom Filter (Baseline)
//=============================================================================

class BlockedBloomFilter : public Filter {
private:
    static constexpr size_t BLOCK_SIZE = 64; // Cache line size
    static constexpr size_t BITS_PER_BLOCK = BLOCK_SIZE * 8;
    
    std::vector<uint8_t> blocks_;
    size_t num_blocks_;
    size_t num_keys_;
    int k_; // Number of hash functions
    
public:
    BlockedBloomFilter(size_t n, double fpr) {
        // Calculate optimal parameters
        if (n == 0) n = 1; // Safety check
        if (fpr <= 0.0) fpr = 0.001; // Minimum FPR
        if (fpr >= 1.0) fpr = 0.5; // Maximum FPR
        num_keys_ = n;
        
        double ln2 = std::log(2.0);
        // Formula: m = -n * ln(p) / (ln(2))^2
        double total_bits_d = -static_cast<double>(n) * std::log(fpr) / (ln2 * ln2);
        size_t total_bits = static_cast<size_t>(std::ceil(total_bits_d));
        
        // Ensure at least one block
        total_bits = std::max(total_bits, BITS_PER_BLOCK);
        
        num_blocks_ = (total_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        blocks_.resize(num_blocks_ * BLOCK_SIZE, 0);
        
        k_ = static_cast<int>(std::ceil(ln2 * total_bits / n));
        k_ = std::max(1, std::min(k_, 16)); // Reasonable bounds
    }
    
    bool insert(uint64_t key) override {
        if (num_blocks_ == 0) return false; // Safety check
        
        uint64_t h1 = hash1(key);
        uint64_t h2 = hash2(key);
        
        size_t block_idx = h1 % num_blocks_;
        uint8_t* block = &blocks_[block_idx * BLOCK_SIZE];
        
        for (int i = 0; i < k_; i++) {
            uint64_t hash = h1 + i * h2;
            size_t bit_idx = hash % BITS_PER_BLOCK;
            block[bit_idx / 8] |= (1 << (bit_idx % 8));
        }
        return true;
    }
    
    bool query(uint64_t key) const override {
        if (num_blocks_ == 0) return false; // Safety check
        
        uint64_t h1 = hash1(key);
        uint64_t h2 = hash2(key);
        
        size_t block_idx = h1 % num_blocks_;
        const uint8_t* block = &blocks_[block_idx * BLOCK_SIZE];
        
        for (int i = 0; i < k_; i++) {
            uint64_t hash = h1 + i * h2;
            size_t bit_idx = hash % BITS_PER_BLOCK;
            if (!(block[bit_idx / 8] & (1 << (bit_idx % 8)))) {
                return false;
            }
        }
        return true;
    }
    
    bool remove(uint64_t key) override {
        (void)key;
        return false; // Bloom filters don't support deletion
    }
    
    size_t size_bytes() const override {
        return blocks_.size();
    }
    
    double bits_per_entry() const override {
        return (blocks_.size() * 8.0) / num_keys_;
    }
    
    std::string name() const override {
        return "BlockedBloom";
    }
    
    void print_stats() const override {
        LOG("Blocked Bloom Filter:\n");
        LOG("  Blocks: " << num_blocks_ << "\n");
        LOG("  Hash functions (k): " << k_ << "\n");
        LOG("  Size: " << size_bytes() / 1024.0 / 1024.0 << " MB\n");
        LOG("  Size bytes: " << size_bytes() << "\n");
        LOG("  Num keys: " << num_keys_ << "\n");
        LOG("  Bits per entry: " << bits_per_entry() << "\n");
    }
    
    size_t get_num_items() const override { return num_keys_; }
};

//=============================================================================
// XOR Filter (Static)
//=============================================================================

class XORFilter : public Filter {
private:
    std::vector<uint8_t> fingerprint_data_;  // Bit-packed storage
    size_t segment_length_;
    int fingerprint_bits_;
    size_t num_keys_;
    size_t num_fingerprints_;
    
    // Helper to get/set fingerprints from bit-packed array
    uint32_t get_fingerprint(size_t index) const {
        size_t bit_pos = index * fingerprint_bits_;
        size_t byte_pos = bit_pos / 8;
        size_t bit_offset = bit_pos % 8;
        
        uint32_t value = 0;
        int bits_remaining = fingerprint_bits_;
        int value_offset = 0;
        
        while (bits_remaining > 0 && byte_pos < fingerprint_data_.size()) {
            int bits_in_byte = std::min(8 - static_cast<int>(bit_offset), bits_remaining);
            uint8_t mask = ((1 << bits_in_byte) - 1);
            uint8_t bits = (fingerprint_data_[byte_pos] >> bit_offset) & mask;
            value |= (static_cast<uint32_t>(bits) << value_offset);
            
            bits_remaining -= bits_in_byte;
            value_offset += bits_in_byte;
            byte_pos++;
            bit_offset = 0;
        }
        
        return value;
    }
    
    void set_fingerprint(size_t index, uint32_t value) {
        size_t bit_pos = index * fingerprint_bits_;
        size_t byte_pos = bit_pos / 8;
        size_t bit_offset = bit_pos % 8;
        
        int bits_remaining = fingerprint_bits_;
        
        while (bits_remaining > 0 && byte_pos < fingerprint_data_.size()) {
            int bits_in_byte = std::min(8 - static_cast<int>(bit_offset), bits_remaining);
            uint8_t mask = ((1 << bits_in_byte) - 1);
            
            // Clear existing bits
            fingerprint_data_[byte_pos] &= ~(mask << bit_offset);
            // Set new bits
            fingerprint_data_[byte_pos] |= ((value & mask) << bit_offset);
            
            value >>= bits_in_byte;
            bits_remaining -= bits_in_byte;
            byte_pos++;
            bit_offset = 0;
        }
    }
    
    void xor_fingerprint(size_t index, uint32_t value) {
        uint32_t current = get_fingerprint(index);
        set_fingerprint(index, current ^ value);
    }
    
public:
    XORFilter(const std::vector<uint64_t>& keys, int fp_bits = 8) 
        : fingerprint_bits_(fp_bits), num_keys_(keys.size()) {
        
        // XOR filter uses ~1.23n entries for 3-wise independence
        size_t capacity = static_cast<size_t>(keys.size() * 1.23);
        segment_length_ = (capacity + 2) / 3;
        num_fingerprints_ = segment_length_ * 3;
        
        // Allocate bit-packed storage
        size_t total_bits = num_fingerprints_ * fingerprint_bits_;
        size_t total_bytes = (total_bits + 7) / 8;
        fingerprint_data_.resize(total_bytes, 0);
        
        build(keys);
    }
    
    void build(const std::vector<uint64_t>& keys) {
        // Simplified XOR filter construction
        std::fill(fingerprint_data_.begin(), fingerprint_data_.end(), 0);
        
        for (uint64_t key : keys) {
            uint64_t h1 = hash1(key);
            uint64_t h2 = hash2(key);
            uint64_t h3 = hash3(key);
            
            size_t idx1 = h1 % segment_length_;
            size_t idx2 = segment_length_ + (h2 % segment_length_);
            size_t idx3 = 2 * segment_length_ + (h3 % segment_length_);
            
            uint32_t fp = fingerprint(h1, fingerprint_bits_);
            if (fp == 0) fp = 1; // Avoid zero fingerprints
            
            // XOR the fingerprint into all three positions
            xor_fingerprint(idx1, fp);
            xor_fingerprint(idx2, fp);
            xor_fingerprint(idx3, fp);
        }
    }
    
    bool insert(uint64_t key) override {
        (void)key;
        return false; // XOR filter is static
    }
    
    bool query(uint64_t key) const override {
        uint64_t h1 = hash1(key);
        uint64_t h2 = hash2(key);
        uint64_t h3 = hash3(key);
        
        size_t idx1 = h1 % segment_length_;
        size_t idx2 = segment_length_ + (h2 % segment_length_);
        size_t idx3 = 2 * segment_length_ + (h3 % segment_length_);
        
        uint32_t fp = fingerprint(h1, fingerprint_bits_);
        if (fp == 0) fp = 1;
        
        // Check if XOR of three positions equals fingerprint
        uint32_t xor_result = get_fingerprint(idx1) ^ get_fingerprint(idx2) ^ get_fingerprint(idx3);
        return xor_result == fp;
    }
    
    bool remove(uint64_t key) override {
        (void)key;
        return false; // XOR filter doesn't support deletion
    }
    
    size_t size_bytes() const override {
        return fingerprint_data_.size();
    }
    
    double bits_per_entry() const override {
        // BPE = (total bits used) / number of keys
        return (fingerprint_data_.size() * 8.0) / static_cast<double>(num_keys_);
    }
    
    std::string name() const override {
        return "XOR";
    }
    
    void print_stats() const override {
        LOG("XOR Filter:\n");
        LOG("  Fingerprints: " << num_fingerprints_ << "\n");
        LOG("  Fingerprint bits: " << fingerprint_bits_ << "\n");
        LOG("  Storage bytes: " << fingerprint_data_.size() << "\n");
        LOG("  Size: " << size_bytes() / 1024.0 / 1024.0 << " MB\n");
        LOG("  Bits per entry: " << bits_per_entry() << "\n");
    }
    
    size_t get_num_items() const override { return num_keys_; }
};

//=============================================================================
// Cuckoo Filter (Dynamic)
//=============================================================================

class CuckooFilter : public Filter {
private:
    static constexpr int BUCKET_SIZE = 4;
    static constexpr int MAX_KICKS = 100;  // Reduced to trigger failures sooner
    
    struct Bucket {
        uint32_t fingerprints[BUCKET_SIZE];
        
        Bucket() {
            std::memset(fingerprints, 0, sizeof(fingerprints));
        }
        
        bool insert(uint32_t fp) {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                if (fingerprints[i] == 0) {
                    fingerprints[i] = fp;
                    return true;
                }
            }
            return false;
        }
        
        bool contains(uint32_t fp) const {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                if (fingerprints[i] == fp) {
                    return true;
                }
            }
            return false;
        }
        
        bool remove(uint32_t fp) {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                if (fingerprints[i] == fp) {
                    fingerprints[i] = 0;
                    return true;
                }
            }
            return false;
        }
    };
    
    std::vector<Bucket> buckets_;
    std::vector<uint32_t> stash_; // Small overflow buffer
    size_t num_items_;
    size_t insertion_failures_;
    int fingerprint_bits_;
    
    // Statistics
    mutable size_t total_kicks_;
    mutable size_t total_insertions_;
    mutable size_t stash_queries_;
    mutable size_t stash_hits_;
    
    size_t alt_index(size_t idx, uint32_t fp) const {
        uint64_t hash = XXH64(&fp, sizeof(fp), 3);
        return (idx ^ hash) % buckets_.size();
    }
    
public:
    CuckooFilter(size_t capacity, int fp_bits = 12) 
        : num_items_(0), insertion_failures_(0), fingerprint_bits_(fp_bits),
          total_kicks_(0), total_insertions_(0), stash_queries_(0), stash_hits_(0) {
        
        size_t num_buckets = capacity / BUCKET_SIZE;
        buckets_.resize(num_buckets);
        stash_.reserve(8); // Small stash
    }
    
    bool insert(uint64_t key) override {
        uint64_t h = hash1(key);
        uint32_t fp = fingerprint(h, fingerprint_bits_);
        if (fp == 0) fp = 1;
        
        size_t idx1 = h % buckets_.size();
        size_t idx2 = alt_index(idx1, fp);
        
        // Try inserting in primary bucket
        if (buckets_[idx1].insert(fp)) {
            num_items_++;
            total_insertions_++;
            return true;
        }
        
        // Try alternate bucket
        if (buckets_[idx2].insert(fp)) {
            num_items_++;
            total_insertions_++;
            return true;
        }
        
        // Cuckoo eviction
        size_t idx = idx1;
        std::mt19937_64 rng(key);
        
        for (int kick = 0; kick < MAX_KICKS; kick++) {
            total_kicks_++;
            // Random eviction slot
            int slot = rng() % BUCKET_SIZE;
            std::swap(fp, buckets_[idx].fingerprints[slot]);
            
            idx = alt_index(idx, fp);
            if (buckets_[idx].insert(fp)) {
                num_items_++;
                total_insertions_++;
                return true;
            }
        }
        
        // Failed - use stash
        if (stash_.size() < stash_.capacity()) {
            stash_.push_back(fp);
            num_items_++;
            total_insertions_++;
            return true;
        }
        
        insertion_failures_++;
        total_insertions_++;  // Count failed insertions too
        return false; // Filter is full
    }
    
    bool query(uint64_t key) const override {
        uint64_t h = hash1(key);
        uint32_t fp = fingerprint(h, fingerprint_bits_);
        if (fp == 0) fp = 1;
        
        size_t idx1 = h % buckets_.size();
        size_t idx2 = alt_index(idx1, fp);
        
        if (buckets_[idx1].contains(fp) || buckets_[idx2].contains(fp)) {
            return true;
        }
        
        // Check stash
        stash_queries_++;
        bool found = std::find(stash_.begin(), stash_.end(), fp) != stash_.end();
        if (found) stash_hits_++;
        return found;
    }
    
    bool remove(uint64_t key) override {
        uint64_t h = hash1(key);
        uint32_t fp = fingerprint(h, fingerprint_bits_);
        if (fp == 0) fp = 1;
        
        size_t idx1 = h % buckets_.size();
        size_t idx2 = alt_index(idx1, fp);
        
        if (buckets_[idx1].remove(fp) || buckets_[idx2].remove(fp)) {
            num_items_--;
            return true;
        }
        
        // Check stash
        auto it = std::find(stash_.begin(), stash_.end(), fp);
        if (it != stash_.end()) {
            stash_.erase(it);
            num_items_--;
            return true;
        }
        
        return false;
    }
    
    size_t size_bytes() const override {
        return buckets_.size() * sizeof(Bucket) + stash_.capacity() * sizeof(uint32_t);
    }
    
    double bits_per_entry() const override {
        if (num_items_ == 0) return 0;
        // BPE = (buckets × slots/bucket × bits/fingerprint) / items
        size_t total_slots = buckets_.size() * BUCKET_SIZE;
        return (total_slots * fingerprint_bits_) / static_cast<double>(num_items_);
    }
    
    double load_factor() const override {
        size_t capacity = buckets_.size() * BUCKET_SIZE;
        return static_cast<double>(num_items_) / capacity;
    }
    
    std::string name() const override {
        return "Cuckoo";
    }
    
    size_t get_num_items() const override { return num_items_; }
    size_t insertion_failures() const override { return insertion_failures_; }
    
    void print_probe_stats() const {
        double avg_kicks = total_insertions_ > 0 ? static_cast<double>(total_kicks_) / total_insertions_ : 0.0;
        LOG("  Average kicks per insertion: " << avg_kicks << "\n");
        if (stash_queries_ > 0) {
            double stash_hit_rate = static_cast<double>(stash_hits_) / stash_queries_ * 100;
            LOG("  Stash hit rate: " << stash_hit_rate << "% (" << stash_hits_ << "/" << stash_queries_ << ")\n");
        }
    }
    
    void print_stats() const override {
        LOG("Cuckoo Filter:\n");
        LOG("  Buckets: " << buckets_.size() << "\n");
        LOG("  Bucket size: " << BUCKET_SIZE << "\n");
        LOG("  Items: " << num_items_ << "\n");
        LOG("  Load factor: " << load_factor() * 100 << "%\n");
        LOG("  Stash size: " << stash_.size() << "\n");
        LOG("  Insertion failures: " << insertion_failures_ << "\n");
        LOG("  Size: " << size_bytes() / 1024.0 / 1024.0 << " MB\n");
        LOG("  Bits per entry: " << bits_per_entry() << "\n");
    }
};

//=============================================================================
// Quotient Filter (Dynamic)
//=============================================================================

class QuotientFilter : public Filter {
private:
    struct Slot {
        uint32_t remainder : 24;
        uint8_t occupied : 1;
        uint8_t continuation : 1;
        uint8_t shifted : 1;
        uint8_t padding : 5;
        
        Slot() : remainder(0), occupied(0), continuation(0), shifted(0), padding(0) {}
    };
    
    std::vector<Slot> table_;
    size_t q_bits_; // Quotient bits
    size_t r_bits_; // Remainder bits
    size_t num_items_;
    
    // Statistics
    mutable std::vector<size_t> cluster_length_histogram_; // cluster_length_histogram_[len] = count
    mutable size_t total_probes_;
    mutable size_t total_queries_;
    
    void get_hash_parts(uint64_t key, size_t& quotient, uint32_t& remainder) const {
        uint64_t h = hash1(key);
        quotient = h % table_.size();
        remainder = (h / table_.size()) & ((1ULL << r_bits_) - 1);
        if (remainder == 0) remainder = 1;
    }
    
    size_t find_run_start(size_t quotient) const {
        size_t b = quotient;
        while (b > 0 && table_[b].shifted) {
            b--;
        }
        
        size_t s = b;
        while (b < quotient) {
            if (table_[b].occupied) {
                do {
                    s = (s + 1) % table_.size();
                } while (table_[s].continuation);
            }
            b++;
        }
        return s;
    }
    
public:
    QuotientFilter(size_t capacity, int remainder_bits = 8) 
        : r_bits_(remainder_bits), num_items_(0), 
          cluster_length_histogram_(100, 0), total_probes_(0), total_queries_(0) {
        
        q_bits_ = static_cast<size_t>(std::ceil(std::log2(capacity)));
        table_.resize(1ULL << q_bits_);
    }
    
    bool insert(uint64_t key) override {
        size_t quotient;
        uint32_t remainder;
        get_hash_parts(key, quotient, remainder);
        
        // Simplified insertion (full implementation needs cluster management)
        size_t idx = find_run_start(quotient);
        
        // Find insertion point in run
        while (table_[idx].continuation && table_[idx].remainder < remainder) {
            idx = (idx + 1) % table_.size();
        }
        
        // Insert if not duplicate
        if (table_[idx].remainder == remainder && 
            (table_[idx].occupied || table_[idx].continuation)) {
            return true; // Already present
        }
        
        // Shift and insert
        Slot new_slot;
        new_slot.remainder = remainder;
        new_slot.occupied = (idx == quotient);
        new_slot.continuation = (idx != quotient) || table_[quotient].occupied;
        new_slot.shifted = (idx != quotient);
        
        // Simple shift (production version needs proper cluster handling)
        if (table_[idx].remainder != 0) {
            // Slot occupied, shift right
            size_t shift_idx = idx;
            while (table_[shift_idx].remainder != 0) {
                shift_idx = (shift_idx + 1) % table_.size();
            }
            while (shift_idx != idx) {
                size_t prev = (shift_idx - 1 + table_.size()) % table_.size();
                table_[shift_idx] = table_[prev];
                table_[shift_idx].shifted = 1;
                shift_idx = prev;
            }
        }
        
        table_[idx] = new_slot;
        table_[quotient].occupied = 1;
        num_items_++;
        return true;
    }
    
    bool query(uint64_t key) const override {
        total_queries_++;
        size_t quotient;
        uint32_t remainder;
        get_hash_parts(key, quotient, remainder);
        
        if (!table_[quotient].occupied) {
            return false;
        }
        
        size_t idx = find_run_start(quotient);
        size_t cluster_start = idx;
        size_t probes = 0;
        
        do {
            probes++;
            total_probes_++;
            if (table_[idx].remainder == remainder) {
                // Track cluster length
                if (probes < cluster_length_histogram_.size()) {
                    cluster_length_histogram_[probes]++;
                }
                return true;
            }
            idx = (idx + 1) % table_.size();
        } while (table_[idx].continuation);
        
        // Track cluster length even on misses
        if (probes < cluster_length_histogram_.size()) {
            cluster_length_histogram_[probes]++;
        }
        return false;
    }
    
    bool remove(uint64_t key) override {
        size_t quotient;
        uint32_t remainder;
        get_hash_parts(key, quotient, remainder);
        
        if (!table_[quotient].occupied) {
            return false;
        }
        
        size_t idx = find_run_start(quotient);
        
        do {
            if (table_[idx].remainder == remainder) {
                // Remove and shift left
                table_[idx].remainder = 0;
                table_[idx].continuation = 0;
                table_[idx].shifted = 0;
                num_items_--;
                
                // Update occupancy bit if run is now empty
                size_t check = find_run_start(quotient);
                if (table_[check].remainder == 0) {
                    table_[quotient].occupied = 0;
                }
                
                return true;
            }
            idx = (idx + 1) % table_.size();
        } while (table_[idx].continuation);
        
        return false;
    }
    
    size_t size_bytes() const override {
        return table_.size() * sizeof(Slot);
    }
    
    double bits_per_entry() const override {
        if (num_items_ == 0) return 0;
        // BPE = (table_size × (remainder_bits + 3 metadata bits)) / items
        // Metadata: occupied, continuation, shifted (3 bits total)
        size_t bits_per_slot = r_bits_ + 3;
        return (table_.size() * bits_per_slot) / static_cast<double>(num_items_);
    }
    
    double load_factor() const override {
        return static_cast<double>(num_items_) / table_.size();
    }
    
    std::string name() const override {
        return "Quotient";
    }
    
    size_t get_num_items() const override { return num_items_; }
    
    void print_probe_stats() const {
        if (total_queries_ > 0) {
            double avg_probes = static_cast<double>(total_probes_) / total_queries_;
            LOG("  Average probes per query: " << avg_probes << "\n");
            
            // Print cluster length histogram (top 10 non-zero buckets)
            LOG("  Cluster length histogram:\n");
            int printed = 0;
            for (size_t i = 1; i < cluster_length_histogram_.size() && printed < 10; i++) {
                if (cluster_length_histogram_[i] > 0) {
                    double pct = 100.0 * cluster_length_histogram_[i] / total_queries_;
                    LOG("    Length " << i << ": " << cluster_length_histogram_[i] 
                        << " (" << pct << "%)\n");
                    printed++;
                }
            }
        }
    }
    
    void print_stats() const override {
        LOG("Quotient Filter:\n");
        LOG("  Table size: " << table_.size() << "\n");
        LOG("  Quotient bits: " << q_bits_ << "\n");
        LOG("  Remainder bits: " << r_bits_ << "\n");
        LOG("  Items: " << num_items_ << "\n");
        LOG("  Load factor: " << load_factor() * 100 << "%\n");
        LOG("  Size: " << size_bytes() / 1024.0 / 1024.0 << " MB\n");
        LOG("  Bits per entry: " << bits_per_entry() << "\n");
    }
};

//=============================================================================
// Benchmark Harness
//=============================================================================

struct BenchmarkConfig {
    std::string filter_type = "all";
    size_t set_size = 1000000;
    double target_fpr = 0.01;
    std::string workload = "readonly"; // readonly, readmostly, balanced
    double negative_rate = 0.5;
    int num_threads = 1;
    int num_runs = 3;
    size_t capacity = 0; // If non-zero, override auto-capacity calculation for dynamic filters
};

void generate_dataset(std::vector<uint64_t>& positive, 
                      std::vector<uint64_t>& negative, 
                      size_t n) {
    std::mt19937_64 rng(42);
    positive.resize(n);
    negative.resize(n);
    
    for (size_t i = 0; i < n; i++) {
        positive[i] = rng();
        negative[i] = rng();
    }
}

struct LatencyStats {
    double throughput_mops;
    double p50_ns;
    double p95_ns;
    double p99_ns;
};

struct DynamicWorkloadStats {
    double ops_per_sec;
    size_t insert_count;
    size_t delete_count;
    size_t insert_failures;
    double insert_failure_rate;
    double avg_probe_length;
};

DynamicWorkloadStats benchmark_dynamic_workload(Filter* filter,
                                                 const std::vector<uint64_t>& insert_keys,
                                                 const std::vector<uint64_t>& delete_keys,
                                                 int num_threads = 1) {
    // Balanced workload: 50% inserts, 50% deletes
    // Limit to 200K operations total (100K insert + 100K delete) for performance
    size_t max_ops_per_type = std::min({insert_keys.size(), delete_keys.size(), size_t(100000)});
    size_t num_ops = max_ops_per_type * 2;
    std::vector<std::pair<bool, uint64_t>> ops; // (is_insert, key)
    ops.reserve(num_ops);
    
    // Interleave insert and delete operations
    for (size_t i = 0; i < max_ops_per_type; i++) {
        ops.push_back({true, insert_keys[i]});
        ops.push_back({false, delete_keys[i]});
    }
    
    // Shuffle for realistic pattern
    std::mt19937_64 rng(456);
    std::shuffle(ops.begin(), ops.end(), rng);
    
    // Execute operations
    size_t insert_count = 0;
    size_t delete_count = 0;
    size_t insert_failures = 0;
    
    Timer timer;
    
    for (const auto& op : ops) {
        if (op.first) {
            // Insert operation
            bool success = filter->insert(op.second);
            insert_count++;
            if (!success) insert_failures++;
        } else {
            // Delete operation (only for dynamic filters)
            filter->remove(op.second);
            delete_count++;
        }
    }
    
    double elapsed = timer.elapsed_seconds();
    double ops_per_sec = num_ops / elapsed;
    
    DynamicWorkloadStats stats;
    stats.ops_per_sec = ops_per_sec;
    stats.insert_count = insert_count;
    stats.delete_count = delete_count;
    stats.insert_failures = insert_failures;
    stats.insert_failure_rate = insert_count > 0 ? 
        static_cast<double>(insert_failures) / insert_count : 0.0;
    stats.avg_probe_length = 0.0; // Will be populated from filter stats
    
    return stats;
}

LatencyStats benchmark_with_latency(Filter* filter,
                                     const std::vector<uint64_t>& positive_set,
                                     const std::vector<uint64_t>& negative_set,
                                     double negative_rate,
                                     int num_threads = 1) {
    // Create mixed query workload
    size_t num_queries = 100000; // Smaller for latency measurement
    std::vector<uint64_t> queries;
    queries.reserve(num_queries);
    
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < num_queries; i++) {
        if (dist(rng) < negative_rate) {
            queries.push_back(negative_set[rng() % negative_set.size()]);
        } else {
            queries.push_back(positive_set[rng() % positive_set.size()]);
        }
    }
    
    // Measure per-query latencies (single-threaded for accurate measurement)
    std::vector<double> latencies_ns;
    latencies_ns.reserve(num_queries);
    
    Timer latency_timer;
    for (size_t i = 0; i < num_queries; i++) {
        latency_timer.reset();
        volatile bool result = filter->query(queries[i]);
        (void)result;
        
        double latency_ns = static_cast<double>(latency_timer.elapsed_nanoseconds());
        latencies_ns.push_back(latency_ns);
    }
    
    // Calculate throughput
    Timer timer;
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < queries.size(); i++) {
        volatile bool result = filter->query(queries[i]);
        (void)result;
    }
    double elapsed = timer.elapsed_seconds();
    double throughput = queries.size() / elapsed / 1e6; // Mops/s
    
    // Calculate percentiles
    LatencyStats stats;
    stats.throughput_mops = throughput;
    stats.p50_ns = percentile(latencies_ns, 0.50);
    stats.p95_ns = percentile(latencies_ns, 0.95);
    stats.p99_ns = percentile(latencies_ns, 0.99);
    
    return stats;
}

void run_benchmark(const BenchmarkConfig& config) {
    LOG("\n========================================\n");
    LOG("Benchmark Configuration:\n");
    LOG("  Filter: " << config.filter_type << "\n");
    LOG("  Set size: " << config.set_size << "\n");
    LOG("  Target FPR: " << config.target_fpr * 100 << "%\n");
    LOG("  Workload: " << config.workload << "\n");
    LOG("  Negative rate: " << config.negative_rate * 100 << "%\n");
    LOG("  Threads: " << config.num_threads << "\n");
    LOG("========================================\n\n");
    
    // Generate datasets
    std::vector<uint64_t> positive_set, negative_set;
    generate_dataset(positive_set, negative_set, config.set_size);
    
    // Build filter
    Filter* filter = nullptr;
    
    if (config.filter_type == "bloom" || config.filter_type == "all") {
        LOG("Building Blocked Bloom Filter...\n");
        auto bloom = new BlockedBloomFilter(config.set_size, config.target_fpr);
        for (uint64_t key : positive_set) {
            bloom->insert(key);
        }
        bloom->print_stats();
        
        double measured_fpr = bloom->measure_fpr(negative_set);
        LOG("Measured FPR: " << measured_fpr * 100 << "%\n");
        
        auto stats = benchmark_with_latency(bloom, positive_set, negative_set, 
                                           config.negative_rate, config.num_threads);
        LOG("Throughput: " << stats.throughput_mops << " Mops/s\n");
        LOG("Latency p50: " << std::fixed << std::setprecision(1) << stats.p50_ns << " ns\n");
        LOG("Latency p95: " << std::fixed << std::setprecision(1) << stats.p95_ns << " ns\n");
        LOG("Latency p99: " << std::fixed << std::setprecision(1) << stats.p99_ns << " ns\n\n");
        
        delete bloom;
    }
    
    if (config.filter_type == "xor" || config.filter_type == "all") {
        LOG("Building XOR Filter...\n");
        // Calculate fingerprint bits: f ≈ log2(1/ε)
        int fp_bits = static_cast<int>(std::ceil(-std::log2(config.target_fpr)));
        fp_bits = std::max(8, std::min(fp_bits, 16)); // Clamp to 8-16 bits
        auto xor_filter = new XORFilter(positive_set, fp_bits);
        xor_filter->print_stats();
        
        double measured_fpr = xor_filter->measure_fpr(negative_set);
        LOG("Measured FPR: " << measured_fpr * 100 << "%\n");
        
        auto stats = benchmark_with_latency(xor_filter, positive_set, negative_set,
                                           config.negative_rate, config.num_threads);
        LOG("Throughput: " << stats.throughput_mops << " Mops/s\n");
        LOG("Latency p50: " << std::fixed << std::setprecision(1) << stats.p50_ns << " ns\n");
        LOG("Latency p95: " << std::fixed << std::setprecision(1) << stats.p95_ns << " ns\n");
        LOG("Latency p99: " << std::fixed << std::setprecision(1) << stats.p99_ns << " ns\n\n");
        
        delete xor_filter;
    }
    
    if (config.filter_type == "cuckoo" || config.filter_type == "all") {
        LOG("Building Cuckoo Filter...\n");
        // Calculate fingerprint bits: f ≈ log2(1/ε) + log2(2b) where b=4
        int fp_bits = static_cast<int>(std::ceil(-std::log2(config.target_fpr) + 3));
        fp_bits = std::max(8, std::min(fp_bits, 16)); // Clamp to 8-16 bits
        // Capacity: use override if provided, else auto-calculate
        size_t capacity;
        if (config.capacity > 0) {
            capacity = config.capacity;
        } else {
            double alpha = 0.95; // Target load factor
            capacity = static_cast<size_t>(config.set_size / alpha);
        }
        auto cuckoo = new CuckooFilter(capacity, fp_bits);
        for (uint64_t key : positive_set) {
            cuckoo->insert(key);
        }
        cuckoo->print_stats();
        
        double measured_fpr = cuckoo->measure_fpr(negative_set);
        LOG("Measured FPR: " << measured_fpr * 100 << "%\n");
        
        // Choose workload type
        if (config.workload == "balanced") {
            // Dynamic workload: 50% insert, 50% delete
            LOG("\n=== Dynamic Workload (Balanced: 50% Insert / 50% Delete) ===\n");
            
            // Generate additional keys for insert/delete operations (limited to 200K total ops)
            std::vector<uint64_t> insert_keys, delete_keys;
            size_t workload_size = std::min(config.set_size / 2, size_t(100000));
            generate_dataset(insert_keys, delete_keys, workload_size);
            
            auto dyn_stats = benchmark_dynamic_workload(cuckoo, insert_keys, positive_set, config.num_threads);
            
            LOG("Dynamic Operations Throughput: " << dyn_stats.ops_per_sec / 1e6 << " Mops/s\n");
            LOG("Insert operations: " << dyn_stats.insert_count << "\n");
            LOG("Delete operations: " << dyn_stats.delete_count << "\n");
            LOG("Insert failures: " << dyn_stats.insert_failures << "\n");
            LOG("Insert failure rate: " << std::fixed << std::setprecision(2) 
                << dyn_stats.insert_failure_rate * 100 << "%\n");
            
            LOG("\nProbe Statistics:\n");
            cuckoo->print_probe_stats();
        } else {
            // Readonly workload
            auto stats = benchmark_with_latency(cuckoo, positive_set, negative_set,
                                               config.negative_rate, config.num_threads);
            LOG("Throughput: " << stats.throughput_mops << " Mops/s\n");
            LOG("Latency p50: " << std::fixed << std::setprecision(1) << stats.p50_ns << " ns\n");
            LOG("Latency p95: " << std::fixed << std::setprecision(1) << stats.p95_ns << " ns\n");
            LOG("Latency p99: " << std::fixed << std::setprecision(1) << stats.p99_ns << " ns\n");
            
            LOG("\nProbe Statistics:\n");
            cuckoo->print_probe_stats();
        }
        LOG("\n");
        
        delete cuckoo;
    }
    
    if (config.filter_type == "quotient" || config.filter_type == "all") {
        LOG("Building Quotient Filter...\n");
        // Calculate remainder bits based on FPR: r ≈ log2(1/ε)
        int r_bits = static_cast<int>(std::ceil(-std::log2(config.target_fpr)));
        r_bits = std::max(8, std::min(r_bits, 16)); // Clamp to 8-16 bits
        // Capacity: use override if provided, else auto-calculate
        size_t capacity;
        if (config.capacity > 0) {
            capacity = config.capacity;
        } else {
            capacity = static_cast<size_t>(config.set_size * 1.1);
        }
        auto quotient = new QuotientFilter(capacity, r_bits);
        for (uint64_t key : positive_set) {
            quotient->insert(key);
        }
        quotient->print_stats();
        
        double measured_fpr = quotient->measure_fpr(negative_set);
        LOG("Measured FPR: " << measured_fpr * 100 << "%\n");
        
        // Choose workload type
        if (config.workload == "balanced") {
            // Dynamic workload: 50% insert, 50% delete
            LOG("\n=== Dynamic Workload (Balanced: 50% Insert / 50% Delete) ===\n");
            
            // Generate additional keys for insert/delete operations (limited to 200K total ops)
            std::vector<uint64_t> insert_keys, delete_keys;
            size_t workload_size = std::min(config.set_size / 2, size_t(100000));
            generate_dataset(insert_keys, delete_keys, workload_size);
            
            auto dyn_stats = benchmark_dynamic_workload(quotient, insert_keys, positive_set, config.num_threads);
            
            LOG("Dynamic Operations Throughput: " << dyn_stats.ops_per_sec / 1e6 << " Mops/s\n");
            LOG("Insert operations: " << dyn_stats.insert_count << "\n");
            LOG("Delete operations: " << dyn_stats.delete_count << "\n");
            LOG("Insert failures: " << dyn_stats.insert_failures << "\n");
            LOG("Insert failure rate: " << std::fixed << std::setprecision(2) 
                << dyn_stats.insert_failure_rate * 100 << "%\n");
            
            LOG("\nProbe Statistics:\n");
            quotient->print_probe_stats();
        } else {
            // Readonly workload
            auto stats = benchmark_with_latency(quotient, positive_set, negative_set,
                                               config.negative_rate, config.num_threads);
            LOG("Throughput: " << stats.throughput_mops << " Mops/s\n");
            LOG("Latency p50: " << std::fixed << std::setprecision(1) << stats.p50_ns << " ns\n");
            LOG("Latency p95: " << std::fixed << std::setprecision(1) << stats.p95_ns << " ns\n");
            LOG("Latency p99: " << std::fixed << std::setprecision(1) << stats.p99_ns << " ns\n");
            
            LOG("\nProbe Statistics:\n");
            quotient->print_probe_stats();
        }
        LOG("\n");
        
        delete quotient;
    }
}

//=============================================================================
// Main
//=============================================================================

void print_usage() {
    std::cout << "Usage: filter_bench [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --mode <benchmark|sweep>       Benchmark mode\n";
    std::cout << "  --filter <bloom|xor|cuckoo|quotient|all>\n";
    std::cout << "  --size <n>                     Set size (default: 1M)\n";
    std::cout << "  --fpr <rate>                   Target FPR (default: 0.01)\n";
    std::cout << "  --workload <readonly|readmostly|balanced>\n";
    std::cout << "  --negative <rate>              Negative query rate\n";
    std::cout << "  --threads <n>                  Number of threads\n";
    std::cout << "  --capacity <n>                 Override capacity for dynamic filters\n";
    std::cout << "  --help                         Show this message\n";
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--filter" && i + 1 < argc) {
            config.filter_type = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            config.set_size = std::stoul(argv[++i]);
        } else if (arg == "--fpr" && i + 1 < argc) {
            config.target_fpr = std::stod(argv[++i]);
        } else if (arg == "--workload" && i + 1 < argc) {
            config.workload = argv[++i];
        } else if (arg == "--negative" && i + 1 < argc) {
            config.negative_rate = std::stod(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--capacity" && i + 1 < argc) {
            config.capacity = std::stoul(argv[++i]);
        }
    }
    
    run_benchmark(config);
    
    return 0;
}
