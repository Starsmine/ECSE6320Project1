# Project A3: Approximate Membership Filters
# XOR, Cuckoo, Quotient, and Blocked Bloom filters

CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS := -fopenmp -lxxhash

TARGET := filter_bench
SOURCES := main.cpp
OBJECTS := $(SOURCES:.cpp=.o)

# Detect architecture
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
    CXXFLAGS += -DARM_NEON
endif

.PHONY: all clean debug perf info

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS := -std=c++17 -O0 -g -march=native -Wall -Wextra -fopenmp
debug: clean $(TARGET)

perf: $(TARGET)
	@echo "=== Building with perf support ==="
	@if [ "$$(uname)" != "Linux" ]; then \
		echo "Warning: perf is only available on Linux"; \
	else \
		perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses \
		./$(TARGET) --mode benchmark --filter all --size 1000000 --fpr 0.01 --workload readonly; \
	fi

info:
	@echo "=== Build Configuration ==="
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "Architecture: $(ARCH)"
	@echo "CPU Info:"
	@if [ "$$(uname)" = "Linux" ]; then \
		lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|MHz|cache"; \
	elif [ "$$(uname)" = "Darwin" ]; then \
		sysctl -a | grep machdep.cpu; \
	fi
	@echo ""
	@echo "xxHash installed:"
	@which xxhsum || echo "  xxhash not found - install with: sudo apt install libxxhash-dev"

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f *.o *.out
	rm -rf results/*.txt results/*.png

help:
	@echo "Available targets:"
	@echo "  make          - Build optimized binary"
	@echo "  make debug    - Build with debug symbols"
	@echo "  make perf     - Run with perf counters (Linux only)"
	@echo "  make info     - Display build configuration"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Usage examples:"
	@echo "  ./filter_bench --help"
	@echo "  ./filter_bench --mode benchmark --filter xor --size 1000000 --fpr 0.01"
	@echo "  ./filter_bench --mode sweep --filter all --sizes 1M,5M,10M --fprs 0.05,0.01,0.001"
