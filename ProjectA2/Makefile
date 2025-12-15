# Makefile for Dense vs Sparse Matrix Multiplication Project
# Supports SIMD (AVX2/AVX-512) and OpenMP multithreading

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native
LDFLAGS = -fopenmp
LIBS = -lopenblas -lpthread

# Target ISA flags (uncomment for specific architectures)
# CXXFLAGS += -mavx2        # AVX2 support
CXXFLAGS += -mavx512f     # AVX-512 support 

# OpenMP support
CXXFLAGS += -fopenmp

# Debug build flags (use 'make debug')
DEBUG_FLAGS = -g -O0 -DDEBUG

# Source files
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = matrix_bench

# Perf measurement targets (Linux only)
PERF_EVENTS = cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses

.PHONY: all clean debug run perf perf-record info help

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS := $(filter-out -O3,$(CXXFLAGS)) $(DEBUG_FLAGS)
debug: clean $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET) *.o

run: $(TARGET)
	./$(TARGET)

# Run with perf counters (Linux only - requires 'perf' tool)
perf: $(TARGET)
	@command -v perf >/dev/null 2>&1 || { echo "Error: 'perf' not found. Install linux-tools or run on Linux."; exit 1; }
	perf stat -e $(PERF_EVENTS) ./$(TARGET)

# Run with detailed perf analysis (Linux only)
perf-record: $(TARGET)
	@command -v perf >/dev/null 2>&1 || { echo "Error: 'perf' not found. Install linux-tools or run on Linux."; exit 1; }
	perf record -e cycles,cache-misses ./$(TARGET)
	perf report

# Check compiler flags and architecture
info:
	@echo "Compiler: $(CXX)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "OpenMP: $(LDFLAGS)"
	@echo "Libraries: $(LIBS)"
	@$(CXX) -march=native -Q --help=target | grep -E "(march|mtune|mavx)"

help:
	@echo "Available targets:"
	@echo "  all          - Build optimized version (default)"
	@echo "  debug        - Build with debug symbols"
	@echo "  clean        - Remove build artifacts"
	@echo "  run          - Build and run the benchmark"
	@echo "  perf         - Run with perf stat counters"
	@echo "  perf-record  - Record and analyze with perf"
	@echo "  info         - Display compiler and architecture info"
	@echo "  help         - Show this help message"
