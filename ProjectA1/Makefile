# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -pthread
LDFLAGS = -pthread -lnuma -lrt -luring

# Target executable
TARGET = os_features_bench

# Source files
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Run all benchmarks
run: $(TARGET)
	@echo "Running all benchmarks..."
	./$(TARGET) --all

# Run individual benchmarks
run-affinity: $(TARGET)
	./$(TARGET) --affinity

run-numa: $(TARGET)
	./$(TARGET) --numa

run-smt: $(TARGET)
	./$(TARGET) --smt

run-hugepages: $(TARGET)
	./$(TARGET) --hugepages

run-asyncio: $(TARGET)
	./$(TARGET) --asyncio

# System setup helpers
setup-hugepages:
	@echo "Configuring huge pages (requires sudo)..."
	sudo sysctl -w vm.nr_hugepages=1000
	@echo "Huge pages configured. Current status:"
	cat /proc/meminfo | grep Huge

check-numa:
	@echo "NUMA Configuration:"
	numactl --hardware

check-cpu:
	@echo "CPU Information:"
	lscpu | grep -E 'CPU\(s\)|Thread|Core|Socket|NUMA'

# Show system info
info:
	@echo "=== System Information ==="
	@echo "Compiler: $(CXX) $(shell $(CXX) --version | head -1)"
	@echo "Kernel: $(shell uname -r)"
	@echo "CPU: $(shell lscpu | grep 'Model name' | cut -d ':' -f2 | xargs)"
	@echo ""
	@$(MAKE) check-cpu
	@echo ""
	@$(MAKE) check-numa

.PHONY: all clean run run-affinity run-numa run-smt run-hugepages setup-hugepages check-numa check-cpu info
