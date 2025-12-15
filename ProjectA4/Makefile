# Project A4: Concurrent Data Structures
# Makefile for hash table benchmarks

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -pthread
LDFLAGS = -pthread

TARGET = hash_bench
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f *.o *.out
	rm -rf results/*.txt results/*.png

help:
	@echo "Project A4: Concurrent Hash Table Benchmarks"
	@echo ""
	@echo "Targets:"
	@echo "  make         - Build the benchmark executable"
	@echo "  make clean   - Remove build artifacts and results"
	@echo "  make help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  ./hash_bench --help"
