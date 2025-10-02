# Project13dpointstencil - 3D Point Stencil SIMD Operations

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

## Overview

3dpoint stencil was interesting as because the opperations did 5 ops per memory access request I was able to get higher GFLOPS showing that SAXPY And Dot Product are memory bound even at the L2 level. I am also showing that weird behavior that I had in SAXPY of the performance tanking when the data nearly filling l2 and crossing into l3

![DotPoduct Performance Results](./plots_output/float32_consolidated.png)
![DotPoduct Performance Results](./plots_output/float64_consolidated.png)
![DotPoduct Performance Results](./plots_output/int32_consolidated.png)
![DotPoduct Performance Results](./plots_output/float32_contiguous_speedup.png)
![DotPoduct Performance Results](./plots_output/float64_contiguous_speedup.png)
![DotPoduct Performance Results](./plots_output/float32_contiguous_speedup.png)
![DotPoduct Performance Results](./plots_output/int32_contiguous_speedup.png)


## Related Projects
- [Dot Product](../Project1DotProduct/README.md) - Fundamental SIMD operations
- [SAXPY](../Project1Saxpy/README.md) - Linear algebra SIMD implementations

---
[← Back to Main README](../README.md)