# Project1DotProduct - SIMD Dot Product Optimization

[![Back to Main README](https://img.shields.io/badge/Back%20to-Main%20README-blue)](../README.md)

Adding charts for Dot Product 

![DotPoduct Performance Results](./plots_output/dotp_performance_float32.png)
![DotPoduct Performance Results](./plots_output/dotp_performance_float64.png)
![DotPoduct Performance Results](./plots_output/dotp_performance_int32.png)

![DotPoduct Performance Results](./plots_output/dotp_speedup_float32.png)
![DotPoduct Performance Results](./plots_output/dotp_speedup_float64.png)
![DotPoduct Performance Results](./plots_output/dotp_speedup_int32.png)

Same problems as with SAXPY, Chrono to granular for L1 cache and gives unreliable data
It is interesting to see the compute to bandwidth bound change from saxpy 
Scaler and SSE2 are always compute bound. 
AVX becomes bandwidth bound at DRAM and AVX bexomes bandwidth bound in L3. 
## Related Projects
- [SAXPY Implementation](../Project1Saxpy/README.md) - Related SIMD optimization project
- [3D Point Stencil](../Project13dpointstencil/README.md) - Advanced SIMD applications

---
[← Back to Main README](../README.md)