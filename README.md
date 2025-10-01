# ECSE6320Project1
## System Info 

CPU: Ryzen 7 7700X
CPU Frequency: Not pinned, turbos to 5.5ghz stable

SIMD Support: MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AES, AVX, AVX2, AVX-512, FMA3, SHA

Caches:

  - L1d: 32 KiB per core 8 way
  - L1i: 32 KiB per core 8 way
  - L2: 1 MiB per core 8 way
  - L3: 32 MiB shared 16 way

RAM: DDR5 6000Mt/s Cl 32

SSD: XPG Gammix S70 Blade 1 TB (Micron TLC B47R 512Gb)

OS: Windows 11 for project 1 and 2
WSL for project 3 (problematic)

Complier: GCC 14.2.0

Complier Flags: for project 1

                "-fdiagnostics-color=always",
                "-g",
                "-O3",
                "-Wpsabi",
                "-mavx2",
                "-mfma",
                "-mavx512f",
                "-fno-tree-vectorize",


