# Vanda A40 FP64 CUDA source

These are the CUDA source files used for the Vanda A40 FP64 attention measurements. They implement the same tiled attention operation as the FP32 CUDA kernel using double-precision arithmetic and cuBLAS double-precision GEMM calls.

Files:

- `attention_cuda.cu` — host-side CUDA runner and timing/validation logic
- `attention_gpu.cuh` — GPU kernel interface
- `attention_common.hpp` — shared options, data, and utility definitions

Build on Vanda from this directory:

```bash
module load gcc/13.1.0
module load cuda12.4/toolkit/12.4.1
nvcc -O3 -std=c++17 -arch=sm_86 attention_cuda.cu -lcublas -o attention_cuda_fp64
```

Example run matching the large-N A40 FP64 measurements:

```bash
./attention_cuda_fp64 --n 32768 --d 64 --dv 64 --bq 128 --bk 256 --repeats 3
```

The CPU FP32 source used in the same-precision comparison is already in `code/vanda_cpu/attention_serial_fp32.cpp`. These CUDA files are the additional A40 FP64 implementation. The larger follow-up jobs were queued separately; this source listing does not imply those queued runs have completed.
