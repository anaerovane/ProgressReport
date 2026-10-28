# Ring Attention CUDA source

This directory contains the standalone source files used for the local RTX 4090 attention run. No Word report or binary is included.

## Files

- `attention_cuda.cu`: CUDA attention runner; builds as single-GPU CUDA when MPI is not enabled, and supports the MPI ring path with `ATTENTION_MPI`.
- `attention_gpu.cuh`: GPU kernel and execution interfaces.
- `attention_common.hpp`: shared argument parsing, input generation, validation, and output utilities.

## Local CUDA build and run

With CUDA Toolkit, cuBLAS, and a C++17 host compiler installed:

```bash
nvcc -O2 -std=c++17 -arch=sm_89 attention_cuda.cu -lcublas -o attention_cuda
./attention_cuda --n 257 --d 31 --dv 19 --bq 37 --bk 53 --validate --repeats 1
./attention_cuda --n 257 --d 31 --dv 19 --bq 37 --bk 53 --causal --validate --repeats 1
./attention_cuda --n 1024 --d 64 --dv 64 --bq 128 --bk 256 --repeats 5
./attention_cuda --n 4096 --d 64 --dv 64 --bq 128 --bk 256 --repeats 5
```

For the multi-rank build, compile with MPI enabled and launch on allocated GPU nodes using one rank per node. The source uses host-staged MPI transfers.
