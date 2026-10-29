# Vanda CPU source files

These C++ source files are the implementations used for the Vanda CPU experiments reported in the follow-up results.

- `attention_serial.cpp`: serial FP64 attention reference.
- `attention_serial_fp32.cpp`: serial FP32 implementation used for precision and scaling measurements.
- `attention_openmp.cpp`: shared-memory OpenMP implementation, including the 72-thread scaling runs.
- `attention_mpi_openmp.cpp`: mixed MPI+OpenMP CPU implementation used for multi-node scaling, including the 10-node run.

The CUDA and MPI+CUDA kernel source is in [`../local_cuda`](../local_cuda). The source files here are provided separately from PBS/bash job scripts; those scripts are not presented as application source code.
