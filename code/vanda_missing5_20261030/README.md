# Vanda additional precision runs

These three PBS scripts cover the five missing measurements:

- CPU serial FP64: N=32768 and N=65536
- CPU serial FP32: N=65536
- A40 CUDA FP64: N=32768 and N=65536

Each large-input measurement uses three repetitions at d=dv=64. The scripts were submitted to the zero-charge `auto_free` route as jobs 1508087, 1508088, and 1508089. At submission time, the PBS server reported `scheduling = False`; all three were queued and had not started. No result is claimed by these scripts or this README.

The scripts expect the existing Vanda experiment source tree at `$HOME/ring_attention_followup_20261029` and write outputs to `$HOME/ring_attention_missing5_20261030/results`. The A40 script uses the FP64 CUDA source already generated in that tree at `src/fp64/attention_cuda.cu`.
