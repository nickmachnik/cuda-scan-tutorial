# cuda-scan-tutorial

This repo contains example code that showcases how parallel algorithms for computing sums can be implemented in CUDA.

## Build with CMake

```console
foo@bar:~/cuda-scan-tutorial$ module load cmake
foo@bar:~/cuda-scan-tutorial$ cmake -S . -B build
foo@bar:~/cuda-scan-tutorial$ cmake --build build
```

## Execute

Make sure that you are on a machine with available GPUs (try running `nvidia-smi`).

```console
foo@bar:~/cuda-scan-tutorial$ ./build/bin/parsum --help
```
