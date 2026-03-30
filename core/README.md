# Core GPU HiPS Engine

This directory contains the CUDA/C++ implementation of the `hips-gpu`
generator used in the paper benchmarks.

## Main Entry Point

- `src/cpp/hipsgen_cuda.cpp`

## Build

```bash
cd core
make
```

Expected output:

```bash
core/bin/hipsgen_cuda
```

## Notes

- The supplied `Makefile` reflects the working research environment.
- You may need to adjust CUDA architecture flags, include paths, and FITS/WCS
  library paths for your own system.
- Helper benchmark scripts in this directory assume that the binary has already
  been built.
