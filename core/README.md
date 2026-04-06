# Core GPU HiPS Engine

This directory contains the CUDA/C++ implementation of the HiPS-GPU
generator. If you only want to build and run the GPU HiPS engine, this is the
only part of the repository you need.

The main executable built here is:

```bash
bin/hipsgen_cuda
```

## Source Layout

- `src/cpp/hipsgen_cuda.cpp`
  main command-line entry point
- `src/cpp/*`
  FITS I/O, WCS transforms, HEALPix helpers, HpxFinder generation, tile
  writing, GPU kernels, and related support code
- `Makefile`
  build rules for the generator

## Build

From this directory:

```bash
make
```

Expected output:

```bash
bin/hipsgen_cuda
```

Clean build artifacts:

```bash
make clean
```

## Required Libraries

The current `Makefile` links against:

- CUDA runtime libraries
- CFITSIO
- WCSLIB
- Healpix C++ library
- OpenMP support

Depending on your system, you may need to edit:

- CUDA architecture flags
- include paths
- library paths
- compiler selections

The current default build target uses `-arch=sm_75`.

## Basic Usage

Minimal command:

```bash
bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips
```

Example with explicit order:

```bash
bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips -order 7
```

Example with recursive scan and filename filter:

```bash
bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips \
  -r \
  -pattern "*-image-r.fits.fz" \
  -order 7
```

## Common Options

- `-order <n>`
  maximum HiPS order
- `-threads <n>`
  host-side thread count
- `-mode <NONE|MEAN|AVERAGE|FADING|ADD>`
  overlay mode for overlapping images
- `-img <file>`
  reference image for initialization defaults
- `-blank <value>`
  explicit blank pixel value
- `-validRange <min> <max>`
  accepted pixel-value range
- `-autoValidRange`
  estimate a valid range from source data
- `-sampleRatio <r>`
  sample ratio for valid-range estimation
- `-r` or `-recursive`
  recursively scan subdirectories
- `-pattern <glob>`
  filename filter
- `-limit <n>`
  process only the first `n` matching files
- `-cache <n>`
  in-memory FITS cache size
- `-progress <n>`
  progress-report interval
- `-no-index`
  skip HpxFinder index generation
- `-no-tiles`
  skip tile generation
- `-cpu`
  force CPU mode
- `-v`
  verbose output

## Input and Output

Input:

- a directory of FITS images
- valid WCS headers
- optionally nested subdirectories when using `-r`

Output:

- `Norder*` HiPS tile directories
- `HpxFinder/` provenance index directories
- `properties`
- optional lower-order derived products and all-sky preview files

## Practical Notes

- Fast local storage is strongly recommended for large runs.
- For large image collections, recursive scan, file filtering, and local
  staging can significantly reduce overhead.
- The benchmark helper scripts in this directory are optional and are not
  required for normal HiPS generation.
