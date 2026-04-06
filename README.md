# HiPS-GPU

HiPS-GPU is a CUDA/C++ software package for generating image HiPS products on
NVIDIA GPUs. The main deliverable in this repository is the command-line
generator `hipsgen_cuda`, which reads a directory of FITS images with valid WCS
headers and writes a HiPS directory tree with tiles, metadata, and HpxFinder
provenance files.

This repository also contains optional Python utilities for benchmarking,
runtime integration, validation, and the case studies used in the accompanying
paper. Those scripts are included as extras. The core package is the GPU HiPS
generator under `core/`.

## What This Repository Is For

- Build and run a GPU-accelerated HiPS generator on your own FITS image sets
- Produce standard-style HiPS directory outputs, including HpxFinder indices
- Compare GPU output with the CDS Java reference implementation when needed
- Reuse the included Python helpers for validation or workflow integration

## Main Components

- `core/`
  - CUDA/C++ source for `hipsgen_cuda`
  - build system and low-level benchmark helpers
- `runtime/`
  - Python helpers that call the compiled generator in larger workflows
- `benchmarks/`
  - scripts for comparing GPU and Java HiPS generation
- `validation/`
  - synthetic validation and figure-generation utilities
- `case_studies/`
  - real-event workflow scripts used in the paper
- `manifest/`
  - bundle inventory files

## System Requirements

HiPS-GPU is currently aimed at Linux systems with an NVIDIA GPU.

Required toolchain and libraries:

- `nvcc`
- `g++`
- CUDA runtime libraries
- CFITSIO
- WCSLIB
- Healpix C++ library

The supplied `core/Makefile` currently links against:

- `-lcfitsio`
- `-lwcs`
- `-lhealpix_cxx`
- CUDA runtime libraries

You may need to adjust include paths, library paths, or the CUDA architecture
flag in `core/Makefile` for your own machine. The current default build uses
`-arch=sm_75`.

## Build

Build the generator from the repository root:

```bash
cd core
make
```

Expected output:

```bash
core/bin/hipsgen_cuda
```

If the build succeeds, the package is ready to use. The Java `Hipsgen.jar` is
not required for normal operation. It is only needed if you want to run the
comparison benchmarks against the CDS reference implementation.

## Quick Start

The generator expects:

- an input directory containing FITS images
- valid WCS headers in those FITS files
- an output directory where the HiPS tree will be written

Minimal example:

```bash
core/bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips
```

Explicit order and overlay mode:

```bash
core/bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips \
  -order 7 \
  -mode MEAN
```

Recursive scan with filename filtering:

```bash
core/bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips \
  -r \
  -pattern "*-image-r.fits.fz" \
  -order 7
```

CPU fallback for debugging:

```bash
core/bin/hipsgen_cuda /path/to/input_fits /path/to/output_hips \
  -order 7 \
  -cpu
```

## Command-Line Interface

Usage:

```bash
hipsgen_cuda <input_dir> <output_dir> [options]
```

Common options:

- `-order <n>`
  set the maximum HEALPix order; if omitted, the code can choose an order
  automatically from image scale
- `-threads <n>`
  set the host-side thread count
- `-mode <NONE|MEAN|AVERAGE|FADING|ADD>`
  choose the overlay mode for overlapping source images
- `-img <file>`
  use a reference image for default initializations
- `-blank <value>`
  set the blank pixel value explicitly
- `-validRange <min> <max>`
  accept only pixels within a given range
- `-autoValidRange`
  estimate a valid range by sampling source data
- `-sampleRatio <r>`
  control the sampling fraction used by `-autoValidRange`
- `-r` or `-recursive`
  scan subdirectories recursively
- `-pattern <glob>`
  process only files matching a glob pattern
- `-limit <n>`
  limit the number of files processed
- `-cache <n>`
  control the in-memory FITS cache size
- `-progress <n>`
  control progress-report frequency
- `-no-index`
  skip the HpxFinder/INDEX stage
- `-no-tiles`
  skip the TILES stage
- `-cpu`
  force CPU mode
- `-v`
  enable verbose logging

## Input Expectations

HiPS-GPU is intended for image collections that:

- are stored as FITS files
- contain valid celestial WCS metadata
- can be projected onto a common HiPS frame

The code is most useful when you want to generate HiPS products from many
input images quickly, especially when overlap is high or when low-latency
incremental generation matters.

## Output Layout

The generator writes a HiPS-style output tree such as:

```text
output_hips/
  HpxFinder/
  Norder0/
  Norder1/
  ...
  NorderN/
  properties
```

Depending on settings and stages, the output may also include:

- `Allsky.fits`
- `index.html`
- lower-order tiles derived from the target order

The `HpxFinder/` subtree stores provenance mappings from HiPS cells back to
contributing source images.

## Typical Workflow

1. Prepare a directory of FITS images on fast local storage.
2. Build `core/bin/hipsgen_cuda`.
3. Run the generator with a target order or allow auto-order selection.
4. Inspect the resulting `Norder*`, `HpxFinder`, and `properties` outputs.
5. Serve or post-process the generated HiPS product in your own environment.

For large datasets, local NVMe or SSD staging is strongly recommended.

## Python Utilities

The Python scripts are optional. They are useful if you want to:

- integrate `hipsgen_cuda` into a larger time-domain workflow
- run synthetic validation
- reproduce the paper benchmarks
- generate analysis figures

Install the optional Python dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Important environment variables used by the Python helpers:

- `HIPSGEN_BIN`
  path to the compiled `hipsgen_cuda` binary
- `HIPS_SCRATCH`
  scratch directory for runtime helpers
- `HIPS_GPU_PAPER_DATA_ROOT`
  location of the companion paper data bundle
- `HIPS_GPU_PAPER_OUTPUT_ROOT`
  output root for generated Python-script products
- `HIPS_GPU_PAPER_DOCS_ROOT`
  figure output directory

If you only want the GPU HiPS generator itself, you can ignore the Python
utilities entirely.

## Benchmarking and Paper Reproduction

This repository can still be used for the paper workflows, but that is not its
only purpose.

Relevant directories:

- `benchmarks/` for GPU vs. Java comparison helpers
- `validation/` for synthetic validation scripts
- `case_studies/` for real-event workflow examples

Some of those scripts expect local datasets that are not redistributed in this
repository.

## License

This repository is released under the MIT License. See `LICENSE`.
