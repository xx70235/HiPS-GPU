# HiPS-GPU: Paper Code Bundle (`v3`)

This repository snapshot packages the code used by the paper
`GPU_HiPS_TimeDomain_Paper_APJS_rev_v3.tex`.

It combines two layers:

- the core CUDA/C++ `hips-gpu` engine for accelerated HiPS generation
- the paper-facing Python scripts used for benchmarks, synthetic validation,
  real-event case studies, and figure generation

This code bundle is intended to be close to a public GitHub release. It is
still a research snapshot rather than a polished software product, but the
main source tree, benchmark scripts, and paper-analysis scripts have been
collected into a single standalone layout.

## Repository Layout

- `core/`
  - main CUDA/C++ HiPS engine source tree
  - `Makefile`
  - benchmark helper scripts
- `benchmarks/`
  - comparison scripts used in the paper benchmark section
- `validation/`
  - synthetic end-to-end validation scripts
  - figure-generation helpers
  - latency-budget measurement script
- `case_studies/`
  - real-event case-study drivers for `EP260110a` and `EP260214b`
- `runtime/`
  - lightweight runtime modules shared by the paper scripts
- `manifest/`
  - inventory files for this bundle
- `project_paths.py`
  - shared path helpers and environment-variable based defaults

## What Is Included

- core engine source code (`core/src/cpp/`)
- build instructions and benchmark helpers (`core/Makefile`, shell scripts)
- Python scripts for:
  - output comparison
  - synthetic injection/recovery validation
  - latency measurement
  - real-event case studies
  - paper figure generation

## What Is Not Included

- the full production follow-up platform
- database schema migrations and API service deployment
- large FITS datasets
- generated PNG, PDF, and CSV products

For figure inputs and paper-ready figure outputs, use the separate data bundle:

- `../paper_figure_sources_v3/`

## Quick Start

### 1. Build the core GPU engine

```bash
cd core
make
```

This should produce:

```bash
core/bin/hipsgen_cuda
```

The provided `Makefile` assumes a local environment with CUDA and FITS/WCS
libraries already available.

Benchmark scripts that compare against the official Java reference path also
require a local copy of `Hipsgen.jar`, which is not redistributed here.

### 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Point the paper scripts to the data bundle

By default, the Python scripts look for data in a sibling directory:

```bash
../paper_figure_sources_v3/data
```

If your data live elsewhere, set:

```bash
export HIPS_GPU_PAPER_DATA_ROOT=/path/to/paper_figure_sources_v3/data
```

Optional output location:

```bash
export HIPS_GPU_PAPER_OUTPUT_ROOT=/path/to/generated_outputs
```

Optional figure output location:

```bash
export HIPS_GPU_PAPER_DOCS_ROOT=/path/to/generated_outputs/docs
```

### 4. Run paper scripts

Examples:

```bash
python validation/generate_real_result_figures.py
python validation/measure_latency_budget.py
python case_studies/ep260110a_generate_outputs.py
```

The two real-event case-study driver scripts require raw input images that are
not redistributed in this repository snapshot. Those scripts therefore need
additional environment variables pointing to local raw data directories.

## Environment Variables

- `HIPSGEN_BIN`
  - path to the compiled `hipsgen_cuda` binary
- `HIPS_SCRATCH`
  - scratch/output path for runtime helpers
- `HIPS_GPU_PAPER_DATA_ROOT`
  - root of the companion paper data bundle
- `HIPS_GPU_PAPER_OUTPUT_ROOT`
  - root for generated outputs
- `HIPS_GPU_PAPER_DOCS_ROOT`
  - directory for generated figure files
- `HIPS_GPU_EP260110A_XL100_DIR`
  - raw XL100 directory for the `EP260110a` case study
- `HIPS_GPU_EP260110A_TRT_DIR`
  - raw TRT directory for the `EP260110a` case study
- `HIPS_GPU_EP260110A_OUTPUT_DIR`
  - output directory for the `EP260110a` case study
- `HIPS_GPU_EP260214B_RAW_ROOT`
  - raw data root for the `EP260214b` case study
- `HIPS_GPU_EP260214B_OUTPUT_DIR`
  - output directory for the `EP260214b` case study

## Notes on Portability

The original working environment used site-specific filesystem layouts. The
most visible absolute paths have been replaced by environment-variable based
defaults or project-relative fallbacks, but this remains a research codebase.
Before a formal public release, you may still want to:

1. audit the build toolchain assumptions in `core/Makefile`
2. test the Python scripts in a fresh environment
3. decide which datasets can be redistributed and which should remain
   fetch-on-demand

## Citation / Provenance

The file inventory for this bundle is listed in:

- `manifest/code_inventory.csv`

The corresponding figure/data bundle is documented separately in:

- `../paper_figure_sources_v3/README.md`
