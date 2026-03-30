#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${ROOT:-${REPO_ROOT}}"
RUN_ID="${1:-20260318}"
WORK_BASE="${WORK_BASE:-${REPO_ROOT}/generated_outputs/benchmarks/${RUN_ID}}"
INPUT_BASE="${WORK_BASE}/inputs"
OUTPUT_BASE="${WORK_BASE}/outputs"
RESULTS_CSV="${WORK_BASE}/benchmark_results.csv"

GPU_BIN="${HIPSGEN_BIN:-${ROOT}/core/bin/hipsgen_cuda}"
JAVA_JAR="${HIPSGEN_JAR:-${ROOT}/Hipsgen.jar}"

mkdir -p "${INPUT_BASE}" "${OUTPUT_BASE}"

echo "experiment_group,dataset,case_name,tool,repeat,images,order,wall_s,top_order_tiles,total_tiles,gpu_total_stage_ms,gpu_coord_ms,gpu_kernel_ms,gpu_copy_ms,gpu_write_ms" > "${RESULTS_CSV}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

input_count() {
  find "${1}" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) | wc -l
}

copy_real_files() {
  local src_dir="$1"
  local dest_dir="$2"
  local count="$3"
  mkdir -p "${dest_dir}"
  find "${src_dir}" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) | sort | head -n "${count}" | while read -r f; do
    [ -e "${f}" ] || continue
    local real
    real="$(readlink -f "${f}")"
    cp -f "${real}" "${dest_dir}/$(basename "${f}")"
  done
}

prepare_inputs() {
  log "Preparing staged inputs under ${INPUT_BASE}"

  if [ ! -d "${INPUT_BASE}/desi_100" ] || [ "$(input_count "${INPUT_BASE}/desi_100")" -lt 100 ]; then
    rm -rf "${INPUT_BASE}/desi_100"
    copy_real_files "${DESI_INPUT_DIR:-${ROOT}/test_input}" "${INPUT_BASE}/desi_100" 100
  fi

  for n in 1 4 16; do
    rm -rf "${INPUT_BASE}/desi_${n}"
    mkdir -p "${INPUT_BASE}/desi_${n}"
    find "${INPUT_BASE}/desi_100" -maxdepth 1 -type f | sort | head -n "${n}" | while read -r f; do
      ln -sf "${f}" "${INPUT_BASE}/desi_${n}/$(basename "${f}")"
    done
  done

  if [ ! -d "${INPUT_BASE}/ep_1000" ] || [ "$(input_count "${INPUT_BASE}/ep_1000")" -lt 1000 ]; then
    rm -rf "${INPUT_BASE}/ep_1000"
    copy_real_files "${EP_INPUT_DIR:-${ROOT}/ep_test_input}" "${INPUT_BASE}/ep_1000" 1000
  fi

  log "Prepared inputs:"
  find "${INPUT_BASE}" -maxdepth 1 -mindepth 1 -type d | sort | while read -r d; do
    printf '  %s: %s files\n' "$(basename "${d}")" "$(input_count "${d}")"
  done
}

extract_gpu_metric() {
  local pattern="$1"
  local log_file="$2"
  grep "${pattern}" "${log_file}" | tail -n 1 | grep -oE '[0-9]+' | tail -n 1 || true
}

run_gpu_case() {
  local group="$1"
  local dataset="$2"
  local case_name="$3"
  local input_dir="$4"
  local order="$5"
  local repeat="$6"

  local out_dir="${OUTPUT_BASE}/${case_name}/gpu_r${repeat}"
  local log_file="${out_dir}.log"
  rm -rf "${out_dir}"
  mkdir -p "$(dirname "${out_dir}")"

  log "GPU ${case_name} repeat ${repeat}"
  /usr/bin/time -f 'WALL_S=%e' \
    "${GPU_BIN}" "${input_dir}" "${out_dir}" -order "${order}" > "${log_file}" 2>&1

  local wall_s top_tiles total_tiles total_stage coord_ms kernel_ms copy_ms write_ms
  wall_s="$(grep 'WALL_S=' "${log_file}" | tail -n 1 | cut -d= -f2)"
  top_tiles="$(find "${out_dir}/Norder${order}" -name 'Npix*.fits' 2>/dev/null | wc -l)"
  total_tiles="$(find "${out_dir}" -name 'Npix*.fits' 2>/dev/null | wc -l)"
  total_stage="$(extract_gpu_metric 'Total Full GPU stage time' "${log_file}")"
  coord_ms="$(extract_gpu_metric 'Coord comp:' "${log_file}")"
  kernel_ms="$(extract_gpu_metric 'GPU kernel:' "${log_file}")"
  copy_ms="$(extract_gpu_metric 'Copy back:' "${log_file}")"
  write_ms="$(extract_gpu_metric 'File write time:' "${log_file}")"

  echo "${group},${dataset},${case_name},gpu,${repeat},$(input_count "${input_dir}"),${order},${wall_s},${top_tiles},${total_tiles},${total_stage:-},${coord_ms:-},${kernel_ms:-},${copy_ms:-},${write_ms:-}" >> "${RESULTS_CSV}"
}

run_java_case() {
  local group="$1"
  local dataset="$2"
  local case_name="$3"
  local input_dir="$4"
  local order="$5"
  local repeat="$6"

  local out_dir="${OUTPUT_BASE}/${case_name}/java_r${repeat}"
  local log_file="${out_dir}.log"
  rm -rf "${out_dir}"
  mkdir -p "$(dirname "${out_dir}")"

  log "Java ${case_name} repeat ${repeat}"
  /usr/bin/time -f 'WALL_S=%e' \
    java -Xmx32g -jar "${JAVA_JAR}" \
      in="${input_dir}" \
      out="${out_dir}" \
      order="${order}" \
      id="TEST/${case_name}/r${repeat}" \
      INDEX TILES > "${log_file}" 2>&1

  local wall_s top_tiles total_tiles
  wall_s="$(grep 'WALL_S=' "${log_file}" | tail -n 1 | cut -d= -f2)"
  top_tiles="$(find "${out_dir}/Norder${order}" -name 'Npix*.fits' 2>/dev/null | wc -l)"
  total_tiles="$(find "${out_dir}" -name 'Npix*.fits' 2>/dev/null | wc -l)"

  echo "${group},${dataset},${case_name},java,${repeat},$(input_count "${input_dir}"),${order},${wall_s},${top_tiles},${total_tiles},,,,," >> "${RESULTS_CSV}"
}

run_pair() {
  local group="$1"
  local dataset="$2"
  local case_name="$3"
  local input_dir="$4"
  local order="$5"
  local java_repeats="$6"
  local gpu_repeats="$7"

  local r
  for r in $(seq 1 "${java_repeats}"); do
    run_java_case "${group}" "${dataset}" "${case_name}" "${input_dir}" "${order}" "${r}"
  done
  for r in $(seq 1 "${gpu_repeats}"); do
    run_gpu_case "${group}" "${dataset}" "${case_name}" "${input_dir}" "${order}" "${r}"
  done
}

prepare_inputs

run_pair bulk desi desi100_o5 "${INPUT_BASE}/desi_100" 5 2 2
run_pair bulk desi desi100_o7 "${INPUT_BASE}/desi_100" 7 2 2
run_pair bulk desi desi100_o9 "${INPUT_BASE}/desi_100" 9 2 2

run_pair incremental desi desi1_o7 "${INPUT_BASE}/desi_1" 7 3 3
run_pair incremental desi desi4_o7 "${INPUT_BASE}/desi_4" 7 3 3
run_pair incremental desi desi16_o7 "${INPUT_BASE}/desi_16" 7 3 3

run_pair bulk ep ep1000_o3 "${INPUT_BASE}/ep_1000" 3 1 2

log "Benchmark run complete: ${RESULTS_CSV}"
cat "${RESULTS_CSV}"
