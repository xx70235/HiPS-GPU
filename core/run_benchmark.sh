#!/bin/bash
# GPU HiPS 加速比 Benchmark 实验
# 测试不同数据量和Order下的性能表现

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="${CORE_ROOT:-${SCRIPT_DIR}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${CORE_ROOT}/.." && pwd)}"
HIPSGEN="${HIPSGEN_BIN:-${CORE_ROOT}/bin/hipsgen_cuda}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/test_input}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/generated_outputs/core_benchmark}"
RESULTS_FILE="${RESULTS_FILE:-${OUTPUT_BASE}/benchmark_results.csv}"

# 检查程序是否存在
if [ ! -f "$HIPSGEN" ]; then
    echo "Error: hipsgen_cuda not found at $HIPSGEN"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 初始化结果文件
echo "experiment,num_images,order,tiles,total_time_ms,gpu_kernel_ms,file_write_ms,coord_precompute_ms" > "$RESULTS_FILE"

# 准备临时输入目录（用于控制图像数量）
prepare_temp_input() {
    local num=$1
    local temp_dir="$OUTPUT_BASE/temp_input_${num}"
    rm -rf "$temp_dir"
    mkdir -p "$temp_dir"
    ls "$INPUT_DIR"/*.fits.fz | head -n $num | while read f; do
        ln -s "$f" "$temp_dir/"
    done
    echo "$temp_dir"
}

# 运行单次benchmark
run_benchmark() {
    local num_images=$1
    local order=$2
    local exp_name=$3
    
    echo ""
    echo "========================================"
    echo "Experiment: $exp_name"
    echo "  Images: $num_images, Order: $order"
    echo "========================================"
    
    local output_dir="$OUTPUT_BASE/${exp_name}"
    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    
    # 准备输入
    local temp_input=$(prepare_temp_input $num_images)
    
    # 运行并捕获输出
    local log_file="$output_dir/run.log"
    
    # 正确的参数格式：位置参数
    "$HIPSGEN" \
        "$temp_input" \
        "$output_dir" \
        -order $order \
        2>&1 | tee "$log_file"
    
    # 解析结果
    local total_time=$(grep "Total GPU stage time:" "$log_file" | grep -oP '\d+' | tail -1)
    local gpu_kernel=$(grep "GPU kernel:" "$log_file" | grep -oP '\d+' | tail -1)
    local file_write=$(grep "File write time:" "$log_file" | grep -oP '\d+' | tail -1)
    local coord_precompute=$(grep "Coord precompute:" "$log_file" | grep -oP '\d+' | tail -1)
    local tiles=$(grep -E "写入.*tiles|Written.*tiles|Total tiles" "$log_file" | grep -oP '\d+' | head -1)
    
    # 如果没找到tiles数，统计输出文件
    if [ -z "$tiles" ]; then
        tiles=$(find "$output_dir" -name "Npix*.fits" 2>/dev/null | wc -l)
    fi
    
    # 清理临时目录
    rm -rf "$temp_input"
    
    echo ""
    echo "Results: $exp_name"
    echo "  Tiles: ${tiles:-0}"
    echo "  Total time: ${total_time:-N/A} ms"
    echo "  GPU kernel: ${gpu_kernel:-N/A} ms"  
    echo "  File write: ${file_write:-N/A} ms"
    echo "  Coord precompute: ${coord_precompute:-N/A} ms"
    
    # 写入CSV
    echo "$exp_name,$num_images,$order,${tiles:-0},${total_time:-0},${gpu_kernel:-0},${file_write:-0},${coord_precompute:-0}" >> "$RESULTS_FILE"
}

echo "Starting GPU HiPS Benchmark..."
echo "Date: $(date)"
echo ""

# ============================================
# 实验1: 固定图像数量(20)，变化Order
# ============================================
echo ""
echo "########################################"
echo "# Experiment 1: Varying Order (20 images)"
echo "########################################"

for order in 3 5 7 9; do
    run_benchmark 20 $order "exp1_order${order}_img20"
done

# ============================================
# 实验2: 固定Order=7，变化图像数量
# ============================================
echo ""
echo "########################################"
echo "# Experiment 2: Varying Images (Order 7)"
echo "########################################"

for num in 10 20 50 100; do
    run_benchmark $num 7 "exp2_img${num}_order7"
done

# ============================================
# 汇总结果
# ============================================
echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Raw Results:"
cat "$RESULTS_FILE" | column -t -s','

# 计算统计
echo ""
echo "========================================"
echo "Performance Analysis"
echo "========================================"
awk -F',' 'NR>1 && $5>0 {
    gpu_pct = ($6>0) ? $6*100/$5 : 0;
    io_pct = ($7>0) ? $7*100/$5 : 0;
    coord_pct = ($8>0) ? $8*100/$5 : 0;
    tiles_per_sec = ($5>0) ? $4*1000/$5 : 0;
    printf "%-20s: %6d tiles in %7.1f sec | GPU %5.1f%% | I/O %5.1f%% | Coord %5.1f%% | %.1f tiles/s\n", 
           $1, $4, $5/1000, gpu_pct, io_pct, coord_pct, tiles_per_sec
}' "$RESULTS_FILE"
