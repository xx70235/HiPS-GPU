#!/bin/bash
# 完整对比测试：GPU vs HipsGen，不同Order

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_ROOT="${CORE_ROOT:-${SCRIPT_DIR}}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${CORE_ROOT}/.." && pwd)}"
HIPSGEN_JAR="${HIPSGEN_JAR:-${PROJECT_ROOT}/Hipsgen.jar}"
GPU_BIN="${HIPSGEN_BIN:-${CORE_ROOT}/bin/hipsgen_cuda}"
INPUT_100="${INPUT_100:-${PROJECT_ROOT}/generated_outputs/core_benchmark/temp_input_100}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/generated_outputs/core_full_benchmark}"
RESULTS="${RESULTS:-${OUTPUT_BASE}/benchmark_full_results.csv}"

echo "order,num_images,hipsgen_time,hipsgen_tiles,gpu_time,gpu_tiles,speedup" > "$RESULTS"

# 测试不同Order
for order in 5 7 9; do
    echo ""
    echo "============================================"
    echo "=== Order $order, 100 images ==="
    echo "============================================"
    
    # HipsGen测试
    echo "--- HipsGen ---"
    HG_OUT="$OUTPUT_BASE/hipsgen_o${order}"
    rm -rf "$HG_OUT"
    
    HG_START=$(date +%s.%N)
    java -Xmx32g -jar "$HIPSGEN_JAR" \
        in="$INPUT_100" \
        out="$HG_OUT" \
        order=$order \
        id="TEST/hipsgen" \
        2>&1 | grep -E "TILES done|THE END"
    HG_END=$(date +%s.%N)
    HG_TIME=$(echo "$HG_END - $HG_START" | bc)
    HG_TILES=$(find "$HG_OUT" -name "Npix*.fits" 2>/dev/null | wc -l)
    echo "HipsGen: ${HG_TIME}s, $HG_TILES tiles"
    
    # GPU测试
    echo "--- GPU ---"
    GPU_OUT="$OUTPUT_BASE/gpu_o${order}"
    rm -rf "$GPU_OUT"
    
    GPU_START=$(date +%s.%N)
    "$GPU_BIN" "$INPUT_100" "$GPU_OUT" -order $order 2>&1 | grep -E "Total Full GPU|File write|HiPS Generation Complete"
    GPU_END=$(date +%s.%N)
    GPU_TIME=$(echo "$GPU_END - $GPU_START" | bc)
    GPU_TILES=$(find "$GPU_OUT" -name "Npix*.fits" 2>/dev/null | wc -l)
    echo "GPU: ${GPU_TIME}s, $GPU_TILES tiles"
    
    # 计算加速比
    SPEEDUP=$(echo "scale=2; $HG_TIME / $GPU_TIME" | bc 2>/dev/null || echo "N/A")
    echo "Speedup: ${SPEEDUP}x"
    
    # 保存结果
    echo "$order,100,$HG_TIME,$HG_TILES,$GPU_TIME,$GPU_TILES,$SPEEDUP" >> "$RESULTS"
done

echo ""
echo "============================================"
echo "=== 完整结果 ==="
echo "============================================"
cat "$RESULTS" | column -t -s','
