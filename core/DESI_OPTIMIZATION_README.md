# DESI DR10 HiPS 生成优化

## 已完成的改进

### 1. ✅ 递归目录扫描
- 添加了 `scanFitsFiles()` 函数支持递归扫描
- 新增 `-recursive` 或 `-r` 参数
- 支持DESI的多层目录结构：`south/coadd/000-359/XXXXpYYY/`

### 2. ✅ 小规模测试模式
- 新增 `-limit <n>` 参数限制处理文件数
- 新增 `-progress <n>` 参数控制进度报告间隔
- 方便快速调试和测试

### 3. ✅ .fits.fz 压缩格式支持
- **验证通过**: CFITSIO 可以透明读取 fpack 压缩文件
- **注意**: 压缩文件的 WCS 信息在 HDU 2（CFITSIO 会自动处理）
- 扫描函数已支持 `.fz` 扩展名

### 4. ✅ 增强的命令行参数
```bash
-recursive, -r  : 递归扫描子目录
-limit <n>      : 限制处理文件数（用于测试）
-cache <n>      : 内存中缓存的FITS文件数（默认:100）
-progress <n>   : 进度报告间隔（默认:100文件）
```

## 编译成功
```bash
cd /path/to/HiPS-GPU/core
make hipsgen
# 生成: bin/hipsgen_cuda
```

## 待完成的关键改进

### 1. 🔄 修复文件扫描集成
**问题**: `scanFitsFiles()` 函数已创建但未被主程序调用  
**位置**: src/cpp/hipsgen_cuda.cpp 第 687, 842 行仍使用旧的 `fs::directory_iterator`

**修复方法**:
```cpp
// 替换第 687 行附近:
// OLD:
//   for (const auto& entry : fs::directory_iterator(config.inputDir)) {
//     if (entry.is_regular_file()) {
//       std::string ext = entry.path().extension().string();
//       ...

// NEW:
std::vector<std::string> filePaths = scanFitsFiles(config);
for (const auto& filePath : filePaths) {
    FitsData fits = FitsReader::readFitsFile(filePath);
    if (config.skipErrors && !fits.isValid) {
        std::cerr << "Skipping invalid file: " << filePath << std::endl;
        continue;
    }
    ...
}
```

### 2. 🔄 流式内存管理
当前代码仍然一次加载所有文件：
```cpp
std::vector<FitsData> allFitsFiles;  // 会占用大量内存
```

建议改为:
- 先扫描获取文件路径列表
- 按需加载每个文件
- 使用 LRU 缓存机制

### 3. 🔄 错误处理增强
添加 try-catch 和错误跳过逻辑

## 测试说明

### 测试 1: 单个 brick（5文件）
```bash
mkdir -p ./test_outputs

CUDA_VISIBLE_DEVICES=1 ./bin/hipsgen_cuda \
  /mnt/mirror/102022-DESI-DR10/south/coadd/317/3173p000 \
  ./test_outputs/test1_single_brick \
  -order 3 \
  -limit 5 \
  -v
```

**当前结果**: 只读取了 2 个 .fits 文件（depth.fits 和 ccds.fits），  
未读取 .fits.fz 文件，因为扫描函数未被调用。

### 测试 2: 单个目录（带递归）
```bash
# 测试递归扫描一个 brick 目录
CUDA_VISIBLE_DEVICES=1 ./bin/hipsgen_cuda \
  /mnt/mirror/102022-DESI-DR10/south/coadd/317/3173p000 \
  ./test_outputs/test2_recursive \
  -r \
  -limit 10 \
  -order 3 \
  -v
```

### 测试 3: 多个 brick（一个子目录）
```bash
# 317/ 目录包含约 1,200 个 brick
CUDA_VISIBLE_DEVICES=1 ./bin/hipsgen_cuda \
  /mnt/mirror/102022-DESI-DR10/south/coadd/317 \
  ./test_outputs/test3_subdir \
  -r \
  -limit 100 \
  -order 5 \
  -progress 10 \
  -v
```

## DESI DR10 完整处理估算

### 数据规模
- **文件数**: ~365,000 bricks × 60 files/brick ≈ 21,900,000 files
- **只处理 image-r**: 365,000 files
- **数据量**: ~74 TB (south/coadd)
- **单文件大小**: ~13 MB (image-r.fits.fz)

### 性能预估
基于测试数据（9,351文件 → 52秒）:
- **单 GPU (A100)**: 365,000/9,351 × 52s ≈ **34 分钟**（只处理 image-r）
- **处理所有波段**: × 4波段 ≈ **2.3 小时**

### 推荐策略
1. **分波段处理**: 每个波段单独生成 HiPS
   ```bash
   # 只处理 g 波段
   -limit 0  # 不限制
   --filter "*-image-g.fits.fz"  # 需要实现文件过滤
   ```

2. **分区域处理**: 按 RA 范围分批
   ```bash
   # 处理 000-099
   -recursive -inputDir /mnt/mirror/.../coadd/0[0-9][0-9]
   ```

3. **并行处理**: 使用多个 GPU
   - GPU 0: 000-089
   - GPU 1: 090-179
   - ...

## 下一步工作

### 优先级 P0（必须）
1. [ ] 修复 `scanFitsFiles()` 集成问题
2. [ ] 添加文件名过滤（只处理 image-*.fits.fz）
3. [ ] 测试完整的单个目录处理

### 优先级 P1（重要）
4. [ ] 实现流式内存管理
5. [ ] 添加详细的进度输出
6. [ ] 错误处理和损坏文件跳过

### 优先级 P2（优化）
7. [ ] 断点续传支持
8. [ ] 多 GPU 并行
9. [ ] 性能分析和瓶颈优化

## 相关测试脚本
- `/tmp/test_desi_fz` - 测试 .fits.fz 读取
- `/tmp/test_fz_hdu` - 检查 HDU 结构
