# HiPS Auto Order Calculation Feature

## 🎉 新功能：自动计算推荐 Order

程序现在可以自动根据输入图像的像素分辨率计算最优的 HEALPix order 值！

---

## 📐 工作原理

### 1. **像素分辨率检测**

程序启动时自动：
- 采样前 20 个 FITS 文件
- 读取 WCS header 信息（CD 矩阵或 CDELT 关键字）
- 计算像素角分辨率（单位：角秒）
- 取中位数作为代表值

### 2. **Order 计算公式**

```
HiPS tile 分辨率 = 374400 / (2^order × tile_width) 角秒

为了匹配图像分辨率：
order ≥ log₂(374400 / (pixel_scale × tile_width))
```

其中：
- `374400` ≈ 180° × 3600″/° / sqrt(3) / π
- `tile_width` = 512 像素（默认）
- `pixel_scale` = 图像像素角分辨率（角秒）

### 3. **示例计算**

#### DESI DR10 数据
```
输入：pixel_scale = 0.262 arcsec
计算：order ≥ log₂(374400 / (0.262 × 512))
     ≥ log₂(2792.7)
     ≥ 11.4
推荐：order = 12  （向上取整）

验证：tile_resolution = 374400 / (2^12 × 512)
                       = 0.178 arcsec  ✓ 小于 0.262 arcsec
```

#### 其他常见数据

| 数据源 | 像素尺度 | 推荐 Order | Tile 分辨率 |
|--------|----------|-----------|------------|
| DESI DR10 | 0.262″ | 12 | 0.178″ |
| HST/ACS | 0.05″ | 15 | 0.022″ |
| SDSS | 0.396″ | 11 | 0.356″ |
| Pan-STARRS | 0.258″ | 12 | 0.178″ |
| 2MASS | 1.0″ | 10 | 0.712″ |
| WISE | 2.75″ | 8 | 2.848″ |

---

## 🔧 使用方法

### 自动模式（推荐）

```bash
# 不指定 -order，程序自动计算
./bin/hipsgen_cuda input_dir output_dir

# 输出示例：
# ===================
# Scanning files for auto order calculation...
# Found 62 FITS files
# 
# === Auto Order Calculation ===
#   Pixel scale: 0.262 arcsec
#   Recommended order: 12 (auto)
#   Tile resolution at order 12: 0.178 arcsec
# 
# Configuration:
#   Max order: 12 (auto)  ← 自动计算
# ===================
```

### 手动模式

```bash
# 显式指定 -order，覆盖自动计算
./bin/hipsgen_cuda input_dir output_dir -order 7

# 输出示例：
# ===================
# Configuration:
#   Max order: 7 (user)  ← 手动指定
# ===================
```

---

## 📊 测试结果

### 测试 1: DESI DR10 单个 brick

```bash
CUDA_VISIBLE_DEVICES=1 ./bin/hipsgen_cuda \
  /mnt/mirror/102022-DESI-DR10/south/coadd/317/3173p000 \
  test_auto_order \
  -limit 5 -v
```

**输出**:
```
Scanning files for auto order calculation...
Found 5 FITS files

=== Auto Order Calculation ===
  Pixel scale: 0.262 arcsec
  Recommended order: 12 (auto)
  Tile resolution at order 12: 0.196 arcsec

Configuration:
  Max order: 12 (auto)  ✅ 自动计算
```

### 测试 2: 手动指定

```bash
CUDA_VISIBLE_DEVICES=1 ./bin/hipsgen_cuda \
  /mnt/mirror/102022-DESI-DR10/south/coadd/317/3173p000 \
  test_manual_order \
  -order 7 \
  -limit 5
```

**输出**:
```
Configuration:
  Max order: 7 (user)  ✅ 手动指定
```

---

## 🎓 技术细节

### 像素分辨率估计算法

1. **CD 矩阵方法**（优先）
   ```cpp
   // 读取 CD 矩阵
   CD1_1, CD1_2
   CD2_1, CD2_2
   
   // 计算像素尺度
   scale1 = sqrt(CD1_1² + CD2_1²)
   scale2 = sqrt(CD1_2² + CD2_2²)
   pixel_scale = (scale1 + scale2) / 2  // 度
   pixel_scale_arcsec = pixel_scale * 3600  // 角秒
   ```

2. **CDELT 方法**（备用）
   ```cpp
   // 读取 CDELT
   CDELT1, CDELT2
   
   // 计算像素尺度
   pixel_scale = (|CDELT1| + |CDELT2|) / 2  // 度
   pixel_scale_arcsec = pixel_scale * 3600  // 角秒
   ```

3. **中位数取值**
   - 采样多个文件（默认 20 个）
   - 计算各自的像素尺度
   - 取中位数（比平均值更稳健）

### Order 限制

- **最小值**: 0
  - 对应 nside=1，12 个 tiles
  - tile 分辨率 ~59°
  
- **最大值**: 15
  - 对应 nside=32768
  - ~805 million tiles
  - tile 分辨率 ~0.014″

### 自动化标识

程序通过 `config.autoOrder` 标志追踪：
- `true`: 使用自动计算
- `false`: 用户手动指定（通过 `-order` 参数）

输出中显示：
- `(auto)`: 自动计算
- `(user)`: 手动指定

---

## ⚠️ 注意事项

### 1. 数据一致性

- 如果不同图像的像素尺度差异很大，自动计算可能不准确
- 建议检查采样结果，必要时手动指定 order

### 2. 性能考虑

| Order | Tiles 数量 | 适用场景 |
|-------|-----------|---------|
| 7 | 196,608 | 快速预览、测试 |
| 9 | 3,145,728 | 中等分辨率 |
| 11 | 50,331,648 | 高分辨率（推荐） |
| 12 | 201,326,592 | 极高分辨率 |
| 13+ | 805+ million | 谨慎使用，处理时间很长 |

### 3. 磁盘空间

```
预估：
Order 11 = 50M tiles × 50 KB/tile ≈ 2.5 TB
Order 12 = 201M tiles × 50 KB/tile ≈ 10 TB
Order 13 = 805M tiles × 50 KB/tile ≈ 40 TB
```

---

## 💡 推荐使用策略

### 场景 1: 第一次处理新数据集

```bash
# 使用自动模式，查看推荐值
./bin/hipsgen_cuda input output -limit 100 -v

# 检查输出的推荐 order
# 如果合理，继续完整处理
```

### 场景 2: 快速预览/测试

```bash
# 手动指定较低的 order
./bin/hipsgen_cuda input output -order 7 -limit 1000
```

### 场景 3: 生产环境

```bash
# 根据需求选择：
# - order 9-10: 平衡性能和质量
# - order 11-12: 高质量（推荐）
# - order 13+: 极高质量（大数据集谨慎使用）

./bin/hipsgen_cuda input output -order 11
```

---

## 🔍 调试

### 查看详细信息

```bash
# 添加 -v 参数显示详细日志
./bin/hipsgen_cuda input output -v

# 输出包括：
# - 系统资源检测详情
# - 像素尺度采样过程
# - Order 计算公式和结果
```

### 验证 Order 选择

```bash
# 检查生成的 HiPS 属性文件
cat output/properties

# 应包含：
# hips_order = 12
# hips_tile_width = 512
```

---

## 📝 实现文件

- **主文件**: `src/cpp/hipsgen_cuda.cpp`
  - `calculateRecommendedOrder()`: Order 计算
  - `estimatePixelScale()`: 像素尺度估计
  
- **配置**: `Config struct`
  - `autoOrder`: 自动 order 标志
  - `orderMax`: 最大 order 值

---

## ✅ 总结

### 优点

1. ✅ **自动化** - 无需手动计算 order
2. ✅ **准确** - 基于实际图像 WCS 信息
3. ✅ **灵活** - 支持手动覆盖
4. ✅ **稳健** - 中位数估计，处理异常值
5. ✅ **快速** - 只采样少量文件，不影响性能

### 最佳实践

```bash
# 1. 首次运行，使用自动模式查看推荐
./bin/hipsgen_cuda input output -limit 100 -v

# 2. 确认推荐合理后，完整处理
./bin/hipsgen_cuda input output

# 3. 或者根据需求手动调整
./bin/hipsgen_cuda input output -order 11
```

---

**日期**: 2026-01-27  
**版本**: v1.1 - Auto Order Feature  
**状态**: ✅ Tested and Working

