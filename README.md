# RetinaFace ONNX 转换与检测工具

完整的RetinaFace ONNX模型转换和批量人脸检测工具集，支持PyTorch模型转换为ONNX格式，以及基于ONNX的批量人脸检测，具备正脸检测和置信度融合功能。

## 功能特性

### 🔄 模型转换功能
- **PyTorch转ONNX**: 将训练好的PyTorch RetinaFace模型转换为ONNX格式
- **配置文件支持**: 基于JSON配置文件进行模型转换
- **动态批处理**: 支持动态批处理维度的ONNX导出
- **权重兼容性**: 自动处理DataParallel等训练方式的权重格式

### 🎯 批量检测功能
- **批量处理**: 支持单张图像或整个目录的批量处理
- **递归搜索**: 可选择是否递归搜索子目录
- **正脸检测**: 基于关键点分析的正脸置信度计算
- **置信度融合**: 结合检测置信度和正脸置信度的综合评分
- **灵活输出**: 支持覆盖原文件或保存到指定目录

### 📊 检测信息
- 边界框坐标
- 检测置信度
- 正脸置信度
- 融合置信度
- 人脸类型（正脸/侧脸）
- 关键点数量和位置

### 🎨 可视化功能
- 彩色边界框（绿色=正脸，橙色=侧脸）
- 多层信息显示（检测/正脸/融合置信度）
- 关键点标注（蓝色=正脸，红色=侧脸）
- 统计信息显示

## 安装要求

### 基础依赖
```bash
pip install opencv-python numpy onnxruntime
```

### 模型转换依赖（如需转换PyTorch模型）
```bash
pip install torch torchvision
```

### GPU加速（可选）
```bash
# ONNX推理GPU加速
pip install onnxruntime-gpu

# PyTorch GPU支持
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 使用方法

### 第一步：模型转换（如果您有PyTorch模型）

#### 基本语法
```bash
python convert_to_onnx.py --model_path <PyTorch模型路径> --config_path <配置文件路径> [选项]
```

#### 必需参数
- `--model_path`: PyTorch模型权重文件路径（.pt或.pth）
- `--config_path`: 模型配置文件路径（JSON格式）

#### 可选参数
- `--output_path`: 输出ONNX模型路径（默认：model.onnx）
- `--input_shape`: 输入张量形状，格式：batch,channel,height,width（默认：1,3,640,640）

#### 转换示例
```bash
# 基本转换
python convert_to_onnx.py --model_path RetinaFace/pytorch_model.pt --config_path RetinaFace/configuration.json

# 指定输出路径和输入尺寸
python convert_to_onnx.py --model_path RetinaFace/pytorch_model.pt --config_path RetinaFace/configuration.json --output_path RetinaFace/model_test.onnx --input_shape 1,3,640,640
```

### 第二步：批量人脸检测

#### 基本语法
```bash
python batch_face_detection.py -m <ONNX模型路径> -i <输入路径> [选项]
```

### 必需参数

- `-m, --model`: ONNX模型文件路径
- `-i, --input`: 输入图像文件或目录路径

### 可选参数

- `-o, --output`: 输出目录路径（默认覆盖原文件）
- `-c, --confidence`: 置信度阈值，范围0-1（默认0.3）
- `--recursive`: 递归搜索子目录（默认启用）
- `--no-recursive`: 不递归搜索子目录
- `--overwrite`: 强制覆盖原文件（忽略--output参数）
- `--device`: 推理设备，cpu或cuda（默认cpu）
- `--suffix`: 输出文件名后缀（默认_detected）

## 使用示例

### 1. 处理单张图像

```bash
# 覆盖原文件
python batch_face_detection.py -m RetinaFace/model_test.onnx -i image.jpg

# 保存到新文件
python batch_face_detection.py -m RetinaFace/model_test.onnx -i image.jpg -o results
```

### 2. 批量处理目录

```bash
# 处理当前目录，不递归搜索
python batch_face_detection.py -m RetinaFace/model_test.onnx -i . --no-recursive -o results

# 递归处理整个目录树
python batch_face_detection.py -m RetinaFace/model_test.onnx -i ./dataset --recursive -o ./output
```

### 3. 自定义置信度阈值

```bash
# 高置信度检测（减少误检）
python batch_face_detection.py -m RetinaFace/model_test.onnx -i ./images -c 0.7 -o results

# 低置信度检测（检测更多人脸）
python batch_face_detection.py -m RetinaFace/model_test.onnx -i ./images -c 0.2 -o results
```

### 4. 覆盖原文件

```bash
# 直接覆盖原文件
python batch_face_detection.py -m RetinaFace/model_test.onnx -i ./images --overwrite
```

### 5. GPU加速

```bash
# 使用CUDA加速（需要CUDA环境）
python batch_face_detection.py -m RetinaFace/model_test.onnx -i ./images --device cuda
```

## 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## 输出信息说明

### 处理过程信息
```
[1/5] 处理中: image.jpg ✓ image.jpg: 3个人脸 (正脸:2, 侧脸:1) 耗时:0.156s -> image_detected.jpg
```

### 统计信息
```
=== 处理完成 ===
总计处理: 5 个文件
成功处理: 5 个文件
失败处理: 0 个文件
总耗时: 0.88秒
平均耗时: 0.175秒/图像
```

## 正脸检测算法

工具使用基于关键点分析的正脸检测算法，评估以下因素：

1. **眼睛对称性**: 双眼是否水平对称
2. **鼻子位置**: 鼻子是否位于人脸中心
3. **嘴部对称性**: 嘴角是否对称
4. **五官排列**: 眼-鼻-嘴的垂直顺序
5. **眼距合理性**: 双眼间距是否符合比例
6. **边界框包含**: 关键点是否在检测框内

正脸置信度范围0-1，>0.5判定为正脸。

## 性能优化建议

1. **批量处理**: 一次处理多个文件比单独处理更高效
2. **合适阈值**: 根据应用场景调整置信度阈值
3. **GPU加速**: 大批量处理时使用CUDA加速
4. **目录结构**: 合理组织输入目录结构

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   错误: 模型文件不存在: model.onnx
   ```
   解决：检查模型文件路径是否正确

2. **输入路径不存在**
   ```
   错误: 输入路径不存在: ./images
   ```
   解决：确认输入路径存在且可访问

3. **无法读取图像**
   ```
   错误: 无法读取图像 image.jpg
   ```
   解决：检查图像文件是否损坏或格式不支持

4. **CUDA不可用**
   ```
   警告: CUDA设备不可用，使用CPU
   ```
   解决：安装CUDA和onnxruntime-gpu

### 性能问题

- **处理速度慢**: 考虑使用GPU加速或降低输入图像分辨率
- **内存不足**: 减少批量处理的文件数量
- **检测效果差**: 调整置信度阈值或检查模型质量

## 文件结构

```
.
├── convert_to_onnx.py           # PyTorch到ONNX转换脚本
├── batch_face_detection.py      # 批量人脸检测脚本
├── inference_onnx.py            # ONNX推理核心模块
├── models.py                    # RetinaFace模型定义
├── test_inference.py            # 单张图像测试脚本
├── README_batch_detection.md    # 项目说明文档
└── RetinaFace/
    ├── pytorch_model.pt         # PyTorch模型权重
    ├── configuration.json       # 模型配置文件
    ├── model_test.onnx         # 转换后的ONNX模型
    └── net.py                  # 原始网络定义
```

## 完整工作流程

### 1. 准备阶段
确保您有以下文件：
- PyTorch训练好的模型权重文件（.pt/.pth）
- 对应的配置文件（JSON格式）
- 测试图像

### 2. 模型转换
```bash
# 将PyTorch模型转换为ONNX格式
python convert_to_onnx.py --model_path RetinaFace/pytorch_model.pt --config_path RetinaFace/configuration.json --output_path RetinaFace/model_test.onnx
```

### 3. 单张测试
```bash
# 测试转换后的ONNX模型
python test_inference.py
```

### 4. 批量处理
```bash
# 批量处理图像
python batch_face_detection.py -m RetinaFace/model_test.onnx -i ./images -o ./results
```

## 技术特点

### 模型转换特性
- **自动权重处理**: 自动处理DataParallel训练的模型权重
- **动态维度支持**: 支持动态批处理维度的ONNX导出
- **ONNX算子兼容**: 使用ONNX opset version 11确保兼容性
- **设备自适应**: 自动检测CUDA可用性并选择合适设备

### 检测算法优势
- **高精度检测**: 基于RetinaFace的高精度人脸检测
- **正脸识别**: 独创的基于关键点分析的正脸置信度算法
- **置信度融合**: 智能融合检测置信度和正脸置信度
- **性能优化**: ONNX推理引擎提供高效的推理性能

## 更新日志

### v1.0.0
- 初始版本发布
- 支持PyTorch到ONNX模型转换
- 支持批量人脸检测
- 集成正脸检测功能
- 提供完整的命令行接口
- 支持多种输出模式

## 许可证

本工具基于现有的RetinaFace ONNX推理代码开发，遵循相应的开源许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue报告问题
- 提交Pull Request贡献代码
- 发送邮件询问使用问题

---

**注意**: 使用前请确保已正确安装所有依赖项，并准备好ONNX模型文件。
