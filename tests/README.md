# 红外可见光Agent系统 - 测试文档

本目录包含红外可见光Agent系统的所有测试文件，已按功能模块进行拆分，便于单独测试和维护。

## 📁 测试文件结构

```
tests/
├── conftest.py              # pytest配置文件，提供共享夹具
├── run_tests.py             # 测试运行脚本
├── test_preprocess.py       # 预处理工具测试
├── test_fusion.py           # 融合工具测试
├── test_segment.py          # 分割工具测试
├── test_metrics.py          # 评测工具测试
├── test_report.py           # 报告生成工具测试
├── test_integration.py      # 集成测试
├── test_data/               # 测试数据目录
├── test_outputs/            # 测试输出目录
├── test_logs/               # 测试日志目录
└── README.md               # 本文件
```

## 🚀 快速开始

### 安装测试依赖

```bash
pip install pytest pytest-cov numpy opencv-python matplotlib seaborn
```

### 运行所有测试

```bash
# 使用测试运行脚本
python tests/run_tests.py --all

# 或直接使用pytest
pytest tests/ -v
```

### 运行特定模块测试

```bash
# 运行预处理模块测试
python tests/run_tests.py --module preprocess

# 运行融合模块测试
python tests/run_tests.py --module fusion

# 运行分割模块测试
python tests/run_tests.py --module segment

# 运行评测模块测试
python tests/run_tests.py --module metrics

# 运行报告模块测试
python tests/run_tests.py --module report

# 运行集成测试
python tests/run_tests.py --module integration
```

### 运行特定测试模式

```bash
# 运行包含"basic"的测试
python tests/run_tests.py --pattern "basic"

# 运行包含"edge"的测试
python tests/run_tests.py --pattern "edge"
```

### 生成覆盖率报告

```bash
# 生成覆盖率报告
python tests/run_tests.py --all --coverage

# 或使用pytest
pytest tests/ --cov=. --cov-report=html --cov-report=term
```

## 📋 测试模块说明

### 1. test_preprocess.py - 预处理测试

测试图像预处理功能，包括：
- 基本预处理功能
- 图像尺寸调整
- 图像增强
- 图像归一化
- 图像配准
- 直方图均衡化
- 降噪处理
- 伽马校正
- 边缘情况处理

**运行方式：**
```bash
python tests/run_tests.py --module preprocess
```

### 2. test_fusion.py - 融合测试

测试图像融合功能，包括：
- 加权平均融合
- 拉普拉斯金字塔融合
- 离散小波变换融合
- 主成分分析融合
- 联合任务融合
- 对比度增强
- 不同权重配置
- 边缘情况处理

**运行方式：**
```bash
python tests/run_tests.py --module fusion
```

### 3. test_segment.py - 分割测试

测试图像分割功能，包括：
- 阈值分割
- 分水岭分割
- K-means分割
- GrabCut分割
- 轮廓检测分割
- 形态学操作
- 后处理
- 置信度计算
- 不同配置测试

**运行方式：**
```bash
python tests/run_tests.py --module segment
```

### 4. test_metrics.py - 评测测试

测试质量评测功能，包括：
- 融合质量评估（熵、互信息、Qabf、SSIM、PSNR）
- 分割质量评估（mIoU、Dice、精确率、召回率、F1分数）
- 灰度图像评测
- 完美匹配测试
- 无重叠测试
- 边缘情况处理
- 配置测试

**运行方式：**
```bash
python tests/run_tests.py --module metrics
```

### 5. test_report.py - 报告测试

测试报告生成功能，包括：
- HTML报告生成
- JSON报告生成
- 图像对比图创建
- 指标可视化
- 执行时间线
- 指标表格
- 不同配置测试
- 边缘情况处理

**运行方式：**
```bash
python tests/run_tests.py --module report
```

### 6. test_integration.py - 集成测试

测试完整处理流程，包括：
- 完整处理流程
- 联合融合流程
- 不同方法组合
- 错误处理
- 数据流验证
- 性能测试
- 系统集成测试

**运行方式：**
```bash
python tests/run_tests.py --module integration
```

## 🔧 测试配置

### conftest.py 配置

`conftest.py` 文件提供了以下共享夹具：

- `sample_images`: 样例图像数据
- `sample_masks`: 样例掩码数据
- `sample_configs`: 样例配置数据
- `temp_dir`: 临时目录
- `sample_metrics`: 样例指标数据
- `sample_execution_steps`: 样例执行步骤数据
- `edge_case_images`: 边缘情况图像数据
- `test_data_path`: 测试数据路径

### 测试标记

系统自动为测试添加以下标记：

- `@pytest.mark.preprocess`: 预处理相关测试
- `@pytest.mark.fusion`: 融合相关测试
- `@pytest.mark.segmentation`: 分割相关测试
- `@pytest.mark.metrics`: 评测相关测试
- `@pytest.mark.report`: 报告相关测试
- `@pytest.mark.integration`: 集成测试
- `@pytest.mark.unit`: 单元测试

### 运行特定标记的测试

```bash
# 运行所有预处理测试
pytest tests/ -m preprocess

# 运行所有融合测试
pytest tests/ -m fusion

# 运行所有集成测试
pytest tests/ -m integration
```

## 📊 测试覆盖率

### 查看覆盖率报告

```bash
# 生成HTML覆盖率报告
pytest tests/ --cov=. --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

### 覆盖率目标

- 单元测试覆盖率：> 90%
- 集成测试覆盖率：> 80%
- 总体覆盖率：> 85%

## 🐛 调试测试

### 详细输出

```bash
# 详细输出
python tests/run_tests.py --verbose

# 或使用pytest
pytest tests/ -v -s
```

### 运行单个测试

```bash
# 运行特定测试函数
pytest tests/test_preprocess.py::TestPreprocess::test_preprocess_basic -v

# 运行特定测试类
pytest tests/test_fusion.py::TestFusion -v
```

### 调试模式

```bash
# 在失败时进入调试器
pytest tests/ --pdb

# 在第一个失败时停止
pytest tests/ -x
```

## 📝 添加新测试

### 1. 创建测试文件

```python
# tests/test_new_module.py
import pytest
import numpy as np
from tools.new_module import new_function

class TestNewModule:
    def test_new_function_basic(self):
        """测试新功能的基本功能"""
        # 测试代码
        pass
    
    def test_new_function_edge_cases(self):
        """测试新功能的边缘情况"""
        # 测试代码
        pass
```

### 2. 使用共享夹具

```python
def test_with_fixtures(sample_images, temp_dir):
    """使用共享夹具的测试"""
    ir_img = sample_images["ir_medium"]
    # 测试代码
    pass
```

### 3. 添加测试标记

```python
@pytest.mark.new_module
def test_new_function():
    """标记为新模块测试"""
    pass
```

## 🔍 测试最佳实践

### 1. 测试结构

- 每个测试函数只测试一个功能
- 使用描述性的测试函数名
- 包含基本功能、边缘情况和错误处理测试

### 2. 测试数据

- 使用 `conftest.py` 中的共享夹具
- 创建有意义的测试数据
- 测试不同尺寸和类型的图像

### 3. 断言

- 使用明确的断言
- 检查返回值的类型和范围
- 验证数据形状和内容

### 4. 错误处理

- 测试异常情况
- 验证错误消息
- 确保系统能够优雅地处理错误

## 📈 性能测试

### 运行性能测试

```bash
# 运行性能测试
python tests/run_tests.py --pattern "performance"
```

### 性能基准

- 预处理时间：< 10秒（512x512图像）
- 融合时间：< 10秒（512x512图像）
- 分割时间：< 10秒（512x512图像）
- 评测时间：< 5秒
- 报告生成时间：< 10秒

## 🚨 常见问题

### 1. 导入错误

确保项目根目录在Python路径中：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 2. 依赖问题

安装所有必要的依赖：

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### 3. 内存问题

对于大图像测试，可能需要增加内存限制或使用较小的测试图像。

### 4. 临时文件清理

测试会自动清理临时文件，但如果测试中断，可能需要手动清理：

```bash
rm -rf tests/test_outputs/*
rm -rf tests/test_logs/*
```

## 📞 支持

如果遇到测试问题，请：

1. 检查测试输出和错误信息
2. 确认所有依赖已正确安装
3. 验证测试数据是否正确
4. 查看相关模块的文档

---

**注意：** 运行测试前请确保已安装所有必要的依赖包，并且项目结构完整。
