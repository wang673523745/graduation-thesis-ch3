# FusionX Agent系统

> 围绕"红外与可见光图像融合与分割"构建的可扩展图像Agent系统，具备**理解—决策—执行—自检**能力，通过工具化封装实现**多任务编排、自动评测与报告生成**。

## 🎯 项目目标

- **目标**：
  1. 提出一套 **视觉 Agent 化** 的系统架构（LLM 调度 + 工具集）
  2. 将**融合**与**分割**模型封装成可调用的工具，并引入**评测/报告**工具


## 🏗️ 系统架构

```
irvis-agent/
├── agent/                    # Agent核心模块
│   ├── planner.py           # 任务规划 & 反思策略
│   ├── tool_router.py       # 工具路由与重试
│   └── prompts/             # 系统/工具/自检/报告生成提示词
├── tools/                   # 工具集
│   ├── preprocess.py        # 配准、增强、归一化、格式转换
│   ├── fuse.py              # 融合：单任务 & 联合任务两套接口
│   ├── segment.py           # 分割：语义/实例/多头输出
│   ├── metrics.py           # 熵、MI、Qabf、SSIM、PSNR、mIoU...
│   └── report.py            # 可视化对比、表格、PDF/HTML报告
├── api/                     # API接口
│   ├── server.py            # FastAPI：统一REST入口
│   └── schemas.py           # Pydantic模型定义
├── configs/                 # 配置文件
│   └── default.yaml         # 默认配置
├── data/                    # 数据目录
│   ├── samples/             # Demo样例
│   └── outputs/             # 中间/最终结果
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包列表
└── README.md               # 项目文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.21+
- FastAPI 0.68+



### 运行系统

#### 1. API服务器模式

```bash
# 启动API服务器
python main.py --mode api

# 访问API文档
# http://localhost:8000/docs
```

#### 2. 演示模式

```bash
# 运行演示处理流程
python main.py --mode demo
```

#### 3. 测试模式

```bash
# 运行系统测试
python main.py --mode test
```

## 📋 核心功能

### 1. 图像预处理 (preprocess.py)

- **功能**：尺寸对齐、配准、去噪、直方图均衡化、归一化
- **接口**：
```python
def preprocess(ir_img: np.ndarray, vis_img: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """return {'ir': ir_p, 'vis': vis_p, 'warp': M, 'debug': {...}}"""
```

### 2. 图像融合 (fuse.py)

- **功能**：支持多种融合算法
- **接口**：
```python
# 单任务融合
def fuse_single(ir: np.ndarray, vis: np.ndarray, cfg: dict) -> np.ndarray

# 联合任务融合
def fuse_joint(ir: np.ndarray, vis: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """return {'fused': F, 'aux': {...}}"""
```

**支持的融合方法**：
- 加权平均融合
- 拉普拉斯金字塔融合
- 离散小波变换融合
- 主成分分析融合

### 3. 图像分割 (segment.py)

- **功能**：语义分割、实例分割
- **接口**：
```python
def segment(img: np.ndarray, cfg: dict) -> np.ndarray:  # 语义分割 mask
```

**支持的分割方法**：
- 阈值分割
- 分水岭分割
- K-means聚类分割
- GrabCut分割
- 轮廓检测分割

### 4. 质量评测 (metrics.py)

- **功能**：计算各种图像质量指标
- **接口**：
```python
def eval_fusion(ir, vis, fused) -> Dict[str, float]:     # 熵、MI、Qabf、SSIM、PSNR
def eval_seg(gt_mask, pred_mask) -> Dict[str, float]:    # mIoU、Dice、Precision、Recall
```

### 5. 报告生成 (report.py)

- **功能**：生成可视化对比、表格、PDF/HTML报告
- **接口**：
```python
def make_report(run_id: str, figures: List[np.ndarray], tables: Dict[str, Any], output_path: str) -> str
```

## 🔧 API接口

### 主要端点

- `GET /` - 系统首页
- `GET /health` - 健康检查
- `GET /tools` - 获取可用工具列表
- `POST /process` - 图像处理接口
- `GET /report/{run_id}` - 获取处理报告
- `GET /status/{run_id}` - 获取任务状态

### 使用示例

```python
import requests
import base64

# 上传图像并处理
with open('ir_image.jpg', 'rb') as f:
    ir_data = f.read()

with open('vis_image.jpg', 'rb') as f:
    vis_data = f.read()

files = {
    'ir_image': ('ir.jpg', ir_data, 'image/jpeg'),
    'vis_image': ('vis.jpg', vis_data, 'image/jpeg')
}

data = {
    'task_type': 'full_pipeline',
    'config': '{"fusion": {"fusion_method": "weighted_average"}}'
}

response = requests.post('http://localhost:8000/process', files=files, data=data)
result = response.json()
print(result)
```

## ⚙️ 配置说明

系统配置位于 `configs/default.yaml`，主要配置项包括：

- **系统配置**：日志级别、超时时间等
- **预处理配置**：重采样方法、增强参数等
- **融合配置**：融合方法、权重参数等
- **分割配置**：分割方法、阈值参数等
- **评测配置**：需要计算的指标
- **报告配置**：输出格式、可视化参数等
- **API配置**：服务器地址、端口等

## 🧪 测试

### 运行单元测试

```bash
pytest tests/
```

### 运行集成测试

```bash
python main.py --mode test
```

## 📊 性能指标

### 融合质量指标

- **熵 (Entropy)**：衡量图像信息量
- **互信息 (MI)**：衡量图像间信息相关性
- **Qabf指标**：衡量边缘保持能力
- **SSIM**：结构相似性指数
- **PSNR**：峰值信噪比

### 分割质量指标

- **mIoU**：平均交并比
- **Dice系数**：分割准确性
- **精确率 (Precision)**：预测为正例中实际为正例的比例
- **召回率 (Recall)**：实际正例中被预测为正例的比例

## 🔄 工作流程

1. **任务解析**：解析用户输入，确定任务类型
2. **步骤规划**：设计最优的处理流程
3. **工具调度**：调用相应的工具函数
4. **异常处理**：处理执行过程中的异常
5. **质量保证**：确保输出结果的质量
6. **报告生成**：生成处理报告

## 🛠️ 扩展开发

### 添加新的融合方法

1. 在 `tools/fuse.py` 中添加新的融合函数
2. 在 `FusionMethod` 枚举中添加新方法
3. 更新配置文件中的相关参数

### 添加新的分割方法

1. 在 `tools/segment.py` 中添加新的分割函数
2. 在 `SegmentationMethod` 枚举中添加新方法
3. 更新配置文件中的相关参数

### 添加新的评测指标

1. 在 `tools/metrics.py` 中添加新的指标计算函数
2. 更新 `MetricsConfig` 配置类
3. 在配置文件中添加相关参数

## 📝 日志

系统日志位于 `logs/irvis_agent.log`，包含：

- 系统启动和关闭信息
- 工具执行状态
- 错误和异常信息
- 性能统计信息

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

**注意**：这是一个研究项目，主要用于学术研究和实验。在生产环境中使用前，请确保充分测试和验证。

