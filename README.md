# 红外可见光Agent系统

> 目标：围绕“红外与可见光图像融合与分割”构建一个可扩展的图像 Agent 系统，使其具备**理解—决策—执行—自检**能力，并通过工具化封装你在第一章与第二章的模型，实现**多任务编排、自动评测与报告生成**。

## 0. 预期与章目标

- **章目标**：
  1. 提出一套 **Agent 化** 的系统架构（LLM 调度 + 工具集）。
  2. 将**融合**与**分割**模型封装成可调用的工具，并引入**评测/报告**工具。
  3. 提供可复现的 **快速 Demo → 正式系统** 迭代路线。
  4. 给出至少 **3 条论文级参考方案**（含非视觉），用于对比与扩展。

## 1. 需求与场景设定

- **核心任务**：输入红外与可见光图像对（或多帧序列），**自动**完成：

  - 图像**配准/预处理**
  - 图像**融合**
  - 目标或语义**分割**
  - **质量评测**与**可视化报告**
  - 可选：**目标检测/跟踪/描述**（扩展）

  **典型场景**：夜间安防监控、低光导航、野外搜救等。

  **非功能性需求**：可扩展、可替换模型、稳定可复现、接口清晰、能自动生成实验报告

## 2. 工程化模块划分与目录结构

```
irvis-agent/
├── agent/
│   ├── planner.py            # 任务规划 & 反思策略
│   ├── tool_router.py        # 工具路由与重试
│   └── prompts/              # 系统/工具/自检/报告生成提示词
├── tools/
│   ├── preprocess.py         # 配准、增强、归一化、格式转换
│   ├── fuse.py               # 融合：单任务 & 联合任务两套接口
│   ├── segment.py            # 分割：语义/实例/多头输出
│   ├── detect.py             # 可选：目标检测/跟踪
│   ├── metrics.py            # 熵、MI、Qabf、SSIM、PSNR、mIoU...
│   └── report.py             # 可视化对比、表格、PDF/HTML报告
├── api/
│   ├── server.py             # FastAPI/Flask：统一REST入口
│   └── schemas.py            # Pydantic模型定义
├── ui/                       # 可选Web前端（Gradio/Streamlit/Next.js）
├── configs/
│   ├── default.yaml
│   └── datasets.yaml
├── data/
│   ├── samples/              # Demo样例
│   └── outputs/              # 中间/最终结果
├── tests/                    # 单元/集成测试
└── README.md

```

## 3. 工具列表与接口（建议最小可用集）

### 3.1 预处理工具（preprocess）

- 功能：尺寸对齐、配准（特征/光流/模板）、去噪、直方图均衡化、归一化。
- 接口（示例）：

```
def preprocess(ir_img: np.ndarray, vis_img: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """return {'ir': ir_p, 'vis': vis_p, 'warp': M, 'debug': {...}}"""
```

### 3.2 融合工具（fuse）

- 两套入口（**单任务** vs **联合模型**）：

```
def fuse_single(ir: np.ndarray, vis: np.ndarray, cfg: dict) -> np.ndarray: ...
def fuse_joint(ir: np.ndarray, vis: np.ndarray, cfg: dict) -> Dict[str, np.ndarray]:
    """return {'fused': F, 'aux': {...}}"""
```

### 3.3 分割工具（segment）

```
def segment(img: np.ndarray, cfg: dict) -> np.ndarray:  # 语义分割 mask
```

### 3.4 评测工具（metrics）

```
def eval_fusion(ir, vis, fused) -> Dict[str, float]:     # 熵、MI、Qabf、SSIM、PSNR
def eval_seg(gt_mask, pred_mask) -> Dict[str, float]:    # mIoU、Dice、Precision、Recall
```

### 3.5 可视化与报告（report）

```
def make_report(run_id: str, figures: List[np.ndarray], tables: Dict[str, Any], out
```

## 4. LLM Agent 规划与提示词（Prompts）

### 4.1 系统提示（System Prompt）示例

> 你是图像智能体调度器。根据用户需求，规划 **预处理→融合→分割→评测→报告** 的最短可行工具链。
> 在每一步：
>
> 1. 明确输入/输出；2) 校验前置条件；3) 若失败，说明原因并回退；4) 若需要，做一次自检（反思）后重试。

### 4.2 工具调用规范（Function/Tool Schema）

为每个工具提供 JSON Schema（参数+返回），便于 LLM 自动组装调用。

### 4.3 自检（Reflection）策略

- 规则：若任一指标缺失/格式异常/像素尺寸不一致，则**回到上一步**重做或切换备用策略（如放弃配准增强、采用备用融合算子）

## 5. 关键代码片段

### 5.1 工具统一包装（异常与计时）

```
import time
from functools import wraps

def tool(name):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                out = fn(*args, **kwargs)
                status = "ok"
            except Exception as e:
                out, status = {"error": str(e)}, "fail"
            dt = time.time() - t0
            return {"tool": name, "status": status, "latency": dt, "output": out}
        return wrapper
    return deco
```

### 5.2 Agent 任务规划（伪代码）

```
def plan_and_execute(task, ir_path, vis_path, cfg):
    # 1) 解析任务（融合/分割/评测/报告）
    steps = [
        ("preprocess", {"ir": ir_path, "vis": vis_path}),
        ("fuse", {"mode": cfg["fuse_mode"]}),
        ("segment", {"target": cfg.get("seg_target", "all")}),
        ("metrics", {}),
        ("report", {}),
    ]
    # 2) 顺序执行 + 自检回退
    results = {}
    for name, params in steps:
        res = ROUTER.call(name, params | results)
        if res["status"] != "ok":
            # 简单回退策略示例
            if name == "preprocess":
                params["fallback"] = True
                res = ROUTER.call(name, params | results)
            if res["status"] != "ok":
                break
        results[name] = res["output"]
    return results
```