"""
红外可见光Agent系统 - 任务规划器
负责解析用户任务、规划执行步骤、处理异常和反思重试
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型枚举"""
    FUSION_ONLY = "fusion_only"
    SEGMENTATION_ONLY = "segmentation_only"
    FUSION_AND_SEGMENTATION = "fusion_and_segmentation"
    FULL_PIPELINE = "full_pipeline"

@dataclass
class TaskStep:
    """任务步骤定义"""
    name: str
    params: Dict[str, Any]
    required: bool = True
    fallback_strategy: Optional[str] = None

class TaskPlanner:
    """任务规划器"""
    
    def __init__(self):
        self.step_definitions = {
            "preprocess": {
                "required": True,
                "fallback": "skip_enhancement"
            },
            "fuse": {
                "required": True,
                "fallback": "simple_average"
            },
            "segment": {
                "required": False,
                "fallback": "basic_threshold"
            },
            "metrics": {
                "required": True,
                "fallback": "basic_metrics"
            },
            "report": {
                "required": True,
                "fallback": "simple_report"
            }
        }
    
    def parse_task(self, user_input: str) -> TaskType:
        """解析用户输入，确定任务类型"""
        user_input = user_input.lower()
        
        if "融合" in user_input or "fuse" in user_input:
            if "分割" in user_input or "segment" in user_input:
                return TaskType.FUSION_AND_SEGMENTATION
            else:
                return TaskType.FUSION_ONLY
        elif "分割" in user_input or "segment" in user_input:
            return TaskType.SEGMENTATION_ONLY
        else:
            return TaskType.FULL_PIPELINE
    
    def plan_steps(self, task_type: TaskType, config: Dict[str, Any]) -> List[TaskStep]:
        """根据任务类型规划执行步骤"""
        steps = []
        
        # 预处理总是需要的
        steps.append(TaskStep(
            name="preprocess",
            params={"ir_path": config.get("ir_path"), "vis_path": config.get("vis_path")},
            required=True,
            fallback_strategy="skip_enhancement"
        ))
        
        if task_type in [TaskType.FUSION_ONLY, TaskType.FUSION_AND_SEGMENTATION, TaskType.FULL_PIPELINE]:
            steps.append(TaskStep(
                name="fuse",
                params={"mode": config.get("fuse_mode", "single")},
                required=True,
                fallback_strategy="simple_average"
            ))
        
        if task_type in [TaskType.SEGMENTATION_ONLY, TaskType.FUSION_AND_SEGMENTATION, TaskType.FULL_PIPELINE]:
            steps.append(TaskStep(
                name="segment",
                params={"target": config.get("seg_target", "all")},
                required=False,
                fallback_strategy="basic_threshold"
            ))
        
        # 评测和报告总是需要的
        steps.append(TaskStep(
            name="metrics",
            params={},
            required=True,
            fallback_strategy="basic_metrics"
        ))
        
        steps.append(TaskStep(
            name="report",
            params={},
            required=True,
            fallback_strategy="simple_report"
        ))
        
        return steps
    
    def reflect_and_retry(self, step_name: str, error: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """反思策略：分析错误并决定重试策略"""
        logger.info(f"步骤 {step_name} 失败，错误: {error}")
        
        reflection = {
            "step": step_name,
            "error": error,
            "timestamp": time.time(),
            "retry_strategy": None,
            "should_retry": False
        }
        
        # 根据错误类型决定重试策略
        if "memory" in error.lower() or "gpu" in error.lower():
            reflection["retry_strategy"] = "reduce_batch_size"
            reflection["should_retry"] = True
        elif "file" in error.lower() or "path" in error.lower():
            reflection["retry_strategy"] = "check_paths"
            reflection["should_retry"] = True
        elif "model" in error.lower() or "load" in error.lower():
            reflection["retry_strategy"] = "use_fallback_model"
            reflection["should_retry"] = True
        else:
            reflection["retry_strategy"] = "skip_step"
            reflection["should_retry"] = False
        
        return reflection

def demo_planner():
    """演示任务规划器功能"""
    planner = TaskPlanner()
    
    # 测试任务解析
    test_inputs = [
        "请对红外和可见光图像进行融合",
        "对图像进行分割处理",
        "融合和分割都要做",
        "完整的处理流程"
    ]
    
    for input_text in test_inputs:
        task_type = planner.parse_task(input_text)
        print(f"输入: {input_text}")
        print(f"任务类型: {task_type.value}")
        
        # 规划步骤
        config = {
            "ir_path": "/path/to/ir.jpg",
            "vis_path": "/path/to/vis.jpg",
            "fuse_mode": "single"
        }
        steps = planner.plan_steps(task_type, config)
        print(f"规划步骤: {[step.name for step in steps]}")
        print("-" * 50)

if __name__ == "__main__":
    demo_planner()
