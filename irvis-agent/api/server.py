"""
红外可见光Agent系统 - FastAPI服务器
提供统一的REST API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import io
import base64
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
import os
from pathlib import Path

# 导入系统模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.planner import TaskPlanner, TaskType
from agent.tool_router import router as tool_router
from tools.preprocess import preprocess
from tools.fuse import fuse_single, fuse_joint
from tools.segment import segment
from tools.metrics import eval_fusion, eval_seg
from tools.report import make_report

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="红外可见光Agent系统",
    description="红外与可见光图像融合与分割的智能体系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
task_planner = TaskPlanner()
output_dir = Path("data/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

def image_to_base64(img: np.ndarray) -> str:
    """将图像转换为base64字符串"""
    if len(img.shape) == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # 确保值在[0, 255]范围内
    img_uint8 = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
    
    _, buffer = cv2.imencode('.png', img_uint8)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def base64_to_image(img_base64: str) -> np.ndarray:
    """将base64字符串转换为图像"""
    # 移除data URL前缀
    if img_base64.startswith('data:image'):
        img_base64 = img_base64.split(',')[1]
    
    img_data = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # 转换为RGB并归一化
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    return img_normalized

@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径，返回简单的HTML页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>红外可见光Agent系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>红外可见光Agent系统</h1>
            <p>欢迎使用红外与可见光图像融合与分割的智能体系统！</p>
            
            <h2>可用API端点：</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> /health - 健康检查
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> /process - 图像处理
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> /docs - API文档
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> /tools - 可用工具列表
            </div>
            
            <p>请访问 <a href="/docs">/docs</a> 查看完整的API文档。</p>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/tools")
async def get_tools():
    """获取可用工具列表"""
    tools = tool_router.list_tools()
    tool_info = {}
    
    for tool_name in tools:
        info = tool_router.get_tool_info(tool_name)
        if info:
            tool_info[tool_name] = {
                "name": info["name"],
                "function": info["function"],
                "doc": info["doc"],
                "config": info["config"]
            }
    
    return {
        "available_tools": tools,
        "tool_details": tool_info
    }

@app.post("/process")
async def process_images(
    ir_image: UploadFile = File(...),
    vis_image: UploadFile = File(...),
    task_type: str = Form("full_pipeline"),
    config: str = Form("{}")
):
    """处理红外和可见光图像"""
    try:
        # 解析配置
        try:
            config_dict = json.loads(config)
        except json.JSONDecodeError:
            config_dict = {}
        
        # 读取图像
        ir_data = await ir_image.read()
        vis_data = await vis_image.read()
        
        ir_array = np.frombuffer(ir_data, dtype=np.uint8)
        vis_array = np.frombuffer(vis_data, dtype=np.uint8)
        
        ir_img = cv2.imdecode(ir_array, cv2.IMREAD_COLOR)
        vis_img = cv2.imdecode(vis_array, cv2.IMREAD_COLOR)
        
        # 转换为RGB并归一化
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 生成任务ID
        run_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 执行处理流程
        results = await execute_processing_pipeline(
            ir_img, vis_img, task_type, config_dict, run_id
        )
        
        return {
            "run_id": run_id,
            "status": "success",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

async def execute_processing_pipeline(
    ir_img: np.ndarray, 
    vis_img: np.ndarray, 
    task_type: str, 
    config: Dict[str, Any],
    run_id: str
) -> Dict[str, Any]:
    """执行处理流程"""
    
    execution_steps = []
    results = {}
    
    try:
        # 1. 预处理
        step_start = datetime.now()
        preprocess_config = config.get("preprocess", {})
        preprocess_result = preprocess(ir_img, vis_img, preprocess_config)
        
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        step_duration = (datetime.now() - step_start).total_seconds()
        execution_steps.append({
            "name": "preprocess",
            "status": "success",
            "duration": step_duration,
            "details": "图像预处理完成"
        })
        
        # 2. 融合
        if task_type in ["fusion_only", "fusion_and_segmentation", "full_pipeline"]:
            step_start = datetime.now()
            fusion_config = config.get("fusion", {"fusion_method": "weighted_average"})
            
            if task_type == "fusion_and_segmentation":
                fused_result = fuse_joint(ir_processed, vis_processed, fusion_config)
                fused_img = fused_result["fused"]
            else:
                fused_img = fuse_single(ir_processed, vis_processed, fusion_config)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            execution_steps.append({
                "name": "fuse",
                "status": "success",
                "duration": step_duration,
                "details": "图像融合完成"
            })
            
            results["fused_image"] = image_to_base64(fused_img)
        
        # 3. 分割
        if task_type in ["segmentation_only", "fusion_and_segmentation", "full_pipeline"]:
            step_start = datetime.now()
            segmentation_config = config.get("segmentation", {"segmentation_method": "threshold"})
            
            input_img = fused_img if "fused_img" in locals() else ir_processed
            segmentation_mask = segment(input_img, segmentation_config)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            execution_steps.append({
                "name": "segment",
                "status": "success",
                "duration": step_duration,
                "details": "图像分割完成"
            })
            
            results["segmentation_mask"] = image_to_base64(segmentation_mask)
        
        # 4. 质量评估
        step_start = datetime.now()
        metrics = {}
        
        if "fused_img" in locals():
            fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
            metrics["fusion"] = fusion_metrics
        
        if "segmentation_mask" in locals():
            # 这里需要真实的分割掩码进行评估
            # 暂时使用模拟数据
            gt_mask = np.random.rand(*segmentation_mask.shape) > 0.5
            seg_metrics = eval_seg(gt_mask, segmentation_mask)
            metrics["segmentation"] = seg_metrics
        
        step_duration = (datetime.now() - step_start).total_seconds()
        execution_steps.append({
            "name": "metrics",
            "status": "success",
            "duration": step_duration,
            "details": "质量评估完成"
        })
        
        results["metrics"] = metrics
        
        # 5. 生成报告
        step_start = datetime.now()
        
        # 准备图像数据
        figures = [ir_processed, vis_processed]
        if "fused_img" in locals():
            figures.append(fused_img)
        if "segmentation_mask" in locals():
            figures.append(segmentation_mask)
        
        tables = {"metrics": metrics}
        
        report_path = make_report(
            run_id, figures, tables, 
            str(output_dir / run_id),
            f"任务类型: {task_type}",
            execution_steps
        )
        
        step_duration = (datetime.now() - step_start).total_seconds()
        execution_steps.append({
            "name": "report",
            "status": "success",
            "duration": step_duration,
            "details": f"报告生成完成: {report_path}"
        })
        
        results["report_path"] = report_path
        results["execution_steps"] = execution_steps
        
        return results
        
    except Exception as e:
        logger.error(f"处理流程失败: {str(e)}")
        execution_steps.append({
            "name": "error",
            "status": "failed",
            "duration": 0,
            "details": f"处理失败: {str(e)}"
        })
        raise

@app.get("/report/{run_id}")
async def get_report(run_id: str):
    """获取处理报告"""
    report_path = output_dir / f"{run_id}.html"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    
    return FileResponse(report_path)

@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """获取任务状态"""
    # 这里可以实现任务状态查询逻辑
    return {
        "run_id": run_id,
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # 注册工具到路由器
    tool_router.register_tool("preprocess", preprocess)
    tool_router.register_tool("fuse_single", fuse_single)
    tool_router.register_tool("fuse_joint", fuse_joint)
    tool_router.register_tool("segment", segment)
    tool_router.register_tool("eval_fusion", eval_fusion)
    tool_router.register_tool("eval_seg", eval_seg)
    
    # 启动服务器
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
