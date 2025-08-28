"""
红外可见光Agent系统 - 主程序入口
提供命令行接口和系统启动功能
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import yaml
import uvicorn

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.planner import TaskPlanner
from agent.tool_router import router as tool_router
from tools.preprocess import preprocess
from tools.fuse import fuse_single, fuse_joint
from tools.segment import segment
from tools.metrics import eval_fusion, eval_seg
from tools.report import make_report

def setup_logging(config: dict):
    """设置日志配置"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建日志目录
    log_file = log_config.get('file', 'logs/irvis_agent.log')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if log_config.get('console', True) else logging.NullHandler()
        ]
    )

def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "configs" / "default.yaml"
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def register_tools():
    """注册所有工具到路由器"""
    logger = logging.getLogger(__name__)
    
    # 注册工具
    tools_to_register = [
        ("preprocess", preprocess),
        ("fuse_single", fuse_single),
        ("fuse_joint", fuse_joint),
        ("segment", segment),
        ("eval_fusion", eval_fusion),
        ("eval_seg", eval_seg),
    ]
    
    for tool_name, tool_func in tools_to_register:
        tool_router.register_tool(tool_name, tool_func)
        logger.info(f"注册工具: {tool_name}")

def demo_processing():
    """演示处理流程"""
    import numpy as np
    
    logger = logging.getLogger(__name__)
    logger.info("开始演示处理流程...")
    
    # 创建模拟图像
    ir_img = np.random.rand(256, 256, 3).astype(np.float32)
    vis_img = np.random.rand(256, 256, 3).astype(np.float32)
    
    logger.info(f"创建模拟图像: IR {ir_img.shape}, VIS {vis_img.shape}")
    
    try:
        # 1. 预处理
        logger.info("执行预处理...")
        preprocess_result = preprocess(ir_img, vis_img, {})
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        logger.info("预处理完成")
        
        # 2. 融合
        logger.info("执行图像融合...")
        fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": "weighted_average"})
        logger.info("图像融合完成")
        
        # 3. 分割
        logger.info("执行图像分割...")
        segmentation_mask = segment(fused_img, {"segmentation_method": "threshold"})
        logger.info("图像分割完成")
        
        # 4. 质量评估
        logger.info("执行质量评估...")
        fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
        logger.info(f"融合质量指标: {fusion_metrics}")
        
        # 5. 生成报告
        logger.info("生成处理报告...")
        figures = [ir_processed, vis_processed, fused_img, segmentation_mask]
        tables = {"metrics": {"fusion": fusion_metrics}}
        
        report_path = make_report(
            "demo_processing",
            figures, tables,
            "demo_report",
            "演示处理流程"
        )
        logger.info(f"报告已生成: {report_path}")
        
        logger.info("演示处理流程完成！")
        
    except Exception as e:
        logger.error(f"演示处理流程失败: {str(e)}")
        raise

def start_api_server(config: dict):
    """启动API服务器"""
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    reload = api_config.get('reload', True)
    log_level = api_config.get('log_level', 'info')
    
    logger = logging.getLogger(__name__)
    logger.info(f"启动API服务器: {host}:{port}")
    
    # 注册工具
    register_tools()
    
    # 启动服务器
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="红外可见光Agent系统")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode", 
        choices=["api", "demo", "test"],
        default="api",
        help="运行模式"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        config['logging']['level'] = args.log_level
        setup_logging(config)
        
        logger = logging.getLogger(__name__)
        logger.info("红外可见光Agent系统启动")
        logger.info(f"运行模式: {args.mode}")
        logger.info(f"配置文件: {args.config}")
        
        # 根据模式执行相应操作
        if args.mode == "api":
            start_api_server(config)
        elif args.mode == "demo":
            register_tools()
            demo_processing()
        elif args.mode == "test":
            logger.info("运行测试模式...")
            # 这里可以添加测试代码
            register_tools()
            logger.info("工具注册完成")
            logger.info(f"可用工具: {tool_router.list_tools()}")
        
    except KeyboardInterrupt:
        logger.info("系统被用户中断")
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
