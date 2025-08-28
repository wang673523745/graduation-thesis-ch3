#!/usr/bin/env python3
"""
红外可见光Agent系统 - 快速启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    print("🚀 红外可见光Agent系统")
    print("=" * 50)
    
    # 检查依赖
    try:
        import numpy as np
        import cv2
        import fastapi
        print("✅ 核心依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖缺失: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 创建必要的目录
    dirs_to_create = [
        "data/outputs",
        "data/samples/ir_images",
        "data/samples/vis_images",
        "data/samples/ground_truth",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ 目录结构检查完成")
    
    # 导入主程序
    try:
        from main import main as main_func
        print("✅ 系统模块加载完成")
    except ImportError as e:
        print(f"❌ 模块加载失败: {e}")
        return
    
    # 启动系统
    print("\n🎯 启动系统...")
    print("可用模式:")
    print("  - api: 启动API服务器")
    print("  - demo: 运行演示")
    print("  - test: 运行测试")
    print("\n默认启动API服务器模式...")
    
    # 设置默认参数
    sys.argv = ["main.py", "--mode", "api"]
    
    # 运行主程序
    main_func()

if __name__ == "__main__":
    main()
