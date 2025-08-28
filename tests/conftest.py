"""
红外可见光Agent系统 - pytest配置文件
提供共享的测试夹具和配置
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def sample_images():
    """提供样例图像数据"""
    return {
        "ir_small": np.random.rand(64, 64, 3).astype(np.float32),
        "vis_small": np.random.rand(64, 64, 3).astype(np.float32),
        "ir_medium": np.random.rand(256, 256, 3).astype(np.float32),
        "vis_medium": np.random.rand(256, 256, 3).astype(np.float32),
        "ir_large": np.random.rand(512, 512, 3).astype(np.float32),
        "vis_large": np.random.rand(512, 512, 3).astype(np.float32),
        "ir_gray": np.random.rand(256, 256).astype(np.float32),
        "vis_gray": np.random.rand(256, 256).astype(np.float32)
    }

@pytest.fixture
def sample_masks():
    """提供样例掩码数据"""
    return {
        "gt_mask": np.random.rand(256, 256) > 0.5,
        "pred_mask": np.random.rand(256, 256) > 0.5,
        "perfect_mask": np.random.rand(256, 256) > 0.5,
        "empty_mask": np.zeros((256, 256), dtype=bool),
        "full_mask": np.ones((256, 256), dtype=bool)
    }

@pytest.fixture
def sample_configs():
    """提供样例配置数据"""
    return {
        "preprocess": {
            "target_size": (512, 512),
            "enhancement": True,
            "normalization": True,
            "registration": True
        },
        "fusion": {
            "fusion_method": "weighted_average",
            "ir_weight": 0.6,
            "vis_weight": 0.4,
            "enhance_contrast": True
        },
        "segmentation": {
            "segmentation_method": "threshold",
            "num_classes": 2,
            "morphology_ops": True,
            "post_process": True
        },
        "metrics": {
            "compute_entropy": True,
            "compute_mi": True,
            "compute_qabf": True,
            "compute_ssim": True,
            "compute_psnr": True
        }
    }

@pytest.fixture
def temp_dir():
    """提供临时目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_metrics():
    """提供样例指标数据"""
    return {
        "fusion": {
            "entropy": 7.234,
            "mutual_information": 2.456,
            "qabf": 0.789,
            "ssim": 0.823,
            "psnr": 28.456
        },
        "segmentation": {
            "miou": 0.756,
            "dice": 0.823,
            "precision": 0.789,
            "recall": 0.812,
            "f1_score": 0.800
        }
    }

@pytest.fixture
def sample_execution_steps():
    """提供样例执行步骤数据"""
    return [
        {
            "name": "preprocess",
            "status": "success",
            "duration": 1.23,
            "details": "图像预处理完成"
        },
        {
            "name": "fuse",
            "status": "success",
            "duration": 2.45,
            "details": "图像融合完成"
        },
        {
            "name": "segment",
            "status": "success",
            "duration": 3.67,
            "details": "图像分割完成"
        },
        {
            "name": "metrics",
            "status": "success",
            "duration": 0.89,
            "details": "质量评估完成"
        },
        {
            "name": "report",
            "status": "success",
            "duration": 1.12,
            "details": "报告生成完成"
        }
    ]

@pytest.fixture
def edge_case_images():
    """提供边缘情况图像数据"""
    return {
        "constant_image": np.ones((256, 256, 3), dtype=np.float32),
        "zero_image": np.zeros((256, 256, 3), dtype=np.float32),
        "random_image": np.random.rand(256, 256, 3).astype(np.float32),
        "high_contrast": np.zeros((256, 256, 3), dtype=np.float32),
        "noisy_image": np.random.rand(256, 256, 3).astype(np.float32) + np.random.normal(0, 0.1, (256, 256, 3))
    }

@pytest.fixture
def test_data_path():
    """提供测试数据路径"""
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir

# 配置pytest
def pytest_configure(config):
    """配置pytest"""
    # 添加自定义标记
    config.addinivalue_line("markers", "slow: 标记为慢速测试")
    config.addinivalue_line("markers", "integration: 标记为集成测试")
    config.addinivalue_line("markers", "unit: 标记为单元测试")
    config.addinivalue_line("markers", "preprocess: 预处理相关测试")
    config.addinivalue_line("markers", "fusion: 融合相关测试")
    config.addinivalue_line("markers", "segmentation: 分割相关测试")
    config.addinivalue_line("markers", "metrics: 评测相关测试")
    config.addinivalue_line("markers", "report: 报告相关测试")

def pytest_collection_modifyitems(config, items):
    """修改测试项"""
    for item in items:
        # 根据文件名自动添加标记
        if "test_preprocess" in item.nodeid:
            item.add_marker(pytest.mark.preprocess)
        elif "test_fusion" in item.nodeid:
            item.add_marker(pytest.mark.fusion)
        elif "test_segment" in item.nodeid:
            item.add_marker(pytest.mark.segmentation)
        elif "test_metrics" in item.nodeid:
            item.add_marker(pytest.mark.metrics)
        elif "test_report" in item.nodeid:
            item.add_marker(pytest.mark.report)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

# 测试会话级别的设置
@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """设置测试会话"""
    print("开始测试会话...")
    
    # 创建必要的测试目录
    test_dirs = [
        "test_data",
        "test_outputs",
        "test_logs"
    ]
    
    for dir_name in test_dirs:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
    
    yield
    
    print("测试会话结束...")

# 测试函数级别的设置
@pytest.fixture(autouse=True)
def setup_test_function():
    """设置每个测试函数"""
    # 在每个测试前设置随机种子以确保可重复性
    np.random.seed(42)
    
    yield
    
    # 在每个测试后清理（如果需要）
    pass
