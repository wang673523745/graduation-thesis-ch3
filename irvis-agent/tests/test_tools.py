"""
红外可见光Agent系统 - 工具测试
测试各个工具模块的功能
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.preprocess import preprocess, ImagePreprocessor
from tools.fuse import fuse_single, fuse_joint, ImageFusion
from tools.segment import segment, ImageSegmenter
from tools.metrics import eval_fusion, eval_seg, ImageMetrics
from tools.report import make_report, ReportGenerator

class TestPreprocess:
    """测试预处理工具"""
    
    def setup_method(self):
        """设置测试环境"""
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.config = {
            "target_size": (512, 512),
            "enhancement": True,
            "normalization": True
        }
    
    def test_preprocess_basic(self):
        """测试基本预处理功能"""
        result = preprocess(self.ir_img, self.vis_img, self.config)
        
        assert "ir" in result
        assert "vis" in result
        assert "debug_info" in result
        assert result["ir"].shape == (512, 512, 3)
        assert result["vis"].shape == (512, 512, 3)
    
    def test_preprocess_without_enhancement(self):
        """测试不进行增强的预处理"""
        config = self.config.copy()
        config["enhancement"] = False
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert result["debug_info"]["enhancement_applied"] == False
    
    def test_preprocess_without_normalization(self):
        """测试不进行归一化的预处理"""
        config = self.config.copy()
        config["normalization"] = False
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert result["debug_info"]["normalization_applied"] == False

class TestFusion:
    """测试融合工具"""
    
    def setup_method(self):
        """设置测试环境"""
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.config = {
            "fusion_method": "weighted_average",
            "ir_weight": 0.6,
            "vis_weight": 0.4
        }
    
    def test_fuse_single_weighted_average(self):
        """测试加权平均融合"""
        fused = fuse_single(self.ir_img, self.vis_img, self.config)
        
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_fuse_single_laplacian(self):
        """测试拉普拉斯金字塔融合"""
        config = {"fusion_method": "laplacian"}
        fused = fuse_single(self.ir_img, self.vis_img, config)
        
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_fuse_joint(self):
        """测试联合任务融合"""
        segmentation_mask = np.random.rand(256, 256, 1) > 0.5
        config = {
            "fusion_method": "weighted_average",
            "segmentation_mask": segmentation_mask
        }
        
        result = fuse_joint(self.ir_img, self.vis_img, config)
        
        assert "fused" in result
        assert "aux" in result
        assert result["fused"].shape == self.ir_img.shape

class TestSegmentation:
    """测试分割工具"""
    
    def setup_method(self):
        """设置测试环境"""
        self.img = np.random.rand(256, 256, 3).astype(np.float32)
        self.config = {
            "segmentation_method": "threshold",
            "num_classes": 2
        }
    
    def test_segment_threshold(self):
        """测试阈值分割"""
        mask = segment(self.img, self.config)
        
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_segment_watershed(self):
        """测试分水岭分割"""
        config = {"segmentation_method": "watershed"}
        mask = segment(self.img, config)
        
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_segment_kmeans(self):
        """测试K-means分割"""
        config = {"segmentation_method": "kmeans", "num_classes": 3}
        mask = segment(self.img, config)
        
        assert mask.shape == (256, 256)
        assert len(np.unique(mask)) <= 3

class TestMetrics:
    """测试评测工具"""
    
    def setup_method(self):
        """设置测试环境"""
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.fused_img = (self.ir_img + self.vis_img) / 2
        self.gt_mask = np.random.rand(256, 256) > 0.5
        self.pred_mask = np.random.rand(256, 256) > 0.5
    
    def test_eval_fusion(self):
        """测试融合质量评估"""
        metrics = eval_fusion(self.ir_img, self.vis_img, self.fused_img)
        
        assert "entropy" in metrics
        assert "mutual_information" in metrics
        assert "qabf" in metrics
        assert "ssim" in metrics
        assert "psnr" in metrics
        
        # 检查指标值是否在合理范围内
        assert 0 <= metrics["entropy"] <= 10
        assert 0 <= metrics["qabf"] <= 1
        assert 0 <= metrics["ssim"] <= 1
        assert metrics["psnr"] >= 0
    
    def test_eval_seg(self):
        """测试分割质量评估"""
        metrics = eval_seg(self.gt_mask, self.pred_mask)
        
        assert "miou" in metrics
        assert "dice" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # 检查指标值是否在合理范围内
        assert 0 <= metrics["miou"] <= 1
        assert 0 <= metrics["dice"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

class TestReport:
    """测试报告生成工具"""
    
    def setup_method(self):
        """设置测试环境"""
        self.run_id = "test_001"
        self.figures = [
            np.random.rand(256, 256, 3),  # 红外图像
            np.random.rand(256, 256, 3),  # 可见光图像
            np.random.rand(256, 256, 3),  # 融合图像
            np.random.rand(256, 256)      # 分割掩码
        ]
        self.tables = {
            "metrics": {
                "fusion": {
                    "entropy": 7.234,
                    "qabf": 0.789,
                    "ssim": 0.823
                }
            }
        }
        self.task_description = "测试处理任务"
    
    def test_make_report_html(self):
        """测试HTML报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_report")
            
            report_path = make_report(
                self.run_id,
                self.figures,
                self.tables,
                output_path,
                self.task_description
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_make_report_json(self):
        """测试JSON报告生成"""
        generator = ReportGenerator({"output_format": "json"})
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_report")
            
            report_path = generator.generate_json_report(
                self.run_id,
                self.task_description,
                {"ir": self.figures[0], "vis": self.figures[1]},
                self.tables,
                [],
                output_path
            )
            
            assert report_path.endswith('.json')
            assert os.path.exists(report_path)

class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整处理流程"""
        # 创建测试图像
        ir_img = np.random.rand(128, 128, 3).astype(np.float32)
        vis_img = np.random.rand(128, 128, 3).astype(np.float32)
        
        # 1. 预处理
        preprocess_result = preprocess(ir_img, vis_img, {})
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        # 2. 融合
        fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": "weighted_average"})
        
        # 3. 分割
        segmentation_mask = segment(fused_img, {"segmentation_method": "threshold"})
        
        # 4. 评测
        fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
        
        # 5. 生成报告
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "integration_test")
            
            report_path = make_report(
                "integration_test",
                [ir_processed, vis_processed, fused_img, segmentation_mask],
                {"metrics": {"fusion": fusion_metrics}},
                output_path,
                "集成测试"
            )
            
            assert os.path.exists(report_path)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空图像
        with pytest.raises(Exception):
            preprocess(np.array([]), np.array([]), {})
        
        # 测试不匹配的图像尺寸
        ir_img = np.random.rand(100, 100, 3).astype(np.float32)
        vis_img = np.random.rand(200, 200, 3).astype(np.float32)
        
        # 这应该能正常处理（会进行尺寸调整）
        result = preprocess(ir_img, vis_img, {})
        assert result["ir"].shape == result["vis"].shape

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
