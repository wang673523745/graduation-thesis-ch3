"""
红外可见光Agent系统 - 评测工具测试
测试图像质量评测模块的功能
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

from tools.metrics import eval_fusion, eval_seg, ImageMetrics

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
    
    def test_eval_fusion_grayscale(self):
        """测试灰度图像融合评估"""
        ir_gray = np.random.rand(256, 256).astype(np.float32)
        vis_gray = np.random.rand(256, 256).astype(np.float32)
        fused_gray = (ir_gray + vis_gray) / 2
        
        metrics = eval_fusion(ir_gray, vis_gray, fused_gray)
        
        assert "entropy" in metrics
        assert "mutual_information" in metrics
        assert "qabf" in metrics
        assert "ssim" in metrics
        assert "psnr" in metrics
    
    def test_eval_fusion_identical_images(self):
        """测试相同图像融合评估"""
        metrics = eval_fusion(self.ir_img, self.ir_img, self.ir_img)
        
        assert "entropy" in metrics
        assert "mutual_information" in metrics
        assert "qabf" in metrics
        assert "ssim" in metrics
        assert "psnr" in metrics
    
    def test_eval_seg_perfect_match(self):
        """测试完美匹配的分割评估"""
        perfect_mask = self.gt_mask.copy()
        metrics = eval_seg(self.gt_mask, perfect_mask)
        
        assert metrics["miou"] == 1.0
        assert metrics["dice"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
    
    def test_eval_seg_no_overlap(self):
        """测试无重叠的分割评估"""
        no_overlap_mask = ~self.gt_mask
        metrics = eval_seg(self.gt_mask, no_overlap_mask)
        
        assert metrics["miou"] == 0.0
        assert metrics["dice"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
    
    def test_eval_fusion_empty_images(self):
        """测试空图像融合评估"""
        with pytest.raises(Exception):
            eval_fusion(np.array([]), np.array([]), np.array([]))
    
    def test_eval_seg_empty_masks(self):
        """测试空掩码分割评估"""
        with pytest.raises(Exception):
            eval_seg(np.array([]), np.array([]))

class TestImageMetrics:
    """测试ImageMetrics类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.metrics = ImageMetrics()
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.fused_img = (self.ir_img + self.vis_img) / 2
        self.gt_mask = np.random.rand(256, 256) > 0.5
        self.pred_mask = np.random.rand(256, 256) > 0.5
    
    def test_calculate_entropy(self):
        """测试熵计算"""
        entropy = self.metrics.calculate_entropy(self.fused_img)
        assert 0 <= entropy <= 10
    
    def test_calculate_entropy_grayscale(self):
        """测试灰度图像熵计算"""
        gray_img = np.random.rand(256, 256).astype(np.float32)
        entropy = self.metrics.calculate_entropy(gray_img)
        assert 0 <= entropy <= 10
    
    def test_calculate_mutual_information(self):
        """测试互信息计算"""
        mi = self.metrics.calculate_mutual_information(self.ir_img, self.fused_img)
        assert mi >= 0
    
    def test_calculate_mutual_information_grayscale(self):
        """测试灰度图像互信息计算"""
        gray1 = np.random.rand(256, 256).astype(np.float32)
        gray2 = np.random.rand(256, 256).astype(np.float32)
        mi = self.metrics.calculate_mutual_information(gray1, gray2)
        assert mi >= 0
    
    def test_calculate_qabf(self):
        """测试Qabf指标计算"""
        qabf = self.metrics.calculate_qabf(self.ir_img, self.vis_img, self.fused_img)
        assert 0 <= qabf <= 1
    
    def test_calculate_ssim(self):
        """测试SSIM计算"""
        ssim = self.metrics.calculate_ssim(self.ir_img, self.fused_img)
        assert 0 <= ssim <= 1
    
    def test_calculate_psnr(self):
        """测试PSNR计算"""
        psnr = self.metrics.calculate_psnr(self.ir_img, self.fused_img)
        assert psnr >= 0
    
    def test_calculate_miou(self):
        """测试mIoU计算"""
        miou = self.metrics.calculate_miou(self.gt_mask, self.pred_mask)
        assert 0 <= miou <= 1
    
    def test_calculate_dice(self):
        """测试Dice系数计算"""
        dice = self.metrics.calculate_dice(self.gt_mask, self.pred_mask)
        assert 0 <= dice <= 1
    
    def test_calculate_precision_recall(self):
        """测试精确率召回率计算"""
        precision, recall = self.metrics.calculate_precision_recall(self.gt_mask, self.pred_mask)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_eval_fusion_method(self):
        """测试融合评估方法"""
        metrics = self.metrics.eval_fusion(self.ir_img, self.vis_img, self.fused_img)
        
        assert "entropy" in metrics
        assert "mutual_information" in metrics
        assert "qabf" in metrics
        assert "ssim" in metrics
        assert "psnr" in metrics
    
    def test_eval_segmentation_method(self):
        """测试分割评估方法"""
        metrics = self.metrics.eval_segmentation(self.gt_mask, self.pred_mask)
        
        assert "miou" in metrics
        assert "dice" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

class TestMetricsEdgeCases:
    """测试评测边缘情况"""
    
    def setup_method(self):
        """设置测试环境"""
        self.metrics = ImageMetrics()
    
    def test_entropy_constant_image(self):
        """测试常数图像熵计算"""
        constant_img = np.ones((256, 256, 3), dtype=np.float32)
        entropy = self.metrics.calculate_entropy(constant_img)
        assert entropy == 0  # 常数图像的熵应该为0
    
    def test_entropy_random_image(self):
        """测试随机图像熵计算"""
        random_img = np.random.rand(256, 256, 3).astype(np.float32)
        entropy = self.metrics.calculate_entropy(random_img)
        assert entropy > 0  # 随机图像的熵应该大于0
    
    def test_miou_perfect_match(self):
        """测试完美匹配的mIoU"""
        mask = np.random.rand(256, 256) > 0.5
        miou = self.metrics.calculate_miou(mask, mask)
        assert miou == 1.0
    
    def test_miou_no_overlap(self):
        """测试无重叠的mIoU"""
        mask1 = np.zeros((256, 256), dtype=bool)
        mask1[50:150, 50:150] = True
        
        mask2 = np.zeros((256, 256), dtype=bool)
        mask2[200:256, 200:256] = True
        
        miou = self.metrics.calculate_miou(mask1, mask2)
        assert miou == 0.0
    
    def test_dice_perfect_match(self):
        """测试完美匹配的Dice系数"""
        mask = np.random.rand(256, 256) > 0.5
        dice = self.metrics.calculate_dice(mask, mask)
        assert dice == 1.0
    
    def test_dice_no_overlap(self):
        """测试无重叠的Dice系数"""
        mask1 = np.zeros((256, 256), dtype=bool)
        mask1[50:150, 50:150] = True
        
        mask2 = np.zeros((256, 256), dtype=bool)
        mask2[200:256, 200:256] = True
        
        dice = self.metrics.calculate_dice(mask1, mask2)
        assert dice == 0.0
    
    def test_precision_recall_edge_cases(self):
        """测试精确率召回率边缘情况"""
        # 空预测
        gt_mask = np.random.rand(256, 256) > 0.5
        pred_mask = np.zeros((256, 256), dtype=bool)
        
        precision, recall = self.metrics.calculate_precision_recall(gt_mask, pred_mask)
        assert precision == 0.0
        assert recall == 0.0
        
        # 空真实标签
        gt_mask = np.zeros((256, 256), dtype=bool)
        pred_mask = np.random.rand(256, 256) > 0.5
        
        precision, recall = self.metrics.calculate_precision_recall(gt_mask, pred_mask)
        assert precision == 0.0
        assert recall == 0.0

class TestMetricsConfigurations:
    """测试评测配置"""
    
    def setup_method(self):
        """设置测试环境"""
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.fused_img = (self.ir_img + self.vis_img) / 2
        self.gt_mask = np.random.rand(256, 256) > 0.5
        self.pred_mask = np.random.rand(256, 256) > 0.5
    
    def test_fusion_metrics_config(self):
        """测试融合指标配置"""
        config = {
            "compute_entropy": True,
            "compute_mi": True,
            "compute_qabf": True,
            "compute_ssim": True,
            "compute_psnr": True
        }
        
        metrics = ImageMetrics(config)
        result = metrics.eval_fusion(self.ir_img, self.vis_img, self.fused_img)
        
        assert "entropy" in result
        assert "mutual_information" in result
        assert "qabf" in result
        assert "ssim" in result
        assert "psnr" in result
    
    def test_segmentation_metrics_config(self):
        """测试分割指标配置"""
        config = {
            "compute_miou": True,
            "compute_dice": True,
            "compute_precision_recall": True
        }
        
        metrics = ImageMetrics(config)
        result = metrics.eval_segmentation(self.gt_mask, self.pred_mask)
        
        assert "miou" in result
        assert "dice" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
    
    def test_partial_metrics_config(self):
        """测试部分指标配置"""
        config = {
            "compute_entropy": True,
            "compute_mi": False,
            "compute_qabf": True,
            "compute_ssim": False,
            "compute_psnr": True
        }
        
        metrics = ImageMetrics(config)
        result = metrics.eval_fusion(self.ir_img, self.vis_img, self.fused_img)
        
        assert "entropy" in result
        assert "mutual_information" not in result
        assert "qabf" in result
        assert "ssim" not in result
        assert "psnr" in result

if __name__ == "__main__":
    # 运行评测测试
    pytest.main([__file__, "-v"])
