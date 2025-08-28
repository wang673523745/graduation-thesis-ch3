"""
红外可见光Agent系统 - 分割工具测试
测试图像分割模块的功能
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

from tools.segment import segment, ImageSegmenter

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
    
    def test_segment_grabcut(self):
        """测试GrabCut分割"""
        config = {"segmentation_method": "grabcut"}
        mask = segment(self.img, config)
        
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_segment_contour(self):
        """测试轮廓检测分割"""
        config = {"segmentation_method": "contour"}
        mask = segment(self.img, config)
        
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_segment_unknown_method(self):
        """测试未知分割方法"""
        config = {"segmentation_method": "unknown_method"}
        mask = segment(self.img, config)
        
        # 应该回退到阈值分割
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_segment_different_num_classes(self):
        """测试不同类别数的分割"""
        config = {"segmentation_method": "kmeans", "num_classes": 5}
        mask = segment(self.img, config)
        
        assert mask.shape == (256, 256)
        assert len(np.unique(mask)) <= 5
    
    def test_segment_grayscale_image(self):
        """测试灰度图像分割"""
        gray_img = np.random.rand(256, 256).astype(np.float32)
        mask = segment(gray_img, self.config)
        
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_segment_with_morphology(self):
        """测试带形态学操作的分割"""
        config = self.config.copy()
        config["morphology_ops"] = True
        
        mask = segment(self.img, config)
        assert mask.shape == (256, 256)
    
    def test_segment_without_morphology(self):
        """测试不带形态学操作的分割"""
        config = self.config.copy()
        config["morphology_ops"] = False
        
        mask = segment(self.img, config)
        assert mask.shape == (256, 256)
    
    def test_segment_with_post_process(self):
        """测试带后处理的分割"""
        config = self.config.copy()
        config["post_process"] = True
        
        mask = segment(self.img, config)
        assert mask.shape == (256, 256)
    
    def test_segment_without_post_process(self):
        """测试不带后处理的分割"""
        config = self.config.copy()
        config["post_process"] = False
        
        mask = segment(self.img, config)
        assert mask.shape == (256, 256)
    
    def test_segment_empty_image(self):
        """测试空图像分割"""
        with pytest.raises(Exception):
            segment(np.array([]), self.config)
    
    def test_segment_none_image(self):
        """测试None图像分割"""
        with pytest.raises(Exception):
            segment(None, self.config)

class TestImageSegmenter:
    """测试ImageSegmenter类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.segmenter = ImageSegmenter()
        self.img = np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_threshold_segmentation(self):
        """测试阈值分割方法"""
        mask = self.segmenter.threshold_segmentation(self.img)
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_watershed_segmentation(self):
        """测试分水岭分割方法"""
        mask = self.segmenter.watershed_segmentation(self.img)
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_kmeans_segmentation(self):
        """测试K-means分割方法"""
        mask = self.segmenter.kmeans_segmentation(self.img)
        assert mask.shape == (256, 256)
        assert len(np.unique(mask)) <= 2  # 默认2类
    
    def test_grabcut_segmentation(self):
        """测试GrabCut分割方法"""
        mask = self.segmenter.grabcut_segmentation(self.img)
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_contour_segmentation(self):
        """测试轮廓检测分割方法"""
        mask = self.segmenter.contour_segmentation(self.img)
        assert mask.shape == (256, 256)
        assert np.all(np.unique(mask) == [0, 1])
    
    def test_apply_morphology(self):
        """测试形态学操作"""
        mask = np.random.rand(256, 256) > 0.5
        processed_mask = self.segmenter.apply_morphology(mask)
        assert processed_mask.shape == mask.shape
    
    def test_post_process(self):
        """测试后处理"""
        mask = np.random.rand(256, 256) > 0.5
        processed_mask = self.segmenter.post_process(mask)
        assert processed_mask.shape == mask.shape
    
    def test_calculate_confidence(self):
        """测试置信度计算"""
        mask = np.random.rand(256, 256) > 0.5
        confidence = self.segmenter.calculate_confidence(self.img, mask)
        assert 0 <= confidence <= 1
    
    def test_segment_method(self):
        """测试分割方法"""
        result = self.segmenter.segment(self.img, "all")
        assert "mask" in result
        assert "confidence" in result
        assert "class_info" in result
        assert result["mask"].shape == (256, 256)

class TestSegmentationEdgeCases:
    """测试分割边缘情况"""
    
    def setup_method(self):
        """设置测试环境"""
        self.img = np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_segment_constant_image(self):
        """测试常数图像分割"""
        constant_img = np.ones((256, 256, 3), dtype=np.float32)
        mask = segment(constant_img, {"segmentation_method": "threshold"})
        assert mask.shape == (256, 256)
    
    def test_segment_zero_image(self):
        """测试零图像分割"""
        zero_img = np.zeros((256, 256, 3), dtype=np.float32)
        mask = segment(zero_img, {"segmentation_method": "threshold"})
        assert mask.shape == (256, 256)
    
    def test_segment_small_image(self):
        """测试小图像分割"""
        small_img = np.random.rand(32, 32, 3).astype(np.float32)
        mask = segment(small_img, {"segmentation_method": "threshold"})
        assert mask.shape == (32, 32)
    
    def test_segment_large_image(self):
        """测试大图像分割"""
        large_img = np.random.rand(512, 512, 3).astype(np.float32)
        mask = segment(large_img, {"segmentation_method": "threshold"})
        assert mask.shape == (512, 512)
    
    def test_segment_high_contrast_image(self):
        """测试高对比度图像分割"""
        # 创建高对比度图像
        high_contrast = np.zeros((256, 256, 3), dtype=np.float32)
        high_contrast[50:200, 50:200] = 1.0
        
        mask = segment(high_contrast, {"segmentation_method": "threshold"})
        assert mask.shape == (256, 256)
    
    def test_segment_noisy_image(self):
        """测试噪声图像分割"""
        # 添加噪声
        noisy_img = self.img + np.random.normal(0, 0.1, self.img.shape)
        noisy_img = np.clip(noisy_img, 0, 1)
        
        mask = segment(noisy_img, {"segmentation_method": "threshold"})
        assert mask.shape == (256, 256)

class TestSegmentationConfigurations:
    """测试分割配置"""
    
    def setup_method(self):
        """设置测试环境"""
        self.img = np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_threshold_configurations(self):
        """测试阈值分割配置"""
        configs = [
            {"segmentation_method": "threshold", "method": "otsu"},
            {"segmentation_method": "threshold", "method": "adaptive"},
            {"segmentation_method": "threshold", "manual_threshold": 128}
        ]
        
        for config in configs:
            mask = segment(self.img, config)
            assert mask.shape == (256, 256)
    
    def test_watershed_configurations(self):
        """测试分水岭分割配置"""
        configs = [
            {"segmentation_method": "watershed", "distance_threshold": 0.5},
            {"segmentation_method": "watershed", "distance_threshold": 0.8},
            {"segmentation_method": "watershed", "min_area": 100}
        ]
        
        for config in configs:
            mask = segment(self.img, config)
            assert mask.shape == (256, 256)
    
    def test_kmeans_configurations(self):
        """测试K-means分割配置"""
        configs = [
            {"segmentation_method": "kmeans", "num_classes": 2},
            {"segmentation_method": "kmeans", "num_classes": 3},
            {"segmentation_method": "kmeans", "num_classes": 4}
        ]
        
        for config in configs:
            mask = segment(self.img, config)
            assert mask.shape == (256, 256)

if __name__ == "__main__":
    # 运行分割测试
    pytest.main([__file__, "-v"])
