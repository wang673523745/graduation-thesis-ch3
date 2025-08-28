"""
红外可见光Agent系统 - 预处理工具测试
测试图像预处理模块的功能
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
    
    def test_preprocess_different_sizes(self):
        """测试不同尺寸图像的预处理"""
        ir_img = np.random.rand(100, 100, 3).astype(np.float32)
        vis_img = np.random.rand(200, 200, 3).astype(np.float32)
        
        result = preprocess(ir_img, vis_img, self.config)
        assert result["ir"].shape == result["vis"].shape
        assert result["ir"].shape == (512, 512, 3)
    
    def test_preprocess_grayscale(self):
        """测试灰度图像预处理"""
        ir_img = np.random.rand(256, 256).astype(np.float32)
        vis_img = np.random.rand(256, 256).astype(np.float32)
        
        result = preprocess(ir_img, vis_img, self.config)
        assert result["ir"].shape == (512, 512, 3)  # 应该转换为3通道
        assert result["vis"].shape == (512, 512, 3)
    
    def test_preprocess_registration(self):
        """测试图像配准功能"""
        config = self.config.copy()
        config["registration"] = True
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert "warp_matrix" in result
        assert result["debug_info"]["registration_applied"] == True
    
    def test_preprocess_no_registration(self):
        """测试不进行配准的预处理"""
        config = self.config.copy()
        config["registration"] = False
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert result["debug_info"]["registration_applied"] == False
    
    def test_preprocess_histogram_equalization(self):
        """测试直方图均衡化"""
        config = self.config.copy()
        config["histogram_equalization"] = True
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert result["debug_info"]["enhancement_applied"] == True
    
    def test_preprocess_noise_reduction(self):
        """测试降噪功能"""
        config = self.config.copy()
        config["noise_reduction"] = True
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert result["debug_info"]["enhancement_applied"] == True
    
    def test_preprocess_gamma_correction(self):
        """测试伽马校正"""
        config = self.config.copy()
        config["gamma_correction"] = 1.5
        
        result = preprocess(self.ir_img, self.vis_img, config)
        assert result["debug_info"]["enhancement_applied"] == True
    
    def test_preprocess_empty_images(self):
        """测试空图像处理"""
        with pytest.raises(Exception):
            preprocess(np.array([]), np.array([]), self.config)
    
    def test_preprocess_none_images(self):
        """测试None图像处理"""
        with pytest.raises(Exception):
            preprocess(None, None, self.config)
    
    def test_preprocess_invalid_config(self):
        """测试无效配置处理"""
        config = {"invalid_param": "invalid_value"}
        
        # 应该使用默认配置
        result = preprocess(self.ir_img, self.vis_img, config)
        assert "ir" in result
        assert "vis" in result

class TestImagePreprocessor:
    """测试ImagePreprocessor类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.preprocessor = ImagePreprocessor()
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_resize_image(self):
        """测试图像尺寸调整"""
        resized = self.preprocessor.resize_image(self.ir_img, (128, 128))
        assert resized.shape == (128, 128, 3)
    
    def test_enhance_image(self):
        """测试图像增强"""
        enhanced = self.preprocessor.enhance_image(self.ir_img)
        assert enhanced.shape == self.ir_img.shape
    
    def test_normalize_image(self):
        """测试图像归一化"""
        normalized = self.preprocessor.normalize_image(self.ir_img)
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
    
    def test_register_images(self):
        """测试图像配准"""
        ir_registered, vis_registered, warp_matrix = self.preprocessor.register_images(
            self.ir_img, self.vis_img
        )
        assert ir_registered.shape == self.ir_img.shape
        assert vis_registered.shape == self.vis_img.shape

if __name__ == "__main__":
    # 运行预处理测试
    pytest.main([__file__, "-v"])
