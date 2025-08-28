"""
红外可见光Agent系统 - 融合工具测试
测试图像融合模块的功能
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

from tools.fuse import fuse_single, fuse_joint, ImageFusion

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
    
    def test_fuse_single_dwt(self):
        """测试离散小波变换融合"""
        config = {"fusion_method": "dwt"}
        fused = fuse_single(self.ir_img, self.vis_img, config)
        
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_fuse_single_pca(self):
        """测试主成分分析融合"""
        config = {"fusion_method": "pca"}
        fused = fuse_single(self.ir_img, self.vis_img, config)
        
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_fuse_single_unknown_method(self):
        """测试未知融合方法"""
        config = {"fusion_method": "unknown_method"}
        fused = fuse_single(self.ir_img, self.vis_img, config)
        
        # 应该回退到加权平均
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
    
    def test_fuse_joint_without_mask(self):
        """测试联合任务融合（无掩码）"""
        config = {"fusion_method": "weighted_average"}
        
        result = fuse_joint(self.ir_img, self.vis_img, config)
        
        assert "fused" in result
        assert "aux" in result
        assert result["fused"].shape == self.ir_img.shape
        assert len(result["aux"]) == 0
    
    def test_fuse_different_weights(self):
        """测试不同权重的融合"""
        config = {
            "fusion_method": "weighted_average",
            "ir_weight": 0.8,
            "vis_weight": 0.2
        }
        
        fused = fuse_single(self.ir_img, self.vis_img, config)
        assert fused.shape == self.ir_img.shape
    
    def test_fuse_enhance_contrast(self):
        """测试增强对比度的融合"""
        config = {
            "fusion_method": "weighted_average",
            "enhance_contrast": True
        }
        
        fused = fuse_single(self.ir_img, self.vis_img, config)
        assert fused.shape == self.ir_img.shape
    
    def test_fuse_grayscale_images(self):
        """测试灰度图像融合"""
        ir_gray = np.random.rand(256, 256).astype(np.float32)
        vis_gray = np.random.rand(256, 256).astype(np.float32)
        
        fused = fuse_single(ir_gray, vis_gray, self.config)
        assert fused.shape == ir_gray.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_fuse_different_sizes(self):
        """测试不同尺寸图像融合"""
        ir_small = np.random.rand(128, 128, 3).astype(np.float32)
        vis_large = np.random.rand(256, 256, 3).astype(np.float32)
        
        # 应该抛出异常或进行尺寸调整
        with pytest.raises(Exception):
            fuse_single(ir_small, vis_large, self.config)
    
    def test_fuse_empty_images(self):
        """测试空图像融合"""
        with pytest.raises(Exception):
            fuse_single(np.array([]), np.array([]), self.config)
    
    def test_fuse_none_images(self):
        """测试None图像融合"""
        with pytest.raises(Exception):
            fuse_single(None, None, self.config)

class TestImageFusion:
    """测试ImageFusion类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.fusion = ImageFusion()
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_weighted_average_fusion(self):
        """测试加权平均融合方法"""
        fused = self.fusion.weighted_average_fusion(self.ir_img, self.vis_img)
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_laplacian_fusion(self):
        """测试拉普拉斯金字塔融合方法"""
        fused = self.fusion.laplacian_fusion(self.ir_img, self.vis_img)
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_dwt_fusion(self):
        """测试离散小波变换融合方法"""
        fused = self.fusion.dwt_fusion(self.ir_img, self.vis_img)
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_pca_fusion(self):
        """测试主成分分析融合方法"""
        fused = self.fusion.pca_fusion(self.ir_img, self.vis_img)
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_enhance_contrast(self):
        """测试对比度增强"""
        enhanced = self.fusion.enhance_contrast(self.ir_img)
        assert enhanced.shape == self.ir_img.shape
    
    def test_fuse_single_method(self):
        """测试单任务融合方法"""
        fused = self.fusion.fuse_single(self.ir_img, self.vis_img)
        assert fused.shape == self.ir_img.shape
        assert np.all(fused >= 0) and np.all(fused <= 1)
    
    def test_fuse_joint_method(self):
        """测试联合任务融合方法"""
        segmentation_mask = np.random.rand(256, 256, 1) > 0.5
        
        result = self.fusion.fuse_joint(self.ir_img, self.vis_img, segmentation_mask)
        assert "fused" in result
        assert "aux" in result
        assert result["fused"].shape == self.ir_img.shape

class TestFusionEdgeCases:
    """测试融合边缘情况"""
    
    def setup_method(self):
        """设置测试环境"""
        self.ir_img = np.random.rand(256, 256, 3).astype(np.float32)
        self.vis_img = np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_fuse_identical_images(self):
        """测试相同图像融合"""
        fused = fuse_single(self.ir_img, self.ir_img, {"fusion_method": "weighted_average"})
        assert fused.shape == self.ir_img.shape
    
    def test_fuse_zero_images(self):
        """测试零图像融合"""
        zero_img = np.zeros_like(self.ir_img)
        fused = fuse_single(self.ir_img, zero_img, {"fusion_method": "weighted_average"})
        assert fused.shape == self.ir_img.shape
    
    def test_fuse_ones_images(self):
        """测试全一图像融合"""
        ones_img = np.ones_like(self.ir_img)
        fused = fuse_single(self.ir_img, ones_img, {"fusion_method": "weighted_average"})
        assert fused.shape == self.ir_img.shape
    
    def test_fuse_extreme_weights(self):
        """测试极端权重融合"""
        # 权重为0
        config = {"fusion_method": "weighted_average", "ir_weight": 0, "vis_weight": 1}
        fused = fuse_single(self.ir_img, self.vis_img, config)
        assert fused.shape == self.ir_img.shape
        
        # 权重为1
        config = {"fusion_method": "weighted_average", "ir_weight": 1, "vis_weight": 0}
        fused = fuse_single(self.ir_img, self.vis_img, config)
        assert fused.shape == self.ir_img.shape

if __name__ == "__main__":
    # 运行融合测试
    pytest.main([__file__, "-v"])
