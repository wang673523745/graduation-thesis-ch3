"""
红外可见光Agent系统 - 图像融合工具
支持单任务融合和联合任务融合
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ImageFusion:
    """图像融合器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_config = {
            "fusion_method": "weighted_average",  # weighted_average, laplacian, dwt, pca
            "ir_weight": 0.5,                    # 红外图像权重
            "vis_weight": 0.5,                   # 可见光图像权重
            "enhance_contrast": True,            # 是否增强对比度
            "preserve_details": True,            # 是否保持细节
            "joint_mode": False                  # 是否为联合模式
        }
        self.config = {**self.default_config, **self.config}
    
    def weighted_average_fusion(self, ir_img: np.ndarray, vis_img: np.ndarray) -> np.ndarray:
        """加权平均融合"""
        ir_weight = self.config["ir_weight"]
        vis_weight = self.config["vis_weight"]
        
        # 确保权重和为1
        total_weight = ir_weight + vis_weight
        ir_weight /= total_weight
        vis_weight /= total_weight
        
        fused = ir_weight * ir_img + vis_weight * vis_img
        return fused
    
    def laplacian_fusion(self, ir_img: np.ndarray, vis_img: np.ndarray) -> np.ndarray:
        """拉普拉斯金字塔融合"""
        # 构建拉普拉斯金字塔
        def build_laplacian_pyramid(img, levels=4):
            pyramid = [img.astype(np.float32)]
            for i in range(levels - 1):
                img = cv2.pyrDown(img)
                pyramid.append(img)
            
            laplacian_pyramid = []
            for i in range(levels - 1):
                size = (pyramid[i].shape[1], pyramid[i].shape[0])
                expanded = cv2.pyrUp(pyramid[i + 1], dstsize=size)
                laplacian = pyramid[i] - expanded
                laplacian_pyramid.append(laplacian)
            laplacian_pyramid.append(pyramid[-1])
            
            return laplacian_pyramid
        
        # 融合拉普拉斯金字塔
        def fuse_laplacian_pyramids(lap1, lap2):
            fused_pyramid = []
            for l1, l2 in zip(lap1, lap2):
                # 使用绝对值最大值作为融合策略
                fused = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
                fused_pyramid.append(fused)
            return fused_pyramid
        
        # 重建图像
        def reconstruct_from_laplacian_pyramid(laplacian_pyramid):
            result = laplacian_pyramid[-1]
            for i in range(len(laplacian_pyramid) - 2, -1, -1):
                size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
                result = cv2.pyrUp(result, dstsize=size)
                result = result + laplacian_pyramid[i]
            return result
        
        # 构建拉普拉斯金字塔
        lap_ir = build_laplacian_pyramid(ir_img)
        lap_vis = build_laplacian_pyramid(vis_img)
        
        # 融合金字塔
        fused_pyramid = fuse_laplacian_pyramids(lap_ir, lap_vis)
        
        # 重建融合图像
        fused = reconstruct_from_laplacian_pyramid(fused_pyramid)
        
        # 确保值在合理范围内
        fused = np.clip(fused, 0, 1)
        
        return fused
    
    def dwt_fusion(self, ir_img: np.ndarray, vis_img: np.ndarray) -> np.ndarray:
        """离散小波变换融合"""
        # 简化的DWT融合实现
        # 在实际应用中，可以使用PyWavelets库
        
        # 这里使用简化的方法：分块处理
        block_size = 8
        h, w = ir_img.shape[:2]
        fused = np.zeros_like(ir_img)
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # 提取块
                ir_block = ir_img[i:i+block_size, j:j+block_size]
                vis_block = vis_img[i:i+block_size, j:j+block_size]
                
                # 计算块的方差作为融合权重
                ir_var = np.var(ir_block)
                vis_var = np.var(vis_block)
                
                total_var = ir_var + vis_var
                if total_var > 0:
                    ir_weight = ir_var / total_var
                    vis_weight = vis_var / total_var
                else:
                    ir_weight = vis_weight = 0.5
                
                # 融合块
                fused_block = ir_weight * ir_block + vis_weight * vis_block
                fused[i:i+block_size, j:j+block_size] = fused_block
        
        return fused
    
    def pca_fusion(self, ir_img: np.ndarray, vis_img: np.ndarray) -> np.ndarray:
        """主成分分析融合"""
        # 将图像重塑为向量
        h, w = ir_img.shape[:2]
        if len(ir_img.shape) == 3:
            ir_flat = ir_img.reshape(h * w, 3)
            vis_flat = vis_img.reshape(h * w, 3)
        else:
            ir_flat = ir_img.reshape(h * w, 1)
            vis_flat = vis_img.reshape(h * w, 1)
        
        # 组合数据
        combined = np.hstack([ir_flat, vis_flat])
        
        # 计算协方差矩阵
        cov_matrix = np.cov(combined.T)
        
        # 计算特征值和特征向量
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 使用最大特征值对应的特征向量作为权重
        max_eigenval_idx = np.argmax(eigenvals)
        weights = eigenvecs[:, max_eigenval_idx]
        
        # 归一化权重
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        
        # 应用权重融合
        if len(ir_img.shape) == 3:
            fused_flat = (weights[0] * ir_flat + weights[1] * vis_flat)
            fused = fused_flat.reshape(h, w, 3)
        else:
            fused_flat = (weights[0] * ir_flat + weights[1] * vis_flat)
            fused = fused_flat.reshape(h, w)
        
        return fused
    
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """增强对比度"""
        if not self.config["enhance_contrast"]:
            return img
        
        # 使用CLAHE（对比度受限的自适应直方图均衡化）
        if len(img.shape) == 3:
            # 彩色图像
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
        
        return enhanced
    
    def fuse_single(self, ir_img: np.ndarray, vis_img: np.ndarray) -> np.ndarray:
        """单任务融合"""
        method = self.config["fusion_method"]
        
        if method == "weighted_average":
            fused = self.weighted_average_fusion(ir_img, vis_img)
        elif method == "laplacian":
            fused = self.laplacian_fusion(ir_img, vis_img)
        elif method == "dwt":
            fused = self.dwt_fusion(ir_img, vis_img)
        elif method == "pca":
            fused = self.pca_fusion(ir_img, vis_img)
        else:
            logger.warning(f"未知的融合方法: {method}，使用加权平均")
            fused = self.weighted_average_fusion(ir_img, vis_img)
        
        # 增强对比度
        if self.config["enhance_contrast"]:
            fused = self.enhance_contrast(fused)
        
        return fused
    
    def fuse_joint(self, ir_img: np.ndarray, vis_img: np.ndarray, 
                   segmentation_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """联合任务融合"""
        # 基础融合
        fused = self.fuse_single(ir_img, vis_img)
        
        # 如果有分割掩码，进行区域特定的融合
        aux_info = {}
        if segmentation_mask is not None:
            # 对分割区域进行特殊处理
            mask_normalized = segmentation_mask.astype(np.float32)
            if len(mask_normalized.shape) == 3:
                mask_normalized = mask_normalized[:, :, 0]
            
            # 在分割区域增强红外信息
            enhanced_fused = fused.copy()
            enhanced_fused = enhanced_fused * (1 + 0.2 * mask_normalized)
            enhanced_fused = np.clip(enhanced_fused, 0, 1)
            
            aux_info["segmentation_enhanced"] = enhanced_fused
            aux_info["segmentation_mask"] = segmentation_mask
        
        return {
            "fused": fused,
            "aux": aux_info
        }

def fuse_single(ir: np.ndarray, vis: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """单任务融合接口"""
    fusion = ImageFusion(cfg)
    return fusion.fuse_single(ir, vis)

def fuse_joint(ir: np.ndarray, vis: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """联合任务融合接口"""
    fusion = ImageFusion(cfg)
    segmentation_mask = cfg.get("segmentation_mask")
    return fusion.fuse_joint(ir, vis, segmentation_mask)

def demo_fusion():
    """演示融合功能"""
    print("=== 图像融合工具演示 ===")
    
    # 创建模拟图像
    ir_img = np.random.rand(256, 256, 3).astype(np.float32)
    vis_img = np.random.rand(256, 256, 3).astype(np.float32)
    
    print(f"红外图像形状: {ir_img.shape}")
    print(f"可见光图像形状: {vis_img.shape}")
    
    # 测试单任务融合
    config_single = {
        "fusion_method": "weighted_average",
        "ir_weight": 0.6,
        "vis_weight": 0.4,
        "enhance_contrast": True
    }
    
    try:
        fused_single = fuse_single(ir_img, vis_img, config_single)
        print(f"单任务融合结果形状: {fused_single.shape}")
        print(f"融合结果范围: [{fused_single.min():.3f}, {fused_single.max():.3f}]")
        
        # 测试联合任务融合
        config_joint = {
            "fusion_method": "laplacian",
            "joint_mode": True,
            "enhance_contrast": True
        }
        
        # 创建模拟分割掩码
        segmentation_mask = np.random.rand(256, 256, 1) > 0.5
        config_joint["segmentation_mask"] = segmentation_mask
        
        fused_joint = fuse_joint(ir_img, vis_img, config_joint)
        print(f"联合任务融合结果形状: {fused_joint['fused'].shape}")
        print(f"辅助信息键: {list(fused_joint['aux'].keys())}")
        
    except Exception as e:
        print(f"融合失败: {str(e)}")

if __name__ == "__main__":
    demo_fusion()
