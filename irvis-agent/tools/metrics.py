"""
红外可见光Agent系统 - 评测工具
计算熵、MI、Qabf、SSIM、PSNR、mIoU等质量指标
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

logger = logging.getLogger(__name__)

class ImageMetrics:
    """图像质量评测器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_config = {
            "compute_entropy": True,
            "compute_mi": True,
            "compute_qabf": True,
            "compute_ssim": True,
            "compute_psnr": True,
            "compute_miou": True,
            "compute_dice": True,
            "compute_precision_recall": True
        }
        self.config = {**self.default_config, **self.config}
    
    def calculate_entropy(self, img: np.ndarray) -> float:
        """计算图像熵"""
        if len(img.shape) == 3:
            # 彩色图像，计算每个通道的熵然后平均
            entropies = []
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                hist, _ = np.histogram(channel, bins=256, range=(0, 1))
                hist = hist[hist > 0]  # 移除零频
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob))
                entropies.append(entropy)
            return np.mean(entropies)
        else:
            # 灰度图像
            hist, _ = np.histogram(img, bins=256, range=(0, 1))
            hist = hist[hist > 0]  # 移除零频
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            return entropy
    
    def calculate_mutual_information(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算互信息"""
        if len(img1.shape) == 3:
            # 彩色图像，计算每个通道的MI然后平均
            mis = []
            for i in range(img1.shape[2]):
                channel1 = img1[:, :, i]
                channel2 = img2[:, :, i]
                mi = self._calculate_channel_mi(channel1, channel2)
                mis.append(mi)
            return np.mean(mis)
        else:
            # 灰度图像
            return self._calculate_channel_mi(img1, img2)
    
    def _calculate_channel_mi(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算单通道互信息"""
        # 计算联合直方图
        hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=256, range=[[0, 1], [0, 1]])
        
        # 计算边缘直方图
        hist1, _ = np.histogram(img1, bins=256, range=(0, 1))
        hist2, _ = np.histogram(img2, bins=256, range=(0, 1))
        
        # 计算概率分布
        total_pixels = img1.size
        p_xy = hist_2d / total_pixels
        p_x = hist1 / total_pixels
        p_y = hist2 / total_pixels
        
        # 计算互信息
        mi = 0
        for i in range(256):
            for j in range(256):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def calculate_qabf(self, ir_img: np.ndarray, vis_img: np.ndarray, fused_img: np.ndarray) -> float:
        """计算Qabf指标"""
        # 简化的Qabf实现
        # 在实际应用中，可以使用更复杂的实现
        
        # 计算边缘强度
        def edge_strength(img):
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Sobel算子
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = np.sqrt(sobelx**2 + sobely**2)
            return edge_map
        
        # 计算各图像的边缘强度
        edge_ir = edge_strength(ir_img)
        edge_vis = edge_strength(vis_img)
        edge_fused = edge_strength(fused_img)
        
        # 计算Qabf
        numerator = np.sum(edge_fused * np.maximum(edge_ir, edge_vis))
        denominator = np.sum(edge_ir + edge_vis)
        
        if denominator > 0:
            qabf = numerator / denominator
        else:
            qabf = 0.0
        
        return qabf
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算结构相似性指数"""
        if len(img1.shape) == 3:
            # 彩色图像，转换为灰度
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            gray2 = img2
        
        # 确保值在[0, 1]范围内
        gray1 = np.clip(gray1, 0, 1)
        gray2 = np.clip(gray2, 0, 1)
        
        # 转换为[0, 255]范围
        gray1_255 = (gray1 * 255).astype(np.uint8)
        gray2_255 = (gray2 * 255).astype(np.uint8)
        
        return ssim(gray1_255, gray2_255)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算峰值信噪比"""
        if len(img1.shape) == 3:
            # 彩色图像，转换为灰度
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = img1
            gray2 = img2
        
        # 确保值在[0, 1]范围内
        gray1 = np.clip(gray1, 0, 1)
        gray2 = np.clip(gray2, 0, 1)
        
        # 转换为[0, 255]范围
        gray1_255 = (gray1 * 255).astype(np.uint8)
        gray2_255 = (gray2 * 255).astype(np.uint8)
        
        return psnr(gray1_255, gray2_255)
    
    def calculate_miou(self, gt_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = 2) -> float:
        """计算平均交并比"""
        # 简化的mIoU计算
        intersection = np.logical_and(gt_mask, pred_mask)
        union = np.logical_or(gt_mask, pred_mask)
        
        if np.sum(union) > 0:
            iou = np.sum(intersection) / np.sum(union)
        else:
            iou = 0.0
        
        return iou
    
    def calculate_dice(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """计算Dice系数"""
        intersection = np.logical_and(gt_mask, pred_mask)
        dice = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask))
        
        return dice
    
    def calculate_precision_recall(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, float]:
        """计算精确率和召回率"""
        intersection = np.logical_and(gt_mask, pred_mask)
        
        # 精确率 = TP / (TP + FP)
        precision = np.sum(intersection) / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0.0
        
        # 召回率 = TP / (TP + FN)
        recall = np.sum(intersection) / np.sum(gt_mask) if np.sum(gt_mask) > 0 else 0.0
        
        return precision, recall
    
    def eval_fusion(self, ir_img: np.ndarray, vis_img: np.ndarray, fused_img: np.ndarray) -> Dict[str, float]:
        """评估融合质量"""
        metrics = {}
        
        if self.config["compute_entropy"]:
            metrics["entropy"] = self.calculate_entropy(fused_img)
        
        if self.config["compute_mi"]:
            metrics["mutual_information"] = self.calculate_mutual_information(ir_img, fused_img)
        
        if self.config["compute_qabf"]:
            metrics["qabf"] = self.calculate_qabf(ir_img, vis_img, fused_img)
        
        if self.config["compute_ssim"]:
            metrics["ssim"] = self.calculate_ssim(ir_img, fused_img)
        
        if self.config["compute_psnr"]:
            metrics["psnr"] = self.calculate_psnr(ir_img, fused_img)
        
        return metrics
    
    def eval_segmentation(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
        """评估分割质量"""
        metrics = {}
        
        if self.config["compute_miou"]:
            metrics["miou"] = self.calculate_miou(gt_mask, pred_mask)
        
        if self.config["compute_dice"]:
            metrics["dice"] = self.calculate_dice(gt_mask, pred_mask)
        
        if self.config["compute_precision_recall"]:
            precision, recall = self.calculate_precision_recall(gt_mask, pred_mask)
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return metrics

def eval_fusion(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> Dict[str, float]:
    """融合质量评估接口"""
    evaluator = ImageMetrics()
    return evaluator.eval_fusion(ir, vis, fused)

def eval_seg(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """分割质量评估接口"""
    evaluator = ImageMetrics()
    return evaluator.eval_segmentation(gt_mask, pred_mask)

def demo_metrics():
    """演示评测功能"""
    print("=== 图像质量评测工具演示 ===")
    
    # 创建模拟图像
    ir_img = np.random.rand(256, 256, 3).astype(np.float32)
    vis_img = np.random.rand(256, 256, 3).astype(np.float32)
    fused_img = (ir_img + vis_img) / 2  # 简单融合
    
    print(f"红外图像形状: {ir_img.shape}")
    print(f"可见光图像形状: {vis_img.shape}")
    print(f"融合图像形状: {fused_img.shape}")
    
    # 测试融合质量评估
    try:
        fusion_metrics = eval_fusion(ir_img, vis_img, fused_img)
        print("\n融合质量指标:")
        for metric, value in fusion_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # 创建模拟分割掩码
        gt_mask = np.random.rand(256, 256) > 0.5
        pred_mask = np.random.rand(256, 256) > 0.5
        
        # 测试分割质量评估
        seg_metrics = eval_seg(gt_mask, pred_mask)
        print("\n分割质量指标:")
        for metric, value in seg_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"评测失败: {str(e)}")

if __name__ == "__main__":
    demo_metrics()
