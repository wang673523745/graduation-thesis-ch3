"""
红外可见光Agent系统 - 图像分割工具
支持语义分割、实例分割和多头输出
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SegmentationType(Enum):
    """分割类型枚举"""
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"

class SegmentationMethod(Enum):
    """分割方法枚举"""
    THRESHOLD = "threshold"
    WATERSHED = "watershed"
    KMEANS = "kmeans"
    GRABCUT = "grabcut"
    CONTOUR = "contour"

class ImageSegmenter:
    """图像分割器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_config = {
            "segmentation_type": "semantic",
            "segmentation_method": "threshold",
            "num_classes": 2,
            "confidence_threshold": 0.5,
            "min_area": 100,
            "max_area": 10000,
            "morphology_ops": True,
            "post_process": True
        }
        self.config = {**self.default_config, **self.config}
    
    def threshold_segmentation(self, img: np.ndarray) -> np.ndarray:
        """阈值分割"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 使用Otsu方法自动确定阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 转换为二值掩码
        mask = (binary > 0).astype(np.uint8)
        
        return mask
    
    def watershed_segmentation(self, img: np.ndarray) -> np.ndarray:
        """分水岭分割"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # 找到未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(img, markers)
        
        # 创建掩码
        mask = (markers > 1).astype(np.uint8)
        
        return mask
    
    def kmeans_segmentation(self, img: np.ndarray) -> np.ndarray:
        """K-means聚类分割"""
        if len(img.shape) == 3:
            # 重塑为二维数组
            h, w = img.shape[:2]
            data = img.reshape(h * w, 3).astype(np.float32)
        else:
            h, w = img.shape
            data = img.reshape(h * w, 1).astype(np.float32)
        
        # 定义终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        
        # 应用K-means
        num_classes = self.config["num_classes"]
        _, labels, centers = cv2.kmeans(data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 重塑回原始形状
        segmented = labels.reshape(h, w)
        
        # 创建掩码（假设第一个类别是背景）
        mask = (segmented > 0).astype(np.uint8)
        
        return mask
    
    def grabcut_segmentation(self, img: np.ndarray) -> np.ndarray:
        """GrabCut分割"""
        if len(img.shape) == 3:
            # 创建掩码
            mask = np.zeros(img.shape[:2], np.uint8)
            
            # 创建临时数组
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 定义矩形区域（可以根据需要调整）
            h, w = img.shape[:2]
            rect = (10, 10, w - 20, h - 20)
            
            # 应用GrabCut
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 创建掩码
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            return mask2
        else:
            # 灰度图像使用阈值分割
            return self.threshold_segmentation(img)
    
    def contour_segmentation(self, img: np.ndarray) -> np.ndarray:
        """轮廓检测分割"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建掩码
        mask = np.zeros_like(gray)
        
        # 过滤轮廓
        min_area = self.config["min_area"]
        max_area = self.config["max_area"]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                cv2.fillPoly(mask, [contour], 255)
        
        # 转换为二值掩码
        mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """应用形态学操作"""
        if not self.config["morphology_ops"]:
            return mask
        
        kernel = np.ones((3, 3), np.uint8)
        
        # 开运算去除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 闭运算填充小孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def post_process(self, mask: np.ndarray) -> np.ndarray:
        """后处理"""
        if not self.config["post_process"]:
            return mask
        
        # 应用形态学操作
        mask = self.apply_morphology(mask)
        
        # 移除小区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 过滤小区域
        min_area = self.config["min_area"]
        filtered_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 跳过背景（标签0）
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == i] = 1
        
        return filtered_mask
    
    def calculate_confidence(self, img: np.ndarray, mask: np.ndarray) -> float:
        """计算分割置信度"""
        # 简化的置信度计算：基于掩码区域的平均强度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 计算前景区域的平均强度
        foreground_pixels = gray[mask > 0]
        if len(foreground_pixels) > 0:
            avg_intensity = np.mean(foreground_pixels)
            # 归一化到[0, 1]范围
            confidence = avg_intensity / 255.0
        else:
            confidence = 0.0
        
        return confidence
    
    def segment(self, img: np.ndarray, target: str = "all") -> Dict[str, Any]:
        """执行分割"""
        method = self.config["segmentation_method"]
        
        # 根据方法选择分割算法
        if method == "threshold":
            mask = self.threshold_segmentation(img)
        elif method == "watershed":
            mask = self.watershed_segmentation(img)
        elif method == "kmeans":
            mask = self.kmeans_segmentation(img)
        elif method == "grabcut":
            mask = self.grabcut_segmentation(img)
        elif method == "contour":
            mask = self.contour_segmentation(img)
        else:
            logger.warning(f"未知的分割方法: {method}，使用阈值分割")
            mask = self.threshold_segmentation(img)
        
        # 后处理
        mask = self.post_process(mask)
        
        # 计算置信度
        confidence = self.calculate_confidence(img, mask)
        
        # 准备输出
        result = {
            "mask": mask,
            "confidence": confidence,
            "class_info": {
                "num_regions": int(np.max(mask)),
                "total_area": int(np.sum(mask)),
                "method": method,
                "target": target
            }
        }
        
        return result

def segment(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """分割工具的主接口"""
    segmenter = ImageSegmenter(cfg)
    target = cfg.get("target", "all")
    result = segmenter.segment(img, target)
    return result["mask"]

def demo_segmentation():
    """演示分割功能"""
    print("=== 图像分割工具演示 ===")
    
    # 创建模拟图像
    img = np.random.rand(256, 256, 3).astype(np.float32)
    
    print(f"输入图像形状: {img.shape}")
    
    # 测试不同的分割方法
    methods = ["threshold", "watershed", "kmeans", "contour"]
    
    for method in methods:
        try:
            config = {
                "segmentation_method": method,
                "num_classes": 3,
                "min_area": 50,
                "max_area": 5000,
                "morphology_ops": True,
                "post_process": True
            }
            
            segmenter = ImageSegmenter(config)
            result = segmenter.segment(img, "all")
            
            print(f"方法: {method}")
            print(f"  掩码形状: {result['mask'].shape}")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  区域数量: {result['class_info']['num_regions']}")
            print(f"  总面积: {result['class_info']['total_area']}")
            print()
            
        except Exception as e:
            print(f"方法 {method} 失败: {str(e)}")

if __name__ == "__main__":
    demo_segmentation()
