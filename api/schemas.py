"""
红外可见光Agent系统 - Pydantic模型定义
定义API请求和响应的数据模型
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime

class TaskType(str, Enum):
    """任务类型枚举"""
    FUSION_ONLY = "fusion_only"
    SEGMENTATION_ONLY = "segmentation_only"
    FUSION_AND_SEGMENTATION = "fusion_and_segmentation"
    FULL_PIPELINE = "full_pipeline"

class FusionMethod(str, Enum):
    """融合方法枚举"""
    WEIGHTED_AVERAGE = "weighted_average"
    LAPLACIAN = "laplacian"
    DWT = "dwt"
    PCA = "pca"

class SegmentationMethod(str, Enum):
    """分割方法枚举"""
    THRESHOLD = "threshold"
    WATERSHED = "watershed"
    KMEANS = "kmeans"
    GRABCUT = "grabcut"
    CONTOUR = "contour"

class PreprocessConfig(BaseModel):
    """预处理配置"""
    resize_method: str = Field(default="bilinear", description="重采样方法")
    enhancement: bool = Field(default=True, description="是否进行图像增强")
    normalization: bool = Field(default=True, description="是否进行归一化")
    registration: bool = Field(default=True, description="是否进行配准")
    target_size: tuple = Field(default=(512, 512), description="目标尺寸")
    histogram_equalization: bool = Field(default=True, description="直方图均衡化")
    noise_reduction: bool = Field(default=True, description="降噪")
    gamma_correction: float = Field(default=1.0, description="伽马校正")

class FusionConfig(BaseModel):
    """融合配置"""
    fusion_method: FusionMethod = Field(default=FusionMethod.WEIGHTED_AVERAGE, description="融合方法")
    ir_weight: float = Field(default=0.5, description="红外图像权重")
    vis_weight: float = Field(default=0.5, description="可见光图像权重")
    enhance_contrast: bool = Field(default=True, description="是否增强对比度")
    preserve_details: bool = Field(default=True, description="是否保持细节")
    joint_mode: bool = Field(default=False, description="是否为联合模式")

class SegmentationConfig(BaseModel):
    """分割配置"""
    segmentation_method: SegmentationMethod = Field(default=SegmentationMethod.THRESHOLD, description="分割方法")
    num_classes: int = Field(default=2, description="类别数量")
    confidence_threshold: float = Field(default=0.5, description="置信度阈值")
    min_area: int = Field(default=100, description="最小区域面积")
    max_area: int = Field(default=10000, description="最大区域面积")
    morphology_ops: bool = Field(default=True, description="形态学操作")
    post_process: bool = Field(default=True, description="后处理")

class MetricsConfig(BaseModel):
    """评测配置"""
    compute_entropy: bool = Field(default=True, description="计算熵")
    compute_mi: bool = Field(default=True, description="计算互信息")
    compute_qabf: bool = Field(default=True, description="计算Qabf")
    compute_ssim: bool = Field(default=True, description="计算SSIM")
    compute_psnr: bool = Field(default=True, description="计算PSNR")
    compute_miou: bool = Field(default=True, description="计算mIoU")
    compute_dice: bool = Field(default=True, description="计算Dice系数")
    compute_precision_recall: bool = Field(default=True, description="计算精确率召回率")

class ProcessingConfig(BaseModel):
    """处理配置"""
    preprocess: Optional[PreprocessConfig] = Field(default=None, description="预处理配置")
    fusion: Optional[FusionConfig] = Field(default=None, description="融合配置")
    segmentation: Optional[SegmentationConfig] = Field(default=None, description="分割配置")
    metrics: Optional[MetricsConfig] = Field(default=None, description="评测配置")

class ProcessingRequest(BaseModel):
    """处理请求"""
    task_type: TaskType = Field(..., description="任务类型")
    config: Optional[ProcessingConfig] = Field(default=None, description="处理配置")
    ir_image_base64: Optional[str] = Field(default=None, description="红外图像base64编码")
    vis_image_base64: Optional[str] = Field(default=None, description="可见光图像base64编码")

class ExecutionStep(BaseModel):
    """执行步骤"""
    name: str = Field(..., description="步骤名称")
    status: str = Field(..., description="执行状态")
    duration: float = Field(..., description="执行时间")
    details: str = Field(default="", description="详细信息")
    error: Optional[str] = Field(default=None, description="错误信息")

class FusionMetrics(BaseModel):
    """融合质量指标"""
    entropy: Optional[float] = Field(default=None, description="熵")
    mutual_information: Optional[float] = Field(default=None, description="互信息")
    qabf: Optional[float] = Field(default=None, description="Qabf指标")
    ssim: Optional[float] = Field(default=None, description="结构相似性")
    psnr: Optional[float] = Field(default=None, description="峰值信噪比")

class SegmentationMetrics(BaseModel):
    """分割质量指标"""
    miou: Optional[float] = Field(default=None, description="平均交并比")
    dice: Optional[float] = Field(default=None, description="Dice系数")
    precision: Optional[float] = Field(default=None, description="精确率")
    recall: Optional[float] = Field(default=None, description="召回率")
    f1_score: Optional[float] = Field(default=None, description="F1分数")

class ProcessingMetrics(BaseModel):
    """处理质量指标"""
    fusion: Optional[FusionMetrics] = Field(default=None, description="融合指标")
    segmentation: Optional[SegmentationMetrics] = Field(default=None, description="分割指标")

class ProcessingResult(BaseModel):
    """处理结果"""
    run_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="处理状态")
    fused_image_base64: Optional[str] = Field(default=None, description="融合图像base64编码")
    segmentation_mask_base64: Optional[str] = Field(default=None, description="分割掩码base64编码")
    metrics: Optional[ProcessingMetrics] = Field(default=None, description="质量指标")
    execution_steps: List[ExecutionStep] = Field(default=[], description="执行步骤")
    report_path: Optional[str] = Field(default=None, description="报告路径")
    timestamp: datetime = Field(..., description="时间戳")

class ProcessingResponse(BaseModel):
    """处理响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[ProcessingResult] = Field(default=None, description="处理结果")
    error: Optional[str] = Field(default=None, description="错误信息")

class ToolInfo(BaseModel):
    """工具信息"""
    name: str = Field(..., description="工具名称")
    function: str = Field(..., description="函数名")
    doc: Optional[str] = Field(default=None, description="文档")
    config: Dict[str, Any] = Field(default={}, description="配置")

class ToolsResponse(BaseModel):
    """工具列表响应"""
    available_tools: List[str] = Field(..., description="可用工具列表")
    tool_details: Dict[str, ToolInfo] = Field(..., description="工具详细信息")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="状态")
    timestamp: datetime = Field(..., description="时间戳")
    version: str = Field(..., description="版本")

class TaskStatus(BaseModel):
    """任务状态"""
    run_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: float = Field(default=0.0, description="进度")
    message: str = Field(default="", description="状态消息")
    timestamp: datetime = Field(..., description="时间戳")

# 示例数据
class ExampleData:
    """示例数据"""
    
    @staticmethod
    def get_preprocess_config() -> PreprocessConfig:
        """获取预处理配置示例"""
        return PreprocessConfig(
            resize_method="bilinear",
            enhancement=True,
            normalization=True,
            registration=True,
            target_size=(512, 512),
            histogram_equalization=True,
            noise_reduction=True,
            gamma_correction=1.0
        )
    
    @staticmethod
    def get_fusion_config() -> FusionConfig:
        """获取融合配置示例"""
        return FusionConfig(
            fusion_method=FusionMethod.WEIGHTED_AVERAGE,
            ir_weight=0.6,
            vis_weight=0.4,
            enhance_contrast=True,
            preserve_details=True,
            joint_mode=False
        )
    
    @staticmethod
    def get_segmentation_config() -> SegmentationConfig:
        """获取分割配置示例"""
        return SegmentationConfig(
            segmentation_method=SegmentationMethod.THRESHOLD,
            num_classes=2,
            confidence_threshold=0.5,
            min_area=100,
            max_area=10000,
            morphology_ops=True,
            post_process=True
        )
    
    @staticmethod
    def get_processing_request() -> ProcessingRequest:
        """获取处理请求示例"""
        return ProcessingRequest(
            task_type=TaskType.FULL_PIPELINE,
            config=ProcessingConfig(
                preprocess=ExampleData.get_preprocess_config(),
                fusion=ExampleData.get_fusion_config(),
                segmentation=ExampleData.get_segmentation_config(),
                metrics=MetricsConfig()
            )
        )
    
    @staticmethod
    def get_processing_result() -> ProcessingResult:
        """获取处理结果示例"""
        return ProcessingResult(
            run_id="demo_20241201_001",
            status="completed",
            metrics=ProcessingMetrics(
                fusion=FusionMetrics(
                    entropy=7.234,
                    mutual_information=2.456,
                    qabf=0.789,
                    ssim=0.823,
                    psnr=28.456
                ),
                segmentation=SegmentationMetrics(
                    miou=0.756,
                    dice=0.823,
                    precision=0.789,
                    recall=0.812,
                    f1_score=0.800
                )
            ),
            execution_steps=[
                ExecutionStep(
                    name="preprocess",
                    status="success",
                    duration=1.23,
                    details="图像预处理完成"
                ),
                ExecutionStep(
                    name="fuse",
                    status="success",
                    duration=2.45,
                    details="图像融合完成"
                ),
                ExecutionStep(
                    name="segment",
                    status="success",
                    duration=3.67,
                    details="图像分割完成"
                ),
                ExecutionStep(
                    name="metrics",
                    status="success",
                    duration=0.89,
                    details="质量评估完成"
                )
            ],
            report_path="/data/outputs/demo_20241201_001.html",
            timestamp=datetime.now()
        )
