"""
红外可见光Agent系统 - 集成测试
测试完整处理流程和系统集成
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

from tools.preprocess import preprocess
from tools.fuse import fuse_single, fuse_joint
from tools.segment import segment
from tools.metrics import eval_fusion, eval_seg
from tools.report import make_report

class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.ir_img = np.random.rand(128, 128, 3).astype(np.float32)
        self.vis_img = np.random.rand(128, 128, 3).astype(np.float32)
    
    def test_full_pipeline(self):
        """测试完整处理流程"""
        # 1. 预处理
        preprocess_result = preprocess(self.ir_img, self.vis_img, {})
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        assert ir_processed.shape == vis_processed.shape
        assert ir_processed.shape[:2] == (512, 512)  # 默认目标尺寸
        
        # 2. 融合
        fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": "weighted_average"})
        
        assert fused_img.shape == ir_processed.shape
        assert np.all(fused_img >= 0) and np.all(fused_img <= 1)
        
        # 3. 分割
        segmentation_mask = segment(fused_img, {"segmentation_method": "threshold"})
        
        assert segmentation_mask.shape == fused_img.shape[:2]
        assert np.all(np.unique(segmentation_mask) == [0, 1])
        
        # 4. 评测
        fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
        
        assert "entropy" in fusion_metrics
        assert "mutual_information" in fusion_metrics
        assert "qabf" in fusion_metrics
        assert "ssim" in fusion_metrics
        assert "psnr" in fusion_metrics
        
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
    
    def test_full_pipeline_with_joint_fusion(self):
        """测试带联合融合的完整流程"""
        # 1. 预处理
        preprocess_result = preprocess(self.ir_img, self.vis_img, {})
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        # 2. 分割（用于联合融合）
        segmentation_mask = segment(ir_processed, {"segmentation_method": "threshold"})
        
        # 3. 联合融合
        fusion_result = fuse_joint(ir_processed, vis_processed, {
            "fusion_method": "weighted_average",
            "segmentation_mask": segmentation_mask
        })
        
        fused_img = fusion_result["fused"]
        assert "aux" in fusion_result
        
        # 4. 评测
        fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
        seg_metrics = eval_seg(segmentation_mask, segmentation_mask)  # 完美匹配
        
        # 5. 生成报告
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "joint_fusion_test")
            
            report_path = make_report(
                "joint_fusion_test",
                [ir_processed, vis_processed, fused_img, segmentation_mask],
                {
                    "metrics": {
                        "fusion": fusion_metrics,
                        "segmentation": seg_metrics
                    }
                },
                output_path,
                "联合融合测试"
            )
            
            assert os.path.exists(report_path)
    
    def test_full_pipeline_different_methods(self):
        """测试不同方法的完整流程"""
        # 测试不同的融合方法
        fusion_methods = ["weighted_average", "laplacian", "dwt", "pca"]
        
        for method in fusion_methods:
            # 1. 预处理
            preprocess_result = preprocess(self.ir_img, self.vis_img, {})
            ir_processed = preprocess_result["ir"]
            vis_processed = preprocess_result["vis"]
            
            # 2. 融合
            fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": method})
            assert fused_img.shape == ir_processed.shape
            
            # 3. 分割
            segmentation_mask = segment(fused_img, {"segmentation_method": "threshold"})
            assert segmentation_mask.shape == fused_img.shape[:2]
            
            # 4. 评测
            fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
            assert len(fusion_metrics) > 0
    
    def test_full_pipeline_different_segmentation(self):
        """测试不同分割方法的完整流程"""
        # 测试不同的分割方法
        segmentation_methods = ["threshold", "watershed", "kmeans", "contour"]
        
        # 1. 预处理
        preprocess_result = preprocess(self.ir_img, self.vis_img, {})
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        # 2. 融合
        fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": "weighted_average"})
        
        for method in segmentation_methods:
            # 3. 分割
            segmentation_mask = segment(fused_img, {"segmentation_method": method})
            assert segmentation_mask.shape == fused_img.shape[:2]
            
            # 4. 评测
            seg_metrics = eval_seg(segmentation_mask, segmentation_mask)  # 完美匹配
            assert seg_metrics["miou"] == 1.0
            assert seg_metrics["dice"] == 1.0
    
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
    
    def test_data_flow(self):
        """测试数据流"""
        # 1. 预处理
        preprocess_result = preprocess(self.ir_img, self.vis_img, {})
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        # 验证数据范围
        assert np.all(ir_processed >= 0) and np.all(ir_processed <= 1)
        assert np.all(vis_processed >= 0) and np.all(vis_processed <= 1)
        
        # 2. 融合
        fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": "weighted_average"})
        
        # 验证数据范围
        assert np.all(fused_img >= 0) and np.all(fused_img <= 1)
        
        # 3. 分割
        segmentation_mask = segment(fused_img, {"segmentation_method": "threshold"})
        
        # 验证数据范围
        assert np.all(np.unique(segmentation_mask) == [0, 1])
        
        # 4. 评测
        fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
        
        # 验证指标范围
        assert 0 <= fusion_metrics["entropy"] <= 10
        assert 0 <= fusion_metrics["qabf"] <= 1
        assert 0 <= fusion_metrics["ssim"] <= 1
        assert fusion_metrics["psnr"] >= 0
    
    def test_performance(self):
        """测试性能"""
        import time
        
        # 记录开始时间
        start_time = time.time()
        
        # 1. 预处理
        preprocess_start = time.time()
        preprocess_result = preprocess(self.ir_img, self.vis_img, {})
        preprocess_time = time.time() - preprocess_start
        
        ir_processed = preprocess_result["ir"]
        vis_processed = preprocess_result["vis"]
        
        # 2. 融合
        fusion_start = time.time()
        fused_img = fuse_single(ir_processed, vis_processed, {"fusion_method": "weighted_average"})
        fusion_time = time.time() - fusion_start
        
        # 3. 分割
        segmentation_start = time.time()
        segmentation_mask = segment(fused_img, {"segmentation_method": "threshold"})
        segmentation_time = time.time() - segmentation_start
        
        # 4. 评测
        metrics_start = time.time()
        fusion_metrics = eval_fusion(ir_processed, vis_processed, fused_img)
        metrics_time = time.time() - metrics_start
        
        # 5. 报告生成
        report_start = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "performance_test")
            report_path = make_report(
                "performance_test",
                [ir_processed, vis_processed, fused_img, segmentation_mask],
                {"metrics": {"fusion": fusion_metrics}},
                output_path,
                "性能测试"
            )
        report_time = time.time() - report_start
        
        total_time = time.time() - start_time
        
        # 验证性能（时间应该在合理范围内）
        assert preprocess_time < 10  # 预处理应该在10秒内
        assert fusion_time < 10      # 融合应该在10秒内
        assert segmentation_time < 10  # 分割应该在10秒内
        assert metrics_time < 5      # 评测应该在5秒内
        assert report_time < 10      # 报告生成应该在10秒内
        assert total_time < 50       # 总时间应该在50秒内
        
        print(f"性能测试结果:")
        print(f"  预处理: {preprocess_time:.2f}秒")
        print(f"  融合: {fusion_time:.2f}秒")
        print(f"  分割: {segmentation_time:.2f}秒")
        print(f"  评测: {metrics_time:.2f}秒")
        print(f"  报告生成: {report_time:.2f}秒")
        print(f"  总时间: {total_time:.2f}秒")

class TestSystemIntegration:
    """系统集成测试"""
    
    def test_tool_compatibility(self):
        """测试工具兼容性"""
        # 测试所有工具是否能正常工作
        ir_img = np.random.rand(64, 64, 3).astype(np.float32)
        vis_img = np.random.rand(64, 64, 3).astype(np.float32)
        
        # 预处理
        preprocess_result = preprocess(ir_img, vis_img, {})
        assert "ir" in preprocess_result
        assert "vis" in preprocess_result
        
        # 融合
        fused_img = fuse_single(preprocess_result["ir"], preprocess_result["vis"], {})
        assert fused_img.shape == preprocess_result["ir"].shape
        
        # 分割
        segmentation_mask = segment(fused_img, {})
        assert segmentation_mask.shape == fused_img.shape[:2]
        
        # 评测
        fusion_metrics = eval_fusion(preprocess_result["ir"], preprocess_result["vis"], fused_img)
        assert len(fusion_metrics) > 0
        
        seg_metrics = eval_seg(segmentation_mask, segmentation_mask)
        assert len(seg_metrics) > 0
    
    def test_configuration_consistency(self):
        """测试配置一致性"""
        # 测试不同配置下的处理结果一致性
        ir_img = np.random.rand(64, 64, 3).astype(np.float32)
        vis_img = np.random.rand(64, 64, 3).astype(np.float32)
        
        # 配置1
        config1 = {
            "target_size": (128, 128),
            "fusion_method": "weighted_average",
            "segmentation_method": "threshold"
        }
        
        # 配置2
        config2 = {
            "target_size": (128, 128),
            "fusion_method": "weighted_average",
            "segmentation_method": "threshold"
        }
        
        # 使用相同配置应该得到相同结果
        result1 = preprocess(ir_img, vis_img, config1)
        result2 = preprocess(ir_img, vis_img, config2)
        
        assert result1["ir"].shape == result2["ir"].shape
        assert result1["vis"].shape == result2["vis"].shape
    
    def test_error_recovery(self):
        """测试错误恢复"""
        # 测试系统在遇到错误时的恢复能力
        
        # 正常流程
        ir_img = np.random.rand(64, 64, 3).astype(np.float32)
        vis_img = np.random.rand(64, 64, 3).astype(np.float32)
        
        try:
            # 预处理
            preprocess_result = preprocess(ir_img, vis_img, {})
            
            # 融合
            fused_img = fuse_single(preprocess_result["ir"], preprocess_result["vis"], {})
            
            # 分割
            segmentation_mask = segment(fused_img, {})
            
            # 评测
            fusion_metrics = eval_fusion(preprocess_result["ir"], preprocess_result["vis"], fused_img)
            
            # 报告生成
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "recovery_test")
                report_path = make_report(
                    "recovery_test",
                    [preprocess_result["ir"], preprocess_result["vis"], fused_img, segmentation_mask],
                    {"metrics": {"fusion": fusion_metrics}},
                    output_path,
                    "错误恢复测试"
                )
            
            assert True  # 如果到达这里，说明没有错误
            
        except Exception as e:
            # 记录错误但不中断测试
            print(f"错误恢复测试中遇到错误: {e}")
            assert False  # 如果有错误，测试失败

if __name__ == "__main__":
    # 运行集成测试
    pytest.main([__file__, "-v"])
