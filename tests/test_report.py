"""
红外可见光Agent系统 - 报告生成工具测试
测试报告生成模块的功能
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.report import make_report, ReportGenerator

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
        self.execution_steps = [
            {"name": "preprocess", "status": "success", "duration": 1.23, "details": "预处理完成"},
            {"name": "fuse", "status": "success", "duration": 2.45, "details": "融合完成"},
            {"name": "segment", "status": "success", "duration": 3.67, "details": "分割完成"}
        ]
    
    def test_make_report_html(self):
        """测试HTML报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_report")
            
            report_path = make_report(
                self.run_id,
                self.figures,
                self.tables,
                output_path,
                self.task_description,
                self.execution_steps
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
            
            # 检查生成的图像文件
            comparison_path = f"{output_path}_comparison.png"
            metrics_path = f"{output_path}_metrics.png"
            timeline_path = f"{output_path}_timeline.png"
            
            assert os.path.exists(comparison_path)
            assert os.path.exists(metrics_path)
            assert os.path.exists(timeline_path)
    
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
                self.execution_steps,
                output_path
            )
            
            assert report_path.endswith('.json')
            assert os.path.exists(report_path)
            
            # 验证JSON内容
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data["run_id"] == self.run_id
            assert data["task_description"] == self.task_description
            assert "execution_summary" in data
            assert "metrics" in data
            assert "image_info" in data
    
    def test_make_report_minimal_data(self):
        """测试最小数据报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "minimal_report")
            
            # 只有两张图像
            minimal_figures = [self.figures[0], self.figures[1]]
            minimal_tables = {}
            
            report_path = make_report(
                "minimal_test",
                minimal_figures,
                minimal_tables,
                output_path,
                "最小数据测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_make_report_no_figures(self):
        """测试无图像报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_figures_report")
            
            report_path = make_report(
                "no_figures_test",
                [],
                self.tables,
                output_path,
                "无图像测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_make_report_no_tables(self):
        """测试无表格报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_tables_report")
            
            report_path = make_report(
                "no_tables_test",
                self.figures,
                {},
                output_path,
                "无表格测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_make_report_no_execution_steps(self):
        """测试无执行步骤报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_steps_report")
            
            report_path = make_report(
                "no_steps_test",
                self.figures,
                self.tables,
                output_path,
                "无执行步骤测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)

class TestReportGenerator:
    """测试ReportGenerator类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.generator = ReportGenerator()
        self.images = {
            "ir": np.random.rand(256, 256, 3),
            "vis": np.random.rand(256, 256, 3),
            "fused": np.random.rand(256, 256, 3)
        }
        self.metrics = {
            "entropy": 7.234,
            "qabf": 0.789,
            "ssim": 0.823,
            "psnr": 28.456
        }
        self.execution_steps = [
            {"name": "preprocess", "status": "success", "duration": 1.23},
            {"name": "fuse", "status": "success", "duration": 2.45},
            {"name": "segment", "status": "failed", "duration": 0.0}
        ]
    
    def test_create_comparison_figure(self):
        """测试图像对比图创建"""
        fig = self.generator.create_comparison_figure(self.images)
        assert fig is not None
        
        # 测试自定义标题
        titles = {"ir": "红外图像", "vis": "可见光图像", "fused": "融合图像"}
        fig = self.generator.create_comparison_figure(self.images, titles)
        assert fig is not None
    
    def test_create_metrics_visualization(self):
        """测试指标可视化创建"""
        fig = self.generator.create_metrics_visualization(self.metrics)
        assert fig is not None
        
        # 测试自定义标题
        fig = self.generator.create_metrics_visualization(self.metrics, "自定义指标")
        assert fig is not None
    
    def test_create_execution_timeline(self):
        """测试执行时间线创建"""
        fig = self.generator.create_execution_timeline(self.execution_steps)
        assert fig is not None
    
    def test_create_metrics_table(self):
        """测试指标表格创建"""
        metrics_data = {
            "fusion": self.metrics,
            "segmentation": {
                "miou": 0.756,
                "dice": 0.823
            }
        }
        
        table_html = self.generator.create_metrics_table(metrics_data)
        assert isinstance(table_html, str)
        assert "table" in table_html.lower()
    
    def test_create_metrics_table_empty(self):
        """测试空指标表格创建"""
        table_html = self.generator.create_metrics_table({})
        assert isinstance(table_html, str)
        assert "无指标数据" in table_html
    
    def test_generate_html_report(self):
        """测试HTML报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "html_report")
            
            report_path = self.generator.generate_html_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                self.execution_steps,
                output_path
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_generate_json_report(self):
        """测试JSON报告生成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "json_report")
            
            report_path = self.generator.generate_json_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                self.execution_steps,
                output_path
            )
            
            assert report_path.endswith('.json')
            assert os.path.exists(report_path)
    
    def test_generate_execution_table_rows(self):
        """测试执行表格行生成"""
        rows = self.generator._generate_execution_table_rows(self.execution_steps)
        assert isinstance(rows, str)
        assert "preprocess" in rows
        assert "fuse" in rows
        assert "segment" in rows

class TestReportConfigurations:
    """测试报告配置"""
    
    def setup_method(self):
        """设置测试环境"""
        self.images = {
            "ir": np.random.rand(256, 256, 3),
            "vis": np.random.rand(256, 256, 3)
        }
        self.metrics = {"entropy": 7.234, "qabf": 0.789}
    
    def test_html_format_config(self):
        """测试HTML格式配置"""
        config = {"output_format": "html"}
        generator = ReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "html_config_test")
            
            report_path = generator.generate_html_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                [],
                output_path
            )
            
            assert report_path.endswith('.html')
    
    def test_json_format_config(self):
        """测试JSON格式配置"""
        config = {"output_format": "json"}
        generator = ReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "json_config_test")
            
            report_path = generator.generate_json_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                [],
                output_path
            )
            
            assert report_path.endswith('.json')
    
    def test_no_visualizations_config(self):
        """测试无可视化配置"""
        config = {"include_visualizations": False}
        generator = ReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_viz_test")
            
            report_path = generator.generate_html_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                [],
                output_path
            )
            
            assert report_path.endswith('.html')
    
    def test_no_metrics_table_config(self):
        """测试无指标表格配置"""
        config = {"include_metrics_table": False}
        generator = ReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "no_table_test")
            
            report_path = generator.generate_html_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                [],
                output_path
            )
            
            assert report_path.endswith('.html')
    
    def test_custom_figure_settings(self):
        """测试自定义图像设置"""
        config = {
            "figure_dpi": 150,
            "figure_size": (8, 6),
            "color_scheme": "plasma"
        }
        generator = ReportGenerator(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "custom_fig_test")
            
            report_path = generator.generate_html_report(
                "test_001",
                "测试任务",
                self.images,
                {"metrics": {"fusion": self.metrics}},
                [],
                output_path
            )
            
            assert report_path.endswith('.html')

class TestReportEdgeCases:
    """测试报告边缘情况"""
    
    def setup_method(self):
        """设置测试环境"""
        self.generator = ReportGenerator()
    
    def test_empty_images(self):
        """测试空图像"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "empty_images_test")
            
            report_path = make_report(
                "empty_images_test",
                [],
                {},
                output_path,
                "空图像测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_single_image(self):
        """测试单张图像"""
        single_image = [np.random.rand(256, 256, 3)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "single_image_test")
            
            report_path = make_report(
                "single_image_test",
                single_image,
                {},
                output_path,
                "单图像测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_large_images(self):
        """测试大图像"""
        large_images = [
            np.random.rand(1024, 1024, 3),
            np.random.rand(1024, 1024, 3)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "large_images_test")
            
            report_path = make_report(
                "large_images_test",
                large_images,
                {},
                output_path,
                "大图像测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_grayscale_images(self):
        """测试灰度图像"""
        gray_images = [
            np.random.rand(256, 256),
            np.random.rand(256, 256)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "gray_images_test")
            
            report_path = make_report(
                "gray_images_test",
                gray_images,
                {},
                output_path,
                "灰度图像测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)
    
    def test_mixed_image_types(self):
        """测试混合图像类型"""
        mixed_images = [
            np.random.rand(256, 256, 3),  # 彩色
            np.random.rand(256, 256),     # 灰度
            np.random.rand(256, 256, 1)   # 单通道
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "mixed_images_test")
            
            report_path = make_report(
                "mixed_images_test",
                mixed_images,
                {},
                output_path,
                "混合图像测试"
            )
            
            assert report_path.endswith('.html')
            assert os.path.exists(report_path)

if __name__ == "__main__":
    # 运行报告测试
    pytest.main([__file__, "-v"])
