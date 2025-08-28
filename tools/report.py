"""
红外可见光Agent系统 - 报告生成工具
生成可视化对比、表格、PDF/HTML报告
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import os
from datetime import datetime
import cv2

logger = logging.getLogger(__name__)

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_config = {
            "output_format": "html",  # html, pdf, json
            "include_visualizations": True,
            "include_metrics_table": True,
            "include_execution_summary": True,
            "figure_dpi": 300,
            "figure_size": (12, 8),
            "color_scheme": "viridis"
        }
        self.config = {**self.default_config, **self.config}
        
        # 设置matplotlib样式
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = self.config["figure_dpi"]
        plt.rcParams['figure.figsize'] = self.config["figure_size"]
    
    def create_comparison_figure(self, images: Dict[str, np.ndarray], 
                                titles: Optional[Dict[str, str]] = None) -> plt.Figure:
        """创建图像对比图"""
        if titles is None:
            titles = {key: key.title() for key in images.keys()}
        
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (key, img) in enumerate(images.items()):
            ax = axes[i]
            
            # 确保图像在[0, 1]范围内
            img_display = np.clip(img, 0, 1)
            
            if len(img_display.shape) == 3:
                ax.imshow(img_display)
            else:
                ax.imshow(img_display, cmap='gray')
            
            ax.set_title(titles[key], fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_metrics_visualization(self, metrics: Dict[str, float], 
                                   title: str = "质量指标") -> plt.Figure:
        """创建指标可视化图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 柱状图
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        
        bars = ax1.bar(metrics_names, metrics_values, color='skyblue', alpha=0.7)
        ax1.set_title(f"{title} - 柱状图", fontsize=14, fontweight='bold')
        ax1.set_ylabel('指标值')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        values = list(metrics_values)
        values += values[:1]  # 闭合图形
        
        ax2.plot(angles, values, 'o-', linewidth=2, color='red', alpha=0.7)
        ax2.fill(angles, values, alpha=0.25, color='red')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics_names)
        ax2.set_title(f"{title} - 雷达图", fontsize=14, fontweight='bold')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_execution_timeline(self, execution_steps: List[Dict[str, Any]]) -> plt.Figure:
        """创建执行时间线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        step_names = [step['name'] for step in execution_steps]
        durations = [step.get('duration', 0) for step in execution_steps]
        statuses = [step.get('status', 'unknown') for step in execution_steps]
        
        # 定义颜色映射
        color_map = {
            'success': 'green',
            'failed': 'red',
            'running': 'orange',
            'unknown': 'gray'
        }
        
        colors = [color_map.get(status, 'gray') for status in statuses]
        
        bars = ax.barh(step_names, durations, color=colors, alpha=0.7)
        ax.set_xlabel('执行时间 (秒)')
        ax.set_title('任务执行时间线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, duration in zip(bars, durations):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{duration:.2f}s', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def create_metrics_table(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """创建指标表格HTML"""
        if not metrics:
            return "<p>无指标数据</p>"
        
        # 创建DataFrame
        df = pd.DataFrame(metrics).T
        
        # 格式化数值
        df = df.round(4)
        
        # 生成HTML表格
        html_table = df.to_html(
            classes=['table', 'table-striped', 'table-bordered'],
            table_id='metrics-table',
            escape=False
        )
        
        return html_table
    
    def generate_html_report(self, run_id: str, task_description: str,
                           images: Dict[str, np.ndarray],
                           metrics: Dict[str, Dict[str, float]],
                           execution_steps: List[Dict[str, Any]],
                           output_path: str) -> str:
        """生成HTML报告"""
        
        # 创建图像对比图
        if self.config["include_visualizations"] and images:
            comparison_fig = self.create_comparison_figure(images)
            comparison_path = f"{output_path}_comparison.png"
            comparison_fig.savefig(comparison_path, dpi=self.config["figure_dpi"], bbox_inches='tight')
            plt.close(comparison_fig)
        
        # 创建指标可视化图
        if self.config["include_visualizations"] and metrics:
            metrics_fig = self.create_metrics_visualization(metrics.get('fusion', {}))
            metrics_path = f"{output_path}_metrics.png"
            metrics_fig.savefig(metrics_path, dpi=self.config["figure_dpi"], bbox_inches='tight')
            plt.close(metrics_fig)
        
        # 创建执行时间线图
        if self.config["include_visualizations"] and execution_steps:
            timeline_fig = self.create_execution_timeline(execution_steps)
            timeline_path = f"{output_path}_timeline.png"
            timeline_fig.savefig(timeline_path, dpi=self.config["figure_dpi"], bbox_inches='tight')
            plt.close(timeline_fig)
        
        # 生成HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>红外可见光Agent系统报告 - {run_id}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                    border-bottom: 2px solid #007bff;
                    padding-bottom: 10px;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #007bff;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .status-success {{ color: green; font-weight: bold; }}
                .status-failed {{ color: red; font-weight: bold; }}
                .status-running {{ color: orange; font-weight: bold; }}
                .timestamp {{
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>红外可见光Agent系统处理报告</h1>
                <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p class="timestamp">任务ID: {run_id}</p>
                
                <div class="section">
                    <h2>任务描述</h2>
                    <p>{task_description}</p>
                </div>
                
                <div class="section">
                    <h2>处理结果对比</h2>
                    <div class="image-container">
                        <img src="{os.path.basename(comparison_path)}" alt="图像对比">
                    </div>
                </div>
                
                <div class="section">
                    <h2>质量指标分析</h2>
                    <div class="image-container">
                        <img src="{os.path.basename(metrics_path)}" alt="质量指标">
                    </div>
                    {self.create_metrics_table(metrics) if self.config["include_metrics_table"] else ""}
                </div>
                
                <div class="section">
                    <h2>执行时间线</h2>
                    <div class="image-container">
                        <img src="{os.path.basename(timeline_path)}" alt="执行时间线">
                    </div>
                    <table>
                        <tr>
                            <th>步骤</th>
                            <th>状态</th>
                            <th>执行时间</th>
                            <th>详细信息</th>
                        </tr>
                        {self._generate_execution_table_rows(execution_steps)}
                    </table>
                </div>
                
                <div class="section">
                    <h2>总结</h2>
                    <p>本次处理共执行了 {len(execution_steps)} 个步骤，</p>
                    <p>成功步骤: {len([s for s in execution_steps if s.get('status') == 'success'])}</p>
                    <p>失败步骤: {len([s for s in execution_steps if s.get('status') == 'failed'])}</p>
                    <p>总执行时间: {sum(s.get('duration', 0) for s in execution_steps):.2f} 秒</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        html_path = f"{output_path}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_execution_table_rows(self, execution_steps: List[Dict[str, Any]]) -> str:
        """生成执行步骤表格行"""
        rows = ""
        for step in execution_steps:
            status_class = f"status-{step.get('status', 'unknown')}"
            rows += f"""
            <tr>
                <td>{step.get('name', 'Unknown')}</td>
                <td class="{status_class}">{step.get('status', 'unknown')}</td>
                <td>{step.get('duration', 0):.2f}s</td>
                <td>{step.get('details', '')}</td>
            </tr>
            """
        return rows
    
    def generate_json_report(self, run_id: str, task_description: str,
                           images: Dict[str, np.ndarray],
                           metrics: Dict[str, Dict[str, float]],
                           execution_steps: List[Dict[str, Any]],
                           output_path: str) -> str:
        """生成JSON报告"""
        
        # 准备报告数据
        report_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "execution_summary": {
                "total_steps": len(execution_steps),
                "successful_steps": len([s for s in execution_steps if s.get('status') == 'success']),
                "failed_steps": len([s for s in execution_steps if s.get('status') == 'failed']),
                "total_duration": sum(s.get('duration', 0) for s in execution_steps)
            },
            "execution_steps": execution_steps,
            "metrics": metrics,
            "image_info": {
                key: {
                    "shape": list(img.shape),
                    "dtype": str(img.dtype),
                    "min_value": float(np.min(img)),
                    "max_value": float(np.max(img))
                }
                for key, img in images.items()
            }
        }
        
        # 保存JSON文件
        json_path = f"{output_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return json_path

def make_report(run_id: str, figures: List[np.ndarray], tables: Dict[str, Any], 
                output_path: str, task_description: str = "", 
                execution_steps: List[Dict[str, Any]] = None) -> str:
    """报告生成主接口"""
    
    if execution_steps is None:
        execution_steps = []
    
    # 准备图像数据
    images = {}
    if len(figures) >= 2:
        images['ir'] = figures[0]
        images['vis'] = figures[1]
    if len(figures) >= 3:
        images['fused'] = figures[2]
    if len(figures) >= 4:
        images['segmented'] = figures[3]
    
    # 准备指标数据
    metrics = tables.get('metrics', {})
    
    # 创建报告生成器
    generator = ReportGenerator()
    
    # 根据配置生成报告
    if generator.config["output_format"] == "html":
        return generator.generate_html_report(
            run_id, task_description, images, metrics, execution_steps, output_path
        )
    elif generator.config["output_format"] == "json":
        return generator.generate_json_report(
            run_id, task_description, images, metrics, execution_steps, output_path
        )
    else:
        raise ValueError(f"不支持的报告格式: {generator.config['output_format']}")

def demo_report():
    """演示报告生成功能"""
    print("=== 报告生成工具演示 ===")
    
    # 创建模拟数据
    run_id = "demo_20241201_001"
    task_description = "红外可见光图像融合与分割处理"
    
    # 模拟图像
    figures = [
        np.random.rand(256, 256, 3),  # 红外图像
        np.random.rand(256, 256, 3),  # 可见光图像
        np.random.rand(256, 256, 3),  # 融合图像
        np.random.rand(256, 256)      # 分割掩码
    ]
    
    # 模拟指标
    tables = {
        "metrics": {
            "fusion": {
                "entropy": 7.234,
                "mutual_information": 2.456,
                "qabf": 0.789,
                "ssim": 0.823,
                "psnr": 28.456
            },
            "segmentation": {
                "miou": 0.756,
                "dice": 0.823,
                "precision": 0.789,
                "recall": 0.812,
                "f1_score": 0.800
            }
        }
    }
    
    # 模拟执行步骤
    execution_steps = [
        {"name": "preprocess", "status": "success", "duration": 1.23, "details": "图像预处理完成"},
        {"name": "fuse", "status": "success", "duration": 2.45, "details": "图像融合完成"},
        {"name": "segment", "status": "success", "duration": 3.67, "details": "图像分割完成"},
        {"name": "metrics", "status": "success", "duration": 0.89, "details": "质量评估完成"}
    ]
    
    try:
        # 生成HTML报告
        output_path = "demo_report"
        report_path = make_report(
            run_id, figures, tables, output_path, 
            task_description, execution_steps
        )
        
        print(f"报告已生成: {report_path}")
        print(f"报告类型: {os.path.splitext(report_path)[1]}")
        
        # 生成JSON报告
        generator = ReportGenerator({"output_format": "json"})
        json_path = generator.generate_json_report(
            run_id, task_description, 
            {"ir": figures[0], "vis": figures[1], "fused": figures[2]}, 
            tables, execution_steps, "demo_report_json"
        )
        
        print(f"JSON报告已生成: {json_path}")
        
    except Exception as e:
        print(f"报告生成失败: {str(e)}")

if __name__ == "__main__":
    demo_report()
