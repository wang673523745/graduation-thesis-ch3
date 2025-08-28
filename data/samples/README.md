# 样例数据目录

此目录用于存放演示和测试用的样例图像数据。

## 目录结构

```
samples/
├── ir_images/          # 红外图像样例
├── vis_images/         # 可见光图像样例
├── ground_truth/       # 真实标签（用于评测）
└── README.md          # 本文件
```

## 数据格式

- **图像格式**：支持 JPG、PNG、BMP、TIFF 等常见格式
- **图像尺寸**：建议 256x256 到 2048x2048 像素
- **数据类型**：8位或16位无符号整数

## 命名规范

- 红外图像：`ir_*.jpg`
- 可见光图像：`vis_*.jpg`
- 真实标签：`gt_*.png`

## 使用说明

1. 将红外和可见光图像对放在相应目录中
2. 确保图像对的文件名对应（除了前缀）
3. 运行系统时会自动加载样例数据

## 示例

```
ir_images/
├── ir_sample1.jpg
├── ir_sample2.jpg
└── ir_sample3.jpg

vis_images/
├── vis_sample1.jpg
├── vis_sample2.jpg
└── vis_sample3.jpg

ground_truth/
├── gt_sample1.png
├── gt_sample2.png
└── gt_sample3.png
```
