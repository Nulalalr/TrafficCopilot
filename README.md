# 🚦 TrafficCopilot: 面向自动驾驶的鲁棒交警手势识别系统

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Supported-lightgrey.svg)](https://onnxruntime.ai/)
[![Demo Video](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://youtu.be/placeholder)

</div>

## 📌 项目简介

随着高阶自动驾驶的发展，车辆对动态、非标准化的生物指令（如交警现场手势）的理解已成为 L4 级无人驾驶全场景落地的关键瓶颈。

**TrafficCopilot** 是一个高鲁棒性、强实时性的交警手势识别系统，旨在作为独立感知模块无缝嵌入无人驾驶平台。项目不仅实现了高精度的静态手势分类，更设计了**连续手势序列的时空意图解析机制**，以应对复杂光照、天气变化及个体差异带来的挑战。

> 🔗 **技术意义**：填补了自动驾驶在 **V2P (Vehicle-to-Pedestrian) 高级交互**中的感知空白，是推动无人车在信号灯故障、交通管制等场景下安全通行的关键技术。

## ✨ 核心特性

- **🎯 高精度手势识别**：基于改进的轻量级骨干网络，融合 **坐标注意力机制 (Coordinate Attention)**，精准聚焦交警手臂关键区域。
- **⚡ 车规级实时性能**：支持 **TensorRT / ONNX** 模型加速与量化部署，在边缘设备 (Jetson Orin) 上可达 200+ FPS。
- **🌧️ 强鲁棒性设计**：内置复杂环境数据增强管道，有效抵御雨雾、强光、运动模糊等干扰。
- **⏳ 时序意图理解**：独创的轻量级状态机，将单帧识别结果转化为连续、稳定的驾驶控制意图（如“停止待转”）。
- **🔌 即插即用**：提供标准化的 ROS 2 消息接口设计，方便对接 Apollo 或 Autoware 等自动驾驶框架。

## 🏗️ 系统架构

本项目采用模块化设计，清晰定义了从数据输入到车辆控制指令的完整闭环。

```text
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Camera Input   │ --> │  Preprocessing &    │ --> │  Lightweight Model  │
│ (RGB Stream)    │     │  Data Augmentation  │     │  (MobileNet + CA)   │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ Vehicle Control │ <-- │  Intent Decoder     │ <-- │  Skeleton Auxiliary │
│  (ROS2 Node)    │     │  (Temporal FSM)     │     │  Branch (MediaPipe) │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## 📂 项目文件结构

```bash
TrafficCopilot/
├── config/                 # 模型与训练配置文件
├── core/                   # 核心算法代码
│   ├── model/              # 骨干网络、注意力模块定义
│   ├── utils/              # 数据增强、评估指标工具
│   └── deploy/             # 模型轻量化与推理脚本
├── web/                    # Web 演示与 API 服务 (Flask)
├── docs/                   # 系统方案报告与PPT
├── scripts/                # 训练、测试、导出脚本
├── test_images/            # 独立测试图片集
├── requirements.txt        # Python 依赖包列表
└── README.md
```

## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆仓库
git clone https://github.com/your_org/TrafficCopilot.git
cd TrafficCopilot

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行 Web (图片推理)
我们提供了一个简易的 Web 交互界面，可上传交警图片并查看识别结果与时序意图输出。

```bash
python web/app.py
```
打开浏览器访问 `http://127.0.0.1:5000`，上传图片即可体验。

### 3. 批量测试与准确率评估
```bash
python scripts/evaluate.py --model_path weights/best_model.onnx --data_dir test_images/
```
运行后将输出混淆矩阵、准确率 (Accuracy) 及各类别 F1-Score。

## 📊 实验性能

在自建测试集（包含 5 类常见手势：停止、直行、左转、右转、变道）上的评估结果：

| 模型版本 | 输入尺寸 | 参数量 | **Top-1 准确率** | **推理延迟 (GPU)** |
| :--- | :--- | :--- | :--- | :--- |
| **Ours (FP32)** | 224x224 | 3.2M | **95.6%** | 4.2 ms |
| **Ours (INT8)** | 224x224 | 3.2M | **94.8%** | **1.8 ms** |

> ⚠️ 注：实际性能受硬件环境及图像质量影响。


## 📜 引用与致谢

如果本项目对您的研究或开发有所帮助，欢迎引用：

```
@misc{TrafficCopilot2026,
  author = {Team TrafficCopilot},
  title = {TrafficCopilot: A Robust Traffic Police Gesture Recognition System for Autonomous Driving},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/your_org/TrafficCopilot}}
}
```

本项目的部分数据来源于开源数据集 **HaGRID**，特此致谢。

## 📄 许可证

本项目采用 [MIT License](LICENSE)。您可自由使用、修改及分发代码，但需保留原始版权声明。
