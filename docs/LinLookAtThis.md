# 指导书：从原始视频数据到训练、评估与评估报告（police_gesture_v1）

本文以 `D:\my_projects\TrafficCopilot\data\raw\police_gesture_v1` 为起点，给出从原始数据 → 训练模型 → 评估性能 → 生成评估报告的完整流程（PowerShell 命令）。

## 0. 环境准备

在项目根目录执行：

```powershell
cd D:\my_projects\TrafficCopilot
conda activate TrafficCopilot
pip install -r .\requirements.txt
```

## 1. 原始数据约定（必须满足）

数据根目录：

```
data/raw/police_gesture_v1/
  train/
    xxx.mp4
    xxx.csv
  test/
    yyy.mp4
    yyy.csv
```

要求：
- 每个视频 `xxx.mp4` 必须有同名标签文件 `xxx.csv`
- `csv` 为单行整数序列：每个数对应一帧的类别 ID
  - `0`：背景/无动作（评估时会忽略）
  - `1..8`：8 类手势

类别映射（与代码一致）：
- 1 change lanes
- 2 go straight
- 3 pull over
- 4 slow down
- 5 stop
- 6 turn left
- 7 turn right
- 8 wait for  left turn

## 2. 路线：骨骼关键点降维 → Pose 时序模型（GRU）

如果你希望模型显式利用“动作变化”（时序），可以做 Pose 时序模型。它的输入不是 RGB 图像，而是每帧 33 点骨骼坐标序列。

### 2.1 从原始视频提取骨骼序列缓存

```powershell
cd D:\my_projects\TrafficCopilot
conda run -n TrafficCopilot python .\scripts\extract_pose_sequences_from_videos.py --split train
conda run -n TrafficCopilot python .\scripts\extract_pose_sequences_from_videos.py --split test
```

预期产物：
- `output/pose_sequences/train/*.npz`
- `output/pose_sequences/test/*.npz`

### 2.2 训练 Pose-GRU（你负责训练）

```powershell
cd D:\my_projects\TrafficCopilot
conda run -n TrafficCopilot python .\scripts\train_pose_sequence_classifier.py --config .\config\pose_sequence_classifier.yaml
```

### 2.3 评估 Pose-GRU（帧级准确率，忽略背景帧）

只需要把 `--config` 换成 Pose 时序 Web 配置即可：

```powershell
cd D:\my_projects\TrafficCopilot
conda run -n TrafficCopilot python .\scripts\run_labeled_video_realtime.py `
  --config .\config\web_pose_sequence.yaml `
  --dataset-root .\data\raw\police_gesture_v1 `
  --split test `
  --no-display `
  --sample-every 1 `
  --smooth-window 5 `
  --output-json .\outputs\eval\pose_gru_test.json
```

说明：
- `--sample-every 1`：每帧都做一次骨骼提取与预测（更符合时序模型的假设）
- `--smooth-window 5`：对最近 5 次预测做投票平滑（减少抖动）
- `--output-json`：保存机器可读结果，用于生成报告

如果你想边看边评估（弹窗显示 GT/PRED/ACC）：

```powershell
cd D:\my_projects\TrafficCopilot
conda run -n TrafficCopilot python .\scripts\run_labeled_video_realtime.py `
  --config .\config\web_pose_sequence.yaml `
  --video .\data\raw\police_gesture_v1\test\002.mp4 `
  --sample-every 1 `
  --smooth-window 5
```

### 2.4 生成评估报告（Markdown）

```powershell
cd D:\my_projects\TrafficCopilot
conda run -n TrafficCopilot python .\scripts\generate_video_eval_report.py `
  --eval-json .\outputs\eval\pose_gru_test.json `
  --output-md .\outputs\reports\pose_gru_test_report.md `
  --title "骨骼时序模型评估报告（Pose-GRU）"
```

### 2.5 Web 演示（上传视频看识别效果）

启动 Web（Pose 时序版本）：

```powershell
cd D:\my_projects\TrafficCopilot
conda run -n TrafficCopilot python .\web\app_pose_sequence.py
```

说明：
- Pose 时序 predictor 会先积累 `clip_len=16` 帧，再开始输出非 UNKNOWN 的预测
- Web 更偏“演示”；准确率闭环以 2.3/2.4 的脚本为准

## 3. 常见问题（和准确率直接相关）

- 准确率不高时优先调这两个：
  - `--sample-every`（Pose 时序一般用 1）
  - `--smooth-window`（建议从 5 开始）
- Web 上传视频“看起来识别不稳”，但脚本评估很高：
  - Web 更偏展示，可能受到帧率、压缩、分辨率等影响；以脚本评估结果作为可量化结论
- 不建议把 `data/`（尤其是抽帧数据集）直接上传 GitHub：
  - 数据通常是 GB 级，普通 GitHub 仓库无法承载；建议用网盘/共享盘分发数据，仓库只保存脚本与配置
