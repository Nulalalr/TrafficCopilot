# 路线A：视频数据集抽帧 → 训练 → 实时演示（对齐摄像头分布）

本路线用于将 `data/raw/police_gesture_v1` 的标注视频转换为“图片分类数据集”（Roboflow `_classes.csv` 格式），从而复用现有 MobileNetV3 训练脚本，并在训练完成后切换到视频帧模型进行 Web/摄像头演示。

## 1. 数据准备与目录约定

- 视频数据集目录（已落盘）：
  - `data/raw/police_gesture_v1/train/*.mp4 + *.csv`
  - `data/raw/police_gesture_v1/test/*.mp4 + *.csv`
- 抽帧输出目录（脚本生成）：
  - `data/video_frames_ctp_v1/{train,valid,test}/`

## 2. 生成抽帧图片分类数据集（已完成/可重复执行）

在仓库根目录执行：

```powershell
conda run -n TrafficCopilot python scripts/build_video_frame_dataset.py
```

说明：

- 该脚本会从 `data/raw/police_gesture_v1` 读取视频与逐帧标签（CSV 第一行是一串 label_id）。
- `label_id <= 0` 视为背景帧并跳过。
- 默认每 5 帧抽 1 帧，并限制每个视频每个类别最多保存 80 帧，避免单视频过度重复。
- 从训练帧中切出 10% 作为 `valid`。

验收检查点：

- `data/video_frames_ctp_v1/train/_classes.csv` 存在且首行包含 8 类列名。
- `data/video_frames_ctp_v1/valid/_classes.csv`、`data/video_frames_ctp_v1/test/_classes.csv` 同样存在。

## 3. 训练（你后续单独执行）

训练配置已经准备好：

- `config/mobilenetv3_video_frames.yaml`（`data.dataset_root = data/video_frames_ctp_v1`）

训练输出约定（训练脚本默认行为）：

- `experiments/mobilenetv3_video_frames/`
  - `best_model.pth`
  - `last_model.pth`
  - `training_log.csv`
  - `metrics.json`
  - `class_names.json`

## 4. 切换到视频帧模型进行实时演示

训练完成后，直接运行：

```powershell
conda run -n TrafficCopilot python web/app_video_frames.py
```

该入口固定加载：

- `config/web_video_frames.yaml`
- `config/camera_video_frames.yaml`
- `config/video_video_frames.yaml`

其中：

- 模型 checkpoint 默认指向 `experiments/mobilenetv3_video_frames/best_model.pth`
- 数据集默认指向 `data/video_frames_ctp_v1`
- 摄像头/视频推理默认使用“整帧 ROI”（避免裁剪策略与训练分布不一致）

## 5. 快速评测（可选）

Web 服务启动后，可通过评测接口快速查看在当前数据集上的指标（accuracy、macro-F1、混淆矩阵、延迟统计等）：

- `GET /api/evaluate/train`
- `GET /api/evaluate/valid`
- `GET /api/evaluate/test`

## 6. 常见问题

- 如果实时效果仍然不稳定，优先检查：
  - 摄像头画面是否与训练数据在视角/距离/光照上差异很大（域偏移）
  - `thresholds.confidence` 与 `thresholds.margin` 是否过严导致大量 `UNKNOWN`
  - 是否需要把摄像头分辨率/帧率固定到更稳定的输入（见 `config/camera_video_frames.yaml` 的 `camera_width/camera_height/camera_fps`）

