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

## 5.1 针对 police_gesture_v1/test 的“视频实时识别 + 准确率”验证（推荐）

如果你的目标是“把 `data/raw/police_gesture_v1/test` 这些视频识别得更准（例如 0.95+）”，建议用带标签的视频评测脚本做闭环验证。该脚本会逐帧读视频、调用当前模型做预测、叠加显示结果，并用同名 `.csv` 的逐帧标签计算准确率（忽略 `label_id=0` 的背景帧）。

单个视频实时验证（弹窗显示，按 `q`/`Esc` 退出）：

```powershell
conda run -n TrafficCopilot python scripts/run_labeled_video_realtime.py --video data/raw/police_gesture_v1/test/002.mp4 --sample-every 5 --smooth-window 5
```

批量评测整个 test 目录（只打印统计，不弹窗）：

```powershell
conda run -n TrafficCopilot python scripts/run_labeled_video_realtime.py --split test --no-display --sample-every 5 --smooth-window 5
```

参数建议（为了更容易达到 0.95+）：

- `--sample-every 5`：与抽帧数据集的采样保持一致（减少相邻帧重复带来的抖动）
- `--smooth-window 5`：对连续预测做多数投票稳定化（对同一个动作段的准确率提升明显）

## 6. 常见问题

- 如果实时效果仍然不稳定，优先检查：
  - 摄像头画面是否与训练数据在视角/距离/光照上差异很大（域偏移）
  - `thresholds.confidence` 与 `thresholds.margin` 是否过严导致大量 `UNKNOWN`
  - 是否需要把摄像头分辨率/帧率固定到更稳定的输入（见 `config/camera_video_frames.yaml` 的 `camera_width/camera_height/camera_fps`）

## 7. 进阶：用“骨骼时序模型”直接做视频识别（可选）

如果你希望模型显式利用“动作变化”而不是只看单帧，可以走 Pose 时序模型路线：

1) 从视频生成骨骼序列缓存（每个视频一个 `.npz`，包含每帧 33 点坐标展平后的向量序列）：

```powershell
conda run -n TrafficCopilot python scripts/extract_pose_sequences_from_videos.py --split train
conda run -n TrafficCopilot python scripts/extract_pose_sequences_from_videos.py --split test
```

2) 训练 Pose-GRU 时序模型（你可自行运行）：

```powershell
conda run -n TrafficCopilot python scripts/train_pose_sequence_classifier.py --config config/pose_sequence_classifier.yaml
```

3) 启动 Web 演示（按帧喂入，内部维护 clip_len 滑窗后输出预测）：

```powershell
conda run -n TrafficCopilot python web/app_pose_sequence.py
```
