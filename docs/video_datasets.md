# 可用视频数据集（交警/交通指挥手势）

本页整理适合 TrafficCopilot 的公开视频数据集与补充数据来源，重点关注：连续视频、帧级或时间戳标注、可用于“连续识别 + OoD/Unknown”场景。

## 1) Chinese Traffic Police Gesture Dataset（CTPGesture v1 / v2）

- 形式：连续 RGB 视频 + 标注文件
- 标注：
  - v1：每帧手势标签（CSV）
  - v2：手势 + 身体朝向（时间戳与伪帧标注均提供）
- 类别：与中国交警指挥语义贴合（含 stop/forward/left turn/left turn waiting/right turn/lane changing/slow down/pull over 等）
- 适配价值：非常适合你们下一步的“视频训练 + 时序意图解析 + Unknown/OoD”路线
- 下载：见作者整理仓库（含 Google Drive 链接）

建议落盘位置：

```text
data/raw/videos/ctpgesture_v1/
data/raw/videos/ctpgesture_v2/
```

## 2) Traffic Gesture Dataset（Uni Ulm，含连续记录与 OoD）

- 形式：以雷达为主，伴随相机与运动真值；另提供连续记录与 OoD/Continuous classification 子集（用于验证连续识别与 OoD）
- 适配价值：如果你们后续要强调“连续识别 + Unknown/OoD”，它的连续子集很有参考意义（即便只用其中的连续标签协议/评测方式）
- 下载：官方页面提供公开下载入口

建议落盘位置：

```text
data/raw/videos/uni_ulm_traffic_gesture/
```

## 3) 补充：图片数据集（用于预训练/补齐类别）

- Kaggle 上存在一些交警手势图片分类数据集（多为图像分类，不一定有视频与帧标注）
- 用途：可用于先做图像 backbone 预训练，再用 CTPGesture 视频做时序微调

## 4) 自建数据（强烈建议）

公开数据集通常存在“拍摄角度单一/背景干净/姿态标准化”等偏差。为了提升真实路口可用性，建议自建一小份“路口视角”视频：

- 拍摄：手机 1080p/30fps，距离覆盖 5m/10m/20m，包含逆光/夜间/雨雾（可用喷水/雾化器模拟）
- 标注：以 1-2 秒为片段标注（start/end），并保留 `video_id`，确保按 `video_id` 分组切分避免泄漏
- 目标：每类 10-30 段短片即可显著提升“工程可用性”与答辩说服力

