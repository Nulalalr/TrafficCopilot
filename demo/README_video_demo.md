# 视频识别 Demo 启动说明

## 1. 默认演示版

默认演示版继续使用当前主模型，不包含视频帧训练模型切换。

启动命令：

```powershell
cd E:\downloads\traffic-gesturedemo\TrafficCopilot
E:\Anacond\envs\tc_pose\python.exe web\app.py
```

该入口读取：

- `config/web.yaml`
- `config/camera.yaml`
- `config/video.yaml`

## 2. 视频识别独立演示版

视频识别版单独使用视频帧训练得到的 `MobileNetV3 Video Frames` 模型，不影响默认演示版。

启动命令：

```powershell
cd E:\downloads\traffic-gesturedemo\TrafficCopilot
E:\Anacond\envs\tc_pose\python.exe web\app_video_frames.py
```

该入口读取：

- `config/web_video_frames.yaml`
- `config/camera_video_frames.yaml`
- `config/video_video_frames.yaml`

## 3. 说明

- 如果你之前在 PowerShell 里执行过：

```powershell
$env:TRAFFICCOPILOT_WEB_CONFIG="config/web_video_frames.yaml"
```

那么这个终端会一直保留该环境变量，导致 `web/app.py` 也会加载视频版配置。

清除命令：

```powershell
Remove-Item Env:TRAFFICCOPILOT_WEB_CONFIG -ErrorAction SilentlyContinue
```

建议做法：

- 默认版直接运行 `web/app.py`
- 视频版直接运行 `web/app_video_frames.py`
- 不再依赖环境变量切换
