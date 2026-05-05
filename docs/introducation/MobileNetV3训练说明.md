# MobileNetV3 训练说明

## 目标

基于仓库内现有数据集 `demo/traffic gestures.v1i.multiclass`，训练一个 `MobileNetV3-Small` 手势分类基线模型。

## 新增文件

- `scripts/train_mobilenetv3.py`
- `config/mobilenetv3_baseline.yaml`
- `core/model/mobilenetv3_classifier.py`
- `core/utils/dataset.py`
- `core/utils/training.py`

## 数据集格式

当前脚本直接读取 Roboflow 导出的多分类格式：

- `train/_classes.csv`
- `valid/_classes.csv`
- `test/_classes.csv`

每一行通过 one-hot 列确定类别，脚本会自动解析为单标签分类任务。

## 安装依赖

至少需要：

```bash
pip install torch torchvision pyyaml pillow numpy
```

如果你要固定安装到仓库环境，也可以更新 `requirements.txt` 后执行：

```bash
pip install -r requirements.txt
```

## 启动训练

在仓库根目录执行：

```bash
python scripts/train_mobilenetv3.py --config config/mobilenetv3_baseline.yaml
```

## 输出结果

默认保存到：

```text
experiments/mobilenetv3_baseline/
├── best_model.pth
├── last_model.pth
├── training_log.csv
├── metrics.json
└── class_names.json
```

## 当前配置

- 骨干网络：`MobileNetV3-Small`
- 输入尺寸：`224`
- epoch：`20`
- batch size：`16`
- 优化器：`AdamW`
- 学习率调度：`CosineAnnealingLR`
- 类别不平衡处理：启用 `class weights`

## 你下一步最可能会做的事

1. 先在本机装好 `torch` 和 `torchvision`
2. 跑通 baseline 训练
3. 看 `metrics.json` 和 `training_log.csv`
4. 再决定是否加入注意力模块或更强增强策略
