# YOLO Small Object Review and Screen Defect Detection

YOLO小目标检测文献综述与车载屏幕异常检测研究代码。包含修改版Ultralytics源码、HG多尺度特征模块、网络配置，以及训练、验证和预测入口。应用类别包括黑屏、花屏、红屏、蓝屏和白屏。

## 研究与实现

小目标综述按特征增强、检测头、损失函数与多尺度融合归纳方法，并汇总文献中的公开数据集结果。相关论文发表于CMC：[10.32604/cmc.2025.074191](https://doi.org/10.32604/cmc.2025.074191)。文献对比不是本仓库统一训练得到的基准。

算法源码版本标识为Ultralytics8.2.14，配置包含C2f等YOLOv8风格模块。HG将多路尺度特征聚合后输出两路特征；结构与版本说明见 [代码文档](docs/project_guide.md)。

## 安装

使用独立Python环境，先安装匹配设备的PyTorch和torchvision，再执行：

```bash
python -m pip install -r requirements.txt
```

从仓库根目录运行，使用随附的修改版 `ultralytics/`，避免另一安装版本覆盖导入路径。

## 使用

复制 `configs/screen.example.yaml` 为本地数据配置，填写目录与真实类别编号。权重和数据未随仓库分发。

```bash
python train.py val --model /path/to/original_best.pt --data configs/screen.local.yaml --device cuda:0
python train.py predict --model /path/to/original_best.pt --source /path/to/images --device cuda:0
python train.py --help
```

训练入口支持 `train`、`val`、`predict`，显式传入model/data/source/device。示例命令不代表仓库提供了可直接推理的原始检查点。

## 已知限制

HG返回列表，而当前YAML和 `parse_model` 尚缺少完整的多输出选支与通道传播处理；当前修改配置未完成端到端构建验证。原网络配置的 `nc:80` 源自上游，需要与实际5类数据配置对齐。原始检查点与对应验证日志未分发，因此不发布未经核对的检测指标。普通语法检查不验证网络构建或模型性能。

## 文档与目录

- [结构 配置与评价](docs/project_guide.md)
- `train.py` 训练、验证与预测命令入口
- `yolohg.yaml` 网络定义
- `ultralytics/nn/modules/block.py::HG` 跨尺度模块
- `ultralytics/nn/tasks.py::parse_model` 配置解析
- `configs/screen.example.yaml` 五类数据配置示例

## 许可证

本仓库保留Ultralytics的 [AGPL-3.0 License](LICENSE) 与原有源码声明。网络修改与相关分发遵循该许可证。
