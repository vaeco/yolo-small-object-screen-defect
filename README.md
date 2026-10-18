# YOLO Small Object Review and Screen Defect Detection

第一作者 YOLO 小目标综述与车载屏幕异常识别算法实践的整理仓库。个人贡献为综述写作，以及参与 YOLO 模块设计、适配和实验跑通；不包含个人独立开发完整桌面系统的声明。

## 结构与来源

源码为已有修改版 Ultralytics，`__version__` 为 8.2.14。保留上游 AGPL-3.0 声明。网络入口在 `yolohg.yaml`，HG 在 `ultralytics/nn/modules/block.py`，模块解析在 `ultralytics/nn/tasks.py`。

报告混用了 YOLOv11/8.2.18 等名称，最终检查点和日志未找到，统一按“改进 YOLO”描述。综述 DOI: 10.32604/cmc.2025.074191。

## 运行条件

使用独立环境并安装匹配的 torch/torchvision。源码依赖见 requirements.txt；不要用另外的 pip ultralytics 覆盖本地修改树。源码较旧，新版 torch 对 checkpoint 反序列化的行为可能不同，不加载来源不明的检查点。

```bash
python train.py val --model /path/to/original_best.pt --data configs/screen.local.yaml --device cuda:0
python train.py predict --model /path/to/original_best.pt --source /path/to/images --device cuda:0
```

复制 `configs/screen.example.yaml` 并按真实类别编号与本地数据修改。默认 YAML 中的 80 类来自原模型配置；训练前必须与真实五类配置对齐。当前命令是接口示例，不表示有可用原权重。

## 本次整理

替换了原 train.py 中硬编码 BCCD/VisDrone 路径，增加显式 mode/model/data/source/device 参数。历史报告记录 mAP50 99.5%、mAP50–95 84.9%，但正文有指标对调且缺少验证日志；主简历不采用该未核准数值。本轮不训练、不宣称验证模型可运行。


## 阅读与面试

- [项目档案](docs/project_guide.md)
- [核心问题与递进追问](docs/interview_qa.md)

## 本次整理的验证边界

采用历史材料，不重跑训练。Python 语法检查不能证明模型效果或历史指标已复现。缺失资源明确保留；源码整理和历史实验结果分别记录。


## 静态发现的网络接线缺口

HG返回列表，而所给YAML后续Concat直接引用此输出；parse_model对HG仅pass，缺少完整的多输出通道/选支处理。当前可能是中间源码，本轮未验证网络可构建，不按猜测重写最终架构。
