# Gitee SpacemiT YOLOv8 示例对照

日期：2026-10-19。对照上游 [`spacemit-demo/examples/CV/yolov8`](https://gitee.com/bianbu/spacemit-demo/tree/main/examples/CV/yolov8) 的 README、模型下载脚本、量化 JSON、Python/C++ demo，以及本项目当前训练/量化实现和已有 ONNX 结构检查报告。

## 上游公开了什么

上游 README 将 `yolov8n_192x320.q.onnx` 和 `yolov8n_320x320.q.onnx` 标为“非官方模型量化后的模型”，提供下载与推理 demo；README 没有公开这些模型对应的训练代码、训练数据、浮点 checkpoint 或完整训练超参数。README 同时区分 Ultralytics 标准输出 `(1,84,2100)` 和非官方模型的三个尺度、每尺度三个输出。因此，不能把它理解为与本项目“同一训练模型，只是量化参数不同”。

本地 ONNX 静态检查显示，高速 320 模型与本项目 `last.pt` 导出图的输入均为 `[1,3,320,320]` float32、opset 17，输出也都是三个尺度共九个张量；但分类通道分别为 80 和 1。高速图包含 43 个 Conv、37 个 LeakyRelu、93/136 个 QuantizeLinear/DequantizeLinear，卷积权重约 181.5 万个；本项目图包含 63 个 Conv、60 个 Sigmoid、57 个 Mul、220/283 个 Q/DQ，卷积权重约 300 万个。高速图可见 `rbr_reparam` 节点，表明其部署拓扑与本项目标准 YOLOv8n 不同。参考模型 IR=8，本项目 IR=9；二者 opset 都为 17。

结构数据完整记录在[参考模型结构报告](reference-models.json)和本项目生成的[对比检查报告](../../outputs/pharmacy_yolo/local-redbox-last/comparison-320.json)。拓扑差别不会因改量化参数或 opset 而消失。

## 本项目训练路线

本项目使用 Ultralytics 8.3.200 的标准 YOLOv8n 预训练权重，按项目类别配置训练；本机这次是 `custom_only: red_box`、15 张训练图、5 张验证图、320×320，CPU batch=4。用户选择了训练产物 `last.pt` 作为导出 checkpoint。Gitee 示例未提供高速模型的同等训练流程，因此当前训练无法声称复现参考模型。

项目训练与推理预处理保持一致：RGB、letterbox、填充值 114、除以 255。上游 Python demo 使用黑色填充及 `cv2.normalize(..., NORM_MINMAX)`；C++ demo 使用黑色填充及 `/255`，两份 demo 自身也不完全一致。上游 C++ 还固定 `classNum=80`；部署本项目 1 类模型时，必须改为模型真实类别数。参见[上游 Python demo](https://gitee.com/bianbu/spacemit-demo/blob/main/examples/CV/yolov8/python/utils.py)与[C++ demo](https://gitee.com/bianbu/spacemit-demo/blob/main/examples/CV/yolov8/cpp/yolov8.cc)。

## 量化配方映射

上游公开的 [`xquant_config.json`](https://gitee.com/bianbu/spacemit-demo/blob/main/examples/CV/yolov8/model/xquant_config.json) 设置 `calibration_type=minmax`、`finetune_level=3`、`max_percentile=0.9995`、RGB 和 mean/std 0/255。它没有显式设置 `precision_level`。文件中的 `onnx_model` 写为 `yolov8n_det_320.q.onnx`，不能仅凭此配置确认它是否精确生成了下载的两个 q 模型。

本项目将其映射为以下 XSlim 2.1.2 参数：

| 参数 | 项目当前默认 | 说明 |
|---|---:|---|
| 工具 | XSlim 2.1.2 | 上游量化模型元数据记录 Xquant 2.0.1；不切换/混装旧依赖 |
| `calibration_type` | `minmax` | 与公开 JSON 一致 |
| `precision_level` | `0` | 上游未设置；采用 XSlim 2.1.2 的静态量化默认级别，不能证明两工具行为等价 |
| `finetune_level` | `3` | 与公开 JSON 一致；影响量化耗时，不代表部署推理速度 |
| `max_percentile` | `0.9995` | 与公开 JSON 一致 |
| 校准图 | 用户指定目录，最多 500 张 | 当前本机配置实际取 91 张；上游另链向 COCO 校准集。量化集应反映目标部署输入分布 |
| 预处理 | RGB、114 letterbox、`/255` | 保持与本项目训练和部署一致；不照搬上游 demo 的黑色 padding |
| 输入/输出合同 | opset 17、静态 batch=1、九输出、float32 I/O | 已有项目 K1 合同；不修改为其他输出形式 |

这是“上游量化参数的 XSlim 映射候选”，不是对上游模型、xquant 版本或量化结果的逐位复现。p1/f2 配方继续作为 XSlim 官方示例对照项。比较时必须使用同一 FP32 checkpoint、同一校准清单和相同 K1 软件/计时设置，并记录精度、EP 子图分配和延迟。

## 这次文档/设置变更的边界

- 更新默认配置，使 XSlim 生成配置包含 `minmax`、level 0、finetune level 3 和 0.9995 percentile。
- 保留 XSlim 2.1.2、opset 17、预处理合同、模型结构和类别处理方式；不把预量化的 `.q.onnx` 作为量化输入。
- 不声称此参数映射会提高准确率或速度。尚未用新默认值重新量化本项目 `last.pt`，也未完成 K1 板端对照。
- 参考训练配方和权重仍未公开；项目现提供根据 q ONNX 复现的可训练近似架构，见下方补充审计与验证说明。

## 2026-10-19 补充：训练资产与 43-Conv 图审计

### 1. 参考模型的训练来源

检查了 Gitee 仓库 `examples/CV/yolov8` 的公开目录和文件：该目录只有 `cpp/`、`data/`、`images/`、`model/`、`python/` 和 `README.md`。`model/` 下公开的是下载两个量化 ONNX 的 shell 脚本和 `xquant_config.json`；没有训练脚本、训练数据配置、网络 YAML、checkpoint 或浮点 ONNX。README 也把这两个下载文件明确标为“非官方模型量化后的模型”，仅给出导出官方模型、量化和 demo 推理的说明。

因此，当前能确认的边界是：量化模型的下载来源、量化配置、推理代码和部署图可查；训练该特定网络的代码、数据集、预训练来源和超参数没有在这个示例目录公开。不能据此断言作者完全没有其他训练仓库，只能说这份公开 demo 没有提供这些训练资产。两个模型的 ONNX 元数据记有 PyTorch 2.5.1、opset 17、Xquant 2.0.1，COCO 80 类和 stride 32。

### 2. 320×320 与 192×320 ONNX 的静态结构

两份本地模型报告在 [`reference-models.json`](reference-models.json)。它们的算子拓扑、卷积权重形状和通道配置一致，改变的是静态空间尺寸及各检测尺度的网格尺寸：

| 项目 | 320×320 | 192×320 |
|---|---:|---:|
| 输入 | `[1,3,320,320]` | `[1,3,192,320]` |
| 检测尺度 | 40×40、20×20、10×10 | 24×40、12×20、6×10 |
| 输出合同 | 每尺度 bbox `[1,64,H,W]`、class `[1,80,H,W]`、score `[1,1,H,W]` | 相同通道合同 |
| Conv 节点 | 43 | 43 |
| LeakyRelu / Sigmoid | 37 / 3 | 37 / 3 |
| Q / DQ | 93 / 136 | 93 / 136 |
| ONNX opset / IR | 17 / 8 | 17 / 8 |

对 320 模型解开权重 Q/DQ 输入后，Conv 权重共 1,815,472 个参数，偏置共 3,264 个，合计 1,818,736 个可训练参数；两个尺寸的各 Conv 权重 shape 相同。可以据此判断这两份是同一通道拓扑的静态分辨率变体，而不是两个不同深度的网络。

按 ONNX 节点名分组，43 个 Conv 可复核为：backbone 的 `model.0`–`model.9` 共 13 个；neck 的 `model.10`、`model.13`–`model.15`、`model.18`–`model.20`、`model.22`–`model.24`、`model.26`、`model.27` 共 12 个；检测头 `model.28` 共 18 个，即 3 个尺度各 6 个卷积（bbox 分支和分类分支各两层特征卷积加一层输出卷积）。三个尺度的输出通道分别为 64（4×16 的 DFL 分布）、80 个类别和 1 个分支分数。Gitee README 将第三个输出解释为该尺度候选框的总分分支。

可直接观察到的结构特征：骨干通道逐级为 16、32、64、96、128；许多骨干/neck 卷积名带 `rbr_reparam`，算子包含 LeakyReLU、MaxPool、Resize、Concat；与标准 Ultralytics YOLOv8n ONNX 的 SiLU（Sigmoid+Mul）和 C2f 图不同。`rbr_reparam` 是 ONNX 节点命名带来的重参数化线索，但单凭量化 ONNX 不能恢复其训练态分支结构、各分支 BN 参数或训练方法。ONNX 也不足以证明每个 `model.N` 层对应的原始 PyTorch/YAML 模块定义，所以上述分组用于描述部署拓扑，不冒充原始训练配置。

当前本地参考 `yolov8n_no_dfl.q.onnx` 是标准 Ultralytics 8.3.200 导出图，包含 63 个 Conv、SiLU 分解出的 60 个 Sigmoid 和 57 个 Mul；它同样是 80 类并具有相同的九路尺度输出合同，但 Conv 权重约 3.0M，明显不同于参考高速模型约 1.819M。两者 opset 都是 17，因此高速模型的结构差异不能归因于 opset 设置。

**可复现结论：** 项目已实现 `yolov8n_k1_fast`，近似复现 backbone/neck 通道宽度、`rbr_reparam` 对应的 RepConv 训练态模块、LeakyReLU、PAN/FPN 路径和三尺度解耦头；80 类导出图的 Conv 权重形状多重集、权重总数 1,815,472 和偏置总数 3,264 与 320×320 参考图一致。训练和量化链路已通过合成数据 smoke，融合导出在 320×320 与 192×320 下均通过 43-Conv 及张量 shape 检查。只把标准 YOLOv8n 的 Conv 数量改成 43，或调整导出/opset/量化配置，都不能复现这张计算图。要做到训练结构和权重精确一致，仍需作者提供训练源码或浮点 checkpoint；当前真实数据精度和 K1 性能还待验证。
