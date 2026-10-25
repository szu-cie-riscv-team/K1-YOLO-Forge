# K1-YOLO-Forge

在 Linux/WSL 中，从 YOLO 格式标注数据完成检查、训练、九输出 ONNX 导出、官方 XSlim 静态 INT8 量化、精度比较与部署打包。

**状态：首版实现，已通过端到端验证。** 用户可选 `fast`（默认）和 `normal` 两套训练量化方案。`fast` 按公开量化 ONNX 近似重建 43-Conv/LeakyReLU/RepConv 部署拓扑；用户实测其 K1 推理速度接近 Gitee 参考模型。`normal` 保留标准 YOLOv8n 与此前 XSlim 配方。详见[使用说明](docs/usage.md)和[验证记录](docs/implementation-validation.md)。

## 快速开始：已有 200 train / 50 val / 500 calibration

在 WSL2 Ubuntu 内操作，建议仓库和数据放在 Linux 文件系统，例如 `~/projects/yolo`。

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/szu-cie-riscv-team/K1-YOLO-Forge.git
cd K1-YOLO-Forge

# 仓库根目录；首次安装系统组件需 sudo，仅 --install-system 会执行 apt。
bash scripts/setup_env.sh --profile auto --install-system

# 修改真实类别名称；此示例保留已有划分，核对 200/50/0 张。
cp configs/wsl-provided.yaml config.yaml

# 放入下方目录规定的图像、标签、类别表和采集组信息后：
bash scripts/check_env.sh --config config.yaml
bash scripts/download_assets.sh --config config.yaml
bash scripts/prepare_data.sh --config config.yaml

# 检查数据报告及标注预览，再开始训练。
bash scripts/run_pipeline.sh --config config.yaml --from train --to package
```

`setup_env.sh` 采用独立的训练/导出、XSlim 环境，不改系统 Python。完整环境需要 Python 3.11/3.12；Ubuntu 22.04 默认 Python 3.10 时，会通过 uv 安装用户目录下的 Python 3.11。GPU profile 使用 PyTorch 2.8.0 的官方 cu126 wheel；CPU profile 使用官方 CPU wheel。`--profile cpu` 强制 CPU，`--profile cuda` 要求 GPU 真正可用，`--profile inspect` 仅安装参考模型检查环境。

```text
data/
├── raw/
│   ├── images/train/        # 200 张原图，例如 1024×1024
│   ├── images/val/          # 50 张原图
│   ├── labels/train/        # 每图一个 YOLO detection TXT
│   ├── labels/val/
│   ├── classes.txt          # 自定义局部 ID 的名称顺序
│   └── groups.csv           # image,group；image 为 images/train/xxx.jpg 等路径
└── calibration/images/     # 500 张独立校准图，无需标签
```

原始图像不修改。训练张量、FP32/INT8 评估和校准分别按配置尺寸处理。`groups.csv` 用于隔离采集批次；没有分组信息时可以显式设 `data.split_by: image`，报告会说明未检查采集组，仍检查跨集重复像素。

## 配置与类别模式

根目录 [config.example.yaml](config.example.yaml) 是完整参数表；使用者在 `config.yaml` 中覆盖需要调整的项。未知参数报错，完整解析值随实验保存。

- `model.input_hw: [高, 宽]`：统一默认尺寸，正的 32 倍数。`train.input_hw` / `export.input_hw` 可分别覆盖。
- `workflow.scheme`：选择 `fast` 或 `normal`，默认 `fast`。方案会自动选择架构和对应 XSlim 配方；训练与量化参数在 `config.example.yaml` 的 `workflow.profiles` 中集中调整。切换方案后需重新训练，不能混用另一方案的 checkpoint。
- `data.split.mode: provided`：保留已有 train/val/test，仅核对 `expected_counts`。
- `counts`：从未划分的 images/labels 中按张数选择，先隔离组，再抽样；数量不足报错。
- `ratios`：按比例选择；整组隔离后无法满足数量时说明原因，不拆组跨集。
- `classes.mode: custom_only`：N 类紧凑头，N 为 custom_names 长度。
- `classes.mode: replace_coco`：保留 COCO 0-based 编号，替换指定槽位，仍为 80 类。见 [替换配置](configs/replace-coco.yaml)。当前此模式要求 provided 划分及原类回放/独立原类验证数据；最终张数包含混合数据。
- `evaluation.max_map_drop`：可接受的量化绝对 mAP50-95 损失；默认 null，报告不宣称业务精度达标。

静态兼容路线固定 opset 17、batch=1、float32 RGB NCHW、letterbox 114、1/255、九输出和外部 DFL/NMS。改变这些合同需实现和验证对应代码，当前不接受任意算子/自定义 domain。

## 阶段、重跑与续训

```bash
bash scripts/train.sh --config config.yaml --run-id RUN_ID
bash scripts/export.sh --config config.yaml --run-id RUN_ID
bash scripts/quantize.sh --config config.yaml --run-id RUN_ID
bash scripts/evaluate.sh --config config.yaml --run-id RUN_ID
bash scripts/package.sh --config config.yaml --run-id RUN_ID

# 独立测试必须有 test 数据，不会自动改用 val。
bash scripts/evaluate.sh --config config.yaml --run-id RUN_ID --split test
```

prepare 默认建立新实验，其余单阶段命令默认使用该项目最新实验；推荐明确传 `--run-id`。流水线失败后修正问题，使用同一编号从失败阶段继续。每阶段保存状态、配置、输入哈希和日志；成功且输入相同会复用结果，数据/配置/源码变化会失效。`--force` 归档原阶段后重跑，旧部署目录也会保留。

断点续训使用同一 run-id，将 `train.resume_from` 指向中断时的 `train/weights/last.pt`。只接受仍包含优化器/进度的未完成 checkpoint；正常训练完成后已 stripped 的 last.pt 用于新一轮微调，不作为优化器断点。

比较新导出尺寸/量化配方时，先准备新实验，再复用原 checkpoint：

```bash
bash scripts/prepare_data.sh --config configs/my-new-size.yaml --run-id size-comparison
bash scripts/run_pipeline.sh --config configs/my-new-size.yaml --run-id size-comparison \
  --from export --to package --checkpoint /absolute/path/to/best.pt
```

首版一次运行一个量化配方。不同配方应使用相同数据划分/校准清单，在独立实验中比较；不自动穷举或宣称找到全局最优。

## 输出与 K1 使用

`outputs/<项目>/<run-id>/` 保存数据清单、预览、best/last、FP32、XSlim JSON/日志、量化模型、逐类评估及环境版本。`dist/<项目>/<run-id>/` 和同名 ZIP 包含模型、标签、推理配置、manifest、评估摘要及独立 N 类推理程序。

在已安装匹配官方 runtime、Python SpaceMIT ORT、NumPy/OpenCV 的 K1 上：

```bash
cd /path/to/unpacked-model-package
python3 run_k1.py --image /path/to/image.jpg --report k1-result.json --profile
```

默认必须初始化 SpaceMITExecutionProvider，拒绝整体退回 CPU；EP 初始化成功仍不证明所有网络算子进入 EP，应另看分图/profile。显式 `--cpu` 可用于主机包检查。推理报告不修改包中的原始验收状态。

**现有 riscv ROS 解码器写死了 80 类。** 包内独立推理器支持 N 类；接入该 ROS 项目时仍须修改引擎 shape 校验、标签和后处理循环。本工程没有修改机器人源码或板端依赖。

## 文档与验证

- [参数、数据、环境及常见问题](docs/usage.md)
- [验证记录与限制](docs/implementation-validation.md)
- [原流程与设计依据](docs/pipeline-design.md)
- [WSL 200/50/500 场景](docs/wsl-walkthrough.md)
- [K1/XSlim 调研](docs/research/2026-10-17-k1-xslim.md)
- [Gitee SpacemiT YOLOv8 对照与配置映射](docs/research/2026-10-19-gitee-yolov8-alignment.md)
- [参考模型结构证据](docs/research/reference-models.json)
- [依赖与许可来源](THIRD_PARTY.md)

最小主机验证可运行 `python -m pytest -q tests/test_contracts.py`。合成烟测数据由 `tests/make_smoke_data.py` 生成；它验证数据流，不证明实物识别准确率。
