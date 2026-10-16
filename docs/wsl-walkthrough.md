# WSL 流程模拟：200 train / 50 val / 500 calibration

本文记录已同意并据此实现的流程。当前可执行命令见根 README；实际验证范围见 implementation-validation.md。假定使用 custom_only，已有正确的 YOLO detection 标签，模型目标分辨率为 320×320。选择保留原类模式时，还需补充旧类回放及独立验证数据。

## 1. 新电脑环境

准备 WSL2 Ubuntu。NVIDIA GPU 路线先由主机安装支持 WSL 的 Windows 驱动；WSL 内用 `nvidia-smi` 和安装后的 PyTorch CUDA 张量运算检查实际可用性。不要在 WSL 内另装 Linux 显卡驱动。[NVIDIA 官方指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

仓库、数据、虚拟环境和运行产物放在 WSL Linux 文件系统，例如 `~/projects/yolo`；这更适合 Linux 数据加载工作负载。[Microsoft 文件系统建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems#file-storage-and-performance-across-file-systems)

在 WSL 内克隆仓库：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/szu-cie-riscv-team/K1-YOLO-Forge.git
cd K1-YOLO-Forge
```

正式安装入口负责检查系统依赖、建立独立训练/导出和量化环境、选择固定的 CPU/CUDA 依赖 profile、运行实际计算检查并保存环境报告。驱动问题和不支持的配置明确报出；指定 CUDA 时不静默切换 CPU。

下载入口依据固定清单取得预训练权重和必要资产，并校验 SHA-256。普通使用者不必自行找 XSlim 源码或手动拼接 pip 依赖。

## 2. 放入数据

```text
data/
├── raw/
│   ├── images/train/      # 200 张 1024×1024 原图
│   ├── images/val/        # 50 张 1024×1024 原图
│   ├── labels/train/      # 对应的 200 份标签，允许已确认的空标签
│   ├── labels/val/        # 对应的 50 份标签
│   ├── classes.txt        # 自定义类别名称，按局部 ID 顺序
│   └── groups.csv         # 图像相对路径与采集组信息，用于检查泄漏
└── calibration/images/    # 500 张校准图片，无需检测标签
```

原始 1024×1024 图像保留。训练与校准时按目标分辨率做同一套 resize/letterbox。对于 320×320 方形输入，方形原图等比例缩小；对于 192×320 输入，需要保持比例并填充，不强行拉伸。YOLO 归一化标签由变换代码处理，不要求用户手动重标。

校准图片应代表真实部署场景，并检查与验证集的重复/重叠。如果校准集使用了验证图片，反复调量化后该验证指标就不能作为独立的最终验收。

## 3. 填配置

复制完整的 config.example.yaml 为 config.yaml，再修改下列片段；其他部分保留。类别名仅作示例，必须与实际标签对应。

```yaml
classes:
  mode: custom_only
  custom_names: [box_a, box_b, box_c, box_d]
model:
  input_hw: [320, 320]
data:
  source_dir: data/raw
  split:
    mode: provided
    expected_counts: {train: 200, val: 50, test: 0}
  calibration_source: directory
  calibration_dir: data/calibration/images
train:
  epochs: 200
  patience: 50
  batch: auto
  device: auto
quantize:
  calibration:
    max_samples: 500
```

provided 模式不打乱现有划分，数量核对失败时停止。此场景没有 test，记录独立测试待补，不将 val 改名成 test。

## 4. 数据检查与预览

数据阶段检查实际张数、解码、标签缺失/越界、逐类图片及框数、跨集重复和采集组交叉；校准目录单独统计。生成实际划分清单、模型类别映射、dataset.yaml 和数据报告。

生成原图及模型输入上的标注预览，让用户确认缩小后的物体仍可辨识。只验证合同与预览无法判断 200 张训练图是否足以达到目标精度，结果不足时应补充场景覆盖或调整尺寸。

## 5. 训练

使用 200 张训练图更新权重，50 张验证图用于评估与选 best，500 张校准图不作为训练样本。按 epochs 上限训练，patience 达到时早停。记录实际 batch、设备、每轮损失和逐类指标，保存 best.pt、last.pt、参数快照及预测预览。

增强不会把 200 张原图变成新的独立采集数据。再次训练时，可复用同一划分；断点续训读取 last.pt 的优化器和进度，修改关键任务配置时按新实验处理。

## 6. 导出与量化

选择 best.pt 导出静态 FP32 ONNX：batch=1、输入 [1,3,320,320]、opset=17、九输出、类别通道为 N。核对张量语义和 PyTorch 对照，合格后才量化。

独立量化环境读取这份 FP32 模型和 500 张校准图，使用与部署一致的预处理。工程生成官方 XSlim JSON 和校准清单，显式指定目标 opset；运行后保存 INT8 ONNX、日志和分析报告。对比配方时始终使用同一校准清单。

## 7. 量化后评估与打包

同一批 50 张验证图分别评估 FP32/INT8，使用一致的框解码与评估协议。报告逐类 AP、召回和性能/大小差异，不预填任何虚构结果。按照用户接受的精度损失选择配方；K1 未测时速度状态仍为待验。

模型包包含量化模型、labels.txt、class_map.json、推理配置、manifest、评估摘要和 K1 使用说明。FP32、checkpoint、完整日志留在 outputs，便于追溯。

## 8. K1 验证与重复实验

将部署包复制到已经具备配套官方 runtime 的 K1，先测图片推理与 EP 接管，再核对真实相机识别、延迟与内存。N 类模型先完成现有解码器的类别数适配。电脑端成功量化不自动表示板端验收成功。

要对比 192×320，可复用训练权重、改 export.input_hw 后重新导出→500 图对应尺寸校准→量化→验证→K1 测量；不保证需要重训，也不保证无需重训，以精度结果决定。若改变 train.input_hw，则建立新训练实验。仅改阈值不需要重训或重量化，但需重新评估；更换类别或网络结构通常需要从训练开始。

正式使用体验应收敛为：克隆→安装环境/下载权重→放数据/填配置→检查数据/预览→一键执行训练到打包→查看报告→K1 验收。每个阶段也提供单独入口，失败可从对应阶段重跑。
