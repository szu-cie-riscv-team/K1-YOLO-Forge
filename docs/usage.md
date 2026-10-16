# 使用、配置与故障处理

## 环境

完整训练/量化环境以 x86_64 Linux、Python 3.11/3.12 为目标。已验证组合以 implementation-validation.md 为准。WSL2 依然使用 Linux 脚本；原生 Windows Python 不在支持范围。

```bash
bash scripts/setup_env.sh --profile cpu
bash scripts/setup_env.sh --profile cuda
# 可选：虚拟环境放在外部 Linux 磁盘。
bash scripts/setup_env.sh --profile auto --env-root /path/to/envs
```

CPU/CUDA 不混装。训练为 Ultralytics 8.3.200、PyTorch 2.8.0；量化为官方 XSlim 2.1.2。独立环境分别使用 ONNX 1.17 与 1.21、ORT 1.20.1 与 1.24.3，不能直接合并。XSlim 所需 setuptools 也显式安装。完整依赖版本记录在环境内 resolved-requirements.txt；项目提供经实际解析的 CPU lock，CUDA 路线未实测时不伪装成已验证锁。

使用 pip 镜像时设置常规 PIP_INDEX_URL / UV_INDEX_URL；PyTorch 来源仍由脚本显式指定官方 CPU/cu126 索引。离线部署先在相同系统/Python/GPU profile 中下载 wheelhouse，按已记录版本安装并运行 check_env.sh；首版不提供自动跨 ABI 生成离线包。

脚本自动读取 .environments.json，不要求手动激活环境。单阶段也可用 TRAIN_PYTHON、QUANT_PYTHON 覆盖解释器。流水线跨环境运行时读取 .environments.json 中的 quant_python。

## 数据与数量

YOLO detection 标签每行五列：`class_id x_center y_center width height`，坐标归一化，矩形必须落在图像范围内。灰度图会转为三通道；原始图片保持只读。

provided 目录为 images/{train,val,test} 与 labels/{train,val,test}；未划分数据使用 images/ 与 labels/，可保留对应子目录层次。classes.txt 如存在，必须与自定义 annotation_id 顺序一致。

groups.csv 示例：

```csv
image,group
images/train/frame001.jpg,capture-01
images/train/frame002.jpg,capture-01
images/val/frame003.jpg,capture-02
```

组名应跨数据来源保持可比较；同一视频不要在 replay 和 retained_val 中改用不同组名。split_by=capture_group 时每张图片都需要组信息。split_by=image 是用户显式选择不检查采集组，不代表不存在相邻帧泄漏。近重复哈希只产生人工复核候选，不能证明图像是重复的。

counts 的单位是最终采用的图片张数，包含背景图；不是目标框数。ratio/counts 不破坏组隔离，必要时舍弃同组中未选的图片而不分给其他集合。数据不足、无法满足组约束或 train/val 有类别无标签时会停止。

replace_coco 首版支持 provided：custom 图标签使用局部连续编号，replay 和 retained_val 图标签使用连续 COCO 0..79 编号。回放中被替换原类标签被移除，只采用仍有保留类标签的回放图。replay_train_fraction 表示最终训练集的回放比例；程序根据自定义训练图数选择不重复回放图。expected_counts 校验混合后的数量，报告另列各来源。

所有有效模型类别都要求 train 和 val 中出现，避免只保留通道却没有训练/验证证据。原类能力仍需对原始教师/预训练模型另外做业务回归；初始化保留不保证零遗忘。

## 分辨率与增强

model.input_hw 为 [高,宽]，train/export.input_hw 可覆盖。固定输入支持矩形；训练数据最终张量及标签坐标会同步变换，并检查实际形状。export 尺寸改变后必须重新导出、按新尺寸校准并评估。

颜色任务保持 hsv_h=0。可调 epochs、patience、batch、device、workers、optimizer、lr0、weight_decay、freeze、增强和随机种子。auto batch 在 CUDA 采用上游显存估计，在 CPU 使用 4；单张显存不足可显式减小。首版只支持单训练设备，不支持逗号分隔的 DDP 多卡列表。

## 量化与评估

量化输入必须是本流程检查过的 FP32 ONNX，不能再次静态量化参考 .q.onnx。XSlim 以 subprocess 调用，JSON、预处理入口、校准清单和日志均保存。

静态 batch=1 下官方 XSlim 2.1.2 的 calibration_step 限于 10..1000。max_samples 指最多采用的不重复图片数，实际不足会提示；少于 10 张停止。校准源可选训练集或独立图片目录，独立目录不得和 val/test 存在相同解码像素。最终 test 不参与配方选择。

默认 precision_level=1、finetune_level=2；它是官方检测示例起点，不是所有数据的最优配方。p0f0 可用于检查流程，p0f2/p1f2 的速度与精度需实际比较。首版不自动搜索配方。

评估使用模型实际九输出解码，外部 DFL、最佳类别锚点、类别内 NMS。AP 为 IoU .50:.05:.95、101 召回点的 COCO-style 指标，YOLO TXT 不包含 COCO crowd/ignore 语义；不要与不同后处理协议的数字直接比较。AP 的低分预筛选与部署 confidence_threshold 分开设置。

max_map_drop 为 0..1 单位的绝对损失：0.02 表示 mAP 下降 2 个百分点。null 表示未设置业务验收阈值。超阈值时评估保存报告但阶段失败，打包拒绝继续。部署置信度对应的逐类 precision/recall 另列。

## 重跑与调试

--run-id 指定当前实验。prepare 无编号时创建新实验；train/export 等默认读取项目 latest.json。prepare 会实际读取图像和标签，不是只数文件。

输入/输出哈希和源码发生变化时不会静默使用旧结果。需要新实验，或用 --force 归档原阶段后重新运行。源码变化采用保守的全链失效策略；这可能要求从 prepare 重新开始。

断点续训保持原 run-id 和数据；中断时 last.pt 需要仍保留 optimizer。完成训练后被上游 stripped 的 last.pt 不能恢复原优化器进度，应作为新的 model.pretrained 微调。

失败信息包含日志路径。需要 Python 调用栈时：

```bash
K1_YOLO_TRACEBACK=1 bash scripts/run_pipeline.sh --config config.yaml --run-id RUN_ID --from export
```

常见问题：

- **无 CUDA / 驱动版本不兼容**：检查 WSL 主机驱动与 nvidia-smi；改用 --profile cpu 或修复 GPU。脚本不会在 WSL 中装 Linux 显卡驱动。
- **缺标签**：空 TXT 才表示确认过的背景，不能自动把缺文件当背景。
- **类别无样本**：补充该类别，或重新定义任务类别；不能靠给类别改名解决。
- **跨集重复/采集组交叉**：修正原始划分；不要通过改文件名规避。
- **opset/domain/shape 不符**：使用本工程的九输出导出，不能只修改 ONNX 版本数字。
- **内存不足**：降低 batch/workers，分开执行阶段，并检查 WSL 内存上限。
- **pip 依赖冲突**：使用两套独立环境，不要把 ROS/系统 Python 的包混入虚拟环境。

## 开发验证

requirements/dev.txt 提供 pytest。tests/test_contracts.py 覆盖关键合同，不覆盖每个实现分支。tests/make_smoke_data.py 生成合成训练/验证/校准数据，用于短训练及全链验证；mAP 为零也不能据此判断真实任务模型质量。

部署包 run_k1.py 使用 NumPy/OpenCV 和官方 ORT；不依赖整套训练环境。--cpu 是明确的主机烟测选项，默认仍要求 SpaceMIT EP。相机/ROS 接入不包含在本 Python 工程的硬件验收中。
