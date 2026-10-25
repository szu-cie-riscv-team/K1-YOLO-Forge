# 工程设计与实现顺序

状态：2026-10-17 初始设计记录。首版主体已实施，当前命令、支持范围及验证结论以根 README、usage.md 和 implementation-validation.md 为准；下文“计划”保留设计阶段语境。

## 1. 使用者入口与目录

用户主要编辑根目录 `config.yaml`、放入图像/标签，然后运行 `scripts/` 内命令。Python 逻辑放在 `src/k1_yolo/`，shell 仅做环境选择、路径定位和参数转发，避免两份业务逻辑。

目标结构如下；标为计划的目录和命令在对应功能实施时建立。

```text
yolo/
├── README.md
├── config.yaml                         # 主配置；当前只有 config.example.yaml 草案
├── pyproject.toml                      # 计划：安装核心 Python 包
├── configs/                            # 计划：模式/尺寸/量化配方示例
├── requirements/
│   ├── inspect.txt                     # 已有
│   ├── train-<profile>.lock            # 计划：CPU/CUDA 的完整验证锁
│   └── quant-xslim-<version>.lock      # 计划：独立 XSlim 环境
├── scripts/
│   ├── setup_env.sh                    # 已有：仅检查环境
│   ├── inspect_models.sh               # 已有：模型结构报告
│   ├── download_assets.sh              # 计划：按固定来源和哈希下载
│   ├── prepare_data.sh                 # 计划
│   ├── train.sh                        # 计划
│   ├── export.sh                       # 计划
│   ├── quantize.sh                     # 计划
│   ├── evaluate.sh                     # 计划
│   ├── package.sh                      # 计划
│   └── run_pipeline.sh                 # 计划：全流程/指定起止阶段
├── src/k1_yolo/                        # 计划：按实际步骤分模块
│   ├── config.py / classes.py
│   ├── data.py / preprocess.py
│   ├── train.py / export.py
│   ├── quantize.py / evaluate.py
│   └── package.py / cli.py
├── tools/inspect_models.py             # 已有：开发/审计工具
├── data/                              # 原始、划分后、回放和校准数据
├── models/                            # 计划：reference/、pretrained/，不纳入源码包
├── tests/                             # 计划：只覆盖类别映射、导出和预处理等关键行为
├── docs/                              # 安装、使用、参数、模型契约和调研证据
├── outputs/<run_id>/                   # 计划：各次训练/导出/量化及指标
└── dist/<run_id>/                      # 计划：最终部署包
```

所有路径默认按项目根目录解析，避免取决于启动时的当前目录；第三方工具需要绝对路径时由入口生成。数据和模型目录可配置为外部磁盘，不嵌入本机用户名、归档位置或 `riscv` 绝对路径。

## 2. 两种类别模式

### replace_coco：保留编号并替换槽位

```yaml
classes:
  mode: replace_coco
  replace_coco:
    keep_original: ["0-2", "8-79"]
    custom_slots:
      3: {name: custom_a, annotation_id: 0}
      4: {name: custom_b, annotation_id: 1}
      5: {name: custom_c, annotation_id: 2}
      6: {name: custom_d, annotation_id: 3}
      7: {name: custom_e, annotation_id: 4}
```

范围为闭区间、0-based，保留槽位直接沿用 COCO 名称及原编号；替换槽位采用用户名称。这个例子仍为 80 类，75 个原类 + 5 个自定义类。被替换的旧类退出模型类别语义。

首版要求保留和替换集合无交集且恰好覆盖 0–79；范围越界、重叠、重复名称、同一 annotation_id 多重解释都在数据准备前报错。范围字符串由明确解析器处理，不使用 eval。缺少的槽位不静默填空；若将来需要压缩保留类和新增类的组合，可再增加显式紧凑映射，不混入本模式。

标注人员只使用自定义数据的局部 ID。程序生成映射：

| 数据来源 | 输入 ID | 输出 ID |
|---|---|---|
| custom | 0 | 3 |
| custom | 1 | 4 |
| custom | 2 | 5 |
| custom | 3 | 6 |
| custom | 4 | 7 |
| COCO replay | 转为连续编号后的 0、1、2、8…79 | 保持相同编号 |

预训练迁移需显式检查结构和类别表，不可默认“同为 80 类，整头直接加载即可”。保留最终分类卷积的对应行，替换行重新初始化；共享特征层可能随训练变化，因此仍需原类回放和验证。缺失回放数据时可以形成探索性实验，但不能宣称满足“保留原类能力”的交付目标。

### custom_only：仅用户 N 类

```yaml
classes:
  mode: custom_only
  custom_names: [box_a, box_b, box_c, box_d]
```

此例为 4 类，输出类别编号 0–3；N 是类别列表长度，不是图像数。可以加载预训练骨干和可兼容的回归/特征层，分类头按 N 重建。用户另行选择完全随机初始化时才进行严格意义的从头训练。

导出仍是 9 个输出，通道从 64/80/1 变为 64/N/1。现有机器人解码器须读取一致的 N；最终部署包生成明确的 labels、class map 和模型契约，避免错把 custom class 3 解释为 COCO motorcycle。

## 3. 分阶段流程与产物

| 阶段 | 必须完成的工作 | 可复查产物 |
|---|---|---|
| 环境 | Linux/Python/驱动检查，独立训练与 XSlim 环境 | 完整依赖锁、硬件和版本信息 |
| 下载 | 固定预训练/工具来源，校验并缓存 | URL、revision、SHA-256、许可清单 |
| 数据 | 标签/框/类别检查，分组划分，生成映射 | dataset.yaml、class_map.json、划分清单、统计 |
| 训练 | 权重迁移、训练、断点续训、验证集选 best | best.pt、last.pt、参数快照、逐类指标 |
| 导出 | 固定 shape/opset、九输出、数值对照 | FP32 ONNX、结构契约、PyTorch 对照结果 |
| 校准 | 实际部署预处理，覆盖场景且固定清单 | 当前尺寸的校准清单与哈希 |
| 量化 | 生成并调用官方 XSlim JSON，保存诊断 | INT8 ONNX、实际配置、版本、日志与分析 |
| 评估 | FP32/INT8 同一验证协议比较 | 逐类 AP/召回、混淆、量化损失 |
| 打包 | 模型与标签、预处理、后处理配置一致 | model.onnx、labels.txt、manifest.json、部署说明 |
| 板端 | EP 接管、速度、内存与真实场景 | runtime/profile、P50/P95、验收状态 |

一键入口按明确阶段顺序执行。缓存是否有效由输入模型、配置和数据清单哈希决定，不能仅看文件存在；变更类别或尺寸后，使相关后续阶段失效。训练断点续训与从某一流水线阶段重跑分别处理。

运行目录采用唯一 run_id；保留原始配置和解析后配置，不覆盖上次成功结果。失败记录阶段、命令、版本和日志位置，修正后可以单阶段重跑。不新增任务队列、插件系统或通用工作流框架。

## 4. 默认配置与优化规则

根配置集中暴露用户实际需要调整的项目：类别、路径、模型、训练、导出尺寸、量化与部署阈值。实现内部常量不逐个暴露；XSlim JSON 由主配置生成，连同版本保存。

### 手动分辨率和样本数量

`model.input_hw: [高, 宽]` 是统一分辨率入口；`train.input_hw`、`export.input_hw` 留空时跟随它，也允许分别覆盖。首版 stride=32 路线要求高宽均为正的 32 倍数；满足此条件仍需经过导出和 K1 验证。长方形训练要检查实际 batch 张量并适配训练数据加载，不能直接把二元素列表传入只接受单一 imgsz 的后端后宣称生效。

`data.split.mode` 在 `ratios`、`counts` 和 `provided` 中选择，非当前模式的数值不参与划分。counts 允许手动指定 train/val/test 的图片张数，单位是原始图片，不是框数或增强后样本数。数量作用于最终混合数据集，包含回放图和已确认的背景图；报告同时输出各来源、各类别的图片数与框数。

provided 用于用户已划分好的 `images/train`、`images/val` 和对应 `labels`，保持集合归属；`expected_counts` 只用于核对，不触发重采样。跨集重复、采集组交叉或类别缺失仍会检查。独立量化图片通过 `calibration_source=directory` 和 `calibration_dir` 指定，不自动混入训练/验证，也不要求目标框标签。

counts 模式先隔离采集批次/视频/实物组，再在分配给各集合的图像中抽样。需满足指定数量、来源比例与必要类别覆盖；数据不足或约束冲突时报出原因，不重复图片凑数，不把同一组拆到不同集合。未选中的原图保持原样，保留在未选清单中。固定 seed 和清单以复现实验。

验证集用于调参和选 best，最终 test 保持独立。若数据不足以建立有代表性的 test，可以显式设 test=0 并报告最终独立验收待补，不能悄悄复用 val 当 test。

### 建议保留的手动参数

| 参数 | 调整目的 |
|---|---|
| 类别模式、保留编号、自定义名称 | 定义模型识别任务与部署类别语义 |
| 预训练路径、freeze、resume_from | 选择初始化、微调范围或断点续训 |
| epochs、patience | 控制最长训练时间与早停 |
| batch、device、workers | 适配显存、CPU 与数据读取能力 |
| optimizer、lr0、weight_decay | 控制优化过程；归为进阶项 |
| 颜色、尺度、翻转、mosaic 增强 | 适应真实变化，保持类别语义 |
| replay_train_fraction | 保留原类模式中原类/新类样本的训练权衡 |
| seed、数据组与划分清单 | 固定对照实验的数据条件 |
| 校准样本量、precision_level、finetune_level | 权衡量化精度、计算耗时和部署速度 |
| confidence_threshold、nms_iou_threshold、max_detections | 调整部署时的误检、漏检和相邻目标保留 |

部署置信度不直接复用于标准 AP 评估的预筛选阈值，避免提前过滤低分检测而扭曲评估。opset、输出格式、归一化、padding 和 reg_max 作为兼容配置保留；修改时必须重新验证整个部署合同，不列为日常调优旋钮。

初始值分三类：

- 有兼容依据：opset 17、静态 batch=1、RGB float32 NCHW、letterbox=114、1/255、九输出。
- 有现有 K1 记录：ORT 1/1、EP 4/1 作为首轮线程基准；不是所有部署负载的最优线程数。
- 需要用项目数据调优：epochs、学习率、增强、batch、量化配方、阈值和 320×320/192×320 的取舍。

颜色任务关闭 hsv_h 是业务约束；其余训练参数只是初始配方。按 CUDA 显存自动选择 batch，CPU 使用较小 batch；资源限制由环境检查说明。不能承诺在不同显卡、不同数据上同一个 batch/epochs 全局最优。

量化先比较 p0f2/p1f2 等有限候选。以验证集精度约束排除不可接受模型，再根据 K1 延迟与资源选择；可配置的可接受损失需要结合实物检测任务确定，不擅自把“下降 1%/2%”当作用户已接受。最终 test 用于选定方案的独立报告。

## 5. 从零复用与部署边界

完整环境目标：x86_64 Linux 开发电脑训练/量化，riscv64 K1 推理。先覆盖 Ubuntu 22.04/24.04 的 CPU 和 NVIDIA GPU 路线；其他 Linux 发行版通过能力检查和独立验证扩展，不宣称所有组合已经可用。

正式环境脚本需支持检查/创建独立 venv、固定版本依赖下载、缓存、明确 CPU/CUDA profile、失败重试说明和版本导出；不静默升级系统驱动或 K1 runtime。网络受限的使用者可使用经过哈希核对的 wheelhouse 与模型缓存。Docker 可在原生路线稳定后作为额外分发方式。

训练环境和量化环境通过 FP32 ONNX 交接；K1 板端不安装整套训练依赖。官方 XSlim 是量化主体，工程只负责前后步骤和校验，不另写一套替代量化器。

部署 manifest 至少记录：模型哈希、架构/类别模式、类别顺序、输入 H/W/dtype/布局、预处理规则、九输出顺序与 reg_max、后处理阈值、训练和 XSlim 版本、opset/IR、数据清单哈希、目标 runtime，以及 `host_validation`、`board_validation` 分开的状态。

## 6. 实现顺序

1. 完成调研与结构审计，明确原类替换/N 类合同。
2. 已实现配置校验、两种类别映射、数据检查/划分和 Linux 正式环境安装。
3. 已打通标准 YOLOv8n 的训练→九输出 FP32 导出→任务评估。
4. 已打通官方 XSlim 量化与量化前后精度报告，并提供 N 类部署产物。
5. 已设定 `fast` 为默认方案、`normal` 为兼容方案；用户反馈 fast 在 K1 上的速度接近 Gitee 参考，仍需记录可复核的板端数值和精度结果。
6. 已按 15/5 小数据集训练、量化近似高速架构；更大独立数据上的精度回归仍待完成。
7. 在另一台干净 Linux 上从零复现，完成完整教程与发布包。

上述顺序围绕当前数据和 K1 合同推进；不把复杂蒸馏、多教师或多结构搜索作为首版必要条件。若原类能力或高速结构的实际结果需要这些方法，再有依据地加入。
