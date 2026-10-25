# 首版验证记录

日期：2026-10-17。**标准 YOLOv8n → 官方 XSlim → 部署包的主机端首版已跑通。** 验证主机为 x86_64 WSL2，完整安装使用独立 Python 3.11.16 CPU 环境。

结构化结果见 [validation-summary.json](validation-summary.json)。本机完整日志保存在 outputs/verification-20261017，模型和实验产物保存在下面列出的 outputs/dist 目录；这些可再生的大文件不进入 Git。

## 已验证结果

| 内容 | 结果与范围 |
|---|---|
| 环境从零安装 | 在新建的训练/量化 venv 中通过，Python 3.11；安装脚本选择 CPU profile |
| 固定依赖 | CPU 全部运行依赖已锁定，两个环境 pip/uv 兼容性检查通过；重复安装通过 |
| 官方权重下载 | v8.3.0/yolov8n.pt，6,549,796 字节，SHA-256 与 assets.json 一致 |
| 关键行为检查 | 12 项通过，Python 3.11.16；此前 Python 3.10 的核心行为检查也通过 |
| custom_only | 1 类、12 张 train、4 张 val、12 张 calibration；实际训练张量 64×96 |
| replace_coco | 保留 75 原类、替换 5 类；最终 4 张混合 train、4 张混合 val、12 张 calibration；64×64 |
| FP32 导出 | 两种模式均通过 ONNX checker、九输出 shape 检查、opset 17、参考算子/属性检查和 PyTorch 数值对照 |
| XSlim p1/f2 + minmax | custom_only 全链通过；保存实际 JSON、校准清单及 Graphwise report |
| XSlim p0/f0 + minmax | replace_coco 全链通过 |
| XSlim p0/f0 + default | 单独量化验证通过；本次 CPU 小数据校准约 405 秒，不外推其他设备/大数据耗时 |
| 量化产物 | 两种模式均保持 opset 17、IR 9、float32 I/O 与九输出，内部有 QDQ |
| FP32/INT8 评估 | 同一验证协议执行通过，保存逐类 AP/precision/recall；合成烟测 AP 为 0，不能作为实物准确率证据 |
| 部署 ZIP | 两种模式均生成，包内哈希与 ZIP CRC 校验通过 |
| 独立推理入口 | 打包后的 run_k1.py 使用显式 --cpu 在主机上运行通过；不是 K1 EP 验收 |
| 缓存 | 重跑同一配置和输入，六阶段均正确复用，不重训/重量化 |
| 输入变化拒绝 | 故意修改准备好的标签后，缓存训练拒绝继续；测试后已恢复文件 |
| 独立测试保护 | 没有 test 时明确失败，不替代为 val，原 val 报告保持不变 |
| 中断续训 | 实际在第 1/3 epoch 保存 checkpoint 后中断，恢复剩余 2 epoch 并再次导出通过 |

最终实验：

- outputs/smoke/release-smoke → dist/smoke/release-smoke.zip
- outputs/replace_smoke/release-replace → dist/replace_smoke/release-replace.zip
- outputs/smoke/resume-check：中断及续训证据

## 2026-10-19 高速架构补充验证

新增架构在本机 CPU 上用合成 12 张 train / 4 张 val 数据训练 1 epoch，完成 FP32 导出、12 张合成校准图的官方 XSlim 量化、验证评估和部署打包。整条链路通过；合成集 mAP 为 0，符合该随机合成烟测用途，不提供真实识别效果证据。量化阶段耗时约 2.4 秒，仅代表这组小模型/小校准集。

另对 80 类随机初始化模型的 320×320 和 192×320 静态导出做结构检查；两种尺寸均通过 43 Conv、37 LeakyRelu、三尺度九输出合同，网格尺寸分别为 40/20/10 与 24×40、12×20、6×10。所有这些仍是主机静态/合成数据验证，不是 K1 验收。

随后使用本机 15 张 train / 5 张 val 的 red_box 数据训练了 `yolov8n_k1_fast` 200 epoch，并使用 XSlim 2.1.2、opset 17 和目录中实际检出的 91 张校准图完成量化。导出及量化合同通过；FP32 为 opset 17 / IR 8，量化图为 opset 17 / IR 9，均为 43 Conv、37 LeakyRelu、静态 320×320、九路单类别输出。用户在 K1 上实测反馈 fast 量化模型速度接近 Gitee 参考模型；当前没有保存可复核的逐项延迟数据。5 张验证图上的 best mAP50-95 为 0.832，只能作为这批小数据的训练诊断。

正式提供 `workflow.scheme: fast|normal`：fast 是默认方案，使用 K1-fast 架构和 p0/f3 + minmax + percentile 配方；normal 保留此前标准 YOLOv8n 与 XSlim p1/f2 + default 配方。两套配置集中在 `config.example.yaml` 的 `workflow.profiles`，训练、导出和量化产物都会记录方案名。

## 实施中定位并修复的问题

- ONNX Runtime 1.24.3 没有 Python 3.10 Linux wheel，因此完整安装采用独立 Python 3.11/3.12，并自动处理 Ubuntu 22.04 的默认 3.10。
- XSlim 导入 PyTorch cpp_extension 需要 setuptools，已显式固定并安装。
- C2f.chunk 的原始导出带 Shape/Gather/Slice 等形状计算，改为固定通道的 forward_split；最终主网络为 63 Conv、57 Mul、60 Sigmoid，与标准官方参考模型的对应结构一致。
- 新增 `yolov8n_k1_fast` 近似架构：CPU smoke 构建时检测头 stride 为 8/16/32，重参数化融合后为 43 个 Conv；80 类 ONNX smoke 通过静态检查，包含 37 LeakyRelu、3 Sigmoid、3 ReduceSum+Clip 和参考九输出 shape。合成数据训练和 XSlim smoke 也通过，但不代表真实训练精度或 K1 性能。
- 训练/验证数据最后执行固定高宽变换，标签与 ratio_pad 同步更新；验证禁用上游可变 batch padding，避免矩形输入再次缩小有效内容。
- 续训会覆盖 last.pt，缓存键需记录实际消费的 checkpoint 哈希，不能把续训完成后的 checkpoint 当作当时输入。

## 未验证与明确限制

- 未使用用户的真实药盒数据，未证明真实识别效果或原 COCO 能力保留。
- 没有 CUDA GPU 实测；脚本提供 cu126 路线并实际检查 CUDA 可用性，但不宣称此主机已验证 GPU。
- 未进行另一台独立电脑复现；当前是同一 WSL 主机上的干净环境安装验证。
- 用户已在 K1 上对 fast 量化模型做推理速度实测，反馈接近 Gitee 参考；本验证记录没有可复核的逐项延迟数值。K1 EP 分图、相机、内存和长期运行仍未验证。
- 未修改 riscv ROS 代码。独立推理支持 N 类，现有 ROS 的 80 类限制仍需配套适配。
- `yolov8n_k1_fast` 是按公开部署 ONNX 近似重建；已有小规模真实数据训练和 XSlim 量化记录，但这不证明充分收敛或业务精度；原始训练源码与浮点 checkpoint 未公开。
- 当前保留原类模式仅支持 provided 自定义划分，并需要原类回放和独立原类验证数据。
- AP 是检测 TXT 场景的 COCO-style 101 点协议，不包含 COCO crowd/ignore 元数据。
- fast/normal 两套 XSlim 配方已在配置中分开；normal 复用原 p1/f2 配置，fast 已在本机真实小数据链路上量化。精度选择仍需更多独立数据验证。

本轮没有修改原始三份参考模型、riscv 工程或 K1 板端环境。
