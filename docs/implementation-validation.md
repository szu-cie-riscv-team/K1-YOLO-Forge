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

## 实施中定位并修复的问题

- ONNX Runtime 1.24.3 没有 Python 3.10 Linux wheel，因此完整安装采用独立 Python 3.11/3.12，并自动处理 Ubuntu 22.04 的默认 3.10。
- XSlim 导入 PyTorch cpp_extension 需要 setuptools，已显式固定并安装。
- C2f.chunk 的原始导出带 Shape/Gather/Slice 等形状计算，改为固定通道的 forward_split；最终主网络为 63 Conv、57 Mul、60 Sigmoid，与标准官方参考模型的对应结构一致。
- 训练/验证数据最后执行固定高宽变换，标签与 ratio_pad 同步更新；验证禁用上游可变 batch padding，避免矩形输入再次缩小有效内容。
- 续训会覆盖 last.pt，缓存键需记录实际消费的 checkpoint 哈希，不能把续训完成后的 checkpoint 当作当时输入。

## 未验证与明确限制

- 未使用用户的真实药盒数据，未证明真实识别效果或原 COCO 能力保留。
- 没有 CUDA GPU 实测；脚本提供 cu126 路线并实际检查 CUDA 可用性，但不宣称此主机已验证 GPU。
- 未进行另一台独立电脑复现；当前是同一 WSL 主机上的干净环境安装验证。
- 未进行 K1 EP 接管、分图、相机、延迟/内存和长期测试。
- 未修改 riscv ROS 代码。独立推理支持 N 类，现有 ROS 的 80 类限制仍需配套适配。
- 43-Conv/LeakyReLU 高速参考结构尚无经过验证的训练实现，当前 architecture 只接受 yolov8n_standard。
- 当前保留原类模式仅支持 provided 自定义划分，并需要原类回放和独立原类验证数据。
- AP 是检测 TXT 场景的 COCO-style 101 点协议，不包含 COCO crowd/ignore 元数据。
- 默认 p1/f2 + default 的特定组合尚未单独实测；已分别验证 p1/f2 + minmax 及 p0/f0 + default。配方仍需真实数据精度和 K1 性能选择。

本轮没有修改原始三份参考模型、riscv 工程或 K1 板端环境。
