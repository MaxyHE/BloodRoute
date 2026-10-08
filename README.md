# BloodRoute：血细胞 VLM 适配与选择性复核

用语言侧 LoRA 将通用视觉语言模型适配到血细胞八分类，再用 CNN 的置信度决定哪些图像需要交给 VLM 复核。

项目基于 **BloodMNIST 224、Qwen3.5-4B 和 WideResNet-50-2**，包含数据构建、答案监督微调、CNN 对照、开发集选模与门控策略、真实串行路由测速、配对预测回放，以及 rank 消融。核心问题是：**少量可训练参数能带来多少领域适配收益？CNN 与适配后的 VLM 能否通过选择性调用形成互补？**

[主要结果](#主要结果) · [方法设计](#方法设计) · [Rank 对照](#rank-对照) · [开始运行](#开始运行) · [实验协议](docs/PROTOCOL.md)

## 主要结果

同一批 **3,421 张官方测试图**，训练后的模型和路由策略均由开发集选定。原始 Qwen 与 LoRA 使用相同提示词、图像处理和解码设置。

| 方法 | Accuracy | Macro-F1 | VLM 调用比例 |
|---|---:|---:|---:|
| 原始 Qwen3.5-4B | 58.52% | 55.40% | 100% |
| Qwen3.5-4B + 完整训练集 LoRA | **97.19%** | **97.01%** | 100% |
| 冻结 WR50 特征 + 线性分类器 | 93.66% | 93.25% | 0% |
| CNN–VLM 真实串行路由 | **97.49%** | **97.49%** | **23.41%** |
| 全量微调 WR50 | **98.07%** | **98.11%** | 0% |

![五种方法的官方测试表现与 VLM 调用比例](assets/test-results.svg)

- **领域适配：** 使用官方完整训练集 11,959 张图，保留自然类别比例，仅更新 458,752 个参数；相较未微调 VLM，准确率从 58.52% 提升至 **97.19%**。
- **选择性复核：** 仅 801/3,421 张图调用 VLM，准确率从冻结 CNN 的 93.66% 提升至 **97.49%**；纠正 164 张、引入 33 张错误，净增加 131 个正确预测。
- **实际推理成本：** 同一 RTX 3090、batch=1、并发=1、无预取，端到端平均延迟由全量 VLM 的 **533.09 ms** 降至路由的 **139.04 ms**，降低 **73.92%**。

| 实际推理路径 | 平均延迟 | P95 | VLM 调用 |
|---|---:|---:|---:|
| 冻结 CNN | 10.61 ms | 16.57 ms | 0% |
| 全量微调 VLM | 533.09 ms | 662.72 ms | 100% |
| CNN → 条件 VLM | **139.04 ms** | 651.80 ms | **23.41%** |

计时从读图到最终类别输出，显式同步 CUDA，排除模型加载、5 张暖机和预测写文件；OS 文件缓存未清空。P95 仍约 0.65 秒，接管样本仍需运行 VLM。

**历史全量微调 WR50 的 98.07% 仍是最强固定分类对照。** 最新 VLM 使用 11,959 张训练图；冻结 CNN 和历史全量微调 CNN 使用 4,000 张。该结果展示领域适配与选择性调用，不能解释为同训练量的纯模型优劣比较。历史 4,000 图 LoRA 的 94.68%、离线路由的 96.32% 与 13.21% 调用率保留在[历史记录](docs/EXPERIMENTS.md)。VisA 回归和 rank 消融也对应旧训练版本。

完整指标与配对错误见 [实验结果](docs/EXPERIMENTS.md)，机器可读汇总见 [主实验指标](results/metrics.json)。

## 方法设计

![领域适配与 CNN–VLM 置信度门控流程](assets/method-overview.svg)

### 1. 用答案监督完成领域适配

每张图像配一个固定的八分类问题，以官方类别名作为答案。冻结 Qwen3.5-4B 原始参数，在语言侧 8 个 full-attention 层的 `q_proj`、`v_proj` 上加入 LoRA；主实验采用 `rank=4`、`alpha=8`、`dropout=0.05`。训练只监督答案及 EOS，视觉编码器保持冻结。

最新实验从 base 新建 LoRA，使用官方全部 11,959 张训练图和固定 400 张开发图，训练两轮共 23,918 步；前一轮 lr=2e-4，后一轮 lr=5e-5，每 3,000 步及末步评估，按开发集 Macro-F1 选定 step_18000。旧 2,000→4,000 图热启动实验独立保留。

### 2. 用 CNN 置信度决定是否复核

ImageNet 预训练 WR50 提取 2,048 维冻结特征，`StandardScaler + LogisticRegression` 在历史 4,000 张训练图上拟合，输出八类概率。开发集比较最高类别概率与前两类概率差，选择并冻结门控规则：

```text
CNN 最高类别概率 < 阈值 τ  → 采用适配 VLM 的预测
CNN 最高类别概率 ≥ 阈值 τ  → 保留 CNN 的预测
```

新适配器对应阈值约为 **0.9950**，精确值、调用预算和选型规则见 [新路由协议](docs/FULLTRAIN_REPRODUCE.md#门控选择与实际串行推理)。门控直接使用 CNN 概率，无需额外训练路由模型；开发集选定策略后，既保存配对预测回放，也实际运行 CNN 与条件 VLM。batch/TF32 数值差异导致历史回放 800 次与实际 801 次调用相差一张，最终分类一致；两种记录分别保留。

### 3. 用对照实验检验方案

历史全量微调 WR50 提供 4,000 图专用分类器对照；独立 rank 实验检验 LoRA 容量；VisA 工业图像子集用于补充外域回归检查。三类实验分别回答分类性能、参数选择和外域变化，结果分别记录。

## Rank 对照

`rank=2/4/8` 三组均从同一 Qwen base 新建 LoRA，固定训练 4,000 图、开发 400 图、seed=42、`alpha/r=2`，各训练 8,000 步。按开发集 macro-F1 选择 checkpoint。

| Rank | Alpha | 可训练参数 | 最佳 step | Dev Macro-F1 | Dev Accuracy |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 229,376 | 8,000 | 96.9708% | 97.00% |
| 4 | 8 | 458,752 | 5,000 | **97.0140%** | 97.00% |
| 8 | 16 | 917,504 | 8,000 | 96.6821% | 96.75% |

r=2 与 r=4 接近，r=8 在本轮对照中没有带来收益。r=4 保留为默认配置；r=2 是更省参数的近似可行点。r=4 与 r=2 的 macro-F1 差距仅 0.0432 个百分点，当前单 seed、400 图开发集支持的是这一容量取舍。

这是独立的**从 base 开始训练、只评估开发集**的实验；首页测试结果属于后续完整训练集模型，不沿用 rank 实验的开发成绩。详细设置和聚合混淆矩阵见 [Rank 消融](docs/RANK_ABLATION.md)。

## 开始运行

参考环境为 **Python 3.10、PyTorch 2.4.0+cu121、单张 RTX 3090**。准备 BloodMNIST 224、Qwen3.5-4B 本地权重和 WR50 ImageNet 权重后，从仓库根目录执行：

```bash
git clone https://github.com/MaxyHE/BloodRoute.git
cd BloodRoute
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export MODEL_DIR=/path/to/Qwen3.5-4B
```

最新完整训练集与串行测速见[新实验复现](docs/FULLTRAIN_REPRODUCE.md)。[历史复现步骤](docs/REPRODUCE.md)按顺序提供：数据构建 → VLM 试训与续训 → 固定模型测试 → CNN 对照 → 路由回放。GPU 环境的 PyTorch 安装需与本机驱动匹配，Slurm 示例位于 [configs/slurm/gpu.sbatch](configs/slurm/gpu.sbatch)。

无需模型的路由单元测试：

```bash
python -m unittest discover -s tests -v
```

测试覆盖严格阈值边界、无效预测计错和样本 ID 配对。无需模型的最新结果重算：

```bash
python scripts/recompute_fulltrain_metrics.py
```

仓库公开最新逐图类别、调用记录与计时，可独立核算指标；图像和模型权重需自行准备。

## 代码与实验记录

| 入口 | 内容 |
|---|---|
| [scripts/](scripts/) | Qwen 训练、固定模型评估、路由回放、配图生成 |
| [medical_blood_pilot/](medical_blood_pilot/) | 数据导出、来源索引、冻结与全量微调 CNN 对照 |
| [评估协议](docs/PROTOCOL.md) | 数据隔离、答案解析、选模和门控规则 |
| [实验结果](docs/EXPERIMENTS.md) | 五种方法、配对纠错、VisA 回归及耗时记录 |
| [Rank 消融](docs/RANK_ABLATION.md) | 独立容量对照、固定设置和结果解释 |
| [复现步骤](docs/REPRODUCE.md) | 主实验命令、依赖和输入输出路径 |
| [results/](results/) | 主实验、rank 对照和外域检查的聚合指标 |

技术栈：PyTorch / Transformers / PEFT、torchvision、scikit-learn、NumPy / Pillow、Slurm 与 Matplotlib。

## 补充评估与研究范围

在固定 723 张 VisA 工业图像子集（643 正常 / 80 异常）上，旧 4,000 图血细胞 LoRA 的 balanced accuracy 为 53.13% → 55.00%，异常召回为 6.25% → 10.00%；逐图比较纠正 3 张、没有新增错误。该子集未观察到退化，但两者异常召回均低；这项检查只提供有限外域证据。详见 [VisA 实验记录](docs/EXPERIMENTS.md#外域回归visa-工业图像二分类)。

BloodRoute 是公开数据上的八类细胞形态分类研究，疾病诊断需要独立临床验证。后续工作包括跨 seed 与作用层对照、更广泛的外域评估，以及更多并发与硬件条件下的推理测量。

## License

代码采用 [MIT License](LICENSE)。数据来源：[MedMNIST / BloodMNIST](https://medmnist.com/)，使用与引用遵循其 CC BY 4.0 要求；模型权重遵循各自许可。
