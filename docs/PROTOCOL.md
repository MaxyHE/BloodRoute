# 评估协议

## 最新完整训练集实验

2026-10-05 的主结果使用全部 11,959 张官方训练图、原固定 400 图开发集和 3,421 图测试集。从 base 新建 LoRA，r=4、alpha=8，视觉与基座冻结，训练两轮 23,918 步；首轮 lr=2e-4，第二轮 lr=5e-5，切换时保留优化器状态，每 3,000 步及末步评估，按 dev macro-F1、accuracy、较早 step 选定 step_18000。CNN 沿用历史 4,000 图冻结特征线性分类器。新阈值 0.9950015136477659 在开发集确定，再用于离线回放和实际串行推理；实际调用为 801/3421。

参考代码和命令见[完整训练集复现](FULLTRAIN_REPRODUCE.md)，新结果与历史对照见[实验记录](EXPERIMENTS.md)。下方描述历史 2,000→4,000 图热启动、独立 rank 和旧路由协议，不作为最新模型的训练设置。

## 数据

使用 MedMNIST+ 的 BloodMNIST 224 文件。官方 train/val/test 分别为 11,959/1,712/3,421 张。该 224 版本来自原始图像的处理，不是将旧 28×28 文件简单放大。

类别 ID 0–7：basophil、eosinophil、erythroblast、immature_granulocytes、lymphocyte、monocyte、neutrophil、platelet。erythroblast 不应解释为通常所说的成熟红细胞。

每类使用 `Random(42 + class_id)` 对官方 train 索引打乱，先选前 250 张构建 2,000 图试训集，再选前 500 张构建 4,000 图扩量集；旧集合是新集合的子集。开发集从官方 val 同法选每类 50 张，共 400 张，整个训练过程中不变。测试使用官方全部 3,421 张；类别数量依次为 244/624/311/579/243/284/666/470。

训练、开发、测试按官方来源及图像 ID 隔离；不声称患者级独立划分。构建脚本保存来源索引和标签，图像、标签不能跨集合混用。

## VLM 训练

固定问题：

```text
Which blood cell type is shown? Answer with exactly one of: basophil; eosinophil; erythroblast; immature_granulocytes; lymphocyte; monocyte; neutrophil; platelet.
```

答案为官方类别名。没有逐图 LLM 教师标注或生成的推理链。

基础模型为 Qwen3.5-4B。LoRA 作用于语言侧 8 个 full-attention 层的 q/v 投影；r=4、alpha=8、dropout=0.05，共 458,752 个可训练参数。原始模型和视觉编码器冻结。BF16、batch=1、AdamW、weight decay=0.01，gradient norm clip=1。

试训 2,000 图，lr=2e-4，共 2 个 epoch，按 dev macro-F1 选择第一轮。扩量至 4,000 图后从该适配器热启动，重建 AdamW，lr=5e-5，共新增 8,000 步，每 1,000 步评估保存。按 macro-F1、accuracy、较早 step 顺序选择，旧适配器也作为 step=0 候选。

参考实验在全部训练结束前固定新增 step_03000 做测试；训练完成后 dev 最优仍是它。公开脚本保留参考协议的 checkpoint 名称和冻结元数据。复现实验的选模必须在其测试前完成，不能根据测试结果更换 checkpoint。

## 独立 rank 对照

rank=2/4/8 的容量对照从同一 base 新建适配器，alpha 分别为 4/8/16，固定 alpha/r=2。三组保持同样的语言侧作用层、dropout、数据、seed 与生成设置，完整训练 8,000 步；前 4,000 步 lr=2e-4，后 4,000 步 lr=5e-5，学习率切换时保留 AdamW 状态。

每 1,000 步在同一 400 图开发集评估，按 macro-F1、accuracy、较早 step 选择 checkpoint。本轮不读取测试集；它是独立的容量实验，主实验仍采用上一节所述热启动适配器。完整记录见 [Rank 消融](RANK_ABLATION.md)。

## VLM 评估

原始模型和适配器均使用同样的处理器、RGB 图像、最小像素 65,536、最大像素 448²、seed=42；thinking=false、greedy、max_new_tokens=32。只有规范八类标签可被解析为有效输出，其余计 invalid 且计错。

Accuracy 统计全部样本；macro-F1 对八类等权；balanced accuracy 为八类 recall 的平均。测试集类别不均衡，不将 accuracy 与 balanced accuracy 混为一谈。

## CNN 对照

冻结基线：ImageNet WR50 去除原分类头，输出 2,048 维特征；只在训练特征拟合 StandardScaler 和 LogisticRegression（C=1、max_iter=1000、lbfgs、seed=42）。CNN 不更新。恢复保存的线性分类器后，先在固定 dev 上核验预测与概率复现，再评估测试集。

补充微调对照：同一 4,000 图，WR50 全主干及八分类头参与训练；backbone lr=1e-5、head lr=1e-4、AdamW wd=1e-4，BF16，batch=32，10 个 epoch；无增强，保持 224 Resize/ToTensor/ImageNet normalization。按 dev macro-F1、accuracy、较早 epoch 选第 6 轮后测试。

## 路由

仅使用 CNN 输出概率进行门控，低分转交固定的血细胞 LoRA。比较 max probability 与 top-2 margin，在 dev 上探索 10%/20%/30%/50%/100% 调用预算；同分样本整体处理，不借助 GT 拆分。

每一预算内按 dev macro-F1、accuracy、较少调用、较低阈值选型；主推荐在不超过 30% dev 调用预算的策略中按已定规则选出。参考推荐为 max probability < 0.960442447942174，dev 调用 50/400，test 实际调用 452/3421。测试调用比例不强制维持 dev 预算值。

该实验发生在看过两种模型测试总体成绩之后，属于后续探索。具体规则只根据 dev 选择并保存，再应用到已有 test 预测；不使用 test 在多个策略中重选推荐。回放不等于现场端到端推理测量。
