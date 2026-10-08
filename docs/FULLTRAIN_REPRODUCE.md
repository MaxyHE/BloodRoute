# 完整训练集模型与真实串行路由复现

最新实验使用 11,959 张官方训练图、原固定 400 图开发集和完整 3,421 图测试集。VLM 从 base 新建 LoRA；CNN 沿用历史 4,000 图冻结特征线性分类器。参考环境为 Python 3.10、PyTorch 2.4.0+cu121、Torchvision 0.19.0、Transformers 5.5.4、单 RTX 3090。

## 不加载模型的结果重算

```bash
python scripts/recompute_fulltrain_metrics.py
```

公开附件保存逐图 ID、真实标签、预测类别、是否调用 VLM、CNN 置信度与端到端计时；不包含图像或模型。命令重算 accuracy、macro-F1、调用数、平均/P95 延迟及配对纠错，并对照归档聚合值。

## 准备完整训练集

模型、NPZ 与 CNN 权重路径由用户提供；官方文件来源和 MD5 校验见[历史复现](REPRODUCE.md#1-准备资产)。每类上限设为 11,959，使全部官方训练图保留，而非均衡抽样；开发集仍为每类 50 张。

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export BLOOD_NPZ=/path/to/bloodmnist_224.npz
export WR50_WEIGHTS=/path/to/wide_resnet50_2-95faca4d.pth
python medical_blood_pilot/build_bloodmnist_pilot.py \
  --source-npz "$BLOOD_NPZ" --output-dir data/blood_fulltrain \
  --provenance-dir data/blood_fulltrain/provenance \
  --train-cap-per-class 11959 --dev-cap-per-class 50 \
  --verified-md5 b718ff6835fcbdb22ba9eacccd7b2601
python scripts/run_qwen_fulltrain.py --rank 4 \
  --train-json data/blood_fulltrain/bloodmnist_train_manifest.json \
  --dev-json data/blood_fulltrain/bloodmnist_dev_50_per_class_manifest.json \
  --image-root data/blood_fulltrain --base-model "$MODEL_DIR" \
  --output-dir runs/blood_fulltrain
```

两轮共 23,918 步；首轮 lr=2e-4，后轮 5e-5，优化器状态保留；每 3,000 步及末步评估。开发 macro-F1、accuracy、较早 step 依次选模，参考选择为 step_18000。公开脚本使用仓库 `vlm_common.py` 替代私有 checkout 的同功能 helper 导入，不含集群账号和绝对路径；原实验设置保留在[训练元数据](../results/fulltrain-20261005/training-metadata.json)。GPU 命令需提交到计算节点。

## 选模后评估测试集

```bash
python medical_blood_pilot/build_bloodmnist_test_frozen.py \
  --source-npz "$BLOOD_NPZ" --output-dir data/blood_test_fulltrain \
  --provenance-dir data/blood_test_fulltrain/provenance \
  --train-manifest data/blood_fulltrain/bloodmnist_train_manifest.json \
  --dev-manifest data/blood_fulltrain/bloodmnist_dev_50_per_class_manifest.json \
  --verified-md5 b718ff6835fcbdb22ba9eacccd7b2601
python scripts/eval_fulltrain_test.py --train-run runs/blood_fulltrain \
  --test-json data/blood_test_fulltrain/bloodmnist_test_manifest.json \
  --image-root data/blood_test_fulltrain --base-model "$MODEL_DIR" \
  --adapter runs/blood_fulltrain/step_18000 --output-dir runs/test_fulltrain
```

复现时必须使用自己开发集选定的 checkpoint；评估入口验证它与 `best_checkpoint.json` 一致。这里的 step_18000 是原实验参考，不能在看到测试结果后更换 checkpoint。

## 门控选择与实际串行推理

先按[历史 CNN 复现](REPRODUCE.md#6-冻结-cnn-特征与线性分类器)准备原 4,000 图线性分类器、开发预测和测试预测。新 VLM 开发预测使用 `runs/blood_fulltrain/step_18000_dev_predictions.jsonl`。路由脚本先读 dev 选择并保存策略，再读 test 回放；新实验参考阈值为 0.9950015136477659。

```bash
python scripts/replay_routing.py \
  --qwen-dev runs/blood_fulltrain/step_18000_dev_predictions.jsonl \
  --cnn-dev runs/wr50_linear/dev_predictions.jsonl \
  --qwen-test runs/test_fulltrain/test_predictions.jsonl \
  --cnn-test runs/wr50_linear_test/test_predictions.jsonl \
  --output runs/routing_fulltrain
```

实际测速按三种模式分别执行以下命令，将 `route` 换成 `cnn`、`vlm` 并使用各自新输出目录。脚本读取已冻结的 policy，不重新选择阈值。

```bash
python scripts/benchmark_fulltrain.py --mode route --project . \
  --root data/blood_test_fulltrain --base "$MODEL_DIR" \
  --adapter runs/blood_fulltrain/step_18000 \
  --probe runs/wr50_linear/linear_probe.npz --weights "$WR50_WEIGHTS" \
  --policy runs/routing_fulltrain/policy.json \
  --cnn-reference runs/wr50_linear_test/test_predictions.jsonl \
  --vlm-reference runs/test_fulltrain/test_predictions.jsonl \
  --output runs/live_fulltrain/route
```

5 图暖机后按 source ID 排序，batch=1、并发=1、无预取；从读图到最终类别输出同步 CUDA 计时，排除加载、暖机和写文件。记录环境、峰值显存、逐图时延与参考预测差异。文件缓存未清空，硬件和磁盘条件变化时不要求精确复现每个毫秒。

参考原回放为 800 次 VLM 调用，本次真实串行为 801 次；batch/TF32 引起一张图的门控概率跨界，最终类别一致。原阈值不变。[实验结果](EXPERIMENTS.md)分别保留回放和实测口径。

## 版本边界

历史 4,000 图 LoRA、rank 消融和 VisA 回归保持原成绩。新训练同时改变规模、分布、初始化与更新过程；测试集已有历史使用。历史全量微调 CNN 为 98.07%，高于新路由的 97.49%。本次同步是实验材料和可移植脚本发布，没有重新训练模型或重新测量 GPU 性能。
