# 复现步骤

以下命令从仓库根目录执行，所有模型、数据和输出路径由使用者提供。参考环境为 Python 3.10、PyTorch 2.4.0+cu121、Torchvision 0.19.0、Transformers 5.5.4，单张 RTX 3090。具体依赖见根目录 requirements.txt；PyTorch 的 CUDA wheel 按本机驱动选择。

在共享集群中，将 GPU 命令交给 Slurm，不在登录节点执行模型。提供的 `configs/slurm/gpu.sbatch` 是通用单卡示例，需按本地环境调整 partition；提交前创建 `logs/`。数据导出可放 CPU 作业。

## 1. 准备资产

```bash
export MODEL_DIR=/path/to/Qwen3.5-4B
export BLOOD_NPZ=/path/to/bloodmnist_224.npz
export WR50_WEIGHTS=/path/to/wide_resnet50_2-95faca4d.pth
```

BloodMNIST 224 官方下载地址：[Zenodo](https://zenodo.org/records/10519652/files/bloodmnist_224.npz)。文件约 1.5 GB，参考 MD5 为 `b718ff6835fcbdb22ba9eacccd7b2601`；使用前校验下载文件，构建脚本的 `--verified-md5` 是记录字段，不替代校验。WR50 使用 Torchvision 对应的 ImageNet 预训练权重，Qwen 使用完整本地模型目录。

## 2. 构建试训与扩量数据

输出目录必须不存在。两次使用同一抽样方法，250/class 是 500/class 的子集，dev 均为相同 50/class。

```bash
python medical_blood_pilot/build_bloodmnist_pilot.py \
  --source-npz "$BLOOD_NPZ" \
  --output-dir data/blood_pilot \
  --provenance-dir data/blood_pilot/provenance \
  --train-cap-per-class 250 --dev-cap-per-class 50 \
  --verified-md5 b718ff6835fcbdb22ba9eacccd7b2601

python medical_blood_pilot/build_bloodmnist_pilot.py \
  --source-npz "$BLOOD_NPZ" \
  --output-dir data/blood_expanded \
  --provenance-dir data/blood_expanded/provenance \
  --train-cap-per-class 500 --dev-cap-per-class 50 \
  --verified-md5 b718ff6835fcbdb22ba9eacccd7b2601
```

## 3. VLM 初训

```bash
python scripts/run_medical_blood_pilot.py \
  --train-json data/blood_pilot/bloodmnist_train_manifest.json \
  --dev-json data/blood_pilot/bloodmnist_dev_50_per_class_manifest.json \
  --image-root data/blood_pilot --base-model "$MODEL_DIR" \
  --output-dir runs/blood_pilot
```

原始参考实验的最佳适配器为 `epoch_01`。下一步复现的是该固定分支，不会自动根据测试结果换成其他初训轮次。

## 4. 扩量续训

```bash
python scripts/run_medical_blood_expand.py \
  --train-json data/blood_expanded/bloodmnist_train_manifest.json \
  --dev-json data/blood_expanded/bloodmnist_dev_50_per_class_manifest.json \
  --reference-dev-json data/blood_pilot/bloodmnist_dev_50_per_class_manifest.json \
  --reference-epoch01-metrics runs/blood_pilot/epoch_01_metrics.json \
  --image-root data/blood_expanded --base-model "$MODEL_DIR" \
  --hot-start-adapter runs/blood_pilot/epoch_01 \
  --output-dir runs/blood_expanded
```

脚本先复测旧适配器，核验与初训记录一致，再运行新增 8,000 步。每 1,000 步保存、评估并更新 `best_checkpoint.json`。参考结果固定在 `step_03000`；本仓库测试入口保留这个参考实验约束，不能在查看测试后挑其他 step。

## 5. 冻结测试数据并评估 VLM

选模完成后再构建测试数据：

```bash
python medical_blood_pilot/build_bloodmnist_test_frozen.py \
  --source-npz "$BLOOD_NPZ" \
  --output-dir data/blood_test \
  --provenance-dir data/blood_test/provenance \
  --train-manifest data/blood_expanded/bloodmnist_train_manifest.json \
  --dev-manifest data/blood_pilot/bloodmnist_dev_50_per_class_manifest.json \
  --verified-md5 b718ff6835fcbdb22ba9eacccd7b2601

python scripts/eval_medical_blood_test.py \
  --test-json data/blood_test/bloodmnist_test_manifest.json \
  --image-root data/blood_test --base-model "$MODEL_DIR" \
  --base-only --output-dir runs/test_base

python scripts/eval_medical_blood_test.py \
  --test-json data/blood_test/bloodmnist_test_manifest.json \
  --image-root data/blood_test --base-model "$MODEL_DIR" \
  --adapter runs/blood_expanded/step_03000 \
  --output-dir runs/test_lora
```

输出 `test_metrics.json`、`test_predictions.jsonl`、`run_metadata.json`。base-only 不加载适配器。二者不能混用不同问题或图像版本。

## 6. 冻结 CNN 特征与线性分类器

```bash
python medical_blood_pilot/run_wr50_linear_probe.py \
  --manifest data/blood_expanded/bloodmnist_train_val_manifest.json \
  --dev-source-ids data/blood_expanded/val_dev_50_per_class.json \
  --data-root data/blood_expanded --weights "$WR50_WEIGHTS" \
  --output runs/wr50_linear --max-train-per-class 500 \
  --max-dev-per-class 50 --batch-size 64 --num-workers 4 --seed 42

python medical_blood_pilot/run_wr50_frozen_test.py \
  --manifest data/blood_test/bloodmnist_test_manifest.json \
  --data-root data/blood_test --weights "$WR50_WEIGHTS" \
  --frozen-probe runs/wr50_linear/linear_probe.npz \
  --parity-manifest data/blood_expanded/bloodmnist_train_val_manifest.json \
  --parity-data-root data/blood_expanded \
  --parity-dev-source-ids data/blood_expanded/val_dev_50_per_class.json \
  --parity-predictions runs/wr50_linear/dev_predictions.jsonl \
  --output runs/wr50_linear_test
```

## 7. 路由回放

同一 source_id 的图像、GT 必须一致。先用 dev 选定策略并保存，再打开 test 预测，不能根据 test 从多个策略中挑报告点。

```bash
python scripts/replay_routing.py \
  --qwen-dev runs/blood_expanded/step_03000_dev_predictions.jsonl \
  --cnn-dev runs/wr50_linear/dev_predictions.jsonl \
  --qwen-test runs/test_lora/test_predictions.jsonl \
  --cnn-test runs/wr50_linear_test/test_predictions.jsonl \
  --output runs/routing_replay
```

此命令不加载模型，不测端到端速度。VLM 调用比例和配对纠错数来自固定门控对已有预测的选择。

## 8. 补充：全量微调 CNN

```bash
python medical_blood_pilot/run_wr50_finetune.py train \
  --manifest data/blood_expanded/bloodmnist_train_val_manifest.json \
  --data-root data/blood_expanded \
  --dev-source-ids data/blood_expanded/val_dev_50_per_class.json \
  --weights "$WR50_WEIGHTS" --output runs/wr50_finetune \
  --batch-size 32

python medical_blood_pilot/run_wr50_finetune.py test \
  --manifest data/blood_test/bloodmnist_test_manifest.json \
  --data-root data/blood_test --train-output runs/wr50_finetune \
  --output runs/wr50_finetune_test --batch-size 32
```

训练入口只读 train/val，测试入口要求训练完成标记并读取开发集最佳模型，不重新拟合任何参数。

## 重新生成配图

配图读取 `results/metrics.json` 中的聚合结果，输出 SVG 与 PNG。绘图依赖可安装在单独环境中：

```bash
pip install matplotlib
python scripts/make_figures.py
```
