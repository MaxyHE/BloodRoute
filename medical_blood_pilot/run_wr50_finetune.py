#!/usr/bin/env python3
"""Fixed full WideResNet-50-2 fine-tuning and separated frozen test evaluation.

The ``train`` and ``smoke`` subcommands accept only the expanded train-plus-val
manifest.  The ``test`` subcommand accepts only the frozen official-test
manifest and loads the unique development-selected checkpoint from ``best.json``.
It never selects an epoch from test results.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_IDS = tuple(range(8))
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def common_data_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--manifest", type=Path, required=True)
        target.add_argument("--data-root", type=Path, required=True)
        target.add_argument("--batch-size", type=int, default=32)
        target.add_argument("--num-workers", type=int, default=4)

    smoke = subparsers.add_parser("smoke", help="one no-update training batch; does not read test")
    common_data_arguments(smoke)
    smoke.add_argument("--dev-source-ids", type=Path, required=True)
    smoke.add_argument("--weights", type=Path, required=True)

    train = subparsers.add_parser("train", help="fixed train/dev fine-tuning; does not read test")
    common_data_arguments(train)
    train.add_argument("--dev-source-ids", type=Path, required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--backbone-lr", type=float, default=1e-5)
    train.add_argument("--head-lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)

    test = subparsers.add_parser("test", help="frozen best-checkpoint official-test evaluation")
    common_data_arguments(test)
    test.add_argument("--train-output", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path, allowed_splits: set[str]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    required = {"source_id", "image", "label", "split", "official_split"}
    if not isinstance(records, list) or any(not isinstance(record, dict) or not required <= set(record) for record in records):
        raise ValueError(f"{path} must be a record list with fields {sorted(required)}")
    if not records:
        raise ValueError(f"{path} has no records")
    if any(record["split"] not in allowed_splits or record["official_split"] != record["split"] for record in records):
        raise ValueError(f"{path} must contain only official {sorted(allowed_splits)} records")
    if any(not isinstance(record["label"], int) or record["label"] not in CLASS_IDS for record in records):
        raise ValueError("labels must be integer IDs from 0 through 7")
    if len({record["source_id"] for record in records}) != len(records):
        raise ValueError("manifest source_id values must be unique")
    names: dict[int, str] = {}
    for record in records:
        name = record.get("label_name", record.get("class_name"))
        if not isinstance(name, str) or not name:
            raise ValueError("each record must provide label_name or class_name")
        known = names.setdefault(record["label"], name)
        if known != name:
            raise ValueError(f"inconsistent class name for label {record['label']}")
        if "image_shape" in record and record["image_shape"] != [IMAGE_SIZE, IMAGE_SIZE, 3]:
            raise ValueError(f"image_shape must be [{IMAGE_SIZE}, {IMAGE_SIZE}, 3] when provided")
    if set(names) != set(CLASS_IDS):
        raise ValueError("manifest must contain all eight classes")
    return sorted(records, key=lambda record: record["source_id"]), names


def read_dev_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_ids = payload.get("source_ids") if isinstance(payload, dict) else None
    if not isinstance(source_ids, list) or not all(isinstance(source_id, str) for source_id in source_ids):
        raise ValueError("development source-id file must contain a string list at source_ids")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("development source IDs must be unique")
    return source_ids


def load_train_and_dev(manifest: Path, dev_ids_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    records, names = read_manifest(manifest, {"train", "val"})
    train = [record for record in records if record["split"] == "train"]
    by_id = {record["source_id"]: record for record in records}
    dev_ids = read_dev_ids(dev_ids_path)
    if any(source_id not in by_id for source_id in dev_ids):
        raise ValueError("development source IDs are absent from the manifest")
    dev = [by_id[source_id] for source_id in dev_ids]
    if any(record["split"] != "val" or record["official_split"] != "val" for record in dev):
        raise ValueError("development IDs must select official validation records")
    train_counts = Counter(record["label"] for record in train)
    dev_counts = Counter(record["label"] for record in dev)
    if len(train) != 4000 or any(train_counts[class_id] != 500 for class_id in CLASS_IDS):
        raise ValueError("fixed expanded training set must contain 4,000 images, 500 per class")
    if len(dev) != 400 or any(dev_counts[class_id] != 50 for class_id in CLASS_IDS):
        raise ValueError("fixed development set must contain 400 images, 50 per class")
    if set(record["source_id"] for record in train) & set(record["source_id"] for record in dev):
        raise ValueError("training and development source IDs overlap")
    return train, sorted(dev, key=lambda record: record["source_id"]), names


class ManifestImages(Dataset):
    def __init__(self, records: list[dict[str, Any]], data_root: Path):
        self.records = records
        self.data_root = data_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.data_root / record["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "label": record["label"], "source_id": record["source_id"]}


def loader(records: list[dict[str, Any]], root: Path, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ManifestImages(records, root), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0,
    )


def assert_cuda_bf16() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a Slurm-allocated CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 autocast is required by this fixed protocol but is unsupported on this GPU")
    return torch.device("cuda:0")


def make_pretrained_model(weights: Path, device: torch.device) -> nn.Module:
    if not weights.is_file():
        raise FileNotFoundError(f"missing local ImageNet weights: {weights}")
    model = models.wide_resnet50_2(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_IDS))
    return model.to(device)


def make_checkpoint_model(device: torch.device) -> nn.Module:
    model = models.wide_resnet50_2(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_IDS))
    return model.to(device)


def metrics(labels: np.ndarray, predictions: np.ndarray, names: dict[int, str]) -> dict[str, Any]:
    matrix = np.zeros((8, 9), dtype=int)
    per_class: dict[str, Any] = {}
    f1s: list[float] = []
    recalls: list[float] = []
    invalid = ~np.isin(predictions, CLASS_IDS)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction) if prediction in CLASS_IDS else 8] += 1
    for class_id in CLASS_IDS:
        actual = labels == class_id
        predicted = predictions == class_id
        true_positive = int(np.sum(actual & predicted))
        false_positive = int(np.sum(~actual & predicted))
        false_negative = int(np.sum(actual & ~predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        recalls.append(recall)
        per_class[str(class_id)] = {
            "label_name": names[class_id], "precision": precision, "recall": recall,
            "f1": f1, "support": int(np.sum(actual)),
        }
    return {
        "images": int(len(labels)), "correct": int(np.sum(labels == predictions)), "invalid": int(np.sum(invalid)),
        "accuracy": float(np.mean(labels == predictions)), "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)), "confusion_rows": [str(value) for value in CLASS_IDS],
        "confusion_columns": [str(value) for value in CLASS_IDS] + ["invalid"],
        "confusion_8_plus_invalid": matrix.tolist(), "per_class": per_class,
    }


def evaluate(model: nn.Module, data: DataLoader, device: torch.device, names: dict[int, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in data:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            predicted = np.argmax(probabilities, axis=1).astype(int)
            batch_labels = batch["label"].numpy()
            labels.append(batch_labels)
            predictions.append(predicted)
            rows.extend({
                "source_id": source_id, "label": int(label), "prediction": int(prediction),
                "valid": bool(prediction in CLASS_IDS), "confidence": float(np.max(probability)),
                "probabilities": [float(value) for value in probability],
            } for source_id, label, prediction, probability in zip(batch["source_id"], batch_labels, predicted, probabilities))
    torch.cuda.synchronize(device)
    summary = metrics(np.concatenate(labels), np.concatenate(predictions), names)
    summary["inference_seconds"] = time.perf_counter() - started
    return summary, rows


def config(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    return {
        "architecture": "WideResNet-50-2", "initialization": "supplied local ImageNet weights", "image_size": [224, 224],
        "transform": "Resize(224,224), ToTensor, Normalize(ImageNet mean/std)", "augmentation": "none",
        "batch_norm": "train mode during train batches; normal running-stat updates", "amp": "cuda bfloat16 autocast",
        "batch_size": args.batch_size, "num_workers": args.num_workers, "gpu": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
    }


def smoke(args: argparse.Namespace) -> None:
    device = assert_cuda_bf16()
    train_records, _, _ = load_train_and_dev(args.manifest, args.dev_source_ids)
    model = make_pretrained_model(args.weights, device)
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.train()
    criterion = nn.CrossEntropyLoss()
    batch = next(iter(loader(train_records, args.data_root, args.batch_size, args.num_workers, shuffle=False)))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(batch["image"].to(device, non_blocking=True)), batch["label"].to(device, non_blocking=True))
    loss.backward()
    torch.cuda.synchronize(device)
    print(json.dumps({"status": "smoke_passed", "batch_size": args.batch_size, "loss": float(loss.detach().cpu()), "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)), **config(args, device)}))


def train(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output}")
    if args.epochs != 10 or args.seed != 42 or args.backbone_lr != 1e-5 or args.head_lr != 1e-4 or args.weight_decay != 1e-4:
        raise ValueError("this fixed protocol requires 10 epochs, seed 42, backbone LR 1e-5, head LR 1e-4, and weight decay 1e-4")
    if min(args.batch_size, args.num_workers) <= 0:
        raise ValueError("batch size and workers must be positive")
    seed_everything(args.seed)
    device = assert_cuda_bf16()
    train_records, dev_records, names = load_train_and_dev(args.manifest, args.dev_source_ids)
    model = make_pretrained_model(args.weights, device)
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = AdamW([
        {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")], "lr": args.backbone_lr},
        {"params": model.fc.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(train_records, args.data_root, args.batch_size, args.num_workers, shuffle=True)
    dev_loader = loader(dev_records, args.data_root, args.batch_size, args.num_workers, shuffle=False)
    args.output.mkdir(parents=True)
    run_metadata = {
        "protocol": {
            "scope": "fixed full fine-tuning development experiment; no augmentation, early stopping, or hyperparameter search",
            "seed": args.seed, "epochs": args.epochs, "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
            "weight_decay": args.weight_decay, "all_cnn_parameters": all_parameters,
            "trainable_cnn_parameters": trainable_parameters, **config(args, device),
        },
        "inputs": {"manifest": str(args.manifest), "dev_source_ids": str(args.dev_source_ids), "weights": str(args.weights)},
        "train": {"images": len(train_records), "per_class": {str(key): value for key, value in sorted(Counter(record["label"] for record in train_records).items())}},
        "development": {"images": len(dev_records), "per_class": {str(key): value for key, value in sorted(Counter(record["label"] for record in dev_records).items())}},
        "selection": ["highest development macro_f1", "highest development accuracy", "earlier epoch"],
    }
    (args.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        image_count = 0
        started = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            image_count += len(labels)
        torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        dev_metrics, _ = evaluate(model, dev_loader, device, names)
        epoch_record = {"epoch": epoch, "train_loss": loss_sum / image_count, "train_seconds": train_seconds, "development": dev_metrics}
        checkpoint_name = f"epoch_{epoch:02d}.pt"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": run_metadata["protocol"]}, args.output / checkpoint_name)
        epoch_record["checkpoint"] = checkpoint_name
        (args.output / f"epoch_{epoch:02d}_metrics.json").write_text(json.dumps(epoch_record, indent=2), encoding="utf-8")
        history.append(epoch_record)
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        candidate = epoch_record["development"]
        print(json.dumps({
            "epoch": epoch, "train_loss": epoch_record["train_loss"], "elapsed_seconds": train_seconds,
            "dev_accuracy": candidate["accuracy"], "dev_macro_f1": candidate["macro_f1"],
            "dev_balanced_accuracy": candidate["balanced_accuracy"],
        }), flush=True)
        if best is None or candidate["macro_f1"] > best["development"]["macro_f1"] or (candidate["macro_f1"] == best["development"]["macro_f1"] and candidate["accuracy"] > best["development"]["accuracy"]):
            best = epoch_record
            (args.output / "best.json").write_text(json.dumps({"best_epoch": epoch, "checkpoint": checkpoint_name, "development": candidate, "selection": run_metadata["selection"], "class_names": {str(key): value for key, value in names.items()}}, indent=2), encoding="utf-8")
    if best is None:
        raise RuntimeError("no epoch completed")
    run_metadata["completed_epochs"] = args.epochs
    (args.output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")


def test(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite test directory: {args.output}")
    if min(args.batch_size, args.num_workers) <= 0:
        raise ValueError("batch size and workers must be positive")
    device = assert_cuda_bf16()
    best_path = args.train_output / "best.json"
    run_metadata_path = args.train_output / "run_metadata.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"missing completed training metadata: {run_metadata_path}")
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("completed_epochs") != 10:
        raise ValueError("test requires a successfully completed fixed 10-epoch training run")
    if not best_path.is_file():
        raise FileNotFoundError(f"missing fixed best metadata: {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    checkpoint_name = best.get("checkpoint")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("best metadata must name a checkpoint in the train run directory")
    checkpoint_path = args.train_output / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing development-selected checkpoint: {checkpoint_path}")
    records, names = read_manifest(args.manifest, {"test"})
    if len(records) != 3421:
        raise ValueError(f"frozen official test manifest must contain 3421 images, found {len(records)}")
    stored_names = best.get("class_names")
    if stored_names != {str(key): value for key, value in names.items()}:
        raise ValueError("test class-name mapping differs from development-selected model metadata")
    model = make_checkpoint_model(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("epoch") != best.get("best_epoch") or "model_state_dict" not in payload:
        raise ValueError("checkpoint does not match best metadata")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    summary, rows = evaluate(model, loader(records, args.data_root, args.batch_size, args.num_workers, shuffle=False), device, names)
    args.output.mkdir(parents=True)
    with (args.output / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "split": "test"}) + "\n")
    report = {
        "protocol": {"scope": "test-only inference using the unique development-selected checkpoint; test did not select epoch", **config(args, device)},
        "inputs": {"test_manifest": str(args.manifest), "test_data_root": str(args.data_root), "train_output": str(args.train_output), "best_metadata": str(best_path), "checkpoint": str(checkpoint_path)},
        "best": best, "test": summary,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        smoke(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
