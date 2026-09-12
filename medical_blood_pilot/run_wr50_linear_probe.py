#!/usr/bin/env python3
"""Frozen ImageNet WideResNet-50-2 feature linear probe for BloodMNIST.

This is a fixed development baseline, not a fully optimized CNN or a SOTA
claim. It accepts only a train-plus-validation manifest; test examples are
rejected before images are opened.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_IDS = tuple(range(8))
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dev-source-ids", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-per-class", type=int, default=250)
    parser.add_argument("--max-dev-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    required = {"source_id", "image", "label", "split", "official_split"}
    if not isinstance(records, list) or any(not required <= set(record) for record in records):
        raise ValueError(f"{path} must be a list with fields {sorted(required)}")
    normalized = []
    for record in records:
        copy = dict(record)
        copy["label_name"] = copy.get("label_name", copy.get("class_name"))
        if not isinstance(copy["label_name"], str) or not copy["label_name"]:
            raise ValueError("each record must provide label_name or class_name")
        if "image_shape" in copy and copy["image_shape"] != [IMAGE_SIZE, IMAGE_SIZE, 3]:
            raise ValueError(f"image_shape must be [{IMAGE_SIZE}, {IMAGE_SIZE}, 3] when provided")
        normalized.append(copy)
    records = normalized
    if any(record["split"] not in {"train", "val"} for record in records):
        raise ValueError("manifest must contain train and val records only; test records are not accepted")
    if any(record["official_split"] != record["split"] for record in records):
        raise ValueError("manifest split and official_split must agree")
    if any(not isinstance(record["label"], int) or record["label"] not in CLASS_IDS for record in records):
        raise ValueError("labels must be integers 0 through 7")
    if len({record["source_id"] for record in records}) != len(records):
        raise ValueError("manifest source_id values must be unique")
    return records


def read_dev_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_ids = payload.get("source_ids") if isinstance(payload, dict) else None
    if not isinstance(source_ids, list) or not all(isinstance(value, str) for value in source_ids):
        raise ValueError(f"{path} must contain a string list at source_ids")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("development source_ids must be unique")
    return source_ids


def select_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, str]]:
    records = read_records(args.manifest)
    selected_ids = set(read_dev_ids(args.dev_source_ids))
    by_id = {record["source_id"]: record for record in records}
    missing = selected_ids - set(by_id)
    if missing:
        raise ValueError(f"development source_ids absent from manifest: {len(missing)}")
    train = sorted((record for record in records if record["split"] == "train"), key=lambda item: item["source_id"])
    dev = sorted((by_id[source_id] for source_id in selected_ids), key=lambda item: item["source_id"])
    if any(record["split"] != "val" for record in dev):
        raise ValueError("development source_ids must select official validation records")
    train_counts = Counter(record["label"] for record in train)
    dev_counts = Counter(record["label"] for record in dev)
    if set(train_counts) != set(CLASS_IDS) or set(dev_counts) != set(CLASS_IDS):
        raise ValueError("train and development selections must each contain all eight classes")
    if any(count > args.max_train_per_class for count in train_counts.values()):
        raise ValueError("train manifest exceeds --max-train-per-class")
    if any(count > args.max_dev_per_class for count in dev_counts.values()):
        raise ValueError("development selection exceeds --max-dev-per-class")
    names: dict[int, str] = {}
    for record in train + dev:
        existing = names.setdefault(record["label"], record["label_name"])
        if existing != record["label_name"]:
            raise ValueError(f"inconsistent label_name for class {record['label']}")
    return train, dev, names


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


def load_backbone(weights: Path, device: torch.device) -> torch.nn.Module:
    if not weights.is_file():
        raise FileNotFoundError(f"missing local WideResNet-50-2 weights: {weights}")
    backbone = models.wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def extract_features(
    records: list[dict[str, Any]], data_root: Path, backbone: torch.nn.Module, args: argparse.Namespace, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    loader = DataLoader(
        ManifestImages(records, data_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    source_ids: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = backbone(batch["image"].to(device, non_blocking=True))
            features.append(output.cpu().numpy())
            labels.append(batch["label"].numpy())
            source_ids.extend(batch["source_id"])
    torch.cuda.synchronize(device)
    return np.concatenate(features), np.concatenate(labels), source_ids, time.perf_counter() - started


def evaluate(labels: np.ndarray, predictions: np.ndarray, class_names: dict[int, str]) -> dict[str, Any]:
    invalid = ~np.isin(predictions, CLASS_IDS)
    valid_predictions = predictions.copy()
    valid_predictions[invalid] = -1
    matrix = np.zeros((8, 9), dtype=int)
    for label, prediction in zip(labels, valid_predictions):
        matrix[int(label), int(prediction) if prediction in CLASS_IDS else 8] += 1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, valid_predictions, labels=list(CLASS_IDS), zero_division=0
    )
    correct = int(np.sum(labels == valid_predictions))
    return {
        "images": int(len(labels)),
        "correct": correct,
        "invalid": int(np.sum(invalid)),
        "accuracy": float(accuracy_score(labels, valid_predictions)),
        "macro_f1": float(f1_score(labels, valid_predictions, labels=list(CLASS_IDS), average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_columns": [str(class_id) for class_id in CLASS_IDS] + ["invalid"],
        "confusion_rows": [str(class_id) for class_id in CLASS_IDS],
        "confusion_8_plus_invalid": matrix.tolist(),
        "per_class": {
            str(class_id): {
                "label_name": class_names[class_id],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_id in enumerate(CLASS_IDS)
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This baseline requires a Slurm-allocated CUDA GPU.")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output}")
    if min(args.max_train_per_class, args.max_dev_per_class, args.batch_size) <= 0:
        raise ValueError("sample caps and batch size must be positive")
    seed_everything(args.seed)
    train_records, dev_records, class_names = select_records(args)
    args.output.mkdir(parents=True)
    device = torch.device("cuda:0")
    backbone = load_backbone(args.weights, device)
    train_features, train_labels, train_ids, train_seconds = extract_features(train_records, args.data_root, backbone, args, device)
    dev_features, dev_labels, dev_ids, dev_seconds = extract_features(dev_records, args.data_root, backbone, args, device)

    scaler = StandardScaler().fit(train_features)
    train_features = scaler.transform(train_features)
    dev_features = scaler.transform(dev_features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier = LogisticRegression(C=1, max_iter=1000, solver="lbfgs", random_state=args.seed)
        fit_started = time.perf_counter()
        classifier.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    probabilities = classifier.predict_proba(dev_features)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(int)
    probe_inference_seconds = time.perf_counter() - inference_started

    with (args.output / "dev_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for source_id, label, prediction, probability in zip(dev_ids, dev_labels, predictions, probabilities):
            handle.write(json.dumps({
                "source_id": source_id,
                "split": "val_dev",
                "label": int(label),
                "prediction": int(prediction),
                "valid": bool(prediction in CLASS_IDS),
                "confidence": float(np.max(probability)),
                "probabilities": [float(value) for value in probability],
            }) + "\n")
    np.savez(
        args.output / "linear_probe.npz",
        classes=classifier.classes_,
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    report = {
        "protocol": {
            "name": "frozen ImageNet WideResNet-50-2 feature linear probe",
            "scope": "development baseline; not a fully optimized CNN or SOTA claim",
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "normalization": "ImageNet mean/std",
            "backbone": "WideResNet-50-2 loaded from supplied local weights; all backbone parameters frozen",
            "feature_dimension": int(train_features.shape[1]),
            "classifier": "StandardScaler fit on train features, then LogisticRegression(C=1, max_iter=1000, solver=lbfgs)",
            "seed": args.seed,
            "test_records": "rejected by manifest validation",
        },
        "inputs": {
            "manifest": str(args.manifest),
            "dev_source_ids": str(args.dev_source_ids),
            "weights": str(args.weights),
        },
        "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__, "sklearn": sklearn.__version__, "gpu": torch.cuda.get_device_name(device)},
        "class_names": {str(class_id): class_names[class_id] for class_id in CLASS_IDS},
        "train": {"images": len(train_records), "per_class": {str(key): value for key, value in sorted(Counter(train_labels).items())}, "source_ids": train_ids, "feature_extraction_seconds": train_seconds},
        "development": {"images": len(dev_records), "per_class": {str(key): value for key, value in sorted(Counter(dev_labels).items())}, "source_ids": dev_ids, "feature_extraction_seconds": dev_seconds, "probe_inference_seconds": probe_inference_seconds, **evaluate(dev_labels, predictions, class_names)},
        "classifier_fit": {"seconds": fit_seconds, "n_iter": [int(value) for value in classifier.n_iter_], "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught)},
    }
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
