#!/usr/bin/env python3
"""Evaluate the fixed BloodMNIST WR50 linear probe on an official test manifest.

The backbone, StandardScaler, and logistic-regression parameters are loaded from
the expanded-v2 development run.  This program does not fit any component.  It
first reproduces that run's fixed 400-image development predictions, then opens
the test images only when the exact-prediction parity check has passed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_IDS = tuple(range(8))
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PROBE_KEYS = {"classes", "coef", "intercept", "scaler_mean", "scaler_scale"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="official-test-only manifest")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--frozen-probe", type=Path, required=True)
    parser.add_argument("--parity-manifest", type=Path, required=True, help="expanded-v2 train+val-only manifest")
    parser.add_argument("--parity-data-root", type=Path, required=True)
    parser.add_argument("--parity-dev-source-ids", type=Path, required=True)
    parser.add_argument("--parity-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probability-atol", type=float, default=1e-6)
    return parser.parse_args()


def read_records(path: Path, expected_split: str | None) -> tuple[list[dict[str, Any]], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    required = {"source_id", "image", "label", "split", "official_split"}
    if not isinstance(records, list) or any(not isinstance(record, dict) or not required <= set(record) for record in records):
        raise ValueError(f"{path} must be a list with fields {sorted(required)}")
    if not records:
        raise ValueError(f"{path} has no records")
    allowed_splits = {expected_split} if expected_split is not None else {"train", "val"}
    if any(record["split"] not in allowed_splits or record["official_split"] != record["split"] for record in records):
        allowed_text = "/".join(sorted(allowed_splits))
        raise ValueError(f"{path} must contain only official {allowed_text} records")
    if any(not isinstance(record["label"], int) or record["label"] not in CLASS_IDS for record in records):
        raise ValueError("labels must be integers 0 through 7")
    if len({record["source_id"] for record in records}) != len(records):
        raise ValueError("source_id values must be unique")
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
        raise ValueError("the manifest must contain all eight classes")
    return sorted(records, key=lambda record: record["source_id"]), names


def read_dev_records(manifest: Path, source_ids_path: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    records, names = read_records(manifest, None)
    payload = json.loads(source_ids_path.read_text(encoding="utf-8"))
    source_ids = payload.get("source_ids") if isinstance(payload, dict) else None
    if not isinstance(source_ids, list) or not all(isinstance(source_id, str) for source_id in source_ids):
        raise ValueError("development source-id file must contain a string list at source_ids")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("development source ids must be unique")
    by_id = {record["source_id"]: record for record in records}
    selected = [by_id[source_id] for source_id in source_ids if source_id in by_id]
    if len(selected) != len(source_ids):
        raise ValueError("development source IDs are absent from the parity manifest")
    if any(record["split"] != "val" or record["official_split"] != "val" for record in selected):
        raise ValueError("fixed parity development records must be official validation records")
    counts = Counter(record["label"] for record in selected)
    if any(counts[class_id] != 50 for class_id in CLASS_IDS):
        raise ValueError("fixed parity development set must have 50 images for each class")
    return sorted(selected, key=lambda record: record["source_id"]), names


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
        ManifestImages(records, data_root), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    source_ids: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            features.append(backbone(batch["image"].to(device, non_blocking=True)).cpu().numpy())
            labels.append(batch["label"].numpy())
            source_ids.extend(batch["source_id"])
    torch.cuda.synchronize(device)
    return np.concatenate(features), np.concatenate(labels), source_ids, time.perf_counter() - started


def restore_probe(path: Path) -> tuple[StandardScaler, LogisticRegression, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen probe: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != PROBE_KEYS:
            raise ValueError(f"unexpected frozen-probe keys: {sorted(saved.files)}")
        arrays = {key: saved[key].copy() for key in saved.files}
    classes = arrays["classes"]
    coef = arrays["coef"]
    intercept = arrays["intercept"]
    mean = arrays["scaler_mean"]
    scale = arrays["scaler_scale"]
    if classes.shape != (8,) or not np.array_equal(classes, np.asarray(CLASS_IDS)):
        raise ValueError("frozen probe must contain class IDs 0 through 7 in order")
    if coef.shape != (8, 2048) or intercept.shape != (8,) or mean.shape != (2048,) or scale.shape != (2048,):
        raise ValueError("frozen probe has incompatible WR50 feature dimensions")
    if not all(np.isfinite(value).all() for value in (coef, intercept, mean, scale)) or np.any(scale <= 0):
        raise ValueError("frozen probe contains invalid parameters")

    # Restore sklearn objects so StandardScaler keeps its float32 transform behavior
    # and LogisticRegression uses its native multiclass predict_proba implementation.
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.n_features_in_ = int(mean.shape[0])
    classifier = LogisticRegression(C=1, max_iter=1000, solver="lbfgs", random_state=42)
    classifier.classes_ = classes
    classifier.coef_ = coef
    classifier.intercept_ = intercept
    classifier.n_features_in_ = int(coef.shape[1])
    classifier.n_iter_ = np.zeros(1, dtype=np.int32)
    return scaler, classifier, {key: list(arrays[key].shape) for key in sorted(arrays)}


def predict(features: np.ndarray, scaler: StandardScaler, classifier: LogisticRegression) -> tuple[np.ndarray, np.ndarray]:
    standardized = scaler.transform(features)
    probabilities = classifier.predict_proba(standardized)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(int)
    return predictions, probabilities


def read_saved_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing saved development predictions: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        required = {"source_id", "label", "prediction", "valid", "confidence", "probabilities"}
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"invalid saved prediction at line {line_number}")
        source_id = row["source_id"]
        if not isinstance(source_id, str) or source_id in rows:
            raise ValueError("saved prediction source_ids must be unique strings")
        if row["label"] not in CLASS_IDS or row["prediction"] not in CLASS_IDS or row["valid"] is not True:
            raise ValueError("saved development predictions contain an invalid class")
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        if probabilities.shape != (8,) or not np.isfinite(probabilities).all():
            raise ValueError("saved development probabilities must have eight finite values")
        rows[source_id] = row
    return rows


def verify_parity(
    source_ids: list[str], labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, saved_path: Path, atol: float
) -> dict[str, Any]:
    saved = read_saved_predictions(saved_path)
    if set(source_ids) != set(saved):
        raise ValueError("saved and reproduced development source-id sets differ")
    saved_labels = np.asarray([saved[source_id]["label"] for source_id in source_ids], dtype=int)
    saved_predictions = np.asarray([saved[source_id]["prediction"] for source_id in source_ids], dtype=int)
    saved_probabilities = np.asarray([saved[source_id]["probabilities"] for source_id in source_ids], dtype=np.float64)
    if not np.array_equal(labels, saved_labels):
        raise ValueError("saved and reproduced development labels differ")
    prediction_match = bool(np.array_equal(predictions, saved_predictions))
    max_abs_probability_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
    if not prediction_match or max_abs_probability_difference > atol:
        raise RuntimeError(
            f"frozen-probe parity failed: predictions_match={prediction_match}, "
            f"max_abs_probability_difference={max_abs_probability_difference:.9g}, atol={atol:.9g}"
        )
    return {
        "images": len(source_ids),
        "prediction_match": prediction_match,
        "max_abs_probability_difference": max_abs_probability_difference,
        "probability_atol": atol,
        "status": "passed",
    }


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
    return {
        "images": int(len(labels)),
        "correct": int(np.sum(labels == valid_predictions)),
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
        raise RuntimeError("Frozen WR50 evaluation requires a Slurm-allocated CUDA GPU.")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {args.output}")
    if min(args.batch_size, args.num_workers) <= 0 or args.probability_atol < 0:
        raise ValueError("batch size, workers, and probability tolerance must be valid")

    device = torch.device("cuda:0")
    scaler, classifier, probe_shapes = restore_probe(args.frozen_probe)
    backbone = load_backbone(args.weights, device)

    dev_records, _ = read_dev_records(args.parity_manifest, args.parity_dev_source_ids)
    dev_features, dev_labels, dev_source_ids, dev_seconds = extract_features(
        dev_records, args.parity_data_root, backbone, args, device
    )
    dev_predictions, dev_probabilities = predict(dev_features, scaler, classifier)
    parity = verify_parity(
        dev_source_ids, dev_labels, dev_predictions, dev_probabilities, args.parity_predictions, args.probability_atol
    )
    parity["feature_extraction_seconds"] = dev_seconds

    # The official-test manifest and its images are consumed only after head-recovery parity passes.
    test_records, class_names = read_records(args.manifest, "test")
    if len(test_records) != 3421:
        raise ValueError(f"frozen official test manifest must contain 3421 images, found {len(test_records)}")
    test_features, test_labels, test_source_ids, test_seconds = extract_features(test_records, args.data_root, backbone, args, device)
    inference_started = time.perf_counter()
    test_predictions, test_probabilities = predict(test_features, scaler, classifier)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    metrics = evaluate(test_labels, test_predictions, class_names)

    args.output.mkdir(parents=True)
    with (args.output / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for source_id, label, prediction, probability in zip(test_source_ids, test_labels, test_predictions, test_probabilities):
            handle.write(json.dumps({
                "source_id": source_id,
                "split": "test",
                "label": int(label),
                "prediction": int(prediction),
                "valid": bool(prediction in CLASS_IDS),
                "confidence": float(np.max(probability)),
                "probabilities": [float(value) for value in probability],
            }) + "\n")
    report = {
        "protocol": {
            "name": "frozen ImageNet WideResNet-50-2 feature linear probe official test evaluation",
            "scope": "test-only inference using fixed expanded-v2 scaler and classifier; no fit, refit, or tuning",
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "transform": "Resize(224,224), ToTensor, Normalize(ImageNet mean/std)",
            "backbone": "WideResNet-50-2 loaded from supplied local ImageNet weights; frozen",
            "frozen_probe": str(args.frozen_probe),
            "restored_archive_shapes": probe_shapes,
            "test_records": "official test manifest only; no sampling",
        },
        "inputs": {
            "test_manifest": str(args.manifest),
            "test_data_root": str(args.data_root),
            "parity_manifest": str(args.parity_manifest),
            "parity_dev_source_ids": str(args.parity_dev_source_ids),
            "parity_predictions": str(args.parity_predictions),
            "weights": str(args.weights),
        },
        "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__, "sklearn": sklearn.__version__, "gpu": torch.cuda.get_device_name(device)},
        "parity": parity,
        "test": {"feature_extraction_seconds": test_seconds, "probe_inference_seconds": inference_seconds, **metrics},
    }
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
