#!/usr/bin/env python3
"""Replay a development-frozen CNN-confidence-to-VLM routing policy.

The Qwen JSONL format stores the ground-truth numeric label in ``label`` and
the parsed class name in ``parsed``.  The CNN JSONL format stores ``prediction``
plus an eight-class probability vector.  Development data alone chooses the
thresholds; this program writes ``policy.json`` before it opens either test file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CLASS_IDS = tuple(range(8))
BUDGETS = (0.10, 0.20, 0.30, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-dev", type=Path, required=True)
    parser.add_argument("--cnn-dev", type=Path, required=True)
    parser.add_argument("--qwen-test", type=Path, required=True)
    parser.add_argument("--cnn-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    return rows


def qwen_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    required = {"source_id", "gt", "label", "parsed"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("Qwen predictions must contain source_id, gt, label, and parsed")
    output = {row["source_id"]: row for row in rows}
    if len(output) != len(rows) or not all(isinstance(source_id, str) for source_id in output):
        raise ValueError("Qwen source_id values must be unique strings")
    return output


def cnn_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    required = {"source_id", "label", "prediction", "valid", "confidence", "probabilities"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("CNN predictions must contain source_id, label, prediction, valid, confidence, and probabilities")
    output = {row["source_id"]: row for row in rows}
    if len(output) != len(rows) or not all(isinstance(source_id, str) for source_id in output):
        raise ValueError("CNN source_id values must be unique strings")
    return output


def aligned_pair(qwen_rows: list[dict[str, Any]], cnn_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    qwen, cnn = qwen_by_id(qwen_rows), cnn_by_id(cnn_rows)
    if set(qwen) != set(cnn):
        raise ValueError("Qwen and CNN source_id sets differ")
    class_to_id: dict[str, int] = {}
    for row in qwen.values():
        label = row["label"]
        if not isinstance(label, int) or label not in CLASS_IDS or not isinstance(row["gt"], str):
            raise ValueError("Qwen labels must be integer class IDs and gt must be a class name")
        known = class_to_id.setdefault(row["gt"], label)
        if known != label:
            raise ValueError("Qwen ground-truth class mapping is inconsistent")
    if set(class_to_id.values()) != set(CLASS_IDS):
        raise ValueError("Qwen records must cover all eight class IDs")
    source_ids = sorted(qwen)
    labels = np.asarray([qwen[source_id]["label"] for source_id in source_ids], dtype=int)
    qwen_prediction = np.asarray([class_to_id.get(qwen[source_id]["parsed"], -1) for source_id in source_ids], dtype=int)
    cnn_prediction = np.asarray([cnn[source_id]["prediction"] if cnn[source_id]["valid"] else -1 for source_id in source_ids], dtype=int)
    cnn_labels = np.asarray([cnn[source_id]["label"] for source_id in source_ids], dtype=int)
    if not np.array_equal(labels, cnn_labels):
        raise ValueError("Qwen and CNN labels differ after source_id alignment")
    probabilities = np.asarray([cnn[source_id]["probabilities"] for source_id in source_ids], dtype=float)
    confidence = np.asarray([cnn[source_id]["confidence"] for source_id in source_ids], dtype=float)
    if probabilities.shape != (len(source_ids), 8) or not np.isfinite(probabilities).all() or not np.isfinite(confidence).all():
        raise ValueError("CNN probabilities must be finite [N, 8] values")
    if np.max(np.abs(np.max(probabilities, axis=1) - confidence)) > 1e-9:
        raise ValueError("CNN confidence must equal max(probabilities)")
    if np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("CNN probability rows must sum to one")
    ordered = np.sort(probabilities, axis=1)
    return {
        "source_ids": np.asarray(source_ids), "labels": labels, "qwen": qwen_prediction, "cnn": cnn_prediction,
        "max_probability": confidence, "top1_top2_margin": ordered[:, -1] - ordered[:, -2],
    }


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    invalid = ~np.isin(predictions, CLASS_IDS)
    matrix = np.zeros((8, 9), dtype=int)
    f1_values: list[float] = []
    recall_values: list[float] = []
    per_class: dict[str, Any] = {}
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
        f1_values.append(f1)
        recall_values.append(recall)
        per_class[str(class_id)] = {"precision": precision, "recall": recall, "f1": f1, "support": int(actual.sum())}
    return {
        "images": int(len(labels)), "correct": int(np.sum(labels == predictions)), "invalid": int(np.sum(invalid)),
        "accuracy": float(np.mean(labels == predictions)), "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)), "confusion_rows": [str(value) for value in CLASS_IDS],
        "confusion_columns": [str(value) for value in CLASS_IDS] + ["invalid"],
        "confusion_8_plus_invalid": matrix.tolist(), "per_class": per_class,
    }


def replay(labels: np.ndarray, cnn: np.ndarray, qwen: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    calls = score < threshold
    prediction = np.where(calls, qwen, cnn)
    result = metrics(labels, prediction)
    result.update({
        "threshold": float(threshold), "calls": int(np.sum(calls)), "call_rate": float(np.mean(calls)),
        "cnn_to_qwen_corrected": int(np.sum(calls & (cnn != labels) & (qwen == labels))),
        "cnn_to_qwen_harmed": int(np.sum(calls & (cnn == labels) & (qwen != labels))),
    })
    return result


def paired(labels: np.ndarray, cnn: np.ndarray, qwen: np.ndarray) -> dict[str, Any]:
    both_correct = int(np.sum((cnn == labels) & (qwen == labels)))
    cnn_only = int(np.sum((cnn == labels) & (qwen != labels)))
    qwen_only = int(np.sum((cnn != labels) & (qwen == labels)))
    both_wrong = int(np.sum((cnn != labels) & (qwen != labels)))
    return {
        "both_correct": both_correct, "cnn_only_correct": cnn_only, "qwen_only_correct": qwen_only,
        "both_wrong": both_wrong, "oracle_accuracy_upper_bound": (both_correct + cnn_only + qwen_only) / len(labels),
        "oracle_note": "analysis-only upper bound; not a deployable routing result",
    }


def freeze_policy(dev: dict[str, np.ndarray]) -> dict[str, Any]:
    labels, cnn, qwen = dev["labels"], dev["cnn"], dev["qwen"]
    candidates_by_score: dict[str, Any] = {}
    selected: list[dict[str, Any]] = []
    for score_name in ("max_probability", "top1_top2_margin"):
        values = dev[score_name]
        unique = np.unique(values)
        thresholds = [0.0] + [float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:])] + [1.0000001]
        candidates = [replay(labels, cnn, qwen, values, threshold) for threshold in sorted(set(thresholds))]
        candidates_by_score[score_name] = {
            "direction": "route_to_qwen_when_score_less_than_threshold", "score_range": [float(values.min()), float(values.max())],
            "threshold_candidates": candidates,
        }
        for budget in BUDGETS:
            eligible = [candidate for candidate in candidates if candidate["call_rate"] <= budget + 1e-12]
            chosen = min(eligible, key=lambda candidate: (-candidate["macro_f1"], -candidate["accuracy"], candidate["calls"], candidate["threshold"]))
            selected.append({"score": score_name, "budget_upper_bound": budget, "rule": "score < threshold", **chosen})
    recommendation = min(
        [candidate for candidate in selected if candidate["budget_upper_bound"] <= 0.30],
        key=lambda candidate: (-candidate["macro_f1"], -candidate["accuracy"], candidate["calls"], candidate["score"]),
    )
    return {
        "protocol": {
            "selection_data": "development predictions only", "routing_rule": "all-class global score < threshold",
            "scores": ["max_probability", "top1_top2_margin"], "budgets": list(BUDGETS),
            "candidate_thresholds": "0.0, adjacent unique-score midpoints, 1.0000001",
            "selection_order": ["highest macro_f1", "highest accuracy", "fewer calls", "lower threshold"],
            "recommendation_order": ["selected budgets <=30%: highest macro_f1", "highest accuracy", "fewer calls", "score name"],
            "test_selection": "forbidden; this policy is written before test predictions are opened",
        },
        "input_checks": {"development_images": int(len(labels)), "source_id_sets_equal": True, "ground_truth_equal": True},
        "endpoints": {"cnn_only": metrics(labels, cnn), "qwen_full_call": {**metrics(labels, qwen), "calls": int(len(labels)), "call_rate": 1.0}},
        "development_paired": paired(labels, cnn, qwen), "candidate_tables": candidates_by_score,
        "selected_policies": selected, "main_recommendation": recommendation,
    }


def replay_test(test: dict[str, np.ndarray], policy: dict[str, Any]) -> dict[str, Any]:
    labels, cnn, qwen = test["labels"], test["cnn"], test["qwen"]
    cnn_metrics, qwen_metrics = metrics(labels, cnn), metrics(labels, qwen)
    replays: list[dict[str, Any]] = []
    for frozen in policy["selected_policies"]:
        score_name = frozen["score"]
        result = replay(labels, cnn, qwen, test[score_name], frozen["threshold"])
        result.update({
            "score": score_name, "budget_upper_bound_dev": frozen["budget_upper_bound"], "rule": "score < threshold",
            "frozen_threshold": frozen["threshold"],
            "vs_cnn": {key: result[key] - cnn_metrics[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
            "vs_full_qwen": {key: result[key] - qwen_metrics[key] for key in ("accuracy", "macro_f1", "balanced_accuracy")},
        })
        replays.append(result)
    return {
        "protocol": {"scope": "offline replay of development-frozen policies on existing test predictions; not end-to-end latency", "test_policy_selection": "none"},
        "input_checks": {"test_images": int(len(labels)), "source_id_sets_equal": True, "ground_truth_equal": True},
        "endpoints": {"cnn_only": cnn_metrics, "qwen_full_call": {**qwen_metrics, "calls": int(len(labels)), "call_rate": 1.0}},
        "test_paired": paired(labels, cnn, qwen), "frozen_policy_replays": replays,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    dev = aligned_pair(read_jsonl(args.qwen_dev), read_jsonl(args.cnn_dev))
    policy = freeze_policy(dev)
    args.output.mkdir(parents=True)
    (args.output / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    # Test files are opened only after the development-only policy is persisted.
    test = aligned_pair(read_jsonl(args.qwen_test), read_jsonl(args.cnn_test))
    if set(dev["source_ids"]) & set(test["source_ids"]):
        raise ValueError("development and test source_id sets must not overlap")
    report = replay_test(test, policy)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
