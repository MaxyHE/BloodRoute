#!/usr/bin/env python3
"""Build train/validation-only BloodMNIST pilot data from official 224px NPZ.

This builder deliberately never reads the packaged test arrays and creates no test
manifest.  It exports selected official-train images and all official-validation
images as lossless PNG files under a new output directory.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = (
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature_granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
)
SEED = 42
QUESTION = "Which blood cell type is shown? Answer with exactly one of: " + "; ".join(LABELS) + "."


def source_id(official_split, index):
    return f"bloodmnist:{official_split}:{index:05d}"


def record(official_split, index, label, split):
    return {
        "source_id": source_id(official_split, index),
        "image": f"images/{official_split}/{official_split}_{index:05d}.png",
        "label": int(label),
        "class_name": LABELS[int(label)],
        "split": split,
        "official_split": official_split,
        "question": QUESTION,
        "answer": LABELS[int(label)],
        "task": "cell_type_classification",
        "template_id": "bloodmnist_cell_type_v1",
    }


def choose_per_class(labels, cap):
    chosen = []
    for label in range(len(LABELS)):
        candidates = [i for i, value in enumerate(labels) if value == label]
        rng = random.Random(SEED + label)
        rng.shuffle(candidates)
        chosen.extend(sorted(candidates[:cap]))
    return sorted(chosen)


def pngs(images, indices, official_split, output_dir):
    target = output_dir / "images" / official_split
    target.mkdir(parents=True, exist_ok=False)
    for index in indices:
        array = images[index]
        if array.dtype != np.uint8 or array.shape != (224, 224, 3):
            raise ValueError(f"unexpected {official_split} image at {index}: {array.dtype} {array.shape}")
        Image.fromarray(array, "RGB").save(target / f"{official_split}_{index:05d}.png", format="PNG")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--train-cap-per-class", type=int, default=250)
    parser.add_argument("--dev-cap-per-class", type=int, default=50)
    parser.add_argument("--verified-md5", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.train_cap_per_class <= 0 or args.dev_cap_per_class <= 0:
        raise ValueError("per-class caps must be positive")

    args.output_dir.mkdir(parents=True)
    with np.load(args.source_npz, allow_pickle=False) as data:
        required = {"train_images", "train_labels", "val_images", "val_labels"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing required train/val arrays: {sorted(missing)}")
        # Do not access test_images or test_labels: test stays outside development.
        train_labels = data["train_labels"].reshape(-1).astype(int).tolist()
        val_labels = data["val_labels"].reshape(-1).astype(int).tolist()
        if len(train_labels) != 11959 or len(val_labels) != 1712:
            raise ValueError(f"unexpected official counts: train={len(train_labels)}, val={len(val_labels)}")
        if set(train_labels).difference(range(8)) or set(val_labels).difference(range(8)):
            raise ValueError("labels must be integers in [0, 7]")

        train_indices = choose_per_class(train_labels, args.train_cap_per_class)
        dev_indices = choose_per_class(val_labels, args.dev_cap_per_class)
        all_val_indices = list(range(len(val_labels)))
        pngs(data["train_images"], train_indices, "train", args.output_dir)
        pngs(data["val_images"], all_val_indices, "val", args.output_dir)

    train_records = [record("train", i, train_labels[i], "train") for i in train_indices]
    val_records = [record("val", i, val_labels[i], "val") for i in all_val_indices]
    dev_records = [record("val", i, val_labels[i], "dev") for i in dev_indices]
    train_counts = Counter(row["class_name"] for row in train_records)
    val_counts = Counter(row["class_name"] for row in val_records)
    dev_counts = Counter(row["class_name"] for row in dev_records)
    if len(train_records) != sum(min(args.train_cap_per_class, train_labels.count(i)) for i in range(8)):
        raise AssertionError("incorrect train selection count")
    if any(dev_counts[name] != args.dev_cap_per_class for name in LABELS):
        raise ValueError("official validation has fewer than the requested dev cap for a class")

    metadata = {
        "dataset": "BloodMNIST",
        "source": "MedMNIST+ Zenodo record 10519652, bloodmnist_224.npz",
        "license": "CC BY 4.0",
        "seed": SEED,
        "class_names_by_label": list(LABELS),
        "image_shape": [224, 224, 3],
        "test_policy": "No test arrays, images, labels, records, or selections were read or written by this builder.",
    }
    write_json(args.output_dir / "bloodmnist_train_manifest.json", {"metadata": metadata, "records": train_records})
    write_json(args.output_dir / "bloodmnist_dev_50_per_class_manifest.json", {"metadata": metadata, "records": dev_records})
    # The linear probe consumes this plain train+full-validation list and rejects test rows.
    write_json(args.output_dir / "bloodmnist_train_val_manifest.json", train_records + val_records)
    write_json(args.output_dir / "val_dev_50_per_class.json", {
        "dataset": "BloodMNIST",
        "official_split": "val",
        "seed": SEED,
        "selection": "stratified per class using random.Random(42 + label) over original official validation indices",
        "class_names_by_label": list(LABELS),
        "source_ids": [row["source_id"] for row in dev_records],
    })
    write_json(args.output_dir / "build_summary.json", {
        **metadata,
        "train_records": len(train_records),
        "full_val_records": len(val_records),
        "dev_records": len(dev_records),
        "train_per_class": {name: train_counts[name] for name in LABELS},
        "full_val_per_class": {name: val_counts[name] for name in LABELS},
        "dev_per_class": {name: dev_counts[name] for name in LABELS},
        "train_dev_source_id_intersection": len({r["source_id"] for r in train_records}.intersection(r["source_id"] for r in dev_records)),
        "file_md5": args.verified_md5,
    })
    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.provenance_dir / "bloodmnist_224.provenance.json", {
        "url": "https://zenodo.org/records/10519652/files/bloodmnist_224.npz?download=1",
        "filename": "bloodmnist_224.npz",
        "md5": "b718ff6835fcbdb22ba9eacccd7b2601",
        "license": "CC BY 4.0",
        "builder_test_access": "No test arrays were read during construction.",
    })


if __name__ == "__main__":
    main()
