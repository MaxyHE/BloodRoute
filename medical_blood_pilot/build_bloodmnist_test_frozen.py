#!/usr/bin/env python3
"""Build the one authorized frozen BloodMNIST official-test evaluation set."""
import argparse
import json
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
QUESTION = "Which blood cell type is shown? Answer with exactly one of: " + "; ".join(LABELS) + "."
FROZEN_CHECKPOINT = "step_03000"
FROZEN_DATE = "2026-09-12"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_records(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    if isinstance(value, list):
        return value
    raise ValueError(f"expected records list in {path}")


def source_id(index):
    return f"bloodmnist:test:{index:05d}"


def record(index, label):
    return {
        "source_id": source_id(index),
        "image": f"images/test/test_{index:05d}.png",
        "label": int(label),
        "class_name": LABELS[int(label)],
        "split": "test",
        "official_split": "test",
        "question": QUESTION,
        "answer": LABELS[int(label)],
        "task": "cell_type_classification",
        "template_id": "bloodmnist_cell_type_v1",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--verified-md5", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    train_records = read_records(args.train_manifest)
    dev_records = read_records(args.dev_manifest)
    if len(train_records) != 4000 or len(dev_records) != 400:
        raise ValueError(f"expected expanded train/dev counts 4000/400, got {len(train_records)}/{len(dev_records)}")
    if any(row.get("split") != "train" for row in train_records):
        raise ValueError("train manifest contains a non-train row")
    if any(row.get("split") != "dev" for row in dev_records):
        raise ValueError("dev manifest contains a non-dev row")

    args.output_dir.mkdir(parents=True)
    with np.load(args.source_npz, allow_pickle=False) as data:
        required = {"test_images", "test_labels"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing official test arrays: {sorted(missing)}")
        labels = data["test_labels"].reshape(-1).astype(int).tolist()
        if len(labels) != 3421:
            raise ValueError(f"unexpected official test count: {len(labels)}")
        if set(labels).difference(range(8)) or set(labels) != set(range(8)):
            raise ValueError("official test labels must include each class ID 0 through 7")
        target = args.output_dir / "images" / "test"
        target.mkdir(parents=True, exist_ok=False)
        images = data["test_images"]
        for index, label in enumerate(labels):
            image = images[index]
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError(f"unexpected test image at {index}: {image.dtype} {image.shape}")
            Image.fromarray(image, "RGB").save(target / f"test_{index:05d}.png", format="PNG")

    records = [record(index, label) for index, label in enumerate(labels)]
    test_ids = {row["source_id"] for row in records}
    test_images = {row["image"] for row in records}
    train_ids = {row["source_id"] for row in train_records}
    dev_ids = {row["source_id"] for row in dev_records}
    train_images = {row["image"] for row in train_records}
    dev_images = {row["image"] for row in dev_records}
    if test_ids.intersection(train_ids) or test_ids.intersection(dev_ids):
        raise ValueError("official test source IDs overlap development manifests")
    if test_images.intersection(train_images) or test_images.intersection(dev_images):
        raise ValueError("official test image paths overlap development manifests")

    metadata = {
        "dataset": "BloodMNIST",
        "source": "MedMNIST+ Zenodo record 10519652, bloodmnist_224.npz",
        "license": "CC BY 4.0",
        "class_names_by_label": list(LABELS),
        "image_shape": [224, 224, 3],
        "official_split": "test",
        "frozen_checkpoint": FROZEN_CHECKPOINT,
        "checkpoint_status": "fixed non-final-round evaluation checkpoint",
        "frozen_at": FROZEN_DATE,
        "post_test_selection_policy": "Do not choose a later checkpoint from these test results.",
        "verified_source_md5": args.verified_md5,
    }
    write_json(args.output_dir / "bloodmnist_test_manifest.json", {"metadata": metadata, "records": records})
    write_json(args.output_dir / "build_summary.json", {
        **metadata,
        "test_records": len(records),
        "test_per_class": {LABELS[label]: Counter(labels)[label] for label in range(8)},
        "test_train_source_id_intersection": len(test_ids.intersection(train_ids)),
        "test_dev_source_id_intersection": len(test_ids.intersection(dev_ids)),
        "test_train_image_intersection": len(test_images.intersection(train_images)),
        "test_dev_image_intersection": len(test_images.intersection(dev_images)),
        "all_exported_images_exist": all((args.output_dir / row["image"]).is_file() for row in records),
    })
    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.provenance_dir / "bloodmnist_224.provenance.json", {
        "url": "https://zenodo.org/records/10519652/files/bloodmnist_224.npz?download=1",
        "filename": "bloodmnist_224.npz",
        "md5": args.verified_md5,
        "license": "CC BY 4.0",
        "access_scope": "Official test arrays first accessed to build this frozen test-only evaluation set on 2026-09-12.",
        "frozen_checkpoint": FROZEN_CHECKPOINT,
    })


if __name__ == "__main__":
    main()
