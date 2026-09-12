#!/usr/bin/env python3
"""Evaluate the user-frozen BloodMNIST step_03000 adapter on official test only.

There is no training, calibration, development evaluation, or checkpoint selection
in this program.  It accepts one explicit official-test manifest and evaluates
either the raw base or the user-frozen ``step_03000`` adapter.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from vlm_common import IMAGE_SIZE, MIN_PIXELS, load_processor, set_seed


LABELS = (
    "basophil", "eosinophil", "erythroblast", "immature_granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
)
QUESTION = "Which blood cell type is shown? Answer with exactly one of: " + "; ".join(LABELS) + "."
SEED = 42
EXPECTED_TEST_IMAGES = 3_421
FROZEN_ADAPTER_BASENAME = "step_03000"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    variant = parser.add_mutually_exclusive_group(required=True)
    variant.add_argument("--adapter", type=Path,
                         help="User-frozen adapter; only step_03000 is accepted.")
    variant.add_argument("--base-only", action="store_true",
                         help="Evaluate the raw base model without loading an adapter.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def append_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def read_test_manifest(path):
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or set(manifest) != {"metadata", "records"}:
        raise ValueError("test manifest must be an object with exactly metadata and records")
    if (not isinstance(manifest["metadata"], dict) or manifest["metadata"].get("dataset") != "BloodMNIST" or
            manifest["metadata"].get("frozen_checkpoint") != "step_03000" or
            manifest["metadata"].get("frozen_at") != "2026-09-12"):
        raise ValueError("test manifest metadata must identify the 2026-09-12 BloodMNIST step_03000 freeze")
    if not isinstance(manifest["records"], list):
        raise ValueError("test manifest records must be an array")
    return manifest


def safe_test_image(value, source_id):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("images", "test"):
        raise ValueError(f"{source_id}: expected a safe images/test-relative path")
    return path.as_posix()


def validate_test_records(records):
    required = {
        "source_id", "image", "label", "class_name", "split", "official_split",
        "question", "answer", "task", "template_id",
    }
    source_ids, images, counts = set(), set(), Counter()
    cleaned = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("test manifest contains a non-object record")
        missing = required.difference(record)
        if missing:
            raise ValueError(f"test record missing fields {sorted(missing)}")
        source_id, label = record["source_id"], record["label"]
        if not isinstance(source_id, str) or not isinstance(label, int) or label not in range(8):
            raise ValueError("test record has invalid source_id or label")
        image = safe_test_image(record["image"], source_id)
        if (record["split"] != "test" or record["official_split"] != "test" or
                not source_id.startswith("bloodmnist:test:") or
                record["class_name"] != LABELS[label] or record["answer"] != LABELS[label] or
                record["question"] != QUESTION or record["task"] != "cell_type_classification" or
                record["template_id"] != "bloodmnist_cell_type_v1"):
            raise ValueError(f"{source_id}: non-canonical official BloodMNIST test record")
        if source_id in source_ids or image in images:
            raise ValueError(f"{source_id}: duplicate test source ID or image")
        source_ids.add(source_id)
        images.add(image)
        counts[label] += 1
        cleaned.append(dict(record, image=image))
    if len(cleaned) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"official test must contain {EXPECTED_TEST_IMAGES} images, got {len(cleaned)}")
    if set(counts) != set(range(8)):
        raise ValueError(f"official test must contain all eight classes, got {dict(counts)}")
    return cleaned, counts


def load_image(image_root, record):
    from PIL import Image
    path = image_root.joinpath(*PurePosixPath(record["image"]).parts)
    if not path.is_file():
        raise FileNotFoundError(f"missing official test image: {path}")
    with Image.open(path) as opened:
        return opened.convert("RGB")


def model_inputs(batch):
    allowed = {
        "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids",
        "pixel_values_videos", "video_grid_thw",
    }
    return {key: value for key, value in batch.items() if key in allowed}


def move_batch(batch):
    return {key: value.to("cuda") for key, value in model_inputs(batch).items()}


def score(rows):
    confusion = [[0] * 9 for _ in LABELS]
    for row in rows:
        predicted = LABELS.index(row["parsed"]) if row["parsed"] in LABELS else 8
        confusion[row["label"]][predicted] += 1
    recalls, f1_scores = [], []
    for label in range(8):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in range(8) if other != label)
        fn = sum(confusion[label][other] for other in range(9) if other != label)
        total = sum(confusion[label])
        recalls.append(tp / total if total else 0.0)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "n": len(rows),
        "accuracy": sum(confusion[label][label] for label in range(8)) / len(rows),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class_recall": {LABELS[label]: recalls[label] for label in range(8)},
        "per_class_f1": {LABELS[label]: f1_scores[label] for label in range(8)},
        "confusion_labels_columns": list(LABELS) + ["__invalid__"],
        "confusion_8x9": confusion,
        "invalid": sum(row["parsed"] == "__invalid__" for row in rows),
    }


def main(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    if args.adapter is not None and args.adapter.name != FROZEN_ADAPTER_BASENAME:
        raise ValueError(f"test evaluator is frozen to {FROZEN_ADAPTER_BASENAME}, got {args.adapter.name}")
    manifest = read_test_manifest(args.test_json)
    records, class_counts = validate_test_records(manifest["records"])

    import torch
    from peft import PeftModel
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    set_seed(SEED)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    processor = load_processor(args.base_model, MIN_PIXELS, IMAGE_SIZE)
    torch.cuda.reset_peak_memory_stats()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter).eval() if args.adapter else base.eval()
    model.config.use_cache = False
    predictions, generation_ms, e2e_ms = [], [], []
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            example_started = time.perf_counter()
            image = load_image(args.image_root, record)
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUESTION}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            batch = move_batch(processor(
                text=[prompt], images=[image], return_tensors="pt", min_pixels=MIN_PIXELS, max_pixels=IMAGE_SIZE
            ))
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**model_inputs(batch), max_new_tokens=32, do_sample=False)
            torch.cuda.synchronize()
            raw = processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            generated_ms = (time.perf_counter() - generation_started) * 1000
            total_ms = (time.perf_counter() - example_started) * 1000
            row = {
                "source_id": record["source_id"], "image": record["image"], "label": record["label"],
                "gt": record["answer"], "question": QUESTION, "raw_answer": raw,
                "parsed": raw if raw in LABELS else "__invalid__",
                "generation_ms": generated_ms, "e2e_ms": total_ms,
            }
            append_jsonl(handle, row)
            predictions.append(row)
            generation_ms.append(generated_ms)
            e2e_ms.append(total_ms)
            if index % 100 == 0 or index == len(records):
                print(json.dumps({"event": "test_progress", "done": index, "total": len(records)}), flush=True)
    metrics = score(predictions)
    write_json(args.output_dir / "test_metrics.json", metrics)
    runtime = {
        "elapsed_wall_seconds": time.perf_counter() - started,
        "generation_mean_ms": sum(generation_ms) / len(generation_ms),
        "generation_p95_ms": sorted(generation_ms)[int(0.95 * (len(generation_ms) - 1))],
        "e2e_mean_ms": sum(e2e_ms) / len(e2e_ms),
        "e2e_p95_ms": sorted(e2e_ms)[int(0.95 * (len(e2e_ms) - 1))],
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(args.output_dir / "run_metadata.json", {
        "cohort": "official frozen BloodMNIST test only", "test_records": len(records),
        "test_class_counts": {LABELS[label]: class_counts[label] for label in range(8)},
        "manifest": str(args.test_json), "manifest_metadata": manifest["metadata"],
        "variant": "adapter" if args.adapter else "base",
        "base_model": str(args.base_model), "adapter": str(args.adapter) if args.adapter else None,
        "frozen_checkpoint": FROZEN_ADAPTER_BASENAME if args.adapter else None,
        "checkpoint_freeze": {
            "frozen_by_user_on": "2026-09-12",
            "context": "The cohort was frozen when step_03000 was selected by the fixed 400-image dev protocol before the test was opened.",
            "selection_policy": "No test result may select or replace a checkpoint; base-only is the raw pre-SFT comparison arm.",
        },
        "generation": {"prompt": QUESTION, "max_new_tokens": 32, "do_sample": False,
                       "enable_thinking": False, "seed": SEED, "min_pixels": MIN_PIXELS, "max_pixels": IMAGE_SIZE},
        "invalid_policy": "Any raw answer outside the canonical eight labels is counted as __invalid__ and wrong.",
        "runtime": runtime,
    })
    print(json.dumps({"event": "complete", **metrics, "runtime": runtime}), flush=True)


if __name__ == "__main__":
    main(parse_args())
