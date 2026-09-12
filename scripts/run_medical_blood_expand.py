#!/usr/bin/env python3
"""Fixed BloodMNIST 4,000-image LoRA continuation from the audited epoch-01 adapter.

This runner deliberately accepts only official BloodMNIST train/validation manifests.
It never accepts or reads test data.  The starting adapter is a candidate at step 0;
each later candidate is evaluated on the same fixed 400-image development cohort.
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from vlm_common import (
    IMAGE_SIZE,
    MIN_PIXELS,
    build_supervised_batch,
    load_image,
    load_processor,
    model_inputs,
    move_batch,
    set_seed,
)


LABELS = (
    "basophil", "eosinophil", "erythroblast", "immature_granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
)
QUESTION = "Which blood cell type is shown? Answer with exactly one of: " + "; ".join(LABELS) + "."
SEED = 42
TRAIN_PER_CLASS = 500
DEV_PER_CLASS = 50
MAX_STEPS = 8_000
EPOCHS = 2
EVAL_INTERVAL = 1_000
LR = 5e-5
EXPECTED_TRAINABLE_PARAMS = 458_752


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--dev-json", type=Path, required=True)
    parser.add_argument("--reference-dev-json", type=Path, required=True,
                        help="Original 400-image dev manifest; exact source/image equality is required.")
    parser.add_argument("--reference-epoch01-metrics", type=Path, required=True,
                        help="Metrics emitted by the audited old epoch_01 adapter.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--hot-start-adapter", type=Path, required=True,
                        help="Audited old epoch_01 LoRA directory; loaded trainably with fresh AdamW.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_manifest(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected an object with a records array")
    return rows


def safe_relative_image(value, source_id):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{source_id}: image path is not a safe relative path")
    if "test" in {part.lower() for part in path.parts}:
        raise ValueError(f"{source_id}: test image path is forbidden")
    return path.as_posix()


def validate_records(rows, split, per_class):
    required = {
        "source_id", "image", "label", "class_name", "split", "official_split",
        "question", "answer", "task", "template_id",
    }
    seen_sources, seen_images, counts = set(), set(), Counter()
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{split}: manifest includes a non-object record")
        missing = required.difference(row)
        if missing:
            raise ValueError(f"{split}: record missing fields {sorted(missing)}")
        source_id = row["source_id"]
        label = row["label"]
        if not isinstance(source_id, str) or not isinstance(label, int) or label not in range(8):
            raise ValueError(f"{split}: invalid source_id or label")
        image = safe_relative_image(row["image"], source_id)
        expected_official = "train" if split == "train" else "val"
        if (row["split"] != split or row["official_split"] != expected_official or
                row["class_name"] != LABELS[label] or row["answer"] != LABELS[label] or
                row["question"] != QUESTION or row["task"] != "cell_type_classification"):
            raise ValueError(f"{source_id}: non-canonical BloodMNIST question/answer or split metadata")
        if source_id in seen_sources or image in seen_images:
            raise ValueError(f"{split}: duplicate source or image {source_id}")
        if not source_id.startswith(f"bloodmnist:{expected_official}:"):
            raise ValueError(f"{source_id}: source ID does not identify permitted official split")
        seen_sources.add(source_id)
        seen_images.add(image)
        counts[label] += 1
        cleaned.append(dict(row, image=image))
    expected = {label: per_class for label in range(8)}
    if dict(counts) != expected or len(cleaned) != 8 * per_class:
        raise ValueError(f"{split}: require exactly {per_class} records per class; got {dict(counts)}")
    return cleaned


def validate_fixed_dev(dev, reference_dev):
    reference = validate_records(reference_dev, "dev", DEV_PER_CLASS)
    fields = ("source_id", "image", "label", "class_name", "question", "answer", "official_split")
    left = {row["source_id"]: tuple(row[field] for field in fields[1:]) for row in dev}
    right = {row["source_id"]: tuple(row[field] for field in fields[1:]) for row in reference}
    if left != right:
        raise ValueError("expanded dev cohort differs from the original 400-image manifest")


def validate_isolation(train, dev):
    train_sources = {row["source_id"] for row in train}
    dev_sources = {row["source_id"] for row in dev}
    train_images = {row["image"] for row in train}
    dev_images = {row["image"] for row in dev}
    if train_sources & dev_sources or train_images & dev_images:
        raise ValueError("train/dev overlap is forbidden")


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def append_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def expected_targets(base):
    layer_types = list(base.config.text_config.layer_types)
    full_layers = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    if not full_layers:
        raise RuntimeError("base model reports no language full-attention layers")
    targets = [
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in full_layers for projection in ("q_proj", "v_proj")
    ]
    modules = dict(base.named_modules())
    missing = [target for target in targets if target not in modules]
    if missing:
        raise RuntimeError(f"base model missing expected LoRA targets: {missing}")
    shape_expected = sum(
        4 * (module.in_features + module.out_features)
        for target, module in modules.items() if target in targets
    )
    return full_layers, targets, shape_expected


def load_hot_start(base, adapter_dir, targets, shape_expected):
    from peft import PeftConfig, PeftModel
    config = PeftConfig.from_pretrained(adapter_dir)
    target_modules = set(config.target_modules or [])
    if (config.peft_type.value != "LORA" or config.r != 4 or config.lora_alpha != 8 or
            not math.isclose(float(config.lora_dropout), 0.05) or target_modules != set(targets)):
        raise RuntimeError("hot-start adapter does not match fixed r4/alpha8/dropout.05 q/v full-attention protocol")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_count = sum(item["numel"] for item in trainable)
    if (trainable_count != EXPECTED_TRAINABLE_PARAMS or trainable_count != shape_expected or
            not trainable or any("lora_" not in item["name"] or "visual" in item["name"] for item in trainable)):
        raise RuntimeError(
            f"hot-start trainable guard failed: actual={trainable_count}, shape_expected={shape_expected}, "
            f"protocol_expected={EXPECTED_TRAINABLE_PARAMS}"
        )
    return model, trainable


def evaluate(model, processor, records, image_root, output_dir, candidate, global_step):
    import torch
    model.eval()
    predictions = []
    prediction_path = output_dir / f"{candidate}_dev_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            image = load_image(image_root, record)
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUESTION}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            batch = move_batch(processor(
                text=[prompt], images=[image], return_tensors="pt", min_pixels=MIN_PIXELS, max_pixels=IMAGE_SIZE
            ), "cuda")
            with torch.inference_mode():
                generated = model.generate(**model_inputs(batch), max_new_tokens=32, do_sample=False)
            raw = processor.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            parsed = raw if raw in LABELS else "__invalid__"
            row = {
                "candidate": candidate, "global_step": global_step, "source_id": record["source_id"],
                "image": record["image"], "label": record["label"], "gt": record["answer"],
                "question": record["question"], "raw_answer": raw, "parsed": parsed,
            }
            append_jsonl(handle, row)
            predictions.append(row)
            print(json.dumps({"event": "dev_progress", "candidate": candidate, "done": index, "total": len(records)}), flush=True)
    confusion = [[0] * 9 for _ in LABELS]
    for row in predictions:
        prediction = LABELS.index(row["parsed"]) if row["parsed"] in LABELS else 8
        confusion[row["label"]][prediction] += 1
    f1 = []
    for label in range(8):
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in range(8) if other != label)
        fn = sum(confusion[label][other] for other in range(9) if other != label)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    metrics = {
        "candidate": candidate, "global_step": global_step, "n": len(predictions),
        "accuracy": sum(confusion[i][i] for i in range(8)) / len(predictions),
        "macro_f1": sum(f1) / len(f1), "per_class_f1": f1,
        "confusion_labels_columns": list(LABELS) + ["__invalid__"],
        "confusion_8x9": confusion,
        "invalid": sum(row["parsed"] == "__invalid__" for row in predictions),
    }
    write_json(output_dir / f"{candidate}_dev_metrics.json", metrics)
    return metrics


def compare_initial(initial, reference_path):
    with reference_path.open(encoding="utf-8") as handle:
        reference = json.load(handle)
    checked = ("accuracy", "macro_f1", "confusion_8x9", "invalid")
    differences = {key: {"initial": initial.get(key), "reference": reference.get(key)} for key in checked
                   if initial.get(key) != reference.get(key)}
    result = {"reference_epoch01_metrics": str(reference_path), "checked_fields": list(checked),
              "matches_reference": not differences, "differences": differences}
    if differences:
        raise RuntimeError(f"initial hot-start dev metrics differ from old epoch_01: {differences}")
    return result


def is_better(candidate, best):
    """Select higher macro F1, then higher accuracy, then earlier global step."""
    return (-candidate["macro_f1"], -candidate["accuracy"], candidate["global_step"]) < (
        -best["macro_f1"], -best["accuracy"], best["global_step"]
    )


def save_checkpoint(model, processor, output_dir, global_step):
    checkpoint = output_dir / f"step_{global_step:05d}"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {checkpoint}")
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)
    return checkpoint.name


def run_self_test():
    a = {"macro_f1": 0.9, "accuracy": 0.91, "global_step": 0}
    assert is_better({"macro_f1": 0.9, "accuracy": 0.92, "global_step": 1}, a)
    assert is_better({"macro_f1": 0.9, "accuracy": 0.91, "global_step": 2},
                     {"macro_f1": 0.9, "accuracy": 0.91, "global_step": 3})
    assert not is_better({"macro_f1": 0.9, "accuracy": 0.91, "global_step": 4}, a)
    print("BloodMNIST expanded-run metric selection checks passed")


def main(args):
    if args.self_test:
        run_self_test()
        return
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {args.output_dir}")
    started = time.perf_counter()
    train = validate_records(read_manifest(args.train_json), "train", TRAIN_PER_CLASS)
    dev = validate_records(read_manifest(args.dev_json), "dev", DEV_PER_CLASS)
    validate_fixed_dev(dev, read_manifest(args.reference_dev_json))
    validate_isolation(train, dev)
    import torch
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    set_seed(SEED)
    args.output_dir.mkdir(parents=True)
    processor = load_processor(args.base_model, MIN_PIXELS, IMAGE_SIZE)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    full_layers, targets, shape_expected = expected_targets(base)
    model, trainable = load_hot_start(base, args.hot_start_adapter, targets, shape_expected)
    model.config.use_cache = False
    metadata = {
        "protocol": "bloodmnist_expanded_v2_continuation", "seed": SEED,
        "train_json": str(args.train_json), "dev_json": str(args.dev_json),
        "reference_dev_json": str(args.reference_dev_json), "image_root": str(args.image_root),
        "base_model": str(args.base_model), "hot_start_adapter": str(args.hot_start_adapter),
        "data": {"train_records": len(train), "dev_records": len(dev), "train_per_class": TRAIN_PER_CLASS,
                 "dev_per_class": DEV_PER_CLASS, "test_policy": "No test manifest, image, or label is accepted or read."},
        "generation": {"max_new_tokens": 32, "do_sample": False, "enable_thinking": False, "prompt": QUESTION},
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.05, "full_attention_layers": full_layers,
                 "target_modules": targets, "trainable_parameter_count": sum(item["numel"] for item in trainable),
                 "shape_expected_parameter_count": shape_expected, "trainable_parameters": trainable},
        "optimizer": {"name": "AdamW", "fresh_after_hot_start": True, "lr": LR, "weight_decay": 0.01,
                      "batch_size": 1, "bf16": True, "clip_grad_norm": 1.0, "epochs": EPOCHS,
                      "max_steps": MAX_STEPS, "eval_interval_steps": EVAL_INTERVAL},
        "selection": "max macro_f1, then max accuracy, then earlier global_step; includes old epoch_01 at step 0",
    }
    write_json(args.output_dir / "run_metadata.json", metadata)
    initial = evaluate(model, processor, dev, args.image_root, args.output_dir, "initial_step_00000", 0)
    reproducibility = compare_initial(initial, args.reference_epoch01_metrics)
    write_json(args.output_dir / "initial_dev_reproducibility.json", reproducibility)
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.01)
    best = dict(initial, checkpoint=str(args.hot_start_adapter), source="hot_start_epoch_01")
    candidates = [best]
    write_json(args.output_dir / "checkpoint_candidates.json", candidates)
    write_json(args.output_dir / "best_checkpoint.json", best)
    global_step, interval_loss, interval_count = 0, 0.0, 0
    with (args.output_dir / "training_progress.jsonl").open("w", encoding="utf-8") as progress:
        for epoch in range(1, EPOCHS + 1):
            shuffled = list(train)
            random.Random(SEED + epoch).shuffle(shuffled)
            for record in shuffled:
                if global_step >= MAX_STEPS:
                    break
                batch, _ = build_supervised_batch(record, load_image(args.image_root, record), processor,
                                                  type("FixedArgs", (), {"min_pixels": MIN_PIXELS, "max_pixels": IMAGE_SIZE})())
                output = model(**move_batch(batch, "cuda"))
                loss = float(output.loss.item())
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                interval_loss += loss
                interval_count += 1
                if global_step % args.progress_interval == 0 or global_step % EVAL_INTERVAL == 0:
                    event = {"event": "train_progress", "epoch": epoch, "global_step": global_step,
                             "interval_steps": interval_count, "mean_loss": interval_loss / interval_count,
                             "last_loss": loss, "records_per_epoch": len(train)}
                    append_jsonl(progress, event)
                    print(json.dumps(event), flush=True)
                    interval_loss, interval_count = 0.0, 0
                if global_step % EVAL_INTERVAL == 0:
                    candidate_name = f"step_{global_step:05d}"
                    checkpoint = save_checkpoint(model, processor, args.output_dir, global_step)
                    metrics = evaluate(model, processor, dev, args.image_root, args.output_dir, candidate_name, global_step)
                    candidate = dict(metrics, checkpoint=checkpoint, source="expanded_train")
                    candidates.append(candidate)
                    append_jsonl(progress, {"event": "checkpoint_eval", **candidate})
                    print(json.dumps({"event": "checkpoint_eval", **candidate}), flush=True)
                    if is_better(candidate, best):
                        best = candidate
                    write_json(args.output_dir / "checkpoint_candidates.json", candidates)
                    write_json(args.output_dir / "best_checkpoint.json", best)
                    model.train()
                    model.config.use_cache = False
            if global_step >= MAX_STEPS:
                break
    write_json(args.output_dir / "checkpoint_candidates.json", candidates)
    write_json(args.output_dir / "best_checkpoint.json", best)
    metadata.update({"completed_steps": global_step, "best_checkpoint": best,
                     "elapsed_seconds": time.perf_counter() - started})
    write_json(args.output_dir / "run_metadata.json", metadata)
    print(json.dumps({"event": "complete", "completed_steps": global_step, "best": best}), flush=True)


if __name__ == "__main__":
    main(parse_args())
