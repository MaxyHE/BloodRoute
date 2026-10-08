#!/usr/bin/env python3
"""Full official-train BloodMNIST LoRA run; two epochs, natural class distribution.
Same base/LoRA configuration as rank-r4 control; fixed 400-image dev selects checkpoint.
Data size, class proportions, update budget and evaluation schedule differ from the 4k control.
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

# Shared public helpers preserve answer masking, image budgets and device transfer.
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
TRAIN_PER_CLASS = None
DEV_PER_CLASS = 50
TOTAL_STEPS = 23_918
LR_STAGE_1 = 2e-4
LR_STAGE_2 = 5e-5
LR_SWITCH_STEP = 11_959
EVAL_INTERVAL = 3_000
DROPOUT = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True, choices=(2, 4, 8))
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--dev-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def append_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a top-level records array")
    return rows


def safe_relative_image(value, source_id):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{source_id}: image path must be a safe relative path")
    if "test" in {part.lower() for part in path.parts}:
        raise ValueError(f"{source_id}: test image path is forbidden")
    return path.as_posix()


def validate_records(rows, split, per_class):
    expected_official = "train" if split == "train" else "val"
    required = {
        "source_id", "image", "label", "class_name", "split", "official_split",
        "question", "answer", "task", "template_id",
    }
    sources, images, counts, cleaned = set(), set(), Counter(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{split}: found non-object record")
        missing = required.difference(row)
        if missing:
            raise ValueError(f"{split}: missing fields {sorted(missing)}")
        source_id, label = row["source_id"], row["label"]
        if not isinstance(source_id, str) or not isinstance(label, int) or label not in range(8):
            raise ValueError(f"{split}: invalid source id or label")
        image = safe_relative_image(row["image"], source_id)
        if (row["split"] != split or row["official_split"] != expected_official or
                row["class_name"] != LABELS[label] or row["answer"] != LABELS[label] or
                row["question"] != QUESTION or row["task"] != "cell_type_classification"):
            raise ValueError(f"{source_id}: non-canonical BloodMNIST record")
        if source_id in sources or image in images:
            raise ValueError(f"{split}: duplicate source or image")
        if not source_id.startswith(f"bloodmnist:{expected_official}:"):
            raise ValueError(f"{source_id}: source ID does not identify the expected official split")
        sources.add(source_id)
        images.add(image)
        counts[label] += 1
        cleaned.append(dict(row, image=image))
    if (per_class is None and (len(cleaned) != 11959 or set(counts) != set(range(8)))) or (per_class is not None and dict(counts) != {label: per_class for label in range(8)}):
        raise ValueError(f"{split}: expected exactly {per_class} examples per class; got {dict(counts)}")
    return cleaned


def validate_isolation(train, dev):
    if ({row["source_id"] for row in train} & {row["source_id"] for row in dev} or
            {row["image"] for row in train} & {row["image"] for row in dev}):
        raise ValueError("train/dev overlap is forbidden")


def expected_targets(base):
    layer_types = list(base.config.text_config.layer_types)
    full_layers = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    targets = [
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in full_layers for projection in ("q_proj", "v_proj")
    ]
    modules = dict(base.named_modules())
    missing = [target for target in targets if target not in modules]
    if not full_layers or missing:
        raise RuntimeError(f"missing expected full-attention LoRA targets: {missing}")
    return full_layers, targets, modules


def setup_rank_lora(base, rank, targets, modules):
    from peft import LoraConfig, TaskType, get_peft_model

    alpha = 2 * rank  # Keeps alpha/r=2, the original r=4, alpha=8 scale.
    expected = sum(rank * (modules[name].in_features + modules[name].out_features) for name in targets)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=DROPOUT,
        target_modules=targets,
        bias="none",
    ))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    actual = sum(item["numel"] for item in trainable)
    if (actual != expected or not trainable or
            any("lora_" not in item["name"] or "visual" in item["name"] for item in trainable)):
        raise RuntimeError(f"LoRA trainable-parameter guard failed: actual={actual}, expected={expected}")
    return model, alpha, expected, trainable


def evaluate(model, processor, records, image_root, output_dir, candidate, global_step):
    import torch

    model.eval()
    predictions = []
    with (output_dir / f"{candidate}_dev_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            image = load_image(image_root, record)
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUESTION}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            batch = move_batch(processor(
                text=[prompt], images=[image], return_tensors="pt",
                min_pixels=MIN_PIXELS, max_pixels=IMAGE_SIZE,
            ), "cuda")
            with torch.inference_mode():
                generated = model.generate(**model_inputs(batch), max_new_tokens=32, do_sample=False)
            raw = processor.batch_decode(
                generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True,
            )[0].strip()
            parsed = raw if raw in LABELS else "__invalid__"
            row = {
                "candidate": candidate, "global_step": global_step,
                "source_id": record["source_id"], "image": record["image"],
                "label": record["label"], "gt": record["answer"],
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


def is_better(candidate, best):
    return (-candidate["macro_f1"], -candidate["accuracy"], candidate["global_step"]) < (
        -best["macro_f1"], -best["accuracy"], best["global_step"]
    )


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, text=True, capture_output=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(args):
    import torch
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output_dir}")
    if args.progress_interval <= 0:
        raise ValueError("progress interval must be positive")
    started = time.perf_counter()
    train = validate_records(read_records(args.train_json), "train", TRAIN_PER_CLASS)
    dev = validate_records(read_records(args.dev_json), "dev", DEV_PER_CLASS)
    validate_isolation(train, dev)
    args.output_dir.mkdir(parents=True)
    set_seed(SEED)
    processor = load_processor(args.base_model, MIN_PIXELS, IMAGE_SIZE)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    full_layers, targets, modules = expected_targets(base)
    model, alpha, expected, trainable = setup_rank_lora(base, args.rank, targets, modules)
    model.config.use_cache = False
    metadata = {
        "protocol": "bloodmnist_fulltrain_v1_from_scratch",
        "rank": args.rank,
        "alpha": alpha,
        "alpha_over_rank": alpha / args.rank,
        "seed": SEED,
        "base_model": str(args.base_model),
        "train_json": str(args.train_json), "train_json_sha256": sha256_file(args.train_json),
        "dev_json": str(args.dev_json), "dev_json_sha256": sha256_file(args.dev_json),
        "image_root": str(args.image_root),
        "data": {"train_records": len(train), "dev_records": len(dev),
                 "train_per_class": dict(Counter(row["label"] for row in train)), "dev_per_class": DEV_PER_CLASS,
                 "test_policy": "No test manifest, image, label, or checkpoint is accepted or read."},
        "initialization": "Fresh LoRA on the base Qwen3.5-4B checkpoint; no hot-start adapter.",
        "generation": {"max_new_tokens": 32, "do_sample": False, "enable_thinking": False, "prompt": QUESTION},
        "lora": {"rank": args.rank, "alpha": alpha, "dropout": DROPOUT,
                 "full_attention_layers": full_layers, "target_modules": targets,
                 "trainable_parameter_count": sum(item["numel"] for item in trainable),
                 "shape_expected_parameter_count": expected, "trainable_parameters": trainable},
        "optimizer": {"name": "AdamW", "lr_first_epoch": LR_STAGE_1,
                      "lr_second_epoch": LR_STAGE_2, "weight_decay": 0.01,
                      "batch_size": 1, "bf16": True, "clip_grad_norm": 1.0,
                      "total_steps": TOTAL_STEPS, "eval_interval_steps": EVAL_INTERVAL,
                      "optimizer_state": "retained across planned learning-rate change"},
        "selection": "Highest dev macro-F1, then dev accuracy, then earlier step. Test is not read.",
        "gpu_at_start": gpu_snapshot(),
    }
    write_json(args.output_dir / "run_metadata.json", metadata)
    initial = evaluate(model, processor, dev, args.image_root, args.output_dir, "initial_step_00000", 0)
    initial["checkpoint"] = None
    initial["source"] = "fresh_lora_equals_base"
    candidates, best = [initial], initial
    write_json(args.output_dir / "checkpoint_candidates.json", candidates)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LR_STAGE_1, weight_decay=0.01,
    )
    global_step = 0
    interval_loss = interval_count = 0
    with (args.output_dir / "training_progress.jsonl").open("w", encoding="utf-8") as progress:
        for epoch in range(1, 3):
            shuffled = list(train)
            random.Random(SEED + epoch).shuffle(shuffled)
            model.train()
            model.config.use_cache = False
            for record in shuffled:
                if global_step == LR_SWITCH_STEP:
                    for group in optimizer.param_groups:
                        group["lr"] = LR_STAGE_2
                    append_jsonl(progress, {"event": "lr_transition", "global_step": global_step, "lr": LR_STAGE_2})
                    print(json.dumps({"event": "lr_transition", "global_step": global_step, "lr": LR_STAGE_2}), flush=True)
                batch, _ = build_supervised_batch(
                    record, load_image(args.image_root, record), processor,
                    type("FixedArgs", (), {"min_pixels": MIN_PIXELS, "max_pixels": IMAGE_SIZE})(),
                )
                output = model(**move_batch(batch, "cuda"))
                loss = float(output.loss.item())
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                interval_loss += loss
                interval_count += 1
                if global_step % args.progress_interval == 0:
                    event = {"event": "train_progress", "epoch": epoch, "global_step": global_step,
                             "interval_steps": interval_count, "mean_loss": interval_loss / interval_count,
                             "last_loss": loss, "lr": optimizer.param_groups[0]["lr"]}
                    append_jsonl(progress, event)
                    print(json.dumps(event), flush=True)
                    interval_loss = interval_count = 0
                if global_step % EVAL_INTERVAL == 0 or global_step == TOTAL_STEPS:
                    candidate_name = f"step_{global_step:05d}"
                    checkpoint = args.output_dir / candidate_name
                    model.save_pretrained(checkpoint)
                    processor.save_pretrained(checkpoint)
                    metrics = evaluate(model, processor, dev, args.image_root, args.output_dir, candidate_name, global_step)
                    candidate = dict(metrics, checkpoint=candidate_name, source="from_scratch_full_data")
                    candidates.append(candidate)
                    append_jsonl(progress, {"event": "checkpoint_eval", **candidate})
                    print(json.dumps({"event": "checkpoint_eval", **candidate}), flush=True)
                    if is_better(candidate, best):
                        best = candidate
                    write_json(args.output_dir / "checkpoint_candidates.json", candidates)
                    write_json(args.output_dir / "best_checkpoint.json", best)
                    model.train()
                    model.config.use_cache = False
    if global_step != TOTAL_STEPS:
        raise RuntimeError(f"unexpected completed steps: {global_step}")
    torch.cuda.synchronize()
    metadata.update({
        "completed_steps": global_step,
        "best_checkpoint": best,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_at_end": gpu_snapshot(),
    })
    write_json(args.output_dir / "checkpoint_candidates.json", candidates)
    write_json(args.output_dir / "best_checkpoint.json", best)
    write_json(args.output_dir / "run_metadata.json", metadata)
    print(json.dumps({"event": "complete", "rank": args.rank, "completed_steps": global_step, "best": best}), flush=True)


if __name__ == "__main__":
    main(parse_args())
