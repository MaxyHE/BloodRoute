"""Shared Qwen vision-language helpers for the BloodMNIST experiments."""

import json
import random
from pathlib import PurePosixPath

import numpy as np


IMAGE_SIZE = 448 * 448
MIN_PIXELS = 65_536
MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
    "pixel_values_videos",
    "video_grid_thw",
}


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def load_processor(path, min_pixels, max_pixels):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    # Override the pretraining configuration's large longest edge both here and
    # at call sites so image-token budgets stay fixed for this protocol.
    processor.image_processor.size = {
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels,
    }
    return processor


def prompt_messages(record, image):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": record["question"]},
    ]}]


def process_text_and_image(processor, text, image, args):
    return processor(
        text=[text], images=[image], return_tensors="pt", padding=False,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
    )


def ids_have_prompt_prefix(prompt_ids, full_ids):
    import torch
    return full_ids.shape[1] > prompt_ids.shape[1] and torch.equal(
        full_ids[:, :prompt_ids.shape[1]], prompt_ids
    )


def build_supervised_batch(record, image, processor, args):
    """Supervise only a separately verified prompt continuation plus one EOS."""
    prompt = processor.apply_chat_template(
        prompt_messages(record, image), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_inputs = process_text_and_image(processor, prompt, image, args)
    if not processor.tokenizer.eos_token:
        raise RuntimeError("tokenizer has no eos_token text; cannot supervise answer + EOS")
    full_inputs = process_text_and_image(
        processor, prompt + record["answer"] + processor.tokenizer.eos_token, image, args
    )
    if not ids_have_prompt_prefix(prompt_inputs["input_ids"], full_inputs["input_ids"]):
        raise RuntimeError(
            f"{record['source_id']}: prompt token IDs are not a prefix of full token IDs. "
            "Refusing to train with an unverified answer boundary."
        )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    if not (labels != -100).any():
        raise RuntimeError(f"{record['source_id']}: no answer/EOS tokens remain after masking")
    full_inputs["labels"] = labels
    return full_inputs, prompt_len


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items() if key in MODEL_INPUT_KEYS | {"labels"}}


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS}


def load_image(root, record):
    from PIL import Image
    path = root.joinpath(*PurePosixPath(record["image"]).parts)
    if not path.is_file():
        raise FileNotFoundError(f"missing image: {path}")
    with Image.open(path) as opened:
        return opened.convert("RGB")


def setup_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model
    layer_types = list(model.config.text_config.layer_types)
    full_layers = [idx for idx, layer_type in enumerate(layer_types) if layer_type == "full_attention"]
    if not full_layers:
        raise RuntimeError("Qwen config contains no language full-attention layers")
    target_modules = [
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in full_layers for projection in ("q_proj", "v_proj")
    ]
    available = {name for name, _ in model.named_modules()}
    missing = [name for name in target_modules if name not in available]
    if missing:
        raise RuntimeError(f"expected language attention modules missing: {missing}")
    expected_trainable = sum(
        4 * (module.in_features + module.out_features)
        for name, module in model.named_modules() if name in target_modules
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    ))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("visual" in name for name in trainable):
        raise RuntimeError("LoRA unexpectedly left no trainable parameters or changed vision parameters")
    return model, full_layers, target_modules, trainable, expected_trainable


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
