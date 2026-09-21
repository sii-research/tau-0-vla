"""Small-data assistant-only SFT, saving a portable LoRA plus processor."""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import set_seed

from .data import load_images, read_samples, sha256
from .format import answer_text, messages
from .runtime import encode, load_model


def training_batch(row, processor, max_length):
    prompt = messages(row, load_images(row))
    prefix = encode(processor, prompt)
    full = encode(processor, prompt + [{"role": "assistant", "content": [
        {"type": "text", "text": answer_text(row)}
    ]}], generation=False)
    n = prefix.input_ids.shape[1]
    if full.input_ids.shape[1] > max_length:
        raise ValueError(f"Sample {row['id']} exceeds max_length; shorten it rather than truncating its answer")
    if not torch.equal(full.input_ids[:, :n], prefix.input_ids):
        raise ValueError("Chat-template prefix mismatch; refusing incorrect supervision mask")
    labels = full.input_ids.clone()
    labels[:, :n] = -100
    if not (labels != -100).any():
        raise ValueError("Sample has no assistant supervision")
    full["labels"] = labels
    return full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Continue training a previously saved proposal adapter")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps <= 0 or args.rank <= 0 or args.learning_rate <= 0:
        raise ValueError("steps, rank and learning-rate must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    samples = read_samples(args.input, require_answers=True)
    set_seed(args.seed)
    model, processor = load_model(args.model, args.adapter, device=args.device, attention="sdpa",
                                  adapter_trainable=True)
    if not args.adapter:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ))
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    losses = []
    for step in range(args.steps):
        sample = samples[step % len(samples)]
        inputs = training_batch(sample, processor, args.max_length).to(args.device)
        loss = model(**inputs).loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        print(json.dumps({"step": step + 1, "loss": losses[-1], "grad_norm": float(norm)}), flush=True)
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True)
    processor.save_pretrained(args.output)
    (args.output / "training_record.json").write_text(json.dumps({
        "base_model": str(Path(args.model).resolve()), "input_sha256": sha256(args.input),
        "base_weights": {p.name: sha256(p) for p in sorted(Path(args.model).glob("*.safetensors"))},
        "steps": args.steps, "seed": args.seed, "learning_rate": args.learning_rate,
        "losses": losses, "camera_order": ["head", "left", "right"],
        "image_resize_hw": [480, 640], "enable_thinking": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
