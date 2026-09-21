#!/usr/bin/env python3
"""Create a reviewable proposal/full-model and world-model/LoRA release bundle."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--world-lora", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.proposal / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Expected a Qwen3.5 proposal model")
    with safe_open(args.proposal / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        if len(keys) != 760 or sum(k.startswith("model.visual.") for k in keys) != 333:
            raise ValueError("Unexpected proposal tensor structure")
        if any(k.startswith("model.language_model.visual.") for k in keys):
            raise ValueError("Legacy visual keys must not be released")
    with safe_open(args.world_lora, framework="pt") as f:
        metadata = f.metadata() or {}
        if metadata.get("ss_network_dim") != "64" or metadata.get("ss_network_alpha") != "32":
            raise ValueError("Expected the rank-64, alpha-32 robotics LoRA")
        lora_count = len(f.keys())
        if lora_count != 912:
            raise ValueError("Unexpected robotics LoRA tensor structure")
    proposal = args.output / "proposal"
    world = args.output / "world_model"
    proposal.mkdir(parents=True)
    world.mkdir()
    names = ["model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "chat_template.jinja", "processor_config.json"]
    for name in names:
        shutil.copyfile(args.proposal / name, proposal / name)
    allowed = {"ss_network_dim", "ss_network_alpha",
               "ss_network_module", "ss_base_model_version", "ss_discrete_flow_shift",
               "ss_timestep_sampling", "ss_model_prediction_type", "ss_guidance_scale"}
    # Keep tensor values intact; omit local dataset paths and run metadata.
    lora = world / "robotics-lora.safetensors"
    tensors = load_file(str(args.world_lora))
    save_file(tensors, str(lora), metadata={k: v for k, v in metadata.items() if k in allowed})
    import torch
    with safe_open(lora, framework="pt") as check:
        if set(check.keys()) != set(tensors) or not all(torch.equal(check.get_tensor(k), v) for k, v in tensors.items()):
            raise ValueError("LoRA tensor roundtrip mismatch")
    manifest = {
        "proposal": {"type": "full_weights", "model_type": "qwen3_5", "tensor_count": len(keys)},
        "world_model": {"type": "lora", "rank": 64, "alpha": 32,
                        "tensor_count": lora_count, "tensor_values_preserved": True,
                        "base_model": "stepfun-ai/Step1X-Edit-v1p2"},
        "files": {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(args.output.rglob("*")) if p.is_file()},
    }
    (args.output / "weights_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
