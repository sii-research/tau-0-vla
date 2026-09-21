"""Continue the published robotics LoRA using user-provided image pairs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import VENDOR, model_files, read_pairs, sha256


def prepare_data(rows, output, size):
    import toml
    from PIL import Image
    media = output / "data"
    media.mkdir(parents=True)
    metadata = {}
    for row in rows:
        paths = []
        for key in ("image", "target"):
            dst = media / f"{row['id']}_{key}.png"
            with Image.open(row[key]) as image:
                image.convert("RGB").save(dst)
            paths.append(str(dst))
        metadata[paths[1]] = {"caption": row["prompt"], "ref_image_path": paths[0]}
    meta = output / "metadata.json"
    meta.write_text(json.dumps(metadata, indent=2) + "\n")
    config = output / "dataset.toml"
    config.write_text(toml.dumps({"general": {"shuffle_caption": False}, "datasets": [{
        "resolution": [size, size], "batch_size": 1, "edit_dataset": True,
        "enable_bucket": True, "min_bucket_reso": 256, "max_bucket_reso": size,
        "bucket_reso_steps": 64, "subsets": [{"image_dir": str(media), "metadata_file": str(meta)}],
    }]}))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("steps and learning-rate must be positive")
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    files = model_files(args.model)
    if not args.lora.is_file():
        raise FileNotFoundError(args.lora)
    rows = read_pairs(args.input, training=True)
    args.output.mkdir(parents=True)
    config = prepare_data(rows, args.output, args.size)
    command = [
        sys.executable, "-u", str(VENDOR / "finetuning.py"),
        "--pretrained_model_name_or_path", files["dit_path"],
        "--qwen2p5vl", files["qwen2vl_model_path"], "--ae", files["ae_path"],
        "--network_module", "library.lora_module", "--network_weights", str(args.lora.resolve()),
        "--dim_from_weights", "--network_train_unet_only",
        "--dataset_config", str(config), "--output_dir", str(args.output / "weights"),
        "--output_name", "robotics", "--max_train_steps", str(args.steps),
        "--learning_rate", str(args.learning_rate), "--optimizer_type", "adamw",
        "--lr_scheduler", "constant", "--seed", str(args.seed), "--sdpa",
        "--mixed_precision", "bf16", "--save_precision", "bf16", "--save_model_as", "safetensors",
        "--gradient_checkpointing", "--cache_latents", "--cache_text_encoder_outputs",
        "--cache_latents_to_disk", "--cache_text_encoder_outputs_to_disk",
        "--max_data_loader_n_workers", "0", "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.1582", "--model_prediction_type", "raw",
        "--guidance_scale", "1.0",
    ]
    record = {"input_sha256": sha256(args.input), "initial_lora_sha256": sha256(args.lora),
              "samples": len(rows), "command": command, "complete": False}
    record_path = args.output / "training_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(command, cwd=VENDOR, check=True)
    weight = args.output / "weights/robotics.safetensors"
    if not weight.is_file():
        raise FileNotFoundError("Trainer did not save the expected LoRA")
    record.update({"complete": True, "output_lora_sha256": sha256(weight)})
    record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
