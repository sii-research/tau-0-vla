"""Generate one or a batch of goal images with the released robotics LoRA."""
import argparse
import json
from pathlib import Path

from .common import import_vendor, model_files, prompt, read_pairs, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--lora", type=Path, help="Omit for a base-model comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="JSONL batch manifest")
    group.add_argument("--image", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--size", type=int, choices=(256, 512, 768, 1024), default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.steps <= 0 or args.seed < 0 or not 0 <= args.shard < args.num_shards:
        raise ValueError("Invalid steps, seed or shard")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input:
        rows = read_pairs(args.input)
    else:
        if not args.instruction or not args.image.is_file():
            raise ValueError("Single-image mode requires --image and --instruction")
        rows = [{"id": "goal", "image": str(args.image.resolve()), "prompt": prompt(args.instruction)}]
    rows = rows[args.shard::args.num_shards]
    if not rows:
        raise ValueError("Empty shard")
    files = model_files(args.model)
    if args.lora and not args.lora.is_file():
        raise FileNotFoundError(args.lora)
    import torch
    from PIL import Image
    torch.cuda.set_device(args.device)
    import_vendor()
    from inference import ImageGenerator
    generator = ImageGenerator(**files, lora=str(args.lora) if args.lora else None,
                               device=args.device, mode="torch", version="v1p2",
                               max_length=640, quantized=False, offload=False)
    generator.dit.eval()
    generator.ae.eval()
    generator.llm_encoder.eval()
    args.output.mkdir(parents=True)
    manifest = {"model": str(args.model.resolve()), "lora_sha256": sha256(args.lora) if args.lora else None,
                "seed": args.seed, "steps": args.steps, "cfg": args.cfg, "size": args.size, "results": []}
    for row in rows:
        # The VAE samples from the global RNG before diffusion noise is created.
        # Reset per request so a sample is reproducible in any batch or shard.
        torch.manual_seed(args.seed)
        with Image.open(row["image"]) as image:
            images = generator.generate_image(
                prompt=row["prompt"], negative_prompt="", ref_images=image.convert("RGB"),
                num_samples=1, num_steps=args.steps, cfg_guidance=args.cfg, seed=args.seed,
                size_level=args.size, show_progress=True,
            )
        output = args.output / f"{row['id']}.png"
        images[0].save(output)
        manifest["results"].append({"id": row["id"], "image": row["image"],
                                    "prompt": row["prompt"], "output": output.name})
        (args.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
