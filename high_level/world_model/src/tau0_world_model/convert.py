"""Convert an official Step1X-Edit-v1p2 snapshot to the original weight format."""
import argparse
import json
import shutil
from pathlib import Path

from .common import import_vendor, sha256


def load_shards(directory, stem):
    from safetensors.torch import load_file
    index = directory / f"{stem}.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = [f"{stem}.safetensors"]
    state = {}
    for name in names:
        part = load_file(str(directory / name))
        if state.keys() & part.keys():
            raise ValueError("Duplicate tensor across weight shards")
        state.update(part)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import_vendor()
    from convert_to_original import convert_transformer_diffusers_to_original, convert_vae_diffusers_to_original
    from safetensors.torch import save_file
    args.output.mkdir(parents=True)
    for name, converter in (("transformer", convert_transformer_diffusers_to_original), ("vae", convert_vae_diffusers_to_original)):
        state = load_shards(args.input / name, "diffusion_pytorch_model")
        converted = converter(state)
        filename = "step1x-edit-v1p2.safetensors" if name == "transformer" else "vae.safetensors"
        save_file(converted, str(args.output / filename))
        del state, converted
    encoder = args.output / "Qwen2.5-VL-7B-Instruct"
    shutil.copytree(args.input / "text_encoder", encoder)
    # Official snapshots keep processing/tokenization assets separately.
    for file in (args.input / "processor").iterdir():
        if file.is_file():
            if not (encoder / file.name).exists():
                shutil.copyfile(file, encoder / file.name)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        encoder, use_fast=False, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28,
    )
    processor.save_pretrained(encoder)
    (args.output / "conversion.json").write_text(json.dumps({
        "base_model": "stepfun-ai/Step1X-Edit-v1p2",
        "weights": {p.name: sha256(p) for p in sorted(args.output.glob("*.safetensors"))},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
