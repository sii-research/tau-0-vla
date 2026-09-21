"""
Convert diffusers-format Step1X-Edit-v1p2 weights to original format.

Input:  ./models/Step1X-Edit-v1p2/ (diffusers format from HuggingFace)
Output: ./models/Step1X-Edit-v1p2-original/
        ├── step1x-edit-v1p2.safetensors   (transformer, ~24GB)
        └── vae.safetensors                 (VAE, ~330MB)

The text_encoder (Qwen2.5-VL-7B) is used directly via transformers and doesn't need conversion.
"""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def swap_scale_shift(weight):
    """In diffusers, AdaLayerNorm splits into [scale, shift]; original uses [shift, scale]."""
    scale, shift = weight.chunk(2, dim=0)
    return torch.cat([shift, scale], dim=0)


def convert_transformer_diffusers_to_original(
    state_dict, num_layers=19, num_single_layers=38, inner_dim=3072, mlp_ratio=4.0
):
    """Convert diffusers Step1XEditTransformer2DModel state dict to original Step1XEdit format."""
    converted = {}

    # time_embed -> time_in
    converted["time_in.in_layer.weight"] = state_dict.pop("time_embed.in_layer.weight")
    converted["time_in.in_layer.bias"] = state_dict.pop("time_embed.in_layer.bias")
    converted["time_in.out_layer.weight"] = state_dict.pop("time_embed.out_layer.weight")
    converted["time_in.out_layer.bias"] = state_dict.pop("time_embed.out_layer.bias")

    # vec_embed -> vector_in
    converted["vector_in.in_layer.weight"] = state_dict.pop("vec_embed.in_layer.weight")
    converted["vector_in.in_layer.bias"] = state_dict.pop("vec_embed.in_layer.bias")
    converted["vector_in.out_layer.weight"] = state_dict.pop("vec_embed.out_layer.weight")
    converted["vector_in.out_layer.bias"] = state_dict.pop("vec_embed.out_layer.bias")

    # context_embedder -> txt_in
    converted["txt_in.weight"] = state_dict.pop("context_embedder.weight")
    converted["txt_in.bias"] = state_dict.pop("context_embedder.bias")

    # x_embedder -> img_in
    converted["img_in.weight"] = state_dict.pop("x_embedder.weight")
    converted["img_in.bias"] = state_dict.pop("x_embedder.bias")

    # connector keys pass through
    for key in list(state_dict.keys()):
        if "connector" in key:
            converted[key] = state_dict.pop(key)

    # double transformer blocks
    for i in range(num_layers):
        bp = f"transformer_blocks.{i}."
        # norms
        converted[f"double_blocks.{i}.img_mod.lin.weight"] = state_dict.pop(f"{bp}norm1.linear.weight")
        converted[f"double_blocks.{i}.img_mod.lin.bias"] = state_dict.pop(f"{bp}norm1.linear.bias")
        converted[f"double_blocks.{i}.txt_mod.lin.weight"] = state_dict.pop(f"{bp}norm1_context.linear.weight")
        converted[f"double_blocks.{i}.txt_mod.lin.bias"] = state_dict.pop(f"{bp}norm1_context.linear.bias")

        # Q, K, V -> fused qkv
        sample_q = state_dict.pop(f"{bp}attn.to_q.weight")
        sample_k = state_dict.pop(f"{bp}attn.to_k.weight")
        sample_v = state_dict.pop(f"{bp}attn.to_v.weight")
        converted[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.cat([sample_q, sample_k, sample_v], dim=0)

        sample_q_bias = state_dict.pop(f"{bp}attn.to_q.bias")
        sample_k_bias = state_dict.pop(f"{bp}attn.to_k.bias")
        sample_v_bias = state_dict.pop(f"{bp}attn.to_v.bias")
        converted[f"double_blocks.{i}.img_attn.qkv.bias"] = torch.cat([sample_q_bias, sample_k_bias, sample_v_bias], dim=0)

        context_q = state_dict.pop(f"{bp}attn.add_q_proj.weight")
        context_k = state_dict.pop(f"{bp}attn.add_k_proj.weight")
        context_v = state_dict.pop(f"{bp}attn.add_v_proj.weight")
        converted[f"double_blocks.{i}.txt_attn.qkv.weight"] = torch.cat([context_q, context_k, context_v], dim=0)

        context_q_bias = state_dict.pop(f"{bp}attn.add_q_proj.bias")
        context_k_bias = state_dict.pop(f"{bp}attn.add_k_proj.bias")
        context_v_bias = state_dict.pop(f"{bp}attn.add_v_proj.bias")
        converted[f"double_blocks.{i}.txt_attn.qkv.bias"] = torch.cat([context_q_bias, context_k_bias, context_v_bias], dim=0)

        # qk norm
        converted[f"double_blocks.{i}.img_attn.norm.query_norm.scale"] = state_dict.pop(f"{bp}attn.norm_q.weight")
        converted[f"double_blocks.{i}.img_attn.norm.key_norm.scale"] = state_dict.pop(f"{bp}attn.norm_k.weight")
        converted[f"double_blocks.{i}.txt_attn.norm.query_norm.scale"] = state_dict.pop(f"{bp}attn.norm_added_q.weight")
        converted[f"double_blocks.{i}.txt_attn.norm.key_norm.scale"] = state_dict.pop(f"{bp}attn.norm_added_k.weight")

        # ff
        converted[f"double_blocks.{i}.img_mlp.0.weight"] = state_dict.pop(f"{bp}ff.net.0.proj.weight")
        converted[f"double_blocks.{i}.img_mlp.0.bias"] = state_dict.pop(f"{bp}ff.net.0.proj.bias")
        converted[f"double_blocks.{i}.img_mlp.2.weight"] = state_dict.pop(f"{bp}ff.net.2.weight")
        converted[f"double_blocks.{i}.img_mlp.2.bias"] = state_dict.pop(f"{bp}ff.net.2.bias")
        converted[f"double_blocks.{i}.txt_mlp.0.weight"] = state_dict.pop(f"{bp}ff_context.net.0.proj.weight")
        converted[f"double_blocks.{i}.txt_mlp.0.bias"] = state_dict.pop(f"{bp}ff_context.net.0.proj.bias")
        converted[f"double_blocks.{i}.txt_mlp.2.weight"] = state_dict.pop(f"{bp}ff_context.net.2.weight")
        converted[f"double_blocks.{i}.txt_mlp.2.bias"] = state_dict.pop(f"{bp}ff_context.net.2.bias")

        # output projections
        converted[f"double_blocks.{i}.img_attn.proj.weight"] = state_dict.pop(f"{bp}attn.to_out.0.weight")
        converted[f"double_blocks.{i}.img_attn.proj.bias"] = state_dict.pop(f"{bp}attn.to_out.0.bias")
        converted[f"double_blocks.{i}.txt_attn.proj.weight"] = state_dict.pop(f"{bp}attn.to_add_out.weight")
        converted[f"double_blocks.{i}.txt_attn.proj.bias"] = state_dict.pop(f"{bp}attn.to_add_out.bias")

    # single transformer blocks
    for i in range(num_single_layers):
        bp = f"single_transformer_blocks.{i}."
        converted[f"single_blocks.{i}.modulation.lin.weight"] = state_dict.pop(f"{bp}norm.linear.weight")
        converted[f"single_blocks.{i}.modulation.lin.bias"] = state_dict.pop(f"{bp}norm.linear.bias")

        # Q, K, V, mlp -> fused linear1
        q = state_dict.pop(f"{bp}attn.to_q.weight")
        k = state_dict.pop(f"{bp}attn.to_k.weight")
        v = state_dict.pop(f"{bp}attn.to_v.weight")
        mlp = state_dict.pop(f"{bp}proj_mlp.weight")
        converted[f"single_blocks.{i}.linear1.weight"] = torch.cat([q, k, v, mlp], dim=0)

        q_bias = state_dict.pop(f"{bp}attn.to_q.bias")
        k_bias = state_dict.pop(f"{bp}attn.to_k.bias")
        v_bias = state_dict.pop(f"{bp}attn.to_v.bias")
        mlp_bias = state_dict.pop(f"{bp}proj_mlp.bias")
        converted[f"single_blocks.{i}.linear1.bias"] = torch.cat([q_bias, k_bias, v_bias, mlp_bias], dim=0)

        # qk norm
        converted[f"single_blocks.{i}.norm.query_norm.scale"] = state_dict.pop(f"{bp}attn.norm_q.weight")
        converted[f"single_blocks.{i}.norm.key_norm.scale"] = state_dict.pop(f"{bp}attn.norm_k.weight")

        # output projection
        converted[f"single_blocks.{i}.linear2.weight"] = state_dict.pop(f"{bp}proj_out.weight")
        converted[f"single_blocks.{i}.linear2.bias"] = state_dict.pop(f"{bp}proj_out.bias")

    # final layer
    converted["final_layer.linear.weight"] = state_dict.pop("proj_out.weight")
    converted["final_layer.linear.bias"] = state_dict.pop("proj_out.bias")
    converted["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(state_dict.pop("norm_out.linear.weight"))
    converted["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(state_dict.pop("norm_out.linear.bias"))

    # any remaining keys
    if state_dict:
        print(f"  Warning: {len(state_dict)} unconverted keys remaining: {list(state_dict.keys())}")
        for key in list(state_dict.keys()):
            converted[key] = state_dict.pop(key)

    return converted


def convert_vae_diffusers_to_original(state_dict):
    """Convert diffusers AutoencoderKL state dict to original AutoEncoder format."""
    converted = {}

    # encoder conv_in / conv_out (same keys)
    converted["encoder.conv_in.weight"] = state_dict.pop("encoder.conv_in.weight")
    converted["encoder.conv_in.bias"] = state_dict.pop("encoder.conv_in.bias")
    converted["encoder.conv_out.weight"] = state_dict.pop("encoder.conv_out.weight")
    converted["encoder.conv_out.bias"] = state_dict.pop("encoder.conv_out.bias")

    # encoder norm_out
    converted["encoder.norm_out.weight"] = state_dict.pop("encoder.conv_norm_out.weight")
    converted["encoder.norm_out.bias"] = state_dict.pop("encoder.conv_norm_out.bias")

    # encoder down blocks
    for i in range(4):
        for j in range(2):
            for k in range(1, 3):
                converted[f"encoder.down.{i}.block.{j}.conv{k}.weight"] = state_dict.pop(
                    f"encoder.down_blocks.{i}.resnets.{j}.conv{k}.weight")
                converted[f"encoder.down.{i}.block.{j}.conv{k}.bias"] = state_dict.pop(
                    f"encoder.down_blocks.{i}.resnets.{j}.conv{k}.bias")
                converted[f"encoder.down.{i}.block.{j}.norm{k}.weight"] = state_dict.pop(
                    f"encoder.down_blocks.{i}.resnets.{j}.norm{k}.weight")
                converted[f"encoder.down.{i}.block.{j}.norm{k}.bias"] = state_dict.pop(
                    f"encoder.down_blocks.{i}.resnets.{j}.norm{k}.bias")
        # downsample
        if i != 3:
            converted[f"encoder.down.{i}.downsample.conv.weight"] = state_dict.pop(
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight")
            converted[f"encoder.down.{i}.downsample.conv.bias"] = state_dict.pop(
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias")
        # shortcut
        if i == 1 or i == 2:
            converted[f"encoder.down.{i}.block.0.nin_shortcut.weight"] = state_dict.pop(
                f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.weight")
            converted[f"encoder.down.{i}.block.0.nin_shortcut.bias"] = state_dict.pop(
                f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.bias")

    # encoder mid attention
    converted["encoder.mid.attn_1.q.weight"] = state_dict.pop(
        "encoder.mid_block.attentions.0.to_q.weight").unsqueeze(-1).unsqueeze(-1)
    converted["encoder.mid.attn_1.q.bias"] = state_dict.pop("encoder.mid_block.attentions.0.to_q.bias")
    converted["encoder.mid.attn_1.k.weight"] = state_dict.pop(
        "encoder.mid_block.attentions.0.to_k.weight").unsqueeze(-1).unsqueeze(-1)
    converted["encoder.mid.attn_1.k.bias"] = state_dict.pop("encoder.mid_block.attentions.0.to_k.bias")
    converted["encoder.mid.attn_1.v.weight"] = state_dict.pop(
        "encoder.mid_block.attentions.0.to_v.weight").unsqueeze(-1).unsqueeze(-1)
    converted["encoder.mid.attn_1.v.bias"] = state_dict.pop("encoder.mid_block.attentions.0.to_v.bias")
    converted["encoder.mid.attn_1.norm.weight"] = state_dict.pop(
        "encoder.mid_block.attentions.0.group_norm.weight")
    converted["encoder.mid.attn_1.norm.bias"] = state_dict.pop(
        "encoder.mid_block.attentions.0.group_norm.bias")
    converted["encoder.mid.attn_1.proj_out.weight"] = state_dict.pop(
        "encoder.mid_block.attentions.0.to_out.0.weight").unsqueeze(-1).unsqueeze(-1)
    converted["encoder.mid.attn_1.proj_out.bias"] = state_dict.pop(
        "encoder.mid_block.attentions.0.to_out.0.bias")

    # encoder mid blocks
    for i in range(2):
        for j in range(2):
            converted[f"encoder.mid.block_{i+1}.conv{j+1}.weight"] = state_dict.pop(
                f"encoder.mid_block.resnets.{i}.conv{j+1}.weight")
            converted[f"encoder.mid.block_{i+1}.conv{j+1}.bias"] = state_dict.pop(
                f"encoder.mid_block.resnets.{i}.conv{j+1}.bias")
            converted[f"encoder.mid.block_{i+1}.norm{j+1}.weight"] = state_dict.pop(
                f"encoder.mid_block.resnets.{i}.norm{j+1}.weight")
            converted[f"encoder.mid.block_{i+1}.norm{j+1}.bias"] = state_dict.pop(
                f"encoder.mid_block.resnets.{i}.norm{j+1}.bias")

    # decoder conv_in / conv_out
    converted["decoder.conv_in.weight"] = state_dict.pop("decoder.conv_in.weight")
    converted["decoder.conv_in.bias"] = state_dict.pop("decoder.conv_in.bias")
    converted["decoder.conv_out.weight"] = state_dict.pop("decoder.conv_out.weight")
    converted["decoder.conv_out.bias"] = state_dict.pop("decoder.conv_out.bias")

    # decoder norm_out
    converted["decoder.norm_out.weight"] = state_dict.pop("decoder.conv_norm_out.weight")
    converted["decoder.norm_out.bias"] = state_dict.pop("decoder.conv_norm_out.bias")

    # decoder mid attention
    converted["decoder.mid.attn_1.q.weight"] = state_dict.pop(
        "decoder.mid_block.attentions.0.to_q.weight").unsqueeze(-1).unsqueeze(-1)
    converted["decoder.mid.attn_1.q.bias"] = state_dict.pop("decoder.mid_block.attentions.0.to_q.bias")
    converted["decoder.mid.attn_1.k.weight"] = state_dict.pop(
        "decoder.mid_block.attentions.0.to_k.weight").unsqueeze(-1).unsqueeze(-1)
    converted["decoder.mid.attn_1.k.bias"] = state_dict.pop("decoder.mid_block.attentions.0.to_k.bias")
    converted["decoder.mid.attn_1.v.weight"] = state_dict.pop(
        "decoder.mid_block.attentions.0.to_v.weight").unsqueeze(-1).unsqueeze(-1)
    converted["decoder.mid.attn_1.v.bias"] = state_dict.pop("decoder.mid_block.attentions.0.to_v.bias")
    converted["decoder.mid.attn_1.norm.weight"] = state_dict.pop(
        "decoder.mid_block.attentions.0.group_norm.weight")
    converted["decoder.mid.attn_1.norm.bias"] = state_dict.pop(
        "decoder.mid_block.attentions.0.group_norm.bias")
    converted["decoder.mid.attn_1.proj_out.weight"] = state_dict.pop(
        "decoder.mid_block.attentions.0.to_out.0.weight").unsqueeze(-1).unsqueeze(-1)
    converted["decoder.mid.attn_1.proj_out.bias"] = state_dict.pop(
        "decoder.mid_block.attentions.0.to_out.0.bias")

    # decoder mid blocks
    for i in range(2):
        for j in range(2):
            converted[f"decoder.mid.block_{i+1}.conv{j+1}.weight"] = state_dict.pop(
                f"decoder.mid_block.resnets.{i}.conv{j+1}.weight")
            converted[f"decoder.mid.block_{i+1}.conv{j+1}.bias"] = state_dict.pop(
                f"decoder.mid_block.resnets.{i}.conv{j+1}.bias")
            converted[f"decoder.mid.block_{i+1}.norm{j+1}.weight"] = state_dict.pop(
                f"decoder.mid_block.resnets.{i}.norm{j+1}.weight")
            converted[f"decoder.mid.block_{i+1}.norm{j+1}.bias"] = state_dict.pop(
                f"decoder.mid_block.resnets.{i}.norm{j+1}.bias")

    # decoder up blocks (diffusers index is reversed from original)
    for i in range(4):
        for j in range(3):
            for k in range(1, 3):
                converted[f"decoder.up.{i}.block.{j}.conv{k}.weight"] = state_dict.pop(
                    f"decoder.up_blocks.{3-i}.resnets.{j}.conv{k}.weight")
                converted[f"decoder.up.{i}.block.{j}.conv{k}.bias"] = state_dict.pop(
                    f"decoder.up_blocks.{3-i}.resnets.{j}.conv{k}.bias")
                converted[f"decoder.up.{i}.block.{j}.norm{k}.weight"] = state_dict.pop(
                    f"decoder.up_blocks.{3-i}.resnets.{j}.norm{k}.weight")
                converted[f"decoder.up.{i}.block.{j}.norm{k}.bias"] = state_dict.pop(
                    f"decoder.up_blocks.{3-i}.resnets.{j}.norm{k}.bias")
        # upsample
        if i != 0:
            converted[f"decoder.up.{i}.upsample.conv.weight"] = state_dict.pop(
                f"decoder.up_blocks.{3-i}.upsamplers.0.conv.weight")
            converted[f"decoder.up.{i}.upsample.conv.bias"] = state_dict.pop(
                f"decoder.up_blocks.{3-i}.upsamplers.0.conv.bias")
        # shortcut
        if i == 0 or i == 1:
            converted[f"decoder.up.{i}.block.0.nin_shortcut.weight"] = state_dict.pop(
                f"decoder.up_blocks.{3-i}.resnets.0.conv_shortcut.weight")
            converted[f"decoder.up.{i}.block.0.nin_shortcut.bias"] = state_dict.pop(
                f"decoder.up_blocks.{3-i}.resnets.0.conv_shortcut.bias")

    if state_dict:
        print(f"  Warning: {len(state_dict)} unconverted VAE keys remaining: {list(state_dict.keys())}")
        for key in list(state_dict.keys()):
            converted[key] = state_dict.pop(key)

    return converted


def main():
    src_dir = Path("./models/Step1X-Edit-v1p2")
    out_dir = Path("./models/Step1X-Edit-v1p2-original")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert Transformer ---
    print("=" * 60)
    print("Converting Transformer (diffusers -> original)...")
    print("=" * 60)

    transformer_dir = src_dir / "transformer"
    index_file = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    print(f"  Loading {len(shard_files)} shards...")
    state_dict = {}
    for shard in shard_files:
        print(f"    Loading {shard}...")
        state_dict.update(load_file(str(transformer_dir / shard), "cpu"))
    print(f"  Total keys loaded: {len(state_dict)}")

    converted = convert_transformer_diffusers_to_original(state_dict)
    print(f"  Converted keys: {len(converted)}")

    out_path = out_dir / "step1x-edit-v1p2.safetensors"
    print(f"  Saving to {out_path}...")
    save_file(converted, str(out_path))
    print(f"  Done! Size: {out_path.stat().st_size / 1e9:.2f} GB")

    # --- Convert VAE ---
    print()
    print("=" * 60)
    print("Converting VAE (diffusers -> original)...")
    print("=" * 60)

    vae_path = src_dir / "vae" / "diffusion_pytorch_model.safetensors"
    print(f"  Loading {vae_path}...")
    vae_sd = load_file(str(vae_path), "cpu")
    print(f"  Total keys loaded: {len(vae_sd)}")

    converted_vae = convert_vae_diffusers_to_original(vae_sd)
    print(f"  Converted keys: {len(converted_vae)}")

    out_vae_path = out_dir / "vae.safetensors"
    print(f"  Saving to {out_vae_path}...")
    save_file(converted_vae, str(out_vae_path))
    print(f"  Done! Size: {out_vae_path.stat().st_size / 1e6:.1f} MB")

    # --- Symlink text_encoder ---
    te_src = src_dir / "text_encoder"
    te_dst = out_dir / "Qwen2.5-VL-7B-Instruct"
    if not te_dst.exists():
        os.symlink(te_src.resolve(), te_dst)
        print(f"\n  Symlinked text_encoder -> {te_dst}")

    print()
    print("=" * 60)
    print("All done! Original format weights saved to:")
    print(f"  {out_dir}/")
    print(f"    step1x-edit-v1p2.safetensors  (transformer)")
    print(f"    vae.safetensors               (VAE)")
    print(f"    Qwen2.5-VL-7B-Instruct/       (text encoder symlink)")
    print("=" * 60)


if __name__ == "__main__":
    main()
