"""Attention adapter for Tau VLA joint attention.

Wraps HF's standard attention backends (eager / sdpa / flash_attention_2) with
the VLA-internal tensor layout:
  Input:  Q/K/V in [B, L, H, D], bool mask [B, L_q, L_k] (True = visible)
  Output: [B, L, H*D]

Also holds the flow-matching helpers (`create_sinusoidal_pos_embedding`,
`sample_beta`, `make_att_2d_masks`) that used to live in the lingbot_vla
package, moved here verbatim when that architecture was removed.
"""

import math
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward as hf_sdpa_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    eager_attention_forward as hf_eager_attention_forward,
)


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    """Draw ``bsize`` samples from ``Beta(alpha, beta)`` on ``device``.

    Sampling powered uniforms and taking their ratio is not a Beta sampler
    (except for special parameter choices): the powered variables are not
    Gamma-distributed, so the flow-matching time distribution is biased.  Use
    PyTorch's Gamma-based implementation instead and create the concentration
    tensors on the requested device to avoid an implicit host-to-device copy.
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    distribution = torch.distributions.Beta(alpha_t, beta_t)
    return distribution.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def _bool_mask_to_additive(bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool mask [B, Lq, Lk] -> additive float mask [B, 1, Lq, Lk].

    True = visible (0.0), False = masked (finfo.min).
    """
    additive = torch.zeros_like(bool_mask, dtype=dtype)
    additive.masked_fill_(~bool_mask, torch.finfo(dtype).min)
    return additive.unsqueeze(1)


def causal_sdpa_flash_backend_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Causal self-attention via SDPA's Flash backend (no explicit mask).

    Calls ``F.scaled_dot_product_attention`` with ``attn_mask=None`` and
    ``is_causal=True``; on modern GPUs PyTorch dispatches this combination to
    the Flash backend, which is the actual performance win this helper exists
    to capture. Intended for the VLM prefix self-attention only — callers must
    guarantee the sequence is causal and right-padded. Suffix / action
    attention must keep using explicit masks.
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads
    scaling = head_dim**-0.5

    query_hf = query_states.transpose(1, 2)
    key_hf = repeat_kv(key_states.transpose(1, 2), num_key_value_groups)
    value_hf = repeat_kv(value_states.transpose(1, 2), num_key_value_groups)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_hf,
        key_hf,
        value_hf,
        attn_mask=None,
        is_causal=True,
        scale=scaling,
    )
    return attn_output.transpose(1, 2).reshape(bsize, seq_len, num_att_heads * head_dim)


class _DummyModule(nn.Module):
    """Minimal shim so HF attention functions can read ``num_key_value_groups`` and ``scaling``."""

    def __init__(self, num_key_value_groups: int, scaling: float):
        super().__init__()
        self.num_key_value_groups = num_key_value_groups
        self.scaling = scaling
        self.is_causal = False
        self.training = False


def our_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    mode: str = "eager",
):
    """Unified attention adapter for Tau VLA joint attention.

    Args:
        query_states: [B, L, num_heads, head_dim]
        key_states:   [B, L, num_kv_heads, head_dim]
        value_states: [B, L, num_kv_heads, head_dim]
        attention_mask: [B, L_q, L_k] bool (True = visible), or None.
        mode: "eager" | "sdpa" | "flash_attention_2"

    Returns:
        [B, L, num_heads * head_dim]
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads
    scaling = head_dim**-0.5

    # [B, L, H, D] -> [B, H, L, D]  (HF convention)
    query_hf = query_states.transpose(1, 2)
    key_hf = key_states.transpose(1, 2)
    value_hf = value_states.transpose(1, 2)

    # Resolve the HF attention backend
    if mode == "eager":
        attn_fn = hf_eager_attention_forward
    else:
        # "sdpa" and "flash_attention_2" both go through ALL_ATTENTION_FUNCTIONS
        attn_fn = ALL_ATTENTION_FUNCTIONS.get(mode, hf_sdpa_attention_forward)

    # Build a dummy module for HF functions that read module attributes
    dummy = _DummyModule(num_key_value_groups, scaling)

    # Convert bool mask -> additive float mask for eager,
    # or keep bool for sdpa (SDPA accepts bool attn_mask natively).
    if attention_mask is not None:
        if mode == "eager":
            hf_mask = _bool_mask_to_additive(attention_mask, dtype=query_states.dtype)
        else:
            # sdpa / flash_attention_2: bool mask [B, 1, Lq, Lk]
            hf_mask = attention_mask.unsqueeze(1)
    else:
        hf_mask = None

    attn_output, _ = attn_fn(
        dummy,
        query_hf,
        key_hf,
        value_hf,
        hf_mask,
        scaling=scaling,
        dropout=0.0,
    )

    # [B, L, H, D] -> [B, L, H*D]
    attn_output = attn_output.reshape(bsize, seq_len, num_att_heads * head_dim)
    return attn_output
