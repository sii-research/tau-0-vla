"""``Tau0VLAPolicy`` — the sole public symbol here.

Contract for any policy class callers (server / openloop) can swap in:
``from_checkpoint(ckpt) -> policy`` + ``policy.infer(payload) -> {"actions": ndarray}``
+ ``policy.data_spec``.

Copy this file as the starting point for other VLA architectures;
``openloop.py`` takes any class matching that shape via
``--policy-module`` / ``--policy-class``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
del _sys, _Path

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image as PILImage

from tau0_vla.data import encode_payload, load_checkpoint_spec, load_data_spec, restore_action
from tau0_vla.utils.run_spec import load_resolved_args


class Tau0VLAPolicy:
    """``infer`` = encode → tokenize → forward → decode.

    Encode/decode halves are ``tau0_vla.data.encode_payload`` /
    ``restore_action`` — every transform ``data_spec`` declared (prompt
    template, image resize, state pipeline) is applied by construction at
    inference, can't silently skip. VLM tokenization stays tau-0-vla-local —
    the general-pipeline / VLM-side split keeps it out of the dataloader.
    """

    def __init__(self, *, model, processor, data_spec, device: torch.device) -> None:
        # Read cam_keys / target_size / vlm_model_type off self.data_spec
        # directly — no copy-fields, single source of truth.
        self.model = model
        self.processor = processor
        self.data_spec = data_spec
        self.device = device

    @property
    def _model_dtype(self) -> torch.dtype:
        # Cast state tensor to match state_proj.weight dtype — bfloat16 for
        # mixed-precision ckpts, otherwise matmul raises 'mat1 and mat2 must
        # have the same dtype'.
        return next(self.model.parameters()).dtype

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        *,
        route: str | None = None,
        device: str | torch.device | None = None,
    ) -> "Tau0VLAPolicy":
        """Rebuild from ``output_dir`` or ``checkpoint-N/``; both work because
        ``FinchDataSpecCallback`` mirrors deploy artifacts into every
        checkpoint-N and ``load_resolved_args`` falls back to the parent.
        Caller must have run ``deploy._bootstrap.ensure_configs_registered``
        (and for old ckpts, ``ensure_policy_manifest``) first."""
        ckpt_path = Path(ckpt_dir).resolve()

        spec = load_checkpoint_spec(ckpt_path, route=route)
        data_spec = load_data_spec(ckpt_path, route=spec.route)

        model_args, rebuilt_data_args, training_args, _ = load_resolved_args(ckpt_path, inference=True)
        model_args.model_name_or_path = str(ckpt_path)

        from tau0_vla.models.model_builder import ModelBuilder
        mb = ModelBuilder(training_args, model_args, rebuilt_data_args, is_training=False)
        mb.build()

        torch_device = (
            torch.device(device) if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        mb.model.to(torch_device).eval()
        return cls(model=mb.model, processor=mb.processor, data_spec=data_spec, device=torch_device)

    @torch.no_grad()
    def infer(self, payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Native-space payload in, native-space action chunk out."""
        encoded     = encode_payload(payload, self.data_spec)
        model_input = self._tokenize_vlm(encoded)
        raw_action  = self.model.sample_action(model_input).detach().cpu().float().numpy()[0]
        # Unified routes: the relative→absolute inverse needs the ABSOLUTE
        # scattered 40D state (encode_payload's "state_abs"), not the normalized
        # model input. Component routes keep the encoded state (their
        # postprocessor un-normalizes internally).
        restore_state = encoded.get("state_abs", encoded["state"])
        return {"actions": restore_action(raw_action, self.data_spec, state=restore_state)}

    def _tokenize_vlm(self, encoded: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Qwen-VL chat template + pixel preprocessing + device placement."""
        from tau0_vla.vlm.qwenvl_utils import build_vla_inference_data_dict

        cam_keys = list(self.data_spec.cam_keys)
        max_images = self.data_spec.max_images_per_sample
        if max_images is not None:
            cam_keys = cam_keys[: int(max_images)]
        images_pil = [PILImage.fromarray(encoded["images"][key]) for key in cam_keys]
        data_dict = build_vla_inference_data_dict(
            processor=self.processor,
            images_pil=images_pil,
            instruction=encoded["prompt"],
            vlm_model_type=self.data_spec.vlm_model_type,
            cam_view_labels=self._render_cam_view_labels(cam_keys),
        )
        batch: Dict[str, torch.Tensor] = {
            "input_ids": data_dict["input_ids"].to(self.device),
            "attention_mask": data_dict["attention_mask"].to(self.device),
            "state": torch.as_tensor(encoded["state"], dtype=self._model_dtype)
                .reshape(1, 1, -1)
                .to(self.device),
        }
        # Without an action_mask, sample_action skips the inactive-dim pinning
        # (Scheme-C vla_inactive_input_zero ckpts) and padded dims drift OOD
        # during denoising.
        #   - Unified routes: encode_payload supplies the per-route scatter mask
        #     (active slots per registry key + EEF priority).
        #   - Component routes: the training mask is deterministic
        #     first-D-active — synthesize it from the component dims.
        if "action_mask" in encoded:
            batch["action_mask"] = (
                torch.as_tensor(encoded["action_mask"], dtype=self._model_dtype)
                .reshape(1, -1)
                .to(self.device)
            )
        else:
            component_dims = tuple(getattr(self.data_spec, "action_component_dims", ()) or ())
            if component_dims and not getattr(self.data_spec, "unified_registry_key", None):
                active = int(sum(component_dims))
                action_mask = torch.zeros(
                    1, int(self.data_spec.action_dim), dtype=self._model_dtype, device=self.device
                )
                action_mask[:, :active] = 1.0
                batch["action_mask"] = action_mask
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            value = data_dict.get(key)
            if value is not None:
                batch[key] = value.to(self.device)
        return batch

    def _render_cam_view_labels(self, cam_keys: list[str]) -> list[str] | None:
        template = self.data_spec.cam_view_template
        if template is None:
            return None
        names = self.data_spec.cam_view_names or {}
        return [template.format(names.get(key, key.rsplit(".", 1)[-1])) for key in cam_keys]


__all__ = ["Tau0VLAPolicy"]
