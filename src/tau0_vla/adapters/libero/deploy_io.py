"""LIBERO request adaptation for :mod:`deploy.libero_server`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tau0_vla.adapters.libero.layout import (
    LIBERO_STATE_DIM,
    LiberoObservation,
)

_EXPECTED_STATE_INDICES = {
    "eef_pose": tuple(range(6)),
    "gripper": (6, 7),
}


def load_state_field_descriptions(artifacts_dir: str | Path) -> dict[str, Any]:
    path = Path(artifacts_dir) / "field_descriptions.json"
    return json.loads(path.read_text(encoding="utf-8"))["state"]


def state_dim_from_field_descriptions(state_fd: Mapping[str, Any]) -> int:
    return max((max(entry.get("indices") or [-1]) for entry in state_fd.values()), default=-1) + 1


def _validate_state_contract(state_fd: Mapping[str, Any], state_dim: int) -> None:
    if state_dim != LIBERO_STATE_DIM:
        raise ValueError(
            f"LIBERO checkpoint must declare an {LIBERO_STATE_DIM}D raw state "
            f"(6D EEF pose + 2D gripper), got {state_dim}D"
        )
    for name, expected in _EXPECTED_STATE_INDICES.items():
        if name not in state_fd:
            raise KeyError(f"LIBERO checkpoint state is missing field {name!r}")
        actual = tuple(int(index) for index in state_fd[name].get("indices", ()))
        if actual != expected:
            raise ValueError(
                f"LIBERO checkpoint state field {name!r} must use indices {list(expected)}, got {list(actual)}"
            )


def build_payload_adapter(
    *,
    cam_keys: Sequence[str],
    state_fd: Mapping[str, Any],
    state_dim: int,
) -> Callable[[Mapping[str, Any]], dict]:
    _validate_state_contract(state_fd, state_dim)
    cam_keys = tuple(str(key) for key in cam_keys)
    expected_cameras = {"image", "wrist_image"}
    if set(cam_keys) != expected_cameras:
        raise ValueError(f"LIBERO checkpoint cameras must be {sorted(expected_cameras)}, got {list(cam_keys)}")

    def adapt(raw: Mapping[str, Any]) -> dict:
        obs = LiberoObservation.from_payload(raw)
        images = {
            "image": obs.image,
            "wrist_image": obs.wrist_image,
        }
        return {
            "prompt": obs.instruction,
            "images": {key: images[key] for key in cam_keys},
            "state": obs.state,
            "meta": {"embodiment": "libero"},
        }

    return adapt


def canonicalize_action_dict(split: dict) -> dict:
    return dict(split)


def build_sdk_action_perm(data_spec, slices) -> None:
    del data_spec, slices
    return None


def apply_sdk_action_perm(actions, sdk_action_perm) -> np.ndarray:
    if sdk_action_perm is not None:
        raise ValueError("LIBERO component actions do not use an SDK permutation")
    return np.asarray(actions, dtype=np.float32)


__all__ = [
    "apply_sdk_action_perm",
    "build_payload_adapter",
    "build_sdk_action_perm",
    "canonicalize_action_dict",
    "load_state_field_descriptions",
    "state_dim_from_field_descriptions",
]
