"""Layer 3 — G1 / A2D deploy I/O.

Everything the serving path needs to know about *this robot's wire format*:
which keys the SDK sends, what the cameras are called, how the state channels
line up with the checkpoint's ``field_descriptions.json``, and — the dangerous
one — what column order the SDK expects actions back in.

Extracted verbatim out of ``deploy/server.py`` per
the adapter's layer-3 boundary. ``deploy/server.py``
keeps the HTTP surface (routes, wire framing, logging); this module keeps the
embodiment knowledge. No behaviour changed in the move.

For your own robot, copy this file. Four things in it are yours to change, and
they are the four marked "Change this" below:

* ``_STATE_CHANNELS`` — how your SDK's state channels map onto the checkpoint's
  declared state vector
* ``_SDK_IMAGE_ACCESSORS`` and ``_RAW_IMAGE_ALIAS`` — your camera names
* whatever your client expects the action dict to be keyed by, in
  :func:`canonicalize_action_dict`

Everything else is machinery that works the same for any embodiment. The one
piece to read carefully before touching anything is
:func:`build_sdk_action_perm` — see its docstring. Its validation failures are
the only thing standing between a mis-specified registry and a wrong-order
action vector on real hardware; do not soften them into warnings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tau0_vla.adapters.g1.layout import G1Observation

# ──────────────────────────────────────────────────────────────────────────────
# Canonical action-dict merge (the /act wire format)
# ──────────────────────────────────────────────────────────────────────────────

# Change this only to match YOUR client. The left two columns are Unified Layout
# slot names and are fixed by the pipeline; the third is the key your robot's
# client reads. If your client wants per-side keys, it already gets them — this
# merge exists because the G1 client wants them combined.
#
# Unified-40D routes have no declared components, so ``action_slices`` emits
# *per-side* slice names (``left_arm``/``right_arm``/``left_gripper``/...).
# The G01/A2D HTTP client (hil-serl ``corobot_env.component_dict_to_action_dict``)
# reads canonical *combined* keys only — ``arm_joint``(14) / ``eef_pose``(14) /
# ``gripper``(2), plus pass-through ``waist`` / ``chassis_velocity``. Merge
# left+right (left first, matching the client's left[:half]/right[half:] split)
# so the wire dict matches the client contract. Declared-component routes have
# no ``left_*``/``right_*`` keys, so this is a no-op for them.
#
# Public because the merge is a two-way wire contract: whoever consumes ``/act``
# needs the same table to split a canonical key back into its two sides
# (``deploy/openloop_with_server.py`` does exactly that).
UNIFIED_SIDE_TO_CANONICAL = (
    ("left_arm", "right_arm", "arm_joint"),
    ("left_eef", "right_eef", "eef_pose"),
    ("left_gripper", "right_gripper", "gripper"),
)


def canonicalize_action_dict(split: dict) -> dict:
    """Merge per-side unified slices into canonical combined component keys."""
    out = dict(split)
    for left_key, right_key, canonical in UNIFIED_SIDE_TO_CANONICAL:
        left = out.pop(left_key, None)
        right = out.pop(right_key, None)
        if left is None and right is None:
            continue
        if left is None or right is None:
            raise ValueError(
                f"/act: unified action dict has only one of {left_key!r}/"
                f"{right_key!r}; cannot form canonical {canonical!r}."
            )
        merged = np.concatenate(
            [np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)],
            axis=-1,
        )
        out[canonical] = merged.tolist()
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SDK-native action column order
# ──────────────────────────────────────────────────────────────────────────────


def build_sdk_action_perm(data_spec, slices) -> list[int] | None:
    """Column permutation mapping ``restore_action``'s flat output to the A2D
    SDK's native action layout.

    ``/act_lerobot_bytes`` returns a bare positional chunk that the SDK applies
    index-for-index. For UNIFIED-40D routes ``restore_action`` emits columns in
    ``_UNIFIED_NATIVE_ORDER`` (``[gripL, gripR, Larm, Rarm]``), but the SDK
    (and the pre-unified component ckpts) apply actions in registry
    ``action_groups`` order (``[Larm, Rarm, gripL, gripR]`` for a2d). Scatter
    each named slice back to its ``action_groups`` index so the wire vector
    matches the layout the SDK expects.

    Skipping this permutation writes the two gripper commands into the left
    arm's first two joints — silently, and on real hardware. Both ``raise``
    branches below are deliberate: a wrong-order deploy must fail loudly rather
    than move the robot.

    Returns ``None`` for non-unified (component) routes — their
    ``restore_action`` already emits native declared-component order, so the
    flat chunk is passed through unchanged. Raises if a restored slot has no
    ``action_groups`` entry (an EEF slot on a joint-only SDK) or if the SDK
    layout would end up with holes — a wrong-order deploy must fail loudly.
    """
    key = getattr(data_spec, "unified_registry_key", None)
    if key is None:
        return None
    from tau0_vla.data.robots.unified import get_registry_entry

    groups = get_registry_entry(key).get("action_groups", {})
    if not groups:
        return None
    native_indices = [index for indices in groups.values() for index in indices]
    if not native_indices:
        raise ValueError(
            f"/act_lerobot_bytes: registry {key!r} has action_groups but no "
            "native action indices."
        )
    invalid = [
        index
        for index in native_indices
        if isinstance(index, bool) or not isinstance(index, int) or index < 0
    ]
    if invalid:
        raise ValueError(
            f"/act_lerobot_bytes: registry {key!r} has invalid native action "
            f"indices {invalid}; indices must be non-negative integers."
        )
    width = 1 + max(native_indices)
    perm: list[int | None] = [None] * width
    for name, off, dim in slices:
        ix = groups.get(name)
        if ix is None:
            raise KeyError(
                f"/act_lerobot_bytes: restored slot {name!r} has no action_groups "
                f"entry in registry {key!r}; cannot map it to the SDK's native "
                f"action layout (EEF slots aren't supported on a joint SDK endpoint)."
            )
        if len(ix) != dim:
            raise ValueError(
                f"/act_lerobot_bytes: restored slot {name!r} has width {dim}, "
                f"but registry {key!r} maps {len(ix)} native indices; cannot "
                "construct a one-to-one SDK action permutation."
            )
        for j in range(dim):
            native_idx = ix[j]
            if perm[native_idx] is not None:
                raise ValueError(
                    f"/act_lerobot_bytes: registry {key!r} maps multiple restored "
                    f"values to native action index {native_idx}; cannot construct "
                    "a one-to-one SDK action permutation."
                )
            perm[native_idx] = off + j
    holes = [i for i, p in enumerate(perm) if p is None]
    if holes:
        raise ValueError(
            f"/act_lerobot_bytes: action_groups indices {holes} in registry "
            f"{key!r} are not covered by the restored slots; the SDK action "
            f"vector would have holes."
        )
    return [int(p) for p in perm]


def apply_sdk_action_perm(actions, sdk_action_perm: Sequence[int] | None) -> np.ndarray:
    """Reorder a ``[chunk, native_dim]`` action array into SDK column order.

    Identity when ``sdk_action_perm`` is ``None`` (component routes already emit
    native declared-component order).
    """
    arr = np.asarray(actions, dtype=np.float32)
    if sdk_action_perm is not None:
        arr = arr[..., sdk_action_perm]
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# SDK payload -> canonical policy payload
# ──────────────────────────────────────────────────────────────────────────────

# Change this. Where each SDK state channel lands in the checkpoint's declared
# state vector.
# Keys are ``field_descriptions.json`` entries; values pull the matching slice
# off the parsed observation. The gripper arrives as one 2-vector and has to be
# split, because the two effectors are declared as separate fields.
_STATE_CHANNELS: dict[str, Callable[[G1Observation], Any]] = {
    "state/joint/position":          lambda obs: obs.arm_joint_states,
    "state/waist/position":          lambda obs: obs.waist_joint_states,
    "state/head/position":           lambda obs: obs.head_joint_radians,
    "state/left_effector/position":  lambda obs: obs.gripper_states.reshape(-1)[0:1],
    "state/right_effector/position": lambda obs: obs.gripper_states.reshape(-1)[1:2],
}

# Change this. Robot Configs name the cameras ``head`` / ``wrist_left`` /
# ``wrist_right``; ``G1Observation`` parses the SDK's own key names. This is the
# join, and both sides are yours: the keys must match your training config's
# ``camera_keys``, the accessors must match what your SDK sends.
_SDK_IMAGE_ACCESSORS: dict[str, Callable[[G1Observation], Any]] = {
    "head":        lambda obs: obs.head_image,
    "wrist_right": lambda obs: obs.right_hand_image,
    "wrist_left":  lambda obs: obs.left_hand_image,
}

# Change this. Fallback for cam_keys the observation doesn't parse: map the
# config's camera name onto the raw payload key, read straight out of the request
# body. A camera missing from both this and ``_SDK_IMAGE_ACCESSORS`` raises a
# KeyError per request rather than silently serving a black frame — keep that.
_RAW_IMAGE_ALIAS = {
    "head": "observation.images.head",
    "wrist_right": "observation.images.hand_right",
    "wrist_left": "observation.images.hand_left",
    # Legacy full LeRobot keys used by older specs/configs.
    "observation.images.top_head": "observation.images.head",
    "observation.images.hand_right": "observation.images.hand_right",
    "observation.images.hand_left": "observation.images.hand_left",
}


def load_state_field_descriptions(artifacts_dir: str | Path) -> dict[str, Any]:
    """Read the ``state`` block of a checkpoint's ``field_descriptions.json``."""
    path = Path(artifacts_dir) / "field_descriptions.json"
    return json.loads(path.read_text())["state"]


def state_dim_from_field_descriptions(state_fd: Mapping[str, Any]) -> int:
    """Width of the raw state vector = highest declared index + 1."""
    return max((max(e.get("indices") or [-1]) for e in state_fd.values()), default=-1) + 1


def build_payload_adapter(
    *,
    cam_keys: Sequence[str],
    state_fd: Mapping[str, Any],
    state_dim: int,
) -> Callable[[Mapping[str, Any]], dict]:
    """Build the SDK-payload → ``{prompt, images, state, meta}`` adapter.

    Bound once per process to the route's camera list and the checkpoint's
    state field descriptions; the returned callable runs per request.
    """
    cam_keys = tuple(cam_keys)

    def adapt(raw: Mapping[str, Any]) -> dict:
        obs = G1Observation.from_payload(raw)
        state = np.zeros(state_dim, dtype=np.float32)
        for key, accessor in _STATE_CHANNELS.items():
            ix = state_fd.get(key, {}).get("indices") or []
            if ix:
                arr = np.asarray(accessor(obs), dtype=np.float32).reshape(-1)
                state[ix[0]: ix[-1] + 1] = arr[: len(ix)]
        images = {}
        for cam in cam_keys:
            accessor = _SDK_IMAGE_ACCESSORS.get(cam)
            if accessor is not None:
                images[cam] = accessor(obs)
                continue
            images[cam] = raw[_RAW_IMAGE_ALIAS.get(cam, cam)]
        return {
            "prompt": obs.instruction,
            "images": images,
            "state":  state,
            "meta":   {},
        }

    return adapt


__all__ = [
    "UNIFIED_SIDE_TO_CANONICAL",
    "apply_sdk_action_perm",
    "build_payload_adapter",
    "build_sdk_action_perm",
    "canonicalize_action_dict",
    "load_state_field_descriptions",
    "state_dim_from_field_descriptions",
]
