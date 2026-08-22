"""Native LIBERO data and online-observation layout."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Mapping

import numpy as np

from tau0_vla.data.modalities import FieldDescription
from tau0_vla.data.modalities.base import serialize_field_description_map
from tau0_vla.data.robots.base import FinchResolvedRobotConfig, RobotConfig

LIBERO_STATE_DIM = 8
LIBERO_ACTION_DIM = 7


def _flat_f32(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


@dataclasses.dataclass(frozen=True)
class LiberoObservation:
    """One LIBERO request after parsing its wire-format keys."""

    instruction: str
    image: Any
    wrist_image: Any
    eef_pose: np.ndarray
    gripper: np.ndarray

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LiberoObservation":
        state = _flat_f32(
            payload["observation/state"],
            name="observation/state (6D EEF pose + 2D gripper)",
            size=LIBERO_STATE_DIM,
        )
        instruction = str(payload.get("prompt") or payload.get("task") or "")
        if not instruction.strip():
            raise ValueError("LIBERO payload requires a non-empty 'prompt' or 'task'")
        return cls(
            instruction=instruction,
            image=payload["observation/image"],
            wrist_image=payload["observation/wrist_image"],
            eef_pose=state[:6].copy(),
            gripper=state[6:8].copy(),
        )

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.eef_pose, self.gripper]).astype(np.float32, copy=False)


@dataclasses.dataclass(frozen=True)
class LiberoRobot(RobotConfig):
    """LIBERO component route with an 8D EEF-state and 7D delta-EEF action."""

    robot_name: ClassVar[str] = "libero"
    quaternion_order: ClassVar[str] = "xyzw"

    repack: ClassVar[dict[str, object]] = {
        "prompt": "task",
        "images": {
            "image": "image",
            "wrist_image": "wrist_image",
        },
        "state": {
            "raw": "observation.state",
            "semantic": {
                "eef_pose": "eef_pose",
                "gripper": "gripper",
            },
        },
        "action": {
            "raw": "action",
            "semantic": {
                "eef_pose": "eef_pose",
                "gripper": "gripper",
            },
        },
    }

    def _normalize_field_descriptions(self, field_descriptions):
        """Freeze LIBERO's semantic contract even for stale dataset metadata.

        Some existing exports label the eight state columns as seven arm
        joints plus one gripper value. The online policy uses six EEF values
        plus two gripper values. Treating both as an anonymous 8-vector would
        silently train and serve different semantics, so this adapter owns the
        authoritative field descriptions.
        """
        action_payload = (field_descriptions or {}).get("action") or {}
        action_width = (
            max(
                (max((int(i) for i in entry.get("indices", ())), default=-1) for entry in action_payload.values()),
                default=LIBERO_ACTION_DIM - 1,
            )
            + 1
        )
        if action_width != LIBERO_ACTION_DIM:
            raise ValueError(
                f"LIBERO raw action must be {LIBERO_ACTION_DIM}D "
                f"(6D EEF delta + 1D gripper), metadata describes {action_width}D"
            )

        state = {
            "eef_pose": FieldDescription(
                "End-effector pose (xyz + axis-angle)",
                dimensions=6,
                indices=tuple(range(6)),
            ),
            "gripper": FieldDescription(
                "Two gripper joint positions",
                dimensions=2,
                indices=(6, 7),
            ),
        }
        action = {
            "eef_pose": FieldDescription(
                "Delta end-effector pose (xyz + axis-angle)",
                dimensions=6,
                indices=tuple(range(6)),
            ),
            "gripper": FieldDescription(
                "Scalar gripper command",
                dimensions=1,
                indices=(6,),
            ),
        }
        return state, action

    def resolve_output_spec(
        self,
        *,
        field_descriptions: Mapping[str, Any],
    ) -> FinchResolvedRobotConfig:
        """Persist the adapter-owned contract, not stale source metadata."""
        resolved = super().resolve_output_spec(field_descriptions=field_descriptions)
        state, action = self._normalize_field_descriptions(field_descriptions)
        frozen = {
            "state": serialize_field_description_map(state),
            "action": serialize_field_description_map(action),
        }
        return dataclasses.replace(resolved, field_descriptions=frozen)

    @classmethod
    def supports_sdk_payload(cls) -> bool:
        return True

    @classmethod
    def observation_from_payload(cls, payload: Mapping[str, Any]) -> LiberoObservation:
        return LiberoObservation.from_payload(payload)


__all__ = [
    "LIBERO_ACTION_DIM",
    "LIBERO_STATE_DIM",
    "LiberoObservation",
    "LiberoRobot",
]
