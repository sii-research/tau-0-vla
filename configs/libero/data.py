"""LIBERO training route: 8D EEF state and 7D delta-EEF action."""

from __future__ import annotations

import os
from pathlib import Path

from tau0_vla.adapters.libero import LiberoRobot
from tau0_vla.data import EefPose, Gripper, Image, Prompt, PromptSource, register_config
from tau0_vla.data.modalities.image import ResizeWithPad

_DATA = os.environ.get("TAU0_LIBERO_DATA", "<PATH_TO_LIBERO_LEROBOT_V3_DATASET_OR_MANIFEST>")
_NORM_STATS = os.environ.get(
    "TAU0_LIBERO_NORM_STATS",
    str(Path(__file__).with_name("norm_stats.json")),
)


@register_config
def libero_eef_ft() -> LiberoRobot:
    return LiberoRobot(
        repo_id=_DATA,
        images=[
            Image("image", transforms=[ResizeWithPad(224, 224)]),
            Image("wrist_image", transforms=[ResizeWithPad(224, 224)]),
        ],
        prompt_source=PromptSource.from_label(source="parquet"),
        prompt=Prompt(template="What action should the robot take to {instruction}?"),
        state=[
            EefPose(normalize="mean_std"),
            Gripper(normalize="mean_std"),
        ],
        action=[
            # Dataset values are already delta xyz + delta axis-angle.
            EefPose(normalize="mean_std", abs2relative=False),
            Gripper(normalize="mean_std"),
        ],
        # The released Tau0VLA checkpoint was trained with n_action_steps=30.
        # This architecture field cannot use the development LIBERO value (10)
        # when loading the released weights for supervised fine-tuning.
        action_horizon=10,
        state_padding_dim=20,
        action_padding_dim=20,
        norm_stats_path=_NORM_STATS,
        filter_by_segments=False,
        return_all_norm_forms=True,
    )
