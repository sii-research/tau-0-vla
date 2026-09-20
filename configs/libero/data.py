"""LIBERO training route: 8D EEF state and 7D delta-EEF action."""

from __future__ import annotations

import os
from pathlib import Path

from tau0_vla.adapters.libero import LiberoRobot
from tau0_vla.data import EefPose, Gripper, Image, Prompt, PromptSource, register_config
from tau0_vla.data.modalities import AxisAngle2Rot6D, PadToDim, PairToDifference
from tau0_vla.data.modalities.image import ResizeWithPad

_DATA = os.environ.get("TAU0_LIBERO_DATA", "libero")
_NORM_STATS = os.environ.get(
    "TAU0_LIBERO_NORM_STATS",
    str(Path(__file__).with_name("norm_stats.json")),
)


def _robot_prompt(robot_type: str, control_mode: str) -> str:
    return (
        "You are controlling a robot.\n"
        f"Robot type: {robot_type}\n"
        f"Control mode: {control_mode}\n"
        "Whole-body control: disabled\n"
        "Task: {instruction}"
    )


def _libero_eef_config(*, prompt_template: str) -> LiberoRobot:
    return LiberoRobot(
        repo_id=_DATA,
        images=[
            Image("image", transforms=[ResizeWithPad(224, 224)]),
            Image("wrist_image", transforms=[ResizeWithPad(224, 224)]),
        ],
        prompt_source=PromptSource.from_label(source="parquet"),
        prompt=Prompt(template=prompt_template),
        state=[
            # Native xyz+axis-angle -> unified left-EEF xyz+rot6d at slots 0:9.
            # Padding this component to 18 places the next component at slot 18.
            EefPose(normalize="mean_std", transforms=[AxisAngle2Rot6D(), PadToDim(9, 18)]),
            # Two opposing LIBERO finger joints describe one gripper opening.
            Gripper(normalize="mean_std", transforms=[PairToDifference(scale=0.5)]),
        ],
        action=[
            # Dataset values are already delta xyz + delta axis-angle. Convert
            # only the rotation representation; do not relativize a second time.
            EefPose(
                normalize="mean_std",
                abs2relative=False,
                transforms=[AxisAngle2Rot6D(), PadToDim(9, 18)],
            ),
            Gripper(normalize="mean_std"),
        ],
        # The released base checkpoint uses a horizon of 30. ModelBuilder
        # adapts this to 10 while reusing the shared action-token weights.
        action_horizon=10,
        state_padding_dim=40,
        action_padding_dim=40,
        norm_stats_path=_NORM_STATS,
        filter_by_segments=False,
        return_all_norm_forms=True,
    )


@register_config
def libero_eef_ft() -> LiberoRobot:
    """Original question-style LIBERO prompt."""
    return _libero_eef_config(prompt_template="What action should the robot take to {instruction}?")


@register_config
def libero_eef_robot_prompt_ft() -> LiberoRobot:
    """Robot/control-aware prompt matching the unified training template."""
    return _libero_eef_config(prompt_template=_robot_prompt("Panda", "end-effector"))
