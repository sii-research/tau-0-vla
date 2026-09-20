from __future__ import annotations

import numpy as np

from tau0_vla.data.modalities.base import ComponentSpec, ComponentTransform, WorkflowContext


def _current_state_frame(state_reference: np.ndarray, *, context: WorkflowContext) -> np.ndarray:
    """Return a state reference that broadcasts against an action chunk.

    If the owning ``RobotConfig`` declared ``temporal_sequence["state"]``, the
    context carries ``state_has_temporal_axis=True`` and ``state_reference``
    arrives as ``(T_state, state_dim)`` — transforms that *use* state as a
    reference (``RelativeToState`` / ``ImpulseToStep``) need a single frame,
    so collapse to the last entry along axis 0 (the current frame, by
    convention; ``temporal_sequence={"state": N}`` ordering and the
    ``[i/fps for i in range(-K, 1)]`` pattern both put offset 0 last).

    Otherwise leave the array alone. In particular the stats fast path feeds
    batched state ``(batch, 1, state_dim)`` against batched action
    ``(batch, horizon, action_dim)``; its context keeps the flag False so the
    leading axis is treated as the batch axis and broadcasts without
    collapsing.
    """
    state = np.asarray(state_reference)
    if context.state_has_temporal_axis and state.ndim >= 2:
        return state[-1]
    return state


class RelativeToState(ComponentTransform):
    uses_state_reference = True
    applies_to_state_reference = False

    def __init__(self, *, mask: "np.ndarray | list[bool] | tuple[bool, ...] | None" = None) -> None:
        """Optional per-dim mask. When provided, delta is applied only at
        ``mask=True`` indices; ``mask=False`` dims pass through unchanged
        (absolute). Used to match baseline's
        ``DeltaActions(make_bool_mask(..., -N, ...))`` where some dims
        (e.g. grippers) stay absolute while arm joints are delta-fied.

        ``mask`` length must equal the component's output dim (last axis of
        the ``value`` seen at forward/inverse time).
        """
        if mask is None:
            self.mask = None
        else:
            self.mask = np.asarray(mask, dtype=bool)

    def preserves_static_window_range(self, *, component: ComponentSpec) -> bool:
        # Subtracting one anchor-state reference from every item in a future
        # action window preserves per-axis range. EEF pose is different because
        # it rotates deltas into the body frame.
        return component.key != "eef_pose"

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        if state_reference is None:
            raise KeyError("RelativeToState requires a state reference")
        state_reference = _current_state_frame(state_reference, context=context)
        if component.key == "eef_pose":
            return self._forward_eef_pose(value, state_reference)
        if self.mask is None:
            return value - state_reference
        return self._apply_masked(value, state_reference, sign=-1.0)

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        if state_reference is None:
            raise KeyError("RelativeToState requires a state reference")
        state_reference = _current_state_frame(state_reference, context=context)
        if component.key == "eef_pose":
            return self._inverse_eef_pose(value, state_reference)
        if self.mask is None:
            return value + state_reference
        return self._apply_masked(value, state_reference, sign=+1.0)

    def _apply_masked(self, value: np.ndarray, state_reference: np.ndarray, *, sign: float) -> np.ndarray:
        value_arr = np.asarray(value, dtype=np.float32)
        state_arr = np.asarray(state_reference, dtype=np.float32)
        mask = self.mask
        last_dim = value_arr.shape[-1]
        if mask.shape[-1] != last_dim:
            raise ValueError(f"RelativeToState mask length {mask.shape[-1]} does not match value last-axis {last_dim}")
        # Broadcast state_reference over any leading value axes (e.g. action time axis).
        # Use np.add/np.subtract with indexing on the masked last-axis slice.
        result = value_arr.copy()
        delta = sign * state_arr[..., mask]
        result[..., mask] = result[..., mask] + delta
        return result

    @staticmethod
    def _forward_eef_pose(value: np.ndarray, state_reference: np.ndarray) -> np.ndarray:
        pose = np.asarray(value, dtype=np.float32)
        state = np.asarray(state_reference, dtype=np.float32)
        pose_dim = pose.shape[-1]
        state_dim = state.shape[-1]
        if pose_dim != state_dim:
            raise ValueError(
                "RelativeToState for eef_pose requires matching trailing dims, "
                f"got value.shape={pose.shape}, state_reference.shape={state.shape}"
            )
        if pose_dim % 9 != 0:
            raise ValueError(f"RelativeToState for eef_pose expects xyz+rot6d blocks of 9 dims, got shape={pose.shape}")

        pose_blocks = pose.reshape(*pose.shape[:-1], pose_dim // 9, 9)
        state_blocks = state.reshape(*state.shape[:-1], pose_dim // 9, 9)

        pos_diff = pose_blocks[..., :3] - state_blocks[..., :3]
        pose_rotation = RelativeToState._rot6d_to_matrix(pose_blocks[..., 3:])
        state_rotation = RelativeToState._rot6d_to_matrix(state_blocks[..., 3:])
        state_rotation_T = np.swapaxes(state_rotation, -1, -2)
        # Body-frame position delta: R_state^T @ (p_action - p_state)
        delta_xyz = np.matmul(state_rotation_T, pos_diff[..., np.newaxis]).squeeze(-1)
        # Rotation delta: R_state^T @ R_action
        relative_rotation = np.matmul(state_rotation_T, pose_rotation)
        relative_rot6d = RelativeToState._matrix_to_rot6d(relative_rotation)
        return np.concatenate([delta_xyz, relative_rot6d], axis=-1).reshape(*pose.shape[:-1], -1)

    @staticmethod
    def _inverse_eef_pose(value: np.ndarray, state_reference: np.ndarray) -> np.ndarray:
        pose = np.asarray(value, dtype=np.float32)
        state = np.asarray(state_reference, dtype=np.float32)
        pose_dim = pose.shape[-1]
        state_dim = state.shape[-1]
        if pose_dim != state_dim:
            raise ValueError(
                "RelativeToState inverse for eef_pose requires matching trailing dims, "
                f"got value.shape={pose.shape}, state_reference.shape={state.shape}"
            )
        if pose_dim % 9 != 0:
            raise ValueError(
                f"RelativeToState inverse for eef_pose expects xyz+rot6d blocks of 9 dims, got shape={pose.shape}"
            )

        pose_blocks = pose.reshape(*pose.shape[:-1], pose_dim // 9, 9)
        state_blocks = state.reshape(*state.shape[:-1], pose_dim // 9, 9)

        delta_blocks = pose_blocks[..., :3]
        relative_rotation = RelativeToState._rot6d_to_matrix(pose_blocks[..., 3:])
        state_rotation = RelativeToState._rot6d_to_matrix(state_blocks[..., 3:])
        # Inverse body-frame position delta: p_action = R_state @ delta_xyz + p_state
        restored_xyz = np.matmul(state_rotation, delta_blocks[..., np.newaxis]).squeeze(-1) + state_blocks[..., :3]
        # Inverse rotation delta: R_action = R_state @ R_delta
        restored_rotation = np.matmul(state_rotation, relative_rotation)
        restored_rot6d = RelativeToState._matrix_to_rot6d(restored_rotation)
        return np.concatenate([restored_xyz, restored_rot6d], axis=-1).reshape(*pose.shape[:-1], -1)

    @staticmethod
    def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        rot6d = np.asarray(rot6d, dtype=np.float32)
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = Quat2Rot6D._normalize(a1)
        a2_proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = Quat2Rot6D._normalize(a2 - a2_proj)
        b3 = np.cross(b1, b2, axis=-1)
        return np.stack([b1, b2, b3], axis=-1)

    @staticmethod
    def _matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
        rotation = np.asarray(rotation, dtype=np.float32)
        return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)


class ImpulseToStep(ComponentTransform):
    uses_state_reference = True
    applies_to_state_reference = False

    def preserves_static_window_range(self, *, component: ComponentSpec) -> bool:
        del component
        return True

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del component
        if state_reference is None:
            raise KeyError("ImpulseToStep requires a state reference")
        state_reference = _current_state_frame(state_reference, context=context)
        return state_reference + value


class Quat2Rot6D(ComponentTransform):
    def __init__(self, quat_order: str | None = None):
        if quat_order is not None and quat_order not in {"xyzw", "wxyz"}:
            raise ValueError(f"Unsupported quaternion order: {quat_order}")
        self.quat_order = quat_order

    @property
    def resolved_quat_order(self) -> str:
        return self.quat_order if self.quat_order is not None else "xyzw"

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        pose = np.asarray(value, dtype=np.float32)
        pose_dim = pose.shape[-1]
        if pose_dim % 7 != 0:
            raise ValueError(f"Quat2Rot6D expects xyz+quat blocks of 7 dims, got shape={pose.shape}")

        pose_blocks = pose.reshape(*pose.shape[:-1], pose_dim // 7, 7)
        xyz = pose_blocks[..., :3]
        quat = pose_blocks[..., 3:]
        rot6d = self._quat_to_rot6d(quat)
        return np.concatenate([xyz, rot6d], axis=-1).reshape(*pose.shape[:-1], -1)

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        pose = np.asarray(value, dtype=np.float32)
        pose_dim = pose.shape[-1]
        if pose_dim % 9 != 0:
            raise ValueError(f"Quat2Rot6D inverse expects xyz+rot6d blocks of 9 dims, got shape={pose.shape}")

        pose_blocks = pose.reshape(*pose.shape[:-1], pose_dim // 9, 9)
        xyz = pose_blocks[..., :3]
        rot6d = pose_blocks[..., 3:]
        quat = self._rot6d_to_quat(rot6d)
        return np.concatenate([xyz, quat], axis=-1).reshape(*pose.shape[:-1], -1)

    def _quat_to_rot6d(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32)
        if self.resolved_quat_order == "wxyz":
            quat = quat[..., [1, 2, 3, 0]]
        return self._quat_xyzw_to_rot6d(quat)

    def _rot6d_to_quat(self, rot6d: np.ndarray) -> np.ndarray:
        quat_xyzw = self._rot6d_to_quat_xyzw(rot6d)
        # Canonicalize: enforce w >= 0 to avoid sign ambiguity (q ≡ -q).
        w = quat_xyzw[..., 3:4]
        quat_xyzw = np.where(w < 0, -quat_xyzw, quat_xyzw)
        if self.resolved_quat_order == "wxyz":
            return quat_xyzw[..., [3, 0, 1, 2]]
        return quat_xyzw

    @staticmethod
    def _quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float32)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        safe_norm = np.where(norm > 0, norm, 1.0)
        quat = quat / safe_norm

        x = quat[..., 0]
        y = quat[..., 1]
        z = quat[..., 2]
        w = quat[..., 3]

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        xw = x * w
        yw = y * w
        zw = z * w

        rotation = np.stack(
            [
                np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)], axis=-1),
                np.stack([2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)], axis=-1),
                np.stack([2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)], axis=-1),
            ],
            axis=-2,
        )
        return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)

    @staticmethod
    def _rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
        rot6d = np.asarray(rot6d, dtype=np.float32)
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = Quat2Rot6D._normalize(a1)
        a2_proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = Quat2Rot6D._normalize(a2 - a2_proj)
        b3 = np.cross(b1, b2, axis=-1)

        rotation = np.stack([b1, b2, b3], axis=-1)
        return Quat2Rot6D._rotation_matrix_to_quat_xyzw(rotation)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(norm > 0, norm, 1.0)

    @staticmethod
    def _rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
        m00 = rotation[..., 0, 0]
        m01 = rotation[..., 0, 1]
        m02 = rotation[..., 0, 2]
        m10 = rotation[..., 1, 0]
        m11 = rotation[..., 1, 1]
        m12 = rotation[..., 1, 2]
        m20 = rotation[..., 2, 0]
        m21 = rotation[..., 2, 1]
        m22 = rotation[..., 2, 2]

        qw = np.sqrt(np.maximum(0.0, 1.0 + m00 + m11 + m22)) / 2.0
        qx = np.sign(m21 - m12) * np.sqrt(np.maximum(0.0, 1.0 + m00 - m11 - m22)) / 2.0
        qy = np.sign(m02 - m20) * np.sqrt(np.maximum(0.0, 1.0 - m00 + m11 - m22)) / 2.0
        qz = np.sign(m10 - m01) * np.sqrt(np.maximum(0.0, 1.0 - m00 - m11 + m22)) / 2.0

        quat_xyzw = np.stack([qx, qy, qz, qw], axis=-1)
        return Quat2Rot6D._normalize(quat_xyzw).astype(np.float32)


class AxisAngle2Rot6D(ComponentTransform):
    """Convert xyz+axis-angle blocks (6D) to xyz+rot6d blocks (9D)."""

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        from scipy.spatial.transform import Rotation

        pose = np.asarray(value, dtype=np.float32)
        pose_dim = pose.shape[-1]
        if pose_dim % 6 != 0:
            raise ValueError(f"AxisAngle2Rot6D expects xyz+axis-angle blocks of 6 dims, got shape={pose.shape}")
        blocks = pose.reshape(*pose.shape[:-1], pose_dim // 6, 6)
        rotvec = blocks[..., 3:]
        rotation = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(*rotvec.shape[:-1], 3, 3)
        rot6d = RelativeToState._matrix_to_rot6d(rotation)
        return np.concatenate([blocks[..., :3], rot6d], axis=-1).reshape(*pose.shape[:-1], -1).astype(np.float32)

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        from scipy.spatial.transform import Rotation

        pose = np.asarray(value, dtype=np.float32)
        pose_dim = pose.shape[-1]
        if pose_dim % 9 != 0:
            raise ValueError(f"AxisAngle2Rot6D inverse expects xyz+rot6d blocks of 9 dims, got shape={pose.shape}")
        blocks = pose.reshape(*pose.shape[:-1], pose_dim // 9, 9)
        rotation = RelativeToState._rot6d_to_matrix(blocks[..., 3:])
        flat_rotation = rotation.reshape(-1, 3, 3)
        valid = np.isfinite(flat_rotation).all(axis=(1, 2)) & (np.linalg.det(flat_rotation) > 1e-6)
        if not np.all(valid):
            # Decoder construction probes inverse transforms with an all-zero
            # vector to infer native output dimensions. A zero rot6d has no
            # orientation; interpret this (and similarly degenerate model
            # outputs) as identity instead of asking scipy to invert a null
            # coordinate frame.
            flat_rotation = flat_rotation.copy()
            flat_rotation[~valid] = np.eye(3, dtype=np.float32)
        rotvec = Rotation.from_matrix(flat_rotation).as_rotvec().reshape(*blocks.shape[:-1], 3)
        return np.concatenate([blocks[..., :3], rotvec], axis=-1).reshape(*pose.shape[:-1], -1).astype(np.float32)


class PadToDim(ComponentTransform):
    """Right-pad one component so the following component starts at a semantic slot."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim <= 0 or self.output_dim < self.input_dim:
            raise ValueError(f"PadToDim requires 0 < input_dim <= output_dim, got {input_dim}, {output_dim}")

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != self.input_dim:
            raise ValueError(f"PadToDim expected trailing dim {self.input_dim}, got shape={array.shape}")
        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, self.output_dim - self.input_dim)
        return np.pad(array, pad_width, mode="constant", constant_values=0.0).astype(np.float32)

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != self.output_dim:
            raise ValueError(f"PadToDim inverse expected trailing dim {self.output_dim}, got shape={array.shape}")
        return array[..., : self.input_dim]


class PairToDifference(ComponentTransform):
    """Collapse two opposing gripper joints to one signed opening value."""

    def __init__(self, scale: float = 0.5) -> None:
        self.scale = float(scale)
        if self.scale == 0.0:
            raise ValueError("PairToDifference scale must be non-zero")

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != 2:
            raise ValueError(f"PairToDifference expects two gripper joints, got shape={array.shape}")
        return self.scale * (array[..., 0:1] - array[..., 1:2])

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != 1:
            raise ValueError(f"PairToDifference inverse expects one opening value, got shape={array.shape}")
        joint = array / (2.0 * self.scale)
        return np.concatenate([joint, -joint], axis=-1).astype(np.float32)


class Euler2Rot6D(ComponentTransform):
    """Convert xyz+euler blocks (6D) to xyz+rot6d blocks (9D).

    Euler convention: XYZ extrinsic — R = Rz(rz) @ Ry(ry) @ Rx(rx).
    Input block layout: [x, y, z, rx, ry, rz].
    Output block layout: [x, y, z, rot6d(6)].
    """

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        pose = np.asarray(value, dtype=np.float32)
        pose_dim = pose.shape[-1]
        if pose_dim % 6 != 0:
            raise ValueError(
                f"Euler2Rot6D expects xyz+euler blocks of 6 dims, got shape={pose.shape}"
            )

        pose_blocks = pose.reshape(*pose.shape[:-1], pose_dim // 6, 6)
        xyz = pose_blocks[..., :3]
        euler = pose_blocks[..., 3:]
        rot6d = self._euler_xyz_to_rot6d(euler)
        return np.concatenate([xyz, rot6d], axis=-1).reshape(*pose.shape[:-1], -1)

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: ComponentSpec,
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        pose = np.asarray(value, dtype=np.float32)
        pose_dim = pose.shape[-1]
        if pose_dim % 9 != 0:
            raise ValueError(
                f"Euler2Rot6D inverse expects xyz+rot6d blocks of 9 dims, got shape={pose.shape}"
            )

        pose_blocks = pose.reshape(*pose.shape[:-1], pose_dim // 9, 9)
        xyz = pose_blocks[..., :3]
        rot6d = pose_blocks[..., 3:]
        euler = self._rot6d_to_euler_xyz(rot6d)
        return np.concatenate([xyz, euler], axis=-1).reshape(*pose.shape[:-1], -1)

    @staticmethod
    def _euler_xyz_to_rot6d(euler: np.ndarray) -> np.ndarray:
        euler = np.asarray(euler, dtype=np.float32)
        rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        r00 = cz * cy
        r10 = sz * cy
        r20 = -sy
        r01 = cz * sy * sx - sz * cx
        r11 = sz * sy * sx + cz * cx
        r21 = cy * sx

        return np.stack([r00, r10, r20, r01, r11, r21], axis=-1)

    @staticmethod
    def _rot6d_to_euler_xyz(rot6d: np.ndarray) -> np.ndarray:
        rot6d = np.asarray(rot6d, dtype=np.float32)
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = Quat2Rot6D._normalize(a1)
        a2_proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = Quat2Rot6D._normalize(a2 - a2_proj)
        b3 = np.cross(b1, b2, axis=-1)

        r00, r10, r20 = b1[..., 0], b1[..., 1], b1[..., 2]
        r01, r11, r21 = b2[..., 0], b2[..., 1], b2[..., 2]
        _r02, _r12, r22 = b3[..., 0], b3[..., 1], b3[..., 2]

        ry = np.arcsin(np.clip(-r20, -1.0, 1.0))
        rx = np.arctan2(r21, r22)
        rz = np.arctan2(r10, r00)

        return np.stack([rx, ry, rz], axis=-1)
