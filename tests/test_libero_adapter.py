"""Focused regressions for the LIBERO state/action contract."""

import tempfile
import types
import unittest

import numpy as np

import configs.libero.data  # noqa: F401  Registers the public LIBERO route.
from tau0_vla.adapters.libero import LiberoObservation
from tau0_vla.data.data_spec import action_slices, encode_payload, load_data_spec, restore_action, save_data_spec
from tau0_vla.data.modalities import AxisAngle2Rot6D, PadToDim, PairToDifference
from tau0_vla.data.modalities.base import WorkflowContext
from tau0_vla.data.pipeline import _active_indices_mask

_CONTEXT = WorkflowContext(state={}, action={})
_COMPONENT = types.SimpleNamespace(key="eef_pose")


class LiberoTransformsTest(unittest.TestCase):
    def test_axis_angle_round_trip_and_degenerate_inverse(self):
        transform = AxisAngle2Rot6D()
        pose = np.array([0.2, -0.1, 0.4, 0.3, -0.2, 0.1], dtype=np.float32)
        encoded = transform.forward(pose, _CONTEXT, component=_COMPONENT)
        restored = transform.inverse(encoded, _CONTEXT, component=_COMPONENT)
        np.testing.assert_allclose(restored, pose, atol=1e-5)

        degenerate = transform.inverse(np.zeros(9, dtype=np.float32), _CONTEXT, component=_COMPONENT)
        np.testing.assert_allclose(degenerate, np.zeros(6, dtype=np.float32))

    def test_padding_and_gripper_round_trips(self):
        padding = PadToDim(9, 18)
        pose = np.arange(9, dtype=np.float32)
        padded = padding.forward(pose, _CONTEXT, component=_COMPONENT)
        self.assertEqual(padded.shape, (18,))
        np.testing.assert_array_equal(padded[9:], np.zeros(9, dtype=np.float32))
        np.testing.assert_array_equal(padding.inverse(padded, _CONTEXT, component=_COMPONENT), pose)

        gripper = PairToDifference(scale=0.5)
        joints = np.array([0.04, -0.04], dtype=np.float32)
        opening = gripper.forward(joints, _CONTEXT, component=_COMPONENT)
        np.testing.assert_allclose(opening, np.array([0.04], dtype=np.float32))
        np.testing.assert_allclose(gripper.inverse(opening, _CONTEXT, component=_COMPONENT), joints)

    def test_sparse_active_mask_and_validation(self):
        indices = (*range(9), 18)
        mask = _active_indices_mask(19, 40, indices, label="action_mask")
        self.assertEqual(mask.shape, (40,))
        self.assertEqual(float(mask.sum()), 10.0)
        self.assertEqual(mask[18], 1.0)
        self.assertEqual(mask[19], 0.0)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            _active_indices_mask(19, 40, (0, 0), label="action_mask")
        with self.assertRaisesRegex(ValueError, "outside"):
            _active_indices_mask(19, 40, (40,), label="action_mask")


class LiberoPayloadTest(unittest.TestCase):
    def test_training_and_serving_match_with_stale_metadata(self):
        from configs.libero.data import libero_eef_robot_prompt_ft

        config = libero_eef_robot_prompt_ft()
        # Older datasets mislabeled the EEF state as arm joints.
        stale = {
            "state": {
                "arm_joint": {"dimensions": 7, "indices": list(range(7))},
                "gripper": {"dimensions": 1, "indices": [7]},
            }
        }
        fields = config.resolve_output_spec(field_descriptions=stale).field_descriptions
        state = np.array([0.1, -0.2, 0.3, 0.4, -0.2, 0.15, 0.04, -0.03], dtype=np.float32)
        action = np.random.default_rng(7).uniform(-0.4, 0.4, (10, 7)).astype(np.float32)
        action[:, 6] = np.where(action[:, 6] > 0, 1, -1)
        training = config._build_component_assembler(field_descriptions=fields)(
            {"_state_raw": state, "_action_raw": action, "_field_descriptions": fields}
        )
        with tempfile.TemporaryDirectory() as run_dir:
            save_data_spec("libero_eef_robot_prompt_ft", run_dir, vlm_model_type="qwen3.5")
            spec = load_data_spec(run_dir, route="libero_eef_robot_prompt_ft")
            serving = encode_payload(
                {
                    "state": state,
                    "prompt": "pick up the object",
                    "images": {k: np.zeros((32, 32, 3), np.uint8) for k in spec.cam_keys},
                },
                spec,
            )
            for key in ("state", "state_mask", "action_mask"):
                np.testing.assert_allclose(training[key], serving[key], atol=1e-6)
            inactive = np.flatnonzero(training["action_mask"] == 0)
            np.testing.assert_array_equal(training["state"][inactive], 0)
            np.testing.assert_array_equal(training["action"][:, inactive], 0)
            restored = restore_action(training["action"], spec, state=serving["state"])
            np.testing.assert_allclose(restored, action, atol=1e-5)

    def test_public_hardware_contract_and_contiguous_masks(self):
        from deploy.server import _require_public_v1_joint_only

        for unified in (False, True):
            spec = types.SimpleNamespace(
                unified_registry_key="g1" if unified else None,
                unified_has_eef=False,
                is_eef=False,
                finch_config_name="joint",
            )
            _require_public_v1_joint_only(spec)
            spec.unified_has_eef = spec.is_eef = True
            with self.assertRaisesRegex(NotImplementedError, "joint-control"):
                _require_public_v1_joint_only(spec)
        np.testing.assert_array_equal(
            _active_indices_mask(16, 40, None, label="action"), np.r_[np.ones(16), np.zeros(24)]
        )

    def test_payload_contract(self):
        state = np.arange(8, dtype=np.float32)
        observation = LiberoObservation.from_payload(
            {
                "observation/image": np.zeros((4, 4, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
                "observation/state": state,
                "prompt": "pick up the object",
            }
        )
        np.testing.assert_array_equal(observation.eef_pose, state[:6])
        np.testing.assert_array_equal(observation.gripper, state[6:])
        np.testing.assert_array_equal(observation.state, state)

        with self.assertRaisesRegex(ValueError, "exactly 8"):
            LiberoObservation.from_payload(
                {
                    "observation/image": np.zeros((1, 1, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((1, 1, 3), dtype=np.uint8),
                    "observation/state": np.zeros(7, dtype=np.float32),
                    "prompt": "invalid state",
                }
            )

    def test_persisted_data_spec_round_trip(self):
        with tempfile.TemporaryDirectory() as run_dir:
            save_data_spec(
                "libero_eef_robot_prompt_ft",
                run_dir,
                vlm_model_type="qwen3.5",
                cam_keys=("image", "wrist_image"),
                cam_view_template="{} view: <image>",
                cam_view_names={"image": "agent", "wrist_image": "wrist"},
                max_images_per_sample=2,
            )
            spec = load_data_spec(run_dir, route="libero_eef_robot_prompt_ft")

            encoded = encode_payload(
                {
                    "observation": {},
                    "state": np.zeros(8, dtype=np.float32),
                    "prompt": "pick up the object",
                    "images": {
                        "image": np.zeros((32, 32, 3), dtype=np.uint8),
                        "wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
                    },
                },
                spec,
            )
            restored = restore_action(np.zeros((10, 40), dtype=np.float32), spec, state=encoded["state"])

            self.assertEqual(spec.robot_name, "libero")
            self.assertEqual(spec.action_chunk_size, 10)
            self.assertEqual(action_slices(spec), [("eef_pose", 0, 6), ("gripper", 6, 1)])
            self.assertEqual(encoded["state"].shape, (40,))
            self.assertEqual(float(encoded["action_mask"].sum()), 10.0)
            self.assertEqual(restored.shape, (10, 7))
            self.assertTrue(np.isfinite(restored).all())


if __name__ == "__main__":
    unittest.main()
