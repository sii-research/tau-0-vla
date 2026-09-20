"""Evaluation accounting and image parity without a simulator or model."""

import json
import pathlib
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from deploy.libero.main import Args, _prepare_image, _run_episode, eval_libero


class LiberoEvaluationTest(unittest.TestCase):
    def test_image_matches_training(self):
        from tau0_vla.data.pipeline import resize_image_hwc

        image = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        np.testing.assert_array_equal(_prepare_image(image, 224), resize_image_hwc(image[::-1, ::-1], (224, 224)))

    def test_episode_errors_and_success_are_distinct(self):
        obs = {
            "agentview_image": np.zeros((256, 256, 3), np.uint8),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }
        env = Mock()
        env.set_init_state.return_value = obs
        env.step.return_value = (obs, 0, True, {})
        args = Args(num_steps_wait=0)
        for chunk, valid in [
            (np.zeros((10, 7)), True),
            (np.zeros((7, 7)), False),
            (np.zeros((10, 40)), False),
            (np.full((10, 7), np.nan), False),
        ]:
            client = Mock()
            client.infer.return_value = {"actions": chunk}
            with tempfile.TemporaryDirectory() as out, patch("deploy.libero.main.imageio.mimwrite"):
                record = _run_episode(env, client, None, "pick / object", 0, 3, args, 2, pathlib.Path(out))
            self.assertEqual(record["success"], valid)
            self.assertEqual(record["error"] is None, valid)
            self.assertEqual(record["episode_index"], 3)
            self.assertNotIn("/", record["video"])
            client.reset.assert_called_once_with(7)
        client.reset.side_effect = TimeoutError("server unavailable")
        record = _run_episode(env, client, None, "pick", 0, 0, args, 2, pathlib.Path("unused"))
        self.assertIn("TimeoutError", record["error"])
        self.assertFalse(record["success"])

    def test_offset_accounting_and_cleanup_on_error(self):
        suite = Mock(n_tasks=1)
        suite.get_task_init_states.return_value = list(range(5))
        benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_goal": lambda: suite})
        modules = {"libero": types.ModuleType("libero"), "libero.libero": types.ModuleType("libero.libero")}
        modules["libero.libero"].benchmark = benchmark
        for error in (None, "TimeoutError: unavailable"):
            env = Mock()
            record = {
                "task_id": 0,
                "task": "pick",
                "episode_index": 1,
                "success": error is None,
                "error": error,
                "steps": 1,
                "video": "test.mp4",
            }
            with (
                tempfile.TemporaryDirectory() as out,
                patch.dict("sys.modules", modules),
                patch("deploy.libero.main.LiberoHttpPolicy", return_value=Mock(metadata={})),
                patch("deploy.libero.main._get_libero_env", return_value=(env, "pick")),
                patch("deploy.libero.main._run_episode", return_value=record) as run,
            ):
                args = Args(episode_start=1, num_trials_per_task=1, video_out_path=out)
                if error:
                    with self.assertRaisesRegex(RuntimeError, "Evaluation stopped"):
                        eval_libero(args)
                else:
                    eval_libero(args)
                self.assertEqual(run.call_args.args[2], 1)
                env.close.assert_called_once()
                result = json.loads((pathlib.Path(out) / "results.json").read_text())
                self.assertEqual(result["total_errors"], int(error is not None))
                self.assertEqual(result["total_episodes"], 1)
                self.assertTrue((pathlib.Path(out) / "episodes.jsonl").exists())
                with self.assertRaises(FileExistsError):
                    eval_libero(args)


if __name__ == "__main__":
    unittest.main()
