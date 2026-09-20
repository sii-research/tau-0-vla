import collections
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import pickle
import re
import subprocess
import time

import imageio
import numpy as np
import requests
import tqdm
import tyro
from PIL import Image

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class LiberoHttpPolicy:
    """HTTP client for the dedicated LIBERO policy server."""

    def __init__(self, host: str, port: int, timeout: int = 60, wait_timeout: int = 120) -> None:
        self._base_url = f"http://{host}:{port}"
        self._url = f"http://{host}:{port}/act_libero"
        self._timeout = timeout
        self._wait_for_server(host, port, wait_timeout)

    def _wait_for_server(self, host: str, port: int, wait_timeout: int) -> None:
        health_url = f"http://{host}:{port}/health"
        logging.info(f"Waiting for server at {health_url}...")
        deadline = time.monotonic() + wait_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(health_url, timeout=5)
                response.raise_for_status()
                self.metadata = response.json()
                logging.info("Server is ready.")
                return
            except requests.RequestException as exc:
                last_error = exc
                logging.info("Still waiting for server...")
                time.sleep(5)
        raise TimeoutError(f"Server did not become ready within {wait_timeout}s: {health_url}") from last_error

    def reset(self, seed: int) -> None:
        response = requests.post(f"{self._base_url}/reset_episode", params={"seed": seed}, timeout=self._timeout)
        response.raise_for_status()

    def infer(self, obs: dict) -> dict:
        body = pickle.dumps(obs)
        resp = requests.post(
            self._url,
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return {"actions": np.array(resp.json())}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    server_wait_timeout: int = 120
    resize_size: int = 224
    replan_steps: int = 8

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task
    episode_start: int = 0  # First initial-state index; use 1 with four trials to extend a one-trial run.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "outputs/libero_eval/libero_goal"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    from libero.libero import benchmark

    if args.num_trials_per_task < 1 or args.episode_start < 0 or args.replan_steps < 1:
        raise ValueError("trials/replan_steps must be positive and episode_start must be nonnegative")
    if not 0 <= args.seed < 2**32 or args.resize_size < 1 or args.num_steps_wait < 0:
        raise ValueError("seed must be in [0, 2**32), resize_size positive, and num_steps_wait nonnegative")
    # Set random seed
    np.random.seed(args.seed)
    video_out_dir = pathlib.Path(args.video_out_path)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    video_out_dir.mkdir(parents=True, exist_ok=True)
    if (video_out_dir / "episodes.jsonl").exists():
        raise FileExistsError(f"Evaluation records already exist in {video_out_dir}; choose a new output directory")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = LiberoHttpPolicy(args.host, args.port, wait_timeout=args.server_wait_timeout)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parents[2], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=pathlib.Path(__file__).resolve().parents[2], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    (video_out_dir / "run.json").write_text(
        json.dumps(
            {
                "args": dataclasses.asdict(args),
                "code_revision": revision,
                "code_dirty": dirty,
                "client_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "server": client.metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    records = []

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        if args.episode_start + args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"Task {task_id} provides {len(initial_states)} initial states, "
                f"but {args.num_trials_per_task} trials were requested"
            )

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Always release the renderer, including setup, RPC, and video errors.
        try:
            for episode_idx in tqdm.tqdm(range(args.episode_start, args.episode_start + args.num_trials_per_task)):
                record = _run_episode(
                    env,
                    client,
                    initial_states[episode_idx],
                    task_description,
                    task_id,
                    episode_idx,
                    args,
                    max_steps,
                    video_out_dir,
                )
                records.append(record)
                with (video_out_dir / "episodes.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record) + "\n")
                _write_results(video_out_dir, args, records)
                total_episodes += 1
                total_successes += int(record["success"])
                logging.info(
                    "Task %d episode %d: success=%s; total %d/%d",
                    task_id,
                    episode_idx,
                    record["success"],
                    total_successes,
                    total_episodes,
                )
                if record["error"] is not None:
                    raise RuntimeError(
                        f"Evaluation stopped after task {task_id}, episode {episode_idx}: {record['error']}"
                    )
        finally:
            env.close()

    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")

    summary_path = video_out_dir / "results.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"task_suite_name: {args.task_suite_name}",
                f"total_successes: {total_successes}",
                f"total_episodes: {total_episodes}",
                f"total_success_rate: {total_success_rate:.6f}",
                f"total_success_rate_percent: {total_success_rate * 100:.2f}%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Wrote success summary to %s", summary_path)


def _run_episode(env, client, initial_state, task_description, task_id, episode_idx, args, max_steps, video_out_dir):
    replay_images = []
    success = False
    error = None
    video_name = None
    try:
        # Reset simulator and policy RNG for reproducible, independently runnable episodes.
        env.seed(args.seed)
        env.reset()
        client.reset(args.seed)
        obs = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        action_plan = collections.deque()
        for _ in range(max_steps):
            img = _prepare_image(obs["agentview_image"], args.resize_size)
            wrist_img = _prepare_image(obs["robot0_eye_in_hand_image"], args.resize_size)
            replay_images.append(img)
            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                    "prompt": str(task_description),
                }
                chunk = np.asarray(client.infer(element)["actions"])
                if (
                    chunk.ndim != 2
                    or chunk.shape[1] != 7
                    or len(chunk) < args.replan_steps
                    or not np.isfinite(chunk).all()
                ):
                    raise ValueError(f"Invalid LIBERO action chunk: shape={chunk.shape}")
                action_plan.extend(chunk[: args.replan_steps])
            obs, _, done, _ = env.step(action_plan.popleft().tolist())
            if done:
                success = True
                break
    except Exception as exc:
        logging.exception("Episode failed with an exception")
        error = f"{type(exc).__name__}: {exc}"
    if replay_images:
        suffix = "success" if success else "failure"
        segment = re.sub(r"[^A-Za-z0-9._-]+", "_", task_description).strip("._") or f"task_{task_id}"
        video_name = f"task_{task_id:02d}_episode_{episode_idx:03d}_{segment}_{suffix}.mp4"
        try:
            imageio.mimwrite(video_out_dir / video_name, replay_images, fps=10)
        except Exception as exc:
            logging.exception("Could not save episode video")
            error = f"{error + '; ' if error else ''}video: {type(exc).__name__}: {exc}"
            video_name = None
    return {
        "task_id": task_id,
        "task": task_description,
        "episode_index": episode_idx,
        "success": success,
        "error": error,
        "steps": len(replay_images),
        "video": video_name,
    }


def _write_results(output_dir, args, records):
    tasks = []
    for task_id in sorted({row["task_id"] for row in records}):
        rows = [row for row in records if row["task_id"] == task_id]
        tasks.append(
            {
                "task_id": task_id,
                "task": rows[0]["task"],
                "episodes": len(rows),
                "successes": sum(row["success"] for row in rows),
                "errors": sum(row["error"] is not None for row in rows),
            }
        )
    result = {
        "task_suite_name": args.task_suite_name,
        "total_episodes": len(records),
        "total_successes": sum(row["success"] for row in records),
        "total_errors": sum(row["error"] is not None for row in records),
        "tasks": tasks,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _prepare_image(image, size):
    """LIBERO renders square uint8 RGB; match the training pipeline's PIL bilinear resize.

    Keep the simulation client independent of the model's torch/LeRobot stack.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[0] != array.shape[1] or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"Expected square uint8 RGB render, got {array.shape}, {array.dtype}")
    return np.asarray(Image.fromarray(array[::-1, ::-1]).resize((size, size), Image.Resampling.BILINEAR))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    quat = np.array(quat, dtype=np.float64, copy=True)
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
