"""Shared utilities for the custom evaluation scripts.

This module mirrors the conventions used in `test_policy.py` and
`generate_public_rollout.py` (lazy imports, force-cpu handling, scripted
commands via `_force_command`) so the custom evals stay consistent with the
official benchmark pipeline. All custom-eval scripts should call
`setup_runtime_and_env` once and then drive the env through `run_scripted_segments`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Make the repo root importable when scripts/ is run as a file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from course_common import (  # noqa: E402
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround  # noqa: E402


def setup_runtime_and_env(
    config_path: Path,
    stage_name: str,
    force_cpu: bool,
    episode_length_steps: int | None = None,
) -> dict[str, Any]:
    """Build the env, the PPO config, and lazy-imported helper modules.

    Returns a context dict with: env, env_cfg, jax, media, registry.
    """
    config = load_json(config_path)
    config["runtime_overrides"] = {}
    if force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]
    media = stack["media"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, stage_name)
    if episode_length_steps is not None:
        env_cfg.episode_length = int(episode_length_steps)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))

    return {
        "config": config,
        "env": env,
        "env_cfg": env_cfg,
        "registry": registry,
        "jax": jax,
        "media": media,
        "force_cpu": force_cpu,
    }


def load_policy(checkpoint_dir: Path, jax: Any, force_cpu: bool):
    """Restore a Brax PPO policy and optionally jit it."""
    policy = load_policy_with_workaround(checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)
    return policy


def force_command(state: Any, command: np.ndarray, jax: Any) -> Any:
    """Lock the env's command to the given vector for the next step."""
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def run_scripted_segments(
    *,
    env: Any,
    policy: Any,
    jax: Any,
    segments: list[dict[str, Any]],
    seed: int,
    force_cpu: bool,
    keep_trajectory: bool = False,
) -> dict[str, np.ndarray]:
    """Run the policy through a scripted command sequence.

    Each segment is a dict with at least {"command": [vx, vy, yaw], "duration_steps": int}.
    Each segment can also carry an arbitrary "label" string used for grouping in
    downstream summaries. Optional "episode_id" overrides the default sequence
    of episode ids.

    Returns a dict of per-step numpy arrays ready to be saved as `.npz` and
    consumed by `public_eval.compute_metrics`.
    """
    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    rng = jax.random.PRNGKey(int(seed))
    state = reset_fn(rng)

    if not segments:
        raise ValueError("segments must be non-empty")
    state = force_command(state, np.asarray(segments[0]["command"], dtype=np.float32), jax)

    episode_ids: list[int] = []
    labels: list[str] = []
    cmd_xy: list[np.ndarray] = []
    meas_xy: list[np.ndarray] = []
    cmd_yaw: list[float] = []
    meas_yaw: list[float] = []
    fell: list[bool] = []
    joint_torques: list[np.ndarray] = []
    joint_velocities: list[np.ndarray] = []
    foot_slip_speed: list[np.ndarray] = []
    base_height: list[float] = []
    seg_index_per_step: list[int] = []
    trajectory: list[Any] = []
    if keep_trajectory:
        trajectory.append(state)

    for seg_idx, segment in enumerate(segments):
        command = np.asarray(segment["command"], dtype=np.float32)
        duration = int(segment["duration_steps"])
        seg_label = str(segment.get("label", f"seg_{seg_idx}"))
        ep_id = int(segment.get("episode_id", seg_idx))

        for _ in range(duration):
            state = force_command(state, command, jax)
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = force_command(state, command, jax)
            if keep_trajectory:
                trajectory.append(state)

            episode_ids.append(ep_id)
            labels.append(seg_label)
            cmd_xy.append(command[:2])
            meas_xy.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
            cmd_yaw.append(float(command[2]))
            meas_yaw.append(float(np.asarray(env.get_gyro(state.data)[2])))
            joint_torques.append(np.asarray(state.data.actuator_force, dtype=np.float32))
            joint_velocities.append(np.asarray(state.data.qvel[6:], dtype=np.float32))
            feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr], dtype=np.float32)
            foot_slip_speed.append(np.linalg.norm(feet_vel[:, :2], axis=-1).astype(np.float32))
            done = bool(np.asarray(state.done))
            fell.append(done)
            base_height.append(float(np.asarray(state.data.qpos[2])))
            seg_index_per_step.append(seg_idx)

            if done:
                # Reset and continue: the rest of this segment is wasted, but
                # we keep going so per-segment metrics are still well-defined.
                rng, reset_rng = jax.random.split(rng)
                state = reset_fn(reset_rng)
                state = force_command(state, command, jax)
                if keep_trajectory:
                    trajectory.append(state)

    bundle: dict[str, Any] = {
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "segment_index": np.asarray(seg_index_per_step, dtype=np.int32),
        "segment_label": np.asarray(labels),
        "command_lin_vel_xy": np.asarray(cmd_xy, dtype=np.float32),
        "measured_lin_vel_xy": np.asarray(meas_xy, dtype=np.float32),
        "command_yaw_rate": np.asarray(cmd_yaw, dtype=np.float32),
        "measured_yaw_rate": np.asarray(meas_yaw, dtype=np.float32),
        "fell": np.asarray(fell, dtype=bool),
        "joint_torques": np.asarray(joint_torques, dtype=np.float32),
        "joint_velocities": np.asarray(joint_velocities, dtype=np.float32),
        "foot_slip_speed": np.asarray(foot_slip_speed, dtype=np.float32),
        "base_height": np.asarray(base_height, dtype=np.float32),
    }
    if keep_trajectory:
        bundle["_trajectory"] = trajectory  # not saved to .npz, used for rendering
    return bundle


def save_bundle_npz(bundle: dict[str, Any], path: Path) -> None:
    """Save the rollout bundle, dropping any non-array fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saveable = {k: v for k, v in bundle.items() if isinstance(v, np.ndarray)}
    np.savez(path, **saveable)


def metrics_for_mask(
    bundle: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    """Mirror of `public_eval.compute_metrics` over a boolean mask."""
    if not mask.any():
        return {
            "velocity_tracking_error": float("nan"),
            "yaw_tracking_error": float("nan"),
            "fall_rate": float("nan"),
            "energy_proxy": float("nan"),
            "foot_slip_proxy": float("nan"),
            "num_steps": 0,
        }

    cmd_xy = bundle["command_lin_vel_xy"][mask]
    meas_xy = bundle["measured_lin_vel_xy"][mask]
    cmd_yaw = bundle["command_yaw_rate"][mask]
    meas_yaw = bundle["measured_yaw_rate"][mask]
    fell = bundle["fell"][mask].astype(bool)
    torques = bundle["joint_torques"][mask]
    qvel = bundle["joint_velocities"][mask]
    foot_slip = bundle["foot_slip_speed"][mask]

    velocity_tracking_error = float(np.linalg.norm(cmd_xy - meas_xy, axis=-1).mean())
    yaw_tracking_error = float(np.abs(cmd_yaw - meas_yaw).mean())
    fall_rate = float(np.mean(fell.astype(np.float32)))
    energy_proxy = float(np.abs(torques * qvel).sum(axis=-1).mean())
    foot_slip_proxy = float(np.asarray(foot_slip, dtype=np.float32).mean())

    return {
        "velocity_tracking_error": velocity_tracking_error,
        "yaw_tracking_error": yaw_tracking_error,
        "fall_rate": fall_rate,
        "energy_proxy": energy_proxy,
        "foot_slip_proxy": foot_slip_proxy,
        "num_steps": int(mask.sum()),
    }


def write_summary_json(path: Path, payload: Any) -> None:
    """Save a JSON summary, NaN-safe."""
    def clean(value: Any) -> Any:
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return [clean(v) for v in value.tolist()]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    save_json(path, clean(payload))


def seconds_to_steps(seconds: float, dt: float) -> int:
    return max(1, int(round(float(seconds) / float(dt))))
