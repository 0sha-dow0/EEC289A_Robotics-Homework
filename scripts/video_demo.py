#!/usr/bin/env python3
"""Generate the qualitative video demonstration.

Plays a single contiguous rollout of the final policy through scripted
command segments, with the joystick command arrow drawn over each frame.
Every magnitude required by the assignment is exercised:
    vx in {0.6, 0.8, 1.0}
    vy in {0.2, 0.3, 0.4}
    yaw in {0.6, 0.8, 1.0}
plus negative directions (backward, strafe-left, turn-right) and combined
commands (walk + turn, diagonal, full triple).

Each segment is approximately 5 seconds. Total runtime is ~75 seconds.

Usage:
    python scripts/video_demo.py \
        --checkpoint-dir artifacts/run_final/best_checkpoint \
        --output-dir artifacts/video_demo
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np

# Repo-root import dance.
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
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


# (label, [vx, vy, yaw_rate], duration_seconds)
DEMO_SEGMENTS = [
    ("stand_intro",        [ 0.0,  0.0,  0.0], 3.0),
    ("forward_0p6",        [ 0.6,  0.0,  0.0], 5.0),
    ("forward_0p8",        [ 0.8,  0.0,  0.0], 5.0),
    ("forward_1p0",        [ 1.0,  0.0,  0.0], 5.0),
    ("backward_0p8",       [-0.8,  0.0,  0.0], 5.0),
    ("strafe_right_0p2",   [ 0.0,  0.2,  0.0], 5.0),
    ("strafe_right_0p3",   [ 0.0,  0.3,  0.0], 5.0),
    ("strafe_left_0p4",    [ 0.0, -0.4,  0.0], 5.0),
    ("turn_left_0p6",      [ 0.0,  0.0,  0.6], 5.0),
    ("turn_left_0p8",      [ 0.0,  0.0,  0.8], 5.0),
    ("turn_right_1p0",     [ 0.0,  0.0, -1.0], 5.0),
    ("walk_and_turn",      [ 0.6,  0.0,  0.5], 5.0),
    ("diagonal_walk",      [ 0.5,  0.3,  0.0], 5.0),
    ("combined_triple",    [ 0.6,  0.2,  0.5], 5.0),
    ("stand_outro",        [ 0.0,  0.0,  0.0], 3.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-camera", type=str, default="track")
    parser.add_argument(
        "--draw-command-arrow",
        action="store_true",
        default=True,
        help="Draw the joystick command arrow over each frame.",
    )
    parser.add_argument(
        "--no-draw-command-arrow",
        dest="draw_command_arrow",
        action="store_false",
    )
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _force_command(state, command, jax):
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True
    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
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
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)
    # Total expected duration in steps; bump episode_length so the env doesn't
    # auto-truncate before the script finishes.
    total_steps_estimate = int(round(sum(s for _, _, s in DEMO_SEGMENTS) / float(env_cfg.ctrl_dt))) + 50
    env_cfg.episode_length = max(int(env_cfg.episode_length), total_steps_estimate)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))

    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)
    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    rng = jax.random.PRNGKey(int(config["seed"]) + 91)
    state = reset_fn(rng)
    state = _force_command(state, np.asarray(DEMO_SEGMENTS[0][1], dtype=np.float32), jax)

    trajectory = [state]
    modify_scene_fns = [None]

    cmd_xy = []
    meas_xy = []
    cmd_yaw = []
    meas_yaw = []
    fell_log = []
    seg_indices = []
    seg_labels = []

    # Optional joystick arrow overlay
    draw_joystick_command = None
    if args.draw_command_arrow:
        try:
            from mujoco_playground._src.gait import draw_joystick_command as _draw
            draw_joystick_command = _draw
        except Exception as exc:
            print(f"[warn] could not import draw_joystick_command, skipping arrow overlay: {exc}")

    cur_cursor = 0
    for seg_idx, (label, command, seconds) in enumerate(DEMO_SEGMENTS):
        n_steps = max(1, int(round(seconds / float(env.dt))))
        cmd_arr = np.asarray(command, dtype=np.float32)
        for _ in range(n_steps):
            state = _force_command(state, cmd_arr, jax)
            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = _force_command(state, cmd_arr, jax)
            trajectory.append(state)

            cmd_xy.append(cmd_arr[:2].copy())
            meas_xy.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
            cmd_yaw.append(float(cmd_arr[2]))
            meas_yaw.append(float(np.asarray(env.get_gyro(state.data)[2])))
            fell_log.append(bool(np.asarray(state.done)))
            seg_indices.append(seg_idx)
            seg_labels.append(label)

            if draw_joystick_command is not None:
                xyz = np.asarray(state.data.xpos[env._torso_body_id]) + np.array([0.0, 0.0, 0.2])
                x_axis = state.data.xmat[env._torso_body_id, 0]
                yaw = -float(np.arctan2(x_axis[1], x_axis[0]))
                # Scale the arrow by command norm so a stronger command looks bigger.
                cmd_norm = float(np.linalg.norm(cmd_arr))
                scl = max(cmd_norm, 0.05)
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=jax.numpy.asarray(cmd_arr),
                        xyz=xyz,
                        theta=yaw,
                        scl=scl,
                    )
                )
            else:
                modify_scene_fns.append(None)

            cur_cursor += 1

    print(f"rollout finished: {cur_cursor} control steps, {cur_cursor * env.dt:.1f} seconds")

    # Render. Sub-sample frames to keep video size sane (every other step => 25 fps for ctrl_dt=0.02).
    render_every = 1
    fps = int(round(1.0 / float(env.dt) / render_every))
    render_traj = trajectory[::render_every]
    render_mods = modify_scene_fns[::render_every]

    try:
        import mujoco

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    except Exception:
        scene_option = None

    render_kwargs = dict(
        height=int(args.render_height),
        width=int(args.render_width),
        camera=args.render_camera,
    )
    if scene_option is not None:
        render_kwargs["scene_option"] = scene_option
    if any(m is not None for m in render_mods):
        # The Playground render() takes modify_scene_fns parallel to the trajectory.
        render_kwargs["modify_scene_fns"] = render_mods

    frames = env.render(render_traj, **render_kwargs)
    video_path = args.output_dir / "video_demo.mp4"
    media.write_video(video_path, frames, fps=fps)

    # Save bundle and summary
    rollout_npz = args.output_dir / "rollout_video_demo.npz"
    np.savez(
        rollout_npz,
        episode_id=np.asarray(seg_indices, dtype=np.int32),
        segment_index=np.asarray(seg_indices, dtype=np.int32),
        segment_label=np.asarray(seg_labels),
        command_lin_vel_xy=np.asarray(cmd_xy, dtype=np.float32),
        measured_lin_vel_xy=np.asarray(meas_xy, dtype=np.float32),
        command_yaw_rate=np.asarray(cmd_yaw, dtype=np.float32),
        measured_yaw_rate=np.asarray(meas_yaw, dtype=np.float32),
        fell=np.asarray(fell_log, dtype=bool),
    )

    summary = {
        "video_path": str(video_path),
        "rollout_npz": str(rollout_npz),
        "fps": fps,
        "num_steps": int(cur_cursor),
        "duration_seconds": float(cur_cursor * env.dt),
        "segments": [
            {"label": label, "command": list(cmd), "seconds": float(s)}
            for (label, cmd, s) in DEMO_SEGMENTS
        ],
        "drew_command_arrow": draw_joystick_command is not None,
        "any_fall_during_demo": bool(any(fell_log)),
    }
    save_json(args.output_dir / "video_demo_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
