#!/usr/bin/env python3
"""Custom eval: step response to scripted command transitions.

Single contiguous rollout that walks the policy through:
  zero -> +vx 1.0 -> +yaw 1.0 -> negative diagonal (-vx, -vy) -> zero

Plots achieved velocity overlaid on the commanded reference, per axis. This
is the most readable single figure for diagnosing transient response, gait
re-acquisition after a step, and tracking quality at high command magnitudes.

Usage:
    python scripts/eval_step_response.py \
        --checkpoint-dir artifacts/run_final/best_checkpoint \
        --output-dir artifacts/eval_step_response
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_common import (
    load_policy,
    run_scripted_segments,
    save_bundle_npz,
    seconds_to_steps,
    setup_runtime_and_env,
    write_summary_json,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "course_config.json"


SEGMENTS_SECONDS = [
    ("stand", [ 0.0,  0.0,  0.0], 2.0),
    ("forward_max", [ 1.0,  0.0,  0.0], 5.0),
    ("yaw_max", [ 0.0,  0.0,  1.0], 5.0),
    ("backward_strafe", [-0.5, -0.3,  0.0], 5.0),
    ("combined_full", [ 0.6,  0.2,  0.5], 5.0),
    ("stand_recover", [ 0.0,  0.0,  0.0], 3.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ctx = setup_runtime_and_env(
        config_path=args.config,
        stage_name=args.stage_name,
        force_cpu=args.force_cpu,
    )
    env = ctx["env"]
    jax = ctx["jax"]

    segments: list[dict] = []
    boundaries: list[tuple[str, int, int]] = []  # (label, start, end) in steps
    cursor = 0
    for label, command, seconds in SEGMENTS_SECONDS:
        n = seconds_to_steps(seconds, env.dt)
        segments.append({"command": list(command), "duration_steps": n, "label": label})
        boundaries.append((label, cursor, cursor + n))
        cursor += n

    policy = load_policy(args.checkpoint_dir, jax, ctx["force_cpu"])
    bundle = run_scripted_segments(
        env=env,
        policy=policy,
        jax=jax,
        segments=segments,
        seed=int(ctx["config"]["seed"]) + 41,
        force_cpu=ctx["force_cpu"],
    )
    save_bundle_npz(bundle, args.output_dir / "rollout_step_response.npz")

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "segments": [
            {"label": label, "command": list(cmd), "seconds": float(s), "start_step": b[1], "end_step": b[2]}
            for ((label, cmd, s), b) in zip(SEGMENTS_SECONDS, boundaries)
        ],
        "fall_during_rollout": bool(np.any(bundle["fell"])),
        "first_fall_step": int(np.argmax(bundle["fell"])) if bool(np.any(bundle["fell"])) else None,
    }
    write_summary_json(args.output_dir / "step_response_summary.json", summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_axis = np.arange(bundle["command_lin_vel_xy"].shape[0]) * float(env.dt)

        fig, axes_plt = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        # vx
        axes_plt[0].plot(time_axis, bundle["command_lin_vel_xy"][:, 0], "k--", label="commanded vx")
        axes_plt[0].plot(time_axis, bundle["measured_lin_vel_xy"][:, 0], "tab:blue", label="achieved vx")
        axes_plt[0].set_ylabel("vx (m/s)")
        axes_plt[0].legend(loc="upper right", fontsize=8)
        # vy
        axes_plt[1].plot(time_axis, bundle["command_lin_vel_xy"][:, 1], "k--", label="commanded vy")
        axes_plt[1].plot(time_axis, bundle["measured_lin_vel_xy"][:, 1], "tab:orange", label="achieved vy")
        axes_plt[1].set_ylabel("vy (m/s)")
        axes_plt[1].legend(loc="upper right", fontsize=8)
        # yaw
        axes_plt[2].plot(time_axis, bundle["command_yaw_rate"], "k--", label="commanded yaw")
        axes_plt[2].plot(time_axis, bundle["measured_yaw_rate"], "tab:green", label="achieved yaw")
        axes_plt[2].set_ylabel("yaw rate (rad/s)")
        axes_plt[2].legend(loc="upper right", fontsize=8)
        axes_plt[2].set_xlabel("time (s)")

        # Vertical bars at segment boundaries with labels
        for label, start_step, end_step in boundaries:
            t_start = float(start_step) * env.dt
            for ax in axes_plt:
                ax.axvline(t_start, color="grey", alpha=0.3, linewidth=0.8)
            axes_plt[0].text(t_start + 0.05, axes_plt[0].get_ylim()[1] * 0.85, label, fontsize=7, color="grey")

        for ax in axes_plt:
            ax.grid(alpha=0.3)
        fig.suptitle("Step response: commanded vs achieved velocity")
        fig.tight_layout()
        fig.savefig(args.output_dir / "step_response.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed to render plot: {exc}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
