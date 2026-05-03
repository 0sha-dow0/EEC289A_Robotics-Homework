#!/usr/bin/env python3
"""Custom eval: tracking quality on the six signed directions.

The official public benchmark (`public_eval.py`) only tests positive commands:
forward `+vx`, right `+vy`, left-yaw `+yaw`. The assignment text however
explicitly requires `+vx, -vx, +vy, -vy, +yaw_rate, -yaw_rate`. This script
runs scripted single-axis episodes for all six directions (plus a stand
baseline), computes the same five metrics that `public_eval.compute_metrics`
computes, and writes both a per-direction JSON summary and a comparison plot.

Usage:
    python scripts/eval_signed_directions.py \
        --checkpoint-dir artifacts/run_final/best_checkpoint \
        --output-dir artifacts/eval_signed_directions
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
    metrics_for_mask,
    run_scripted_segments,
    save_bundle_npz,
    seconds_to_steps,
    setup_runtime_and_env,
    write_summary_json,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "course_config.json"


# (label, command_vector, magnitude_in_natural_units)
DIRECTIONS = [
    ("stand", [0.0, 0.0, 0.0]),
    ("+vx", [+0.5, 0.0, 0.0]),
    ("-vx", [-0.5, 0.0, 0.0]),
    ("+vy", [0.0, +0.15, 0.0]),
    ("-vy", [0.0, -0.15, 0.0]),
    ("+yaw", [0.0, 0.0, +0.5]),
    ("-yaw", [0.0, 0.0, -0.5]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds per direction.")
    parser.add_argument(
        "--seconds-per-direction",
        type=float,
        default=8.0,
        help="Per-seed rollout duration. Mean is taken over the last 5 s to ignore the transient.",
    )
    parser.add_argument(
        "--steady-state-seconds",
        type=float,
        default=5.0,
        help="Window at the end of each segment used for the metric.",
    )
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

    duration_steps = seconds_to_steps(args.seconds_per_direction, env.dt)
    steady_steps = seconds_to_steps(args.steady_state_seconds, env.dt)
    if steady_steps >= duration_steps:
        raise ValueError("steady_state_seconds must be smaller than seconds_per_direction")

    policy = load_policy(args.checkpoint_dir, jax, ctx["force_cpu"])

    # Build segments: each (direction x seed) is its own segment so episode_id
    # uniquely identifies the trial. The mask for steady-state metrics keeps
    # only the last `steady_steps` of each segment.
    segments: list[dict] = []
    rows: list[tuple[str, int]] = []   # (label, ep_id)
    ep_id = 0
    for label, command in DIRECTIONS:
        for _seed in range(int(args.num_seeds)):
            segments.append(
                {
                    "command": list(command),
                    "duration_steps": duration_steps,
                    "label": label,
                    "episode_id": ep_id,
                }
            )
            rows.append((label, ep_id))
            ep_id += 1

    bundle = run_scripted_segments(
        env=env,
        policy=policy,
        jax=jax,
        segments=segments,
        seed=int(ctx["config"]["seed"]) + 17,
        force_cpu=ctx["force_cpu"],
    )
    save_bundle_npz(bundle, args.output_dir / "rollout_signed_directions.npz")

    # Build per-direction metrics over the steady-state window.
    seg_idx = bundle["segment_index"]
    n_steps = seg_idx.shape[0]
    steady_mask_global = np.zeros(n_steps, dtype=bool)
    # for each segment, keep only the last `steady_steps` steps belonging to it
    for s_idx in np.unique(seg_idx):
        idxs = np.where(seg_idx == s_idx)[0]
        if idxs.size == 0:
            continue
        steady_idxs = idxs[-steady_steps:]
        steady_mask_global[steady_idxs] = True

    per_direction: dict[str, dict] = {}
    for label, _ in DIRECTIONS:
        seeds_metrics = []
        for ep in [r[1] for r in rows if r[0] == label]:
            mask = (bundle["episode_id"] == ep) & steady_mask_global
            seeds_metrics.append(metrics_for_mask(bundle, mask))
        # aggregate across seeds
        keys = [k for k in seeds_metrics[0] if k != "num_steps"]
        agg = {k: float(np.mean([m[k] for m in seeds_metrics])) for k in keys}
        agg_std = {f"{k}_std": float(np.std([m[k] for m in seeds_metrics])) for k in keys}
        agg.update(agg_std)
        agg["num_seeds"] = int(args.num_seeds)
        per_direction[label] = agg

    # Symmetry deltas: how much worse are negatives than positives?
    def delta(pos_label: str, neg_label: str) -> dict[str, float]:
        return {
            "velocity_tracking_error_delta": (
                per_direction[neg_label]["velocity_tracking_error"]
                - per_direction[pos_label]["velocity_tracking_error"]
            ),
            "yaw_tracking_error_delta": (
                per_direction[neg_label]["yaw_tracking_error"]
                - per_direction[pos_label]["yaw_tracking_error"]
            ),
        }

    symmetry = {
        "vx": delta("+vx", "-vx"),
        "vy": delta("+vy", "-vy"),
        "yaw": delta("+yaw", "-yaw"),
    }

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "directions": [d[0] for d in DIRECTIONS],
        "commands": {label: list(cmd) for label, cmd in DIRECTIONS},
        "duration_steps_per_segment": int(duration_steps),
        "steady_state_steps_per_segment": int(steady_steps),
        "per_direction_metrics": per_direction,
        "negative_minus_positive_delta": symmetry,
        "headline": {
            "max_negative_penalty_velocity_tracking": max(
                symmetry["vx"]["velocity_tracking_error_delta"],
                symmetry["vy"]["velocity_tracking_error_delta"],
            ),
            "yaw_negative_penalty": symmetry["yaw"]["yaw_tracking_error_delta"],
        },
    }
    write_summary_json(args.output_dir / "signed_directions_summary.json", summary)

    # Render plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [d[0] for d in DIRECTIONS]
        vel_err = [per_direction[l]["velocity_tracking_error"] for l in labels]
        vel_err_std = [per_direction[l]["velocity_tracking_error_std"] for l in labels]
        yaw_err = [per_direction[l]["yaw_tracking_error"] for l in labels]
        yaw_err_std = [per_direction[l]["yaw_tracking_error_std"] for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(labels))
        axes[0].bar(x, vel_err, yerr=vel_err_std, color="tab:blue", capsize=3)
        axes[0].set_xticks(x, labels)
        axes[0].set_ylabel("velocity tracking error (m/s)")
        axes[0].set_title("Linear-velocity tracking error per direction")
        axes[0].axhline(0.10, color="green", linestyle="--", alpha=0.6, label="benchmark good=0.10")
        axes[0].axhline(0.45, color="red", linestyle="--", alpha=0.6, label="benchmark bad=0.45")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].bar(x, yaw_err, yerr=yaw_err_std, color="tab:orange", capsize=3)
        axes[1].set_xticks(x, labels)
        axes[1].set_ylabel("yaw tracking error (rad/s)")
        axes[1].set_title("Yaw-rate tracking error per direction")
        axes[1].axhline(0.10, color="green", linestyle="--", alpha=0.6)
        axes[1].axhline(0.50, color="red", linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(args.output_dir / "signed_directions.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed to render plot: {exc}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
