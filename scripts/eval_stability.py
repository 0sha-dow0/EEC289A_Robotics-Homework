#!/usr/bin/env python3
"""Custom eval: stability under random commands.

Samples N random command vectors uniformly from the full goal range, runs
each for a fixed duration, and records whether the robot fell, when, and
the final base height. Aggregates fall rate per command-magnitude bucket so
the report can show "stability vs command magnitude" cleanly.

Directly answers the assignment requirement to evaluate "stability (e.g.,
failure / falling behavior)."

Usage:
    python scripts/eval_stability.py \
        --checkpoint-dir artifacts/run_final/best_checkpoint \
        --output-dir artifacts/eval_stability \
        --num-trials 50
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--seconds-per-trial", type=float, default=15.0)
    parser.add_argument("--vx-max", type=float, default=1.0)
    parser.add_argument("--vy-max", type=float, default=0.4)
    parser.add_argument("--yaw-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
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

    duration_steps = seconds_to_steps(args.seconds_per_trial, env.dt)

    rng = np.random.default_rng(args.seed)
    commands = []
    for _ in range(int(args.num_trials)):
        cmd = [
            float(rng.uniform(-args.vx_max, args.vx_max)),
            float(rng.uniform(-args.vy_max, args.vy_max)),
            float(rng.uniform(-args.yaw_max, args.yaw_max)),
        ]
        commands.append(cmd)

    segments = []
    for i, cmd in enumerate(commands):
        segments.append(
            {
                "command": cmd,
                "duration_steps": duration_steps,
                "label": f"trial_{i:03d}",
                "episode_id": i,
            }
        )

    policy = load_policy(args.checkpoint_dir, jax, ctx["force_cpu"])
    bundle = run_scripted_segments(
        env=env,
        policy=policy,
        jax=jax,
        segments=segments,
        seed=int(ctx["config"]["seed"]) + 53,
        force_cpu=ctx["force_cpu"],
    )
    save_bundle_npz(bundle, args.output_dir / "rollout_stability.npz")

    seg_idx = bundle["segment_index"]
    fell = bundle["fell"]
    base_height = bundle["base_height"]

    per_trial = []
    for i, cmd in enumerate(commands):
        idxs = np.where(seg_idx == i)[0]
        if idxs.size == 0:
            continue
        any_fell = bool(np.any(fell[idxs]))
        first_fall = int(np.argmax(fell[idxs])) if any_fell else None
        per_trial.append(
            {
                "trial": i,
                "command": cmd,
                "command_norm": float(np.linalg.norm(cmd)),
                "fell": any_fell,
                "first_fall_step": first_fall,
                "first_fall_seconds": (first_fall * float(env.dt)) if first_fall is not None else None,
                "final_base_height": float(base_height[idxs[-1]]),
                "min_base_height": float(np.min(base_height[idxs])),
            }
        )

    # Aggregate by command-norm bucket.
    norms = np.asarray([t["command_norm"] for t in per_trial])
    buckets = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.8)]
    bucket_summary = []
    for lo, hi in buckets:
        in_bucket = [t for t in per_trial if lo <= t["command_norm"] < hi]
        if not in_bucket:
            continue
        fell_rate = float(np.mean([t["fell"] for t in in_bucket]))
        mean_min_h = float(np.mean([t["min_base_height"] for t in in_bucket]))
        bucket_summary.append(
            {
                "command_norm_bucket": [lo, hi],
                "num_trials": len(in_bucket),
                "fall_rate": fell_rate,
                "mean_min_base_height": mean_min_h,
            }
        )

    overall = {
        "num_trials": len(per_trial),
        "overall_fall_rate": float(np.mean([t["fell"] for t in per_trial])),
        "mean_min_base_height": float(np.mean([t["min_base_height"] for t in per_trial])),
        "buckets": bucket_summary,
    }

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "vx_max": args.vx_max,
        "vy_max": args.vy_max,
        "yaw_max": args.yaw_max,
        "seconds_per_trial": args.seconds_per_trial,
        "overall": overall,
        "per_trial": per_trial,
    }
    write_summary_json(args.output_dir / "stability_summary.json", summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Scatter command norm vs (fell or not), plus fall rate per bucket
        fig, axes_plt = plt.subplots(1, 2, figsize=(11, 4.2))
        axes_plt[0].scatter(
            [t["command_norm"] for t in per_trial],
            [int(t["fell"]) for t in per_trial],
            c=["tab:red" if t["fell"] else "tab:green" for t in per_trial],
            alpha=0.7,
        )
        axes_plt[0].set_xlabel("||command||")
        axes_plt[0].set_ylabel("fell (1) / survived (0)")
        axes_plt[0].set_title("Per-trial outcomes")
        axes_plt[0].grid(alpha=0.3)

        if bucket_summary:
            xs = [f"[{b['command_norm_bucket'][0]}, {b['command_norm_bucket'][1]})" for b in bucket_summary]
            fall_rates = [b["fall_rate"] for b in bucket_summary]
            axes_plt[1].bar(xs, fall_rates, color="tab:red", alpha=0.8)
            axes_plt[1].set_ylim(0, 1)
            axes_plt[1].set_ylabel("fall rate")
            axes_plt[1].set_title("Fall rate by command norm")
            for x, fr, b in zip(xs, fall_rates, bucket_summary):
                axes_plt[1].text(x, fr + 0.02, f"n={b['num_trials']}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "stability.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed to render plot: {exc}")

    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
