#!/usr/bin/env python3
"""Custom eval: per-axis magnitude sweep.

For each of the six signed directions, sweep the command magnitude across
[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] (clamped to the per-axis training max). For
each (direction, magnitude), run a few seconds and record the achieved
velocity over the steady-state window. Plot achieved vs commanded velocity
with a y=x reference line.

This directly answers the assignment's evaluation requirement:
"behavior under different command magnitudes."

Usage:
    python scripts/eval_magnitude_sweep.py \
        --checkpoint-dir artifacts/run_final/best_checkpoint \
        --output-dir artifacts/eval_magnitude_sweep
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


# Per-axis maximum used for clamping. Matches the widened student_stage2_goal.
AXIS_MAX = {"vx": 1.0, "vy": 0.4, "yaw": 1.0}

# (axis_name, sign, command_template)
def _build_axes() -> list[tuple[str, int]]:
    return [
        ("vx", +1),
        ("vx", -1),
        ("vy", +1),
        ("vy", -1),
        ("yaw", +1),
        ("yaw", -1),
    ]


def _command_for(axis: str, value: float) -> list[float]:
    if axis == "vx":
        return [value, 0.0, 0.0]
    if axis == "vy":
        return [0.0, value, 0.0]
    if axis == "yaw":
        return [0.0, 0.0, value]
    raise ValueError(axis)


def _achieved_for(axis: str, bundle: dict) -> np.ndarray:
    if axis == "vx":
        return bundle["measured_lin_vel_xy"][:, 0]
    if axis == "vy":
        return bundle["measured_lin_vel_xy"][:, 1]
    if axis == "yaw":
        return bundle["measured_yaw_rate"]
    raise ValueError(axis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seconds-per-magnitude", type=float, default=6.0)
    parser.add_argument("--steady-state-seconds", type=float, default=4.0)
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
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

    duration_steps = seconds_to_steps(args.seconds_per_magnitude, env.dt)
    steady_steps = seconds_to_steps(args.steady_state_seconds, env.dt)
    if steady_steps >= duration_steps:
        raise ValueError("steady_state_seconds must be smaller than seconds_per_magnitude")

    policy = load_policy(args.checkpoint_dir, jax, ctx["force_cpu"])

    segments: list[dict] = []
    rows: list[dict] = []  # parallel structure: (axis, sign, magnitude, seed, ep_id)
    ep_id = 0
    axes = _build_axes()

    for axis, sign in axes:
        per_axis_max = AXIS_MAX[axis]
        for magnitude in args.magnitudes:
            value = sign * min(float(magnitude), per_axis_max)
            command = _command_for(axis, value)
            for seed_idx in range(int(args.num_seeds)):
                segments.append(
                    {
                        "command": command,
                        "duration_steps": duration_steps,
                        "label": f"{axis}{('+' if sign > 0 else '-')}_m{magnitude:.2f}",
                        "episode_id": ep_id,
                    }
                )
                rows.append(
                    {
                        "axis": axis,
                        "sign": int(sign),
                        "magnitude": float(magnitude),
                        "value": float(value),
                        "seed": int(seed_idx),
                        "episode_id": int(ep_id),
                    }
                )
                ep_id += 1

    bundle = run_scripted_segments(
        env=env,
        policy=policy,
        jax=jax,
        segments=segments,
        seed=int(ctx["config"]["seed"]) + 23,
        force_cpu=ctx["force_cpu"],
    )
    save_bundle_npz(bundle, args.output_dir / "rollout_magnitude_sweep.npz")

    seg_idx = bundle["segment_index"]
    n_steps = seg_idx.shape[0]
    steady_mask = np.zeros(n_steps, dtype=bool)
    for s_idx in np.unique(seg_idx):
        idxs = np.where(seg_idx == s_idx)[0]
        if idxs.size == 0:
            continue
        steady_mask[idxs[-steady_steps:]] = True

    # Aggregate per (axis, sign, magnitude)
    aggregated: dict[str, dict] = {}
    for row in rows:
        key = f"{row['axis']}{'+' if row['sign'] > 0 else '-'}_m{row['magnitude']:.2f}"
        ep_mask = (bundle["episode_id"] == row["episode_id"]) & steady_mask
        if not ep_mask.any():
            continue
        achieved = _achieved_for(row["axis"], bundle)[ep_mask]
        commanded = row["value"]
        record = aggregated.setdefault(
            key,
            {
                "axis": row["axis"],
                "sign": row["sign"],
                "magnitude": row["magnitude"],
                "commanded_value": commanded,
                "achieved_per_seed": [],
                "abs_error_per_seed": [],
            },
        )
        achieved_mean = float(np.mean(achieved))
        record["achieved_per_seed"].append(achieved_mean)
        record["abs_error_per_seed"].append(float(abs(achieved_mean - commanded)))

    # Reduce to mean / std
    for record in aggregated.values():
        record["achieved_mean"] = float(np.mean(record["achieved_per_seed"]))
        record["achieved_std"] = float(np.std(record["achieved_per_seed"]))
        record["abs_error_mean"] = float(np.mean(record["abs_error_per_seed"]))
        record["abs_error_std"] = float(np.std(record["abs_error_per_seed"]))

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "magnitudes": list(args.magnitudes),
        "axis_max_used_for_clamp": AXIS_MAX,
        "by_direction_magnitude": aggregated,
    }
    write_summary_json(args.output_dir / "magnitude_sweep_summary.json", summary)

    # Plot - one panel per axis, two lines per panel (positive and negative direction)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes_plt = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
        for col, axis_name in enumerate(["vx", "vy", "yaw"]):
            ax = axes_plt[col]
            for sign, color, label_suffix in [(+1, "tab:blue", "+"), (-1, "tab:red", "-")]:
                xs, ys, ystds = [], [], []
                for record in aggregated.values():
                    if record["axis"] != axis_name or record["sign"] != sign:
                        continue
                    xs.append(record["commanded_value"])
                    ys.append(record["achieved_mean"])
                    ystds.append(record["achieved_std"])
                order = np.argsort(xs)
                xs = np.asarray(xs)[order]
                ys = np.asarray(ys)[order]
                ystds = np.asarray(ystds)[order]
                ax.errorbar(
                    xs, ys, yerr=ystds, fmt="o-", color=color, label=f"{axis_name}{label_suffix}",
                )
            # y=x reference
            lim = AXIS_MAX[axis_name]
            ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4, label="y=x")
            ax.axhline(0, color="k", linewidth=0.5)
            ax.axvline(0, color="k", linewidth=0.5)
            ax.set_xlim(-lim - 0.05, lim + 0.05)
            ax.set_xlabel("commanded")
            ax.set_ylabel("achieved")
            ax.set_title(f"{axis_name} tracking sweep")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "magnitude_sweep.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed to render plot: {exc}")

    print(json.dumps({"num_records": len(aggregated)}, indent=2))


if __name__ == "__main__":
    main()
