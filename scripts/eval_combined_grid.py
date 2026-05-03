#!/usr/bin/env python3
"""Custom eval: 2D grids over combined commands.

Runs two grids:
- (vx, yaw_rate) at vy=0
- (vx, vy)        at yaw_rate=0
Each grid is 5x5 covering the symmetric range. Reports velocity tracking
error per cell and renders heatmaps. Exposes corner-case behavior the
official benchmark only samples once.

Usage:
    python scripts/eval_combined_grid.py \
        --checkpoint-dir artifacts/run_final/best_checkpoint \
        --output-dir artifacts/eval_combined_grid
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--seconds-per-cell", type=float, default=5.0)
    parser.add_argument("--steady-state-seconds", type=float, default=3.0)
    parser.add_argument("--vx-max", type=float, default=1.0)
    parser.add_argument("--vy-max", type=float, default=0.4)
    parser.add_argument("--yaw-max", type=float, default=1.0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def _build_grid(axis_a: str, lo_a: float, hi_a: float, axis_b: str, lo_b: float, hi_b: float, n: int) -> list[tuple[str, str, float, float, list[float]]]:
    a_values = np.linspace(lo_a, hi_a, n)
    b_values = np.linspace(lo_b, hi_b, n)
    cells = []
    for av in a_values:
        for bv in b_values:
            command = [0.0, 0.0, 0.0]
            command["vx vy yaw".split().index(axis_a)] = float(av)
            command["vx vy yaw".split().index(axis_b)] = float(bv)
            cells.append((axis_a, axis_b, float(av), float(bv), command))
    return cells


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

    duration_steps = seconds_to_steps(args.seconds_per_cell, env.dt)
    steady_steps = seconds_to_steps(args.steady_state_seconds, env.dt)
    if steady_steps >= duration_steps:
        raise ValueError("steady_state_seconds must be smaller than seconds_per_cell")

    policy = load_policy(args.checkpoint_dir, jax, ctx["force_cpu"])

    n = int(args.grid_size)
    grids = {
        "vx_yaw": _build_grid("vx", -args.vx_max, args.vx_max, "yaw", -args.yaw_max, args.yaw_max, n),
        "vx_vy":  _build_grid("vx", -args.vx_max, args.vx_max, "vy",  -args.vy_max,  args.vy_max,  n),
    }

    segments: list[dict] = []
    cell_meta: list[dict] = []
    ep_id = 0
    for grid_name, cells in grids.items():
        for (axis_a, axis_b, av, bv, command) in cells:
            segments.append(
                {
                    "command": command,
                    "duration_steps": duration_steps,
                    "label": f"{grid_name}|{axis_a}={av:.2f}|{axis_b}={bv:.2f}",
                    "episode_id": ep_id,
                }
            )
            cell_meta.append(
                {
                    "grid": grid_name,
                    "axis_a": axis_a,
                    "axis_b": axis_b,
                    "value_a": av,
                    "value_b": bv,
                    "episode_id": ep_id,
                }
            )
            ep_id += 1

    bundle = run_scripted_segments(
        env=env,
        policy=policy,
        jax=jax,
        segments=segments,
        seed=int(ctx["config"]["seed"]) + 31,
        force_cpu=ctx["force_cpu"],
    )
    save_bundle_npz(bundle, args.output_dir / "rollout_combined_grid.npz")

    seg_idx = bundle["segment_index"]
    n_steps = seg_idx.shape[0]
    steady_mask = np.zeros(n_steps, dtype=bool)
    for s_idx in np.unique(seg_idx):
        idxs = np.where(seg_idx == s_idx)[0]
        if idxs.size == 0:
            continue
        steady_mask[idxs[-steady_steps:]] = True

    # Per-cell metrics
    grid_results: dict[str, dict] = {"vx_yaw": {}, "vx_vy": {}}
    for cell in cell_meta:
        ep_mask = (bundle["episode_id"] == cell["episode_id"]) & steady_mask
        m = metrics_for_mask(bundle, ep_mask)
        key = f"{cell['value_a']:+.2f}|{cell['value_b']:+.2f}"
        grid_results[cell["grid"]][key] = {
            "axis_a": cell["axis_a"],
            "axis_b": cell["axis_b"],
            "value_a": cell["value_a"],
            "value_b": cell["value_b"],
            "metrics": m,
        }

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "grid_size": n,
        "vx_max": args.vx_max,
        "vy_max": args.vy_max,
        "yaw_max": args.yaw_max,
        "grids": grid_results,
    }
    write_summary_json(args.output_dir / "combined_grid_summary.json", summary)

    # Heatmap rendering
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes_plt = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (grid_name, cells) in zip(axes_plt, grids.items()):
            a_values = sorted({float(c[2]) for c in cells})
            b_values = sorted({float(c[3]) for c in cells})
            heat = np.full((len(b_values), len(a_values)), np.nan, dtype=np.float32)
            for cell in cell_meta:
                if cell["grid"] != grid_name:
                    continue
                ai = a_values.index(cell["value_a"])
                bi = b_values.index(cell["value_b"])
                key = f"{cell['value_a']:+.2f}|{cell['value_b']:+.2f}"
                m = grid_results[grid_name][key]["metrics"]
                heat[bi, ai] = m["velocity_tracking_error"]
            im = ax.imshow(
                heat,
                origin="lower",
                aspect="auto",
                extent=[a_values[0], a_values[-1], b_values[0], b_values[-1]],
                cmap="viridis_r",
            )
            ax.set_xlabel(cell["axis_a"])
            ax.set_ylabel(cell["axis_b"])
            ax.set_title(f"velocity_tracking_error  ({grid_name})")
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("error (m/s, lower better)")
        fig.tight_layout()
        fig.savefig(args.output_dir / "combined_grid.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed to render plot: {exc}")

    print(json.dumps({"num_cells": sum(len(v) for v in grid_results.values())}, indent=2))


if __name__ == "__main__":
    main()
