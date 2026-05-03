#!/usr/bin/env bash
# Run every evaluation deliverable end-to-end against a single checkpoint.
# Assumes you've already trained and have a best_checkpoint directory.
#
# Usage:
#   bash scripts/run_all_evals.sh artifacts/run_final/best_checkpoint
#
# Output goes under artifacts/. All commands are idempotent on re-run.

set -euo pipefail

CHECKPOINT_DIR="${1:-artifacts/run_final/best_checkpoint}"
CONFIG="${2:-configs/course_config.json}"
ROOT_OUT="${3:-artifacts}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint dir not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

echo "============================================================"
echo "Using checkpoint: $CHECKPOINT_DIR"
echo "Using config:     $CONFIG"
echo "Writing under:    $ROOT_OUT"
echo "============================================================"

# 1. Official public benchmark
echo
echo ">>> [1/7] Official public benchmark rollout"
python generate_public_rollout.py \
    --config "$CONFIG" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --stage-name stage_2 \
    --output-dir "$ROOT_OUT/public_eval_final" \
    --num-episodes 4 \
    --render-first-episode

echo
echo ">>> [2/7] Official public benchmark scoring"
python public_eval.py \
    --config "$CONFIG" \
    --rollout-npz "$ROOT_OUT/public_eval_final/rollout_public_eval.npz" \
    --output-json "$ROOT_OUT/public_eval_final/public_eval.json"

# 3-6. Custom evals
echo
echo ">>> [3/7] Custom: signed direction tracking"
python -m scripts.eval_signed_directions \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --output-dir "$ROOT_OUT/eval_signed_directions"

echo
echo ">>> [4/7] Custom: per-axis magnitude sweep"
python -m scripts.eval_magnitude_sweep \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --output-dir "$ROOT_OUT/eval_magnitude_sweep"

echo
echo ">>> [5/7] Custom: combined-command 2D grids"
python -m scripts.eval_combined_grid \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --output-dir "$ROOT_OUT/eval_combined_grid"

echo
echo ">>> [6/7] Custom: step response + stability"
python -m scripts.eval_step_response \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --output-dir "$ROOT_OUT/eval_step_response"

python -m scripts.eval_stability \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --output-dir "$ROOT_OUT/eval_stability" \
    --num-trials 50

# 7. Qualitative video
echo
echo ">>> [7/7] Qualitative video demo (assignment-spec magnitudes)"
python -m scripts.video_demo \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --output-dir "$ROOT_OUT/video_demo"

echo
echo "============================================================"
echo "All evaluations complete. Artifacts:"
echo "  $ROOT_OUT/public_eval_final/public_eval.json"
echo "  $ROOT_OUT/eval_signed_directions/{signed_directions_summary.json, signed_directions.png}"
echo "  $ROOT_OUT/eval_magnitude_sweep/{magnitude_sweep_summary.json, magnitude_sweep.png}"
echo "  $ROOT_OUT/eval_combined_grid/{combined_grid_summary.json, combined_grid.png}"
echo "  $ROOT_OUT/eval_step_response/{step_response_summary.json, step_response.png}"
echo "  $ROOT_OUT/eval_stability/{stability_summary.json, stability.png}"
echo "  $ROOT_OUT/video_demo/video_demo.mp4"
echo "============================================================"
