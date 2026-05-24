#!/bin/bash
# Single entry-point for InternVL-3.5 baselines.
#
# Usage:
#   bash run_baseline.sh <DS> <T> [method]
#     DS:     COCO | Objects365
#     T:      0.5 | 0.7 | 1.0
#     method: pass@1 | first_valid | vocab_entropy | mean_bbox
#             | basic_mode | smallest | all   (default: all)
#             (pass@1 = the old "first_response" baseline: always pick response_1, bbox_1)
#
# Outputs (new layout):
#   $EXP_DIR/evaluation/selection_csvs/<DS>/<method>_<target>_selection.csv
#   $EXP_DIR/evaluation/eval_results/<DS>/<method>_<target>_evaluation_results/
set -eo pipefail
cd "$(dirname "$0")"
BASE_DIR=$(pwd)
EXP_DIR=$(cd .. && pwd)

DS=${1:-COCO}
T=${2:-0.5}
METHOD=${3:-all}
TARGET=temp_${T}_top_p_0.9_response_10

PYTHON=/home/ty45972/.conda/envs/internvl/bin/python3
MTYPE=internvl
DATA_ROOT=/scratch/ty45972/MLLM_object_detection/object_detection_InternVL/model_response/$DS/testing/bbox_attention_$TARGET

SEL_DIR=$EXP_DIR/evaluation/selection_csvs/$DS
EVAL_DIR=$EXP_DIR/evaluation/eval_results/$DS
LOG_DIR=$BASE_DIR/baseline_logs
mkdir -p "$SEL_DIR" "$EVAL_DIR" "$LOG_DIR"

[ -d "$DATA_ROOT" ] || { echo "ERROR: data not found: $DATA_ROOT"; exit 1; }

run_pass_at_1() {
  local SEL=$SEL_DIR/pass@1_${TARGET}_selection.csv
  echo "[$(date +%H:%M:%S)] pass@1  → $SEL"
  $PYTHON - <<PYEOF
import os, csv
DIR = "$DATA_ROOT"
subs = sorted(d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d)))
rows = []
for s in subs:
    for ext in (".pkl.gz", ".pkl"):
        if os.path.exists(os.path.join(DIR, s, f"response_1{ext}")):
            rows.append((s, f"response_1{ext}", 1)); break
with open("$SEL", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["subdir","pkl_path","bbox_id"]); w.writerows(rows)
print(f"  {len(rows)} rows")
PYEOF
  eval_csv "pass@1"
}

run_select_bbox_baseline() {  # $1 = method name (first_valid | vocab_entropy | smallest)
  local M=$1
  local SEL=$SEL_DIR/${M}_${TARGET}_selection.csv
  echo "[$(date +%H:%M:%S)] select_bbox_baselines --methods $M  → $SEL"
  $PYTHON -u select_bbox_baselines.py \
    --data_root "$DATA_ROOT" --out_dir "$SEL_DIR" \
    --methods "$M" --tag "$TARGET" 2>&1 | tail -3
  mv -f "$SEL_DIR/${M}_${TARGET}.csv" "$SEL" 2>/dev/null || true
  eval_csv "$M"
}

run_basic_mode() {
  local OUT=$EVAL_DIR/basic_mode_${TARGET}_evaluation_results
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] basic_mode (bbox_aggregation_eval --agg basic)"
  $PYTHON -u bbox_aggregation_eval.py \
    --data_root "$DATA_ROOT" --out_dir "$OUT" \
    --model_type "$MTYPE" --agg basic --total_images 2225 \
    --label "basic_mode_${TARGET}" 2>&1 \
    | grep -E "Total images|Detected|Accuracy|Mean IoU" | tail -5
}

run_mean_bbox() {
  # MeanBBox: per-coord arithmetic mean across N candidates → synthetic bbox.
  local OUT=$EVAL_DIR/mean_bbox_${TARGET}_evaluation_results
  mkdir -p "$OUT"
  echo "[$(date +%H:%M:%S)] mean_bbox (bbox_aggregation_eval --agg mean)"
  $PYTHON -u bbox_aggregation_eval.py \
    --data_root "$DATA_ROOT" --out_dir "$OUT" \
    --model_type "$MTYPE" --agg mean --total_images 2225 \
    --label "mean_bbox_${TARGET}" 2>&1 \
    | grep -E "Total images|Detected|Accuracy|Mean IoU" | tail -5
}

eval_csv() {  # $1 = method (uses corresponding *_selection.csv)
  local M=$1
  local SEL=$SEL_DIR/${M}_${TARGET}_selection.csv
  local OUT=$EVAL_DIR/${M}_${TARGET}_evaluation_results
  mkdir -p "$OUT"
  [ -f "$SEL" ] || { echo "  MISSING $SEL"; return; }
  $PYTHON "$EXP_DIR/evaluation/evaluate_single_object.py" \
    --method greedy --dataset "$DS" \
    --data_root "$DATA_ROOT" --selection_csv "$SEL" \
    --output_dir "$OUT" --total_images 2225 2>&1 \
    | grep -E "Total images|Detected|Accuracy|Mean IoU" | tail -5
}

echo "=========================================="
echo "InternVL-3.5 baseline:  $METHOD  on  $DS  t=$T"
echo "=========================================="

case "$METHOD" in
  pass@1|first_response)  run_pass_at_1 ;;
  first_valid)            run_select_bbox_baseline first_valid ;;
  vocab_entropy)          run_select_bbox_baseline vocab_entropy ;;
  mean_bbox)              run_mean_bbox ;;
  smallest)               run_select_bbox_baseline smallest ;;
  basic_mode)             run_basic_mode ;;
  all)
    # Optimization: select_bbox_baselines.py shares pkl loads across methods.
    # Run first_valid + vocab_entropy + smallest in ONE Python call.
    # mean_bbox + basic_mode go through bbox_aggregation_eval (separate Python call, pkl warm by then).
    run_pass_at_1
    echo "[$(date +%H:%M:%S)] select_bbox_baselines (first_valid + vocab_entropy + smallest, batched)"
    $PYTHON -u select_bbox_baselines.py \
      --data_root "$DATA_ROOT" --out_dir "$SEL_DIR" \
      --methods first_valid vocab_entropy smallest --tag "$TARGET" 2>&1 | tail -6
    for m in first_valid vocab_entropy smallest; do
      mv -f "$SEL_DIR/${m}_${TARGET}.csv" "$SEL_DIR/${m}_${TARGET}_selection.csv" 2>/dev/null || true
      eval_csv "$m"
    done
    run_mean_bbox
    run_basic_mode
    ;;
  *) echo "Unknown method: $METHOD"; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] DONE  ($METHOD / $DS / t=$T)"
