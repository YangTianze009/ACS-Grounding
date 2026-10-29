#!/bin/bash
# InternVL regression pipeline:
#   1. test.py            — score each candidate bbox (produces prediction_comparison.csv)
#   2. prediction_to_selection.py — argmax per image (produces regression_*_selection.csv)
#   3. evaluate_single_object.py  — Accuracy + mIoU on the selection
#
# Usage: bash run_test.sh [temp] [dataset]
#   temp:    0.5 (default) | 0.7 | 1.0
#   dataset: COCO (default) | Objects365

set -eo pipefail
cd "$(dirname "$0")"
REG_DIR=$(pwd)
MODEL_DIR=$(cd .. && pwd)
DATA_ROOT=${DATA_ROOT:-$(cd "$MODEL_DIR/.." && pwd)/data}   # folder holding object_detection_<Model>/ (see README)
PYTHON=${PYTHON:-python3}

TEMP=${1:-0.5}
DATASET=${2:-COCO}
TARGET_DIR="temp_${TEMP}_top_p_0.9_response_10"

MODEL_CKPT=${CKPT:-$MODEL_DIR/../weights/acs_learned_internvl3.5-8b.pth}   # released ACS-Learned weights (see README)
# InternVL test.py writes to ../evaluation/regression_results/<DS>/<target_dir> (relative to CWD)
PRED_OUT_DIR=$MODEL_DIR/evaluation/regression_results/$DATASET/$TARGET_DIR
SEL_CSV_DIR=$MODEL_DIR/evaluation/selection_csvs/$DATASET
SEL_CSV=$SEL_CSV_DIR/regression_${TARGET_DIR}_selection.csv
EVAL_DIR=$MODEL_DIR/evaluation/eval_results/$DATASET/regression_${TARGET_DIR}_evaluation_results

mkdir -p "$PRED_OUT_DIR" "$SEL_CSV_DIR" "$EVAL_DIR"

# Override target_dir + dataset in config
TMP_CFG=$(mktemp /tmp/internvl_regression_test.XXXXXX.yaml)
sed -e "s|target_dir:.*|target_dir: \"${TARGET_DIR}\"|" \
    -e "s|dataset:.*|dataset: \"${DATASET}\"|" \
    -e "s|test_results_dir:.*|test_results_dir: \"$MODEL_DIR/evaluation/regression_results\"|" \
    -e 's|^device:.*|device: "cuda:0"|' \
    -e "s|root_dir:.*|root_dir: \"${DATA_ROOT}/object_detection_InternVL/regression_training/processed_data\"|" \
    "$REG_DIR/config.yaml" > "$TMP_CFG"

echo "=========================================="
echo "InternVL regression test"
echo "Dataset:   $DATASET"
echo "Temp dir:  $TARGET_DIR"
echo "Model:     $MODEL_CKPT"
echo "=========================================="

# Step 1: inference
$PYTHON test.py --config "$TMP_CFG" --model "$MODEL_CKPT" --save_predictions 2>&1 | tail -5
rm -f "$TMP_CFG"

PRED_CSV=$PRED_OUT_DIR/prediction_comparison.csv
[ -f "$PRED_CSV" ] || { echo "ERROR: prediction CSV not produced at $PRED_CSV"; exit 1; }

# Step 2: argmax → selection CSV
$PYTHON "prediction_to_selection.py" "$PRED_CSV" "$SEL_CSV"

# Step 3: eval
RESP_DIR=${DATA_ROOT}/object_detection_InternVL/model_response/$DATASET/testing/bbox_attention_${TARGET_DIR}
$PYTHON "$MODEL_DIR/evaluation/evaluate_single_object.py" \
  --method greedy --dataset "$DATASET" \
  --data_root "$RESP_DIR" --selection_csv "$SEL_CSV" \
  --output_dir "$EVAL_DIR" --total_images 2225 2>&1 | grep -E "Total images|Detected|Accuracy|Mean IoU" | tail -5

echo "=========================================="
echo "DONE. Eval: $EVAL_DIR/evaluation_summary_*.txt"
echo "=========================================="
