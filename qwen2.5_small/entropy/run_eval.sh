#!/bin/bash
# Evaluate selection CSV produced by run_select.sh.
#
# Usage: bash run_eval.sh [temp] [dataset]
#   temp:    0.5 (default) | 0.7 | 1.0
#   dataset: COCO (default) | Objects365

set -e
cd "$(dirname "$0")"
DATA_ROOT=${DATA_ROOT:-$(cd ../.. && pwd)/data}   # folder holding object_detection_<Model>/ (see README)
PYTHON=${PYTHON:-python3}

TEMP=${1:-0.5}
DATASET=${2:-COCO}
TARGET_DIR="temp_${TEMP}_top_p_0.9_response_10"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
SEL_CSV="${REPRO_DIR}/../evaluation/entropy_selection_results/${DATASET}/${TARGET_DIR}/entropy_${TARGET_DIR}_selection.csv"
OUT_DIR_ABS="${REPRO_DIR}/../evaluation/eval_results/${DATASET}/entropy_temp_${TEMP}_top_p_0.9_response_10_evaluation_results"

if [ ! -f "$SEL_CSV" ]; then
  echo "ERROR: selection CSV not found: $SEL_CSV"
  echo "Run: bash run_select.sh ${TEMP} ${DATASET}"
  exit 1
fi

RESP_DIR="${DATA_ROOT}/object_detection_Qwen/model_response/${DATASET}/testing/single_object/bbox_attention_temp_${TEMP}_top_p_0.9_response_10"
mkdir -p "$OUT_DIR_ABS"

cd "$REPRO_DIR/.."
$PYTHON evaluation/evaluate_single_object.py \
  --method greedy --dataset "$DATASET" \
  --data_root "$RESP_DIR" \
  --selection_csv "$SEL_CSV" \
  --output_dir "$OUT_DIR_ABS" \
  --total_images 2225 2>&1 | tee "$OUT_DIR_ABS/eval.log"

echo ""
echo "=========================================="
echo "Eval done (${DATASET}, t=${TEMP}). Summary:"
grep -E "Total images|Detected|Accuracy|Mean IoU" "$OUT_DIR_ABS"/evaluation_summary_*.txt | tail -10
echo "=========================================="
