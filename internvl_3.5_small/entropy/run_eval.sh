#!/bin/bash
# Evaluate selection CSV produced by run_select.sh on the InternVL model_response data.
#
# Usage: bash run_eval.sh [temp] [dataset]
#   temp:    0.5 (default) | 0.7 | 1.0
#   dataset: COCO (default) | Objects365

set -e
cd "$(dirname "$0")"

TEMP=${1:-0.5}
DATASET=${2:-COCO}
TARGET_DIR="temp_${TEMP}_top_p_0.9_response_10"
REPRO_DIR="$(cd "$(dirname "$0")" && pwd)"
SEL_CSV="${REPRO_DIR}/../evaluation/entropy_selection_results/${DATASET}/${TARGET_DIR}/entropy_${TARGET_DIR}.csv"
OUT_DIR_ABS="${REPRO_DIR}/../evaluation/eval_results/${DATASET}/entropy_temp_${TEMP}_top_p_0.9_response_10_evaluation_results"

if [ ! -f "$SEL_CSV" ]; then
  echo "ERROR: selection CSV not found: $SEL_CSV"
  echo "Run: bash run_select.sh ${TEMP} ${DATASET}"
  exit 1
fi

DATA_ROOT="/scratch/ty45972/MLLM_object_detection/object_detection_InternVL/model_response/${DATASET}/testing/bbox_attention_temp_${TEMP}_top_p_0.9_response_10"
mkdir -p "$OUT_DIR_ABS"

cd /scratch/ty45972/MLLM_object_detection_emnlp/experiments/internvl_3.5_small
python3 evaluation/evaluate_single_object.py \
  --method greedy --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --selection_csv "$SEL_CSV" \
  --output_dir "$OUT_DIR_ABS" \
  --total_images 2225 2>&1 | tee "$OUT_DIR_ABS/eval.log"

echo ""
echo "=========================================="
echo "Eval done (InternVL ${DATASET}, t=${TEMP}). Summary:"
grep -E "Total images|Detected|Accuracy|Mean IoU" "$OUT_DIR_ABS"/evaluation_summary_*.txt | tail -10
echo "=========================================="
