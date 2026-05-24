#!/bin/bash
# InternVL entropy-based bbox selection - mirrors qwen2.5_small_entropy/manual12_repro/run_select.sh
# but with InternVL's 12 hand-picked heads + num_heads=32.
#
# Usage: bash run_select.sh [temp] [dataset]
#   temp:    0.5 (default) | 0.7 | 1.0
#   dataset: COCO (default) | Objects365

set -e
cd "$(dirname "$0")"

TEMP=${1:-0.5}
DATASET=${2:-COCO}
TARGET_DIR="temp_${TEMP}_top_p_0.9_response_10"

# ------- InternVL 12 hand-picked heads (3 per coord) -------
# Source: object_detection_InternVL/regression_training/select_bbox_by_entropy.sh
HEAD_SELECTION='{"x1":[[21,19], [23, 30], [18, 5]],
                 "y1":[[24,31], [21, 28], [24, 21]],
                 "x2":[[21,19], [21, 23], [22, 0]],
                 "y2":[[21,23], [24, 11], [21, 29]]}'

SAVE_DIR="../evaluation/entropy_selection_results/${DATASET}"
BATCH_SIZE=64
NUM_HEADS=32
SELECTION_MODE="rank"

TMP_CONFIG=$(mktemp /tmp/internvl_manual12_config.XXXXXX.yaml)
sed -e "s|target_dir:.*|target_dir: \"${TARGET_DIR}\"|" \
    -e "s|dataset:.*|dataset: \"${DATASET}\"|" config.yaml > "$TMP_CONFIG"

echo "=========================================="
echo "InternVL manual12_repro selection"
echo "Dataset:        $DATASET"
echo "Target dir:     $TARGET_DIR"
echo "Head Selection: $HEAD_SELECTION"
echo "Mode:           $SELECTION_MODE"
echo "Num Heads:      $NUM_HEADS"
echo "Save Dir:       $SAVE_DIR"
echo "=========================================="

python3 select_bbox_by_entropy.py \
    --config "$TMP_CONFIG" \
    --save_dir "$SAVE_DIR" \
    --batch_size $BATCH_SIZE \
    --num_heads $NUM_HEADS \
    --head_selection "$HEAD_SELECTION" \
    --selection_mode "$SELECTION_MODE"

rm -f "$TMP_CONFIG"

echo "=========================================="
echo "DONE. Final CSV:"
echo "  ${SAVE_DIR}/${TARGET_DIR}/entropy_${TARGET_DIR}.csv"
echo "=========================================="
