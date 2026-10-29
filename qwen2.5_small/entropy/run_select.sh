#!/bin/bash
# ACS-Free: pick one candidate per image by attention entropy on the 12 discovered heads.
#
# Usage: bash run_select.sh [temp] [dataset]
#   temp:    0.5 (default) | 0.7 | 1.0
#   dataset: COCO (default) | Objects365

set -e
cd "$(dirname "$0")"
DATA_ROOT=${DATA_ROOT:-$(cd ../.. && pwd)/data}   # folder holding object_detection_<Model>/ (see README)
PYTHON=${PYTHON:-python3}

TEMP=${1:-0.5}
DATASET=${2:-COCO}
TARGET_DIR="temp_${TEMP}_top_p_0.9_response_10"

# ------- 12 discovered heads (3 per coordinate), from the gradient analysis of the trained regressor -------
HEAD_SELECTION='{"x1":[[20,15], [21, 22], [19, 23]],
                 "y1":[[20,14], [20, 19], [18, 15]],
                 "x2":[[19,27], [19, 23], [18, 15]],
                 "y2":[[20,26], [14, 16], [20, 16]]}'

SAVE_DIR="../evaluation/entropy_selection_results/${DATASET}"
BATCH_SIZE=64
NUM_HEADS=28
SELECTION_MODE="rank"

# Override dataset + target_dir in config without touching the file
TMP_CONFIG=$(mktemp /tmp/acsfree_config.XXXXXX.yaml)
sed -e "s|target_dir:.*|target_dir: \"${TARGET_DIR}\"|" \
    -e "s|dataset:.*|dataset: \"${DATASET}\"|" -e "s|root_dir:.*|root_dir: \"${DATA_ROOT}/object_detection_Qwen/regression_training/processed_data\"|" \
    config.yaml > "$TMP_CONFIG"

echo "=========================================="
echo "ACS-Free selection"
echo "Dataset:        $DATASET"
echo "Target dir:     $TARGET_DIR"
echo "Head Selection: $HEAD_SELECTION"
echo "Mode:           $SELECTION_MODE"
echo "Save Dir:       $SAVE_DIR"
echo "=========================================="

$PYTHON select_bbox_by_entropy.py \
    --config "$TMP_CONFIG" \
    --save_dir "$SAVE_DIR" \
    --batch_size $BATCH_SIZE \
    --num_heads $NUM_HEADS \
    --head_selection "$HEAD_SELECTION" \
    --selection_mode "$SELECTION_MODE"

rm -f "$TMP_CONFIG"

echo "=========================================="
echo "DONE. Final CSV:"
echo "  ${SAVE_DIR}/${TARGET_DIR}/entropy_${TARGET_DIR}_selection.csv"
echo "=========================================="
