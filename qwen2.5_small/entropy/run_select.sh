#!/bin/bash
# Verbatim-logic port of select_bbox_by_entropy.sh (with kind='stable' added
# in select_bbox_by_entropy.py to make tie-breaking deterministic).
#
# Usage: bash run_select.sh [temp] [dataset]
#   temp:    0.5 (default) | 0.7 | 1.0
#   dataset: COCO (default) | Objects365

set -e
cd "$(dirname "$0")"

TEMP=${1:-0.5}
DATASET=${2:-COCO}
TARGET_DIR="temp_${TEMP}_top_p_0.9_response_10"

# ------- 12 hand-picked heads (3 per coord) -------
HEAD_SELECTION='{"x1":[[20,15], [21, 22], [19, 23]],
                 "y1":[[20,14], [20, 19], [18, 15]],
                 "x2":[[19,27], [19, 23], [18, 15]],
                 "y2":[[20,26], [14, 16], [20, 16]]}'

SAVE_DIR="../evaluation/entropy_selection_results/${DATASET}"
BATCH_SIZE=64
NUM_HEADS=28
SELECTION_MODE="rank"

# Override dataset + target_dir in config without touching the file
TMP_CONFIG=$(mktemp /tmp/manual12_repro_config.XXXXXX.yaml)
sed -e "s|target_dir:.*|target_dir: \"${TARGET_DIR}\"|" \
    -e "s|dataset:.*|dataset: \"${DATASET}\"|" config.yaml > "$TMP_CONFIG"

echo "=========================================="
echo "manual12_repro selection"
echo "Dataset:        $DATASET"
echo "Target dir:     $TARGET_DIR"
echo "Head Selection: $HEAD_SELECTION"
echo "Mode:           $SELECTION_MODE"
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
echo "  ${SAVE_DIR}/${TARGET_DIR}/entropy_${TARGET_DIR}_selection.csv"
echo "=========================================="
