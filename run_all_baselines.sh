#!/bin/bash
# Run all 6 baselines (first_response / first_valid / vocab_entropy / mean_bbox
# / basic_mode / smallest) for both models × both datasets × 3 temps = 12 configs.
#
# Usage:
#   bash run_all_baselines.sh                    # all 12 configs, all 6 methods
#   bash run_all_baselines.sh <method>           # only this method, all 12 configs
#   bash run_all_baselines.sh <model> <ds> <t>   # one config, all methods
#   bash run_all_baselines.sh <model> <ds> <t> <method>   # one config, one method
set -eo pipefail
cd "$(dirname "$0")"

MODELS=(qwen2.5_small internvl_3.5_small)
DATASETS=(COCO Objects365)
TEMPS=(0.5 0.7 1.0)

# Parse args
if [ $# -eq 0 ]; then
  M_LIST="${MODELS[*]}"; DS_LIST="${DATASETS[*]}"; T_LIST="${TEMPS[*]}"; METHOD=all
elif [ $# -eq 1 ]; then
  M_LIST="${MODELS[*]}"; DS_LIST="${DATASETS[*]}"; T_LIST="${TEMPS[*]}"; METHOD=$1
elif [ $# -eq 3 ]; then
  M_LIST=$1; DS_LIST=$2; T_LIST=$3; METHOD=all
elif [ $# -eq 4 ]; then
  M_LIST=$1; DS_LIST=$2; T_LIST=$3; METHOD=$4
else
  echo "Usage: bash $(basename $0) [method] | [model ds t [method]]"
  exit 1
fi

echo "=========================================="
echo "Running:  models=[$M_LIST]  datasets=[$DS_LIST]  temps=[$T_LIST]  method=$METHOD"
echo "Started at $(date)"
echo "=========================================="

for MODEL in $M_LIST; do
  for DS in $DS_LIST; do
    for T in $T_LIST; do
      echo ""
      echo "##### $MODEL / $DS / t=$T / $METHOD #####"
      bash "$MODEL/baselines/run_baseline.sh" "$DS" "$T" "$METHOD"
    done
  done
done

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "=========================================="
