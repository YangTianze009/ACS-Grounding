"""Convert per-bbox prediction_comparison.csv (from test.py) into a single-bbox
selection CSV (subdir, pkl_path, bbox_id), one row per image_folder.

For each image, picks the candidate bbox with the highest predicted_iou.

Logic ported verbatim from the original train_model_bbox_selection_single_object.ipynb
in object_detection_Qwen/.

Usage:
    python3 prediction_to_selection.py <prediction_comparison.csv> <output_selection.csv>
"""
import argparse
import os
import re
import sys

import pandas as pd


def select_best_from_comparison(prediction_csv_path: str, save_csv: str) -> None:
    df = pd.read_csv(prediction_csv_path)

    # Per image, pick the row with max predicted_iou
    best_rows = df.loc[df.groupby("image_folder")["predicted_iou"].idxmax()]

    results = []
    for _, row in best_rows.iterrows():
        subdir = row["image_folder"]
        # Extract "response_N.pkl" + bbox_id from pkl_file
        pkl_file = os.path.basename(row["pkl_file"])
        m = re.search(r"response_(\d+)", pkl_file)
        short_name = f"response_{m.group(1)}.pkl" if m else pkl_file
        bbox_id = int(row["bbox_id"])
        results.append({"subdir": subdir, "pkl_path": short_name, "bbox_id": bbox_id})

    out_df = pd.DataFrame(results, columns=["subdir", "pkl_path", "bbox_id"])
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    out_df.to_csv(save_csv, index=False)
    print(f"Wrote {len(out_df)} selections to {save_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prediction_csv")
    p.add_argument("output_csv")
    args = p.parse_args()
    if not os.path.exists(args.prediction_csv):
        sys.exit(f"input not found: {args.prediction_csv}")
    select_best_from_comparison(args.prediction_csv, args.output_csv)


if __name__ == "__main__":
    main()
