"""Single-object detection evaluation.

Computes Accuracy (IoU>=0.5) and Mean IoU from a pre-computed bbox selection CSV.
Reads bbox coordinates + GT from model_response pkl files at the original data location.

Usage:
    python evaluate_single_object.py \\
        --method regression \\
        --dataset COCO \\
        --temperature 1.0 --top_p 0.9 \\
        --base_response 10 --test_response 10
"""
import os
import sys
import pickle
import gzip
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np


# model_response pkls: $DATA_ROOT/object_detection_InternVL/model_response (DATA_ROOT defaults to <repo>/data)
ORIG_DATA_ROOT = os.path.join(os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data")),
                              "object_detection_InternVL", "model_response")

# Output + selection CSV location
EXP_DIR = Path(__file__).resolve().parent.parent  # <repo>/<model>
DETECTION_RESULTS_DIR = EXP_DIR / "evaluation" / "detection_results"


def load_pickle_file(file_path):
    """Load pickle file, supporting both .pkl and .pkl.gz formats"""
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            return pickle.load(f)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def map_bbox_to_original_size(pred_bbox: List[float], processed_size: Tuple[int, int],
                              original_size: Tuple[int, int]) -> List[float]:
    processed_w, processed_h = processed_size
    original_w, original_h = original_size
    if processed_w == 0 or processed_h == 0:
        return pred_bbox
    scale_x = original_w / processed_w
    scale_y = original_h / processed_h
    return (np.array(pred_bbox) * np.array([scale_x, scale_y, scale_x, scale_y])).tolist()


def calculate_max_iou_with_gt(pred_box: List[float], gt_boxes: List[List[float]]) -> float:
    if not gt_boxes:
        return 0.0
    return max(calculate_iou(pred_box, gt_box) for gt_box in gt_boxes)


def evaluate_model_performance(data_root, selection_csv, output_dir, total_images_override=None):
    """Single-object evaluation: Acc@IoU>=0.5 and Mean IoU."""
    os.makedirs(output_dir, exist_ok=True)
    selection_df = pd.read_csv(selection_csv)
    print(f"Loaded {len(selection_df)} images from selection CSV")

    total_images = 0
    detected_images = 0
    total_iou = 0.0
    total_iou_before_mapping = 0.0
    mapped_images = 0
    no_bbox_images = []
    bbox_mapping_log = []
    missing_size_info = []

    for _, row in selection_df.iterrows():
        subdir = row["subdir"]
        pkl_path = row["pkl_path"]
        bbox_id = row["bbox_id"]

        subdir_path = os.path.join(data_root, subdir)
        base_name = os.path.splitext(pkl_path)[0]
        if base_name.endswith(".pkl"):
            base_name = base_name[:-4]

        candidate_pkl = os.path.join(subdir_path, f"{base_name}.pkl")
        candidate_gz = os.path.join(subdir_path, f"{base_name}.pkl.gz")
        if os.path.exists(candidate_pkl):
            full_pkl_path = candidate_pkl
        elif os.path.exists(candidate_gz):
            full_pkl_path = candidate_gz
        else:
            print(f"Warning: missing pkl for {subdir}/{pkl_path}")
            total_images += 1
            no_bbox_images.append(f"{subdir}/{pkl_path} - pkl_not_found")
            continue

        try:
            data = load_pickle_file(full_pkl_path)
            total_images += 1

            extraction_success = data.get("extraction_success", True)
            metadata = data.get("metadata", {})
            extracted_boxes = metadata.get("extracted_boxes", [])
            gt_boxes = metadata.get("gt_boxes_xyxy", metadata.get("gt_boxes", []))

            if not extraction_success or not extracted_boxes:
                no_bbox_images.append(f"{subdir}/{pkl_path} - extraction_failed or no_predictions")
                continue

            if not gt_boxes:
                total_images -= 1
                continue

            pred_box = extracted_boxes[bbox_id - 1]

            # RefCOCO-style processing: InternVL outputs bbox in [0, 1000] range
            token_grid = data.get("token_grid", {})
            original_image_size = token_grid.get("original_image_size", (0, 0))
            if original_image_size == (0, 0):
                missing_size_info.append(f"{subdir}/{pkl_path} - missing original_image_size")
                no_bbox_images.append(f"{subdir}/{pkl_path} - missing size")
                continue

            original_w, original_h = original_image_size
            # Normalize [0,1000] → [0,1] if values are large enough
            if sum(pred_box) >= 4:
                normalized = [x / 1000.0 for x in pred_box]
            else:
                normalized = pred_box
            final_pred_box = [
                normalized[0] * original_w,
                normalized[1] * original_h,
                normalized[2] * original_w,
                normalized[3] * original_h,
            ]
            mapped_images += 1
            bbox_mapping_log.append({
                "image": f"{subdir}/{pkl_path}",
                "bbox_id": bbox_id,
                "original_bbox": pred_box,
                "normalized_bbox": normalized,
                "final_bbox": final_pred_box,
                "original_size": original_image_size,
            })

            max_iou = calculate_max_iou_with_gt(final_pred_box, gt_boxes)
            total_iou += max_iou
            bbox_mapping_log[-1]["iou_after_processing"] = max_iou

            if max_iou > 0.5:
                detected_images += 1
            else:
                no_bbox_images.append(f"{subdir}/{pkl_path} - low_iou_{max_iou:.3f}")
        except Exception as e:
            print(f"Error processing {subdir}/{pkl_path}: {e}")
            total_images += 1
            no_bbox_images.append(f"{subdir}/{pkl_path} - processing_error: {e}")

    # Denominator override (original notebook used 2225 for COCO single-object)
    if total_images_override is not None:
        total_images = total_images_override

    accuracy = detected_images / total_images if total_images > 0 else 0.0
    mean_iou = total_iou / total_images if total_images > 0 else 0.0
    mean_iou_before_mapping = total_iou_before_mapping / total_images if total_images > 0 else 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    with open(summary_file, "w") as f:
        f.write("Enhanced Model Evaluation Results (with BBox Mapping)\n")
        f.write("=" * 53 + "\n\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Detected images (IoU > 0.5): {detected_images}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Mean IoU (after mapping): {mean_iou:.4f}\n")
        f.write(f"Mean IoU (before mapping): {mean_iou_before_mapping:.4f}\n")
        f.write(f"IoU improvement from mapping: {mean_iou - mean_iou_before_mapping:.4f}\n\n")
        f.write("BBox Mapping Statistics:\n")
        f.write(f"  Images with bbox mapping applied: {mapped_images}\n")
        f.write(f"  Images missing size information: {len(missing_size_info)}\n")
        f.write(f"  Failed detections: {len(no_bbox_images)}\n")

    if bbox_mapping_log:
        mapping_log_file = os.path.join(output_dir, f"bbox_mapping_log_{timestamp}.csv")
        pd.DataFrame(bbox_mapping_log).to_csv(mapping_log_file, index=False)
        print(f"BBox mapping details: {mapping_log_file}")

    log_file = os.path.join(output_dir, f"no_bbox_log_{timestamp}.txt")
    with open(log_file, "w") as f:
        f.write(f"Images without successful bbox detection ({len(no_bbox_images)} total):\n")
        f.write("=" * 60 + "\n\n")
        for img in no_bbox_images:
            f.write(f"{img}\n")

    if missing_size_info:
        missing_info_file = os.path.join(output_dir, f"missing_size_info_{timestamp}.txt")
        with open(missing_info_file, "w") as f:
            f.write(f"Images missing size information ({len(missing_size_info)} total):\n")
            f.write("=" * 60 + "\n\n")
            for img in missing_size_info:
                f.write(f"{img}\n")

    print(f"\nEnhanced Evaluation Results:")
    print(f"  Total images: {total_images}")
    print(f"  Detected (IoU > 0.5): {detected_images}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Mean IoU (after mapping): {mean_iou:.4f}")
    print(f"  Mean IoU (before mapping): {mean_iou_before_mapping:.4f}")
    print(f"  Mapped pairs: {mapped_images}   missing size info: {len(missing_size_info)}   failed: {len(no_bbox_images)}")
    print(f"  Results saved to: {output_dir}")

    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "mean_iou_before_mapping": mean_iou_before_mapping,
    }


def resolve_paths(method, dataset, temperature, top_p, base_response, test_response):
    """Compute (data_root, selection_csv, output_dir) from method+config."""
    tag_params = f"temp_{temperature}_top_p_{top_p}"
    base_resp_tag = f"response_{base_response}"
    test_resp_tag = f"response_{test_response}"

    # data_root points to original model_response
    if method == "greedy":
        data_subdir = f"bbox_attention_greedy_response_1"
    elif method == "default":
        data_subdir = f"bbox_attention_default_response_1"
    else:
        data_subdir = f"bbox_attention_{tag_params}_{base_resp_tag}"
    data_root = f"{ORIG_DATA_ROOT}/{dataset}/testing/{data_subdir}"

    # selection_csv is within emnlp
    sel_dir = DETECTION_RESULTS_DIR / dataset / "response_selection"

    # output_dir is within emnlp
    out_dir = DETECTION_RESULTS_DIR / dataset / "results"

    if method == "regression":
        selection_csv = sel_dir / f"regression_{tag_params}_{test_resp_tag}.csv"
        output_dir = out_dir / f"regression_{tag_params}_{test_resp_tag}_evaluation_results"
    elif method == "greedy":
        selection_csv = sel_dir / "greedy_response_1.csv"
        output_dir = out_dir / "greedy_response_1_evaluation_results"
    elif method == "default":
        selection_csv = sel_dir / "default_response_1.csv"
        output_dir = out_dir / "default_response_1_evaluation_results"
    elif method == "entropy":
        selection_csv = sel_dir / f"entropy_{tag_params}_{base_resp_tag}.csv"
        output_dir = out_dir / f"entropy_{tag_params}_{base_resp_tag}_evaluation_results"
    elif method == "first_valid":
        selection_csv = sel_dir / f"first_valid_{tag_params}_{base_resp_tag}.csv"
        output_dir = out_dir / f"first_valid_{tag_params}_{base_resp_tag}_evaluation_results"
    elif method == "smallest":
        selection_csv = sel_dir / f"smallest_{tag_params}_{base_resp_tag}.csv"
        output_dir = out_dir / f"smallest_{tag_params}_{base_resp_tag}_evaluation_results"
    else:
        raise ValueError(f"Unknown method: {method}")

    return data_root, str(selection_csv), str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Single-object detection evaluation")
    parser.add_argument("--method", required=True,
                        choices=["regression", "greedy", "default", "entropy", "first_valid", "smallest"])
    parser.add_argument("--dataset", default="COCO", choices=["COCO", "Objects365"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--base_response", type=int, default=10,
                        help="Number of responses used when generating model_response")
    parser.add_argument("--test_response", type=int, default=10,
                        help="Number of responses used at test time (regression can sub-sample)")
    parser.add_argument("--total_images", type=int, default=2225,
                        help="Fixed denominator for accuracy (original notebook used 2225 for COCO single)")
    parser.add_argument("--data_root", default=None, help="Override data_root (optional)")
    parser.add_argument("--selection_csv", default=None, help="Override selection_csv (optional)")
    parser.add_argument("--output_dir", default=None, help="Override output_dir (optional)")
    args = parser.parse_args()

    data_root, selection_csv, output_dir = resolve_paths(
        args.method, args.dataset, args.temperature, args.top_p,
        args.base_response, args.test_response
    )
    if args.data_root: data_root = args.data_root
    if args.selection_csv: selection_csv = args.selection_csv
    if args.output_dir: output_dir = args.output_dir

    print(f"method={args.method} dataset={args.dataset}")
    print(f"data_root:     {data_root}")
    print(f"selection_csv: {selection_csv}")
    print(f"output_dir:    {output_dir}")

    if not os.path.exists(data_root):
        print(f"ERROR: data_root does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(selection_csv):
        print(f"ERROR: selection_csv does not exist: {selection_csv}", file=sys.stderr)
        sys.exit(1)

    evaluate_model_performance(data_root, selection_csv, output_dir,
                               total_images_override=args.total_images)


if __name__ == "__main__":
    main()
