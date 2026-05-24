"""Selection-style baselines for bbox selection from raw model_response/*.pkl.gz.

Each method picks one (response, bbox) per image; results written as selection CSV
with columns subdir, pkl_path, bbox_id (compatible with evaluate_single_object.py).

Implements (paper's MainTable baselines):
  - first_valid:   first response with a valid bbox (coord-order valid)  → "FirstValid"
  - smallest:      candidate with minimum area                            → "Smallest"
  - vocab_entropy: candidate with lowest summed bbox-token vocab entropy  → "TokEntropy"

For synthetic-bbox baselines (MajVote = basic, MeanBBox = mean),
see bbox_aggregation_eval.py.
"""
import os, gzip, pickle, glob, argparse, re
import pandas as pd
import numpy as np
from pathlib import Path


def load_pkl(p):
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as f: return pickle.load(f)
    with open(p, "rb") as f: return pickle.load(f)


def collect_candidates(image_folder_path):
    """Return list of (response_id, bbox_id, pred_bbox, vocab_entropy_sum)."""
    out = []
    for f in sorted(glob.glob(os.path.join(image_folder_path, "response_*.pkl*"))):
        m = re.search(r"response_(\d+)\.pkl(?:\.gz)?$", os.path.basename(f))
        if not m: continue
        rid = int(m.group(1))
        try:
            data = load_pkl(f)
        except Exception:
            continue
        if not data.get("extraction_success"):
            continue
        bboxes = data.get("bboxes", {})
        for bk, bv in bboxes.items():
            bid_m = re.search(r"bbox_(\d+)", bk)
            bid = int(bid_m.group(1)) if bid_m else 1
            pred_bbox = bv.get("values")
            if not pred_bbox or len(pred_bbox) != 4:
                continue
            # Coord-order check: skip degenerate bboxes (matches old qwen2.5vl notebook is_valid_bbox)
            if pred_bbox[2] <= pred_bbox[0] or pred_bbox[3] <= pred_bbox[1]:
                continue
            ent_sum = 0.0; ent_count = 0
            for c in ["x1", "y1", "x2", "y2"]:
                e = bv.get(f"{c}_entropy")
                if e is not None and len(e) > 0:
                    ent_sum += float(np.mean(e))
                    ent_count += 1
            out.append({
                "response_id": rid, "bbox_id": bid, "pred_bbox": pred_bbox,
                "vocab_entropy_sum": ent_sum if ent_count > 0 else float("inf"),
            })
    return out


def select_first_valid(cands):
    cands = sorted(cands, key=lambda x: (x["response_id"], x["bbox_id"]))
    return cands[0] if cands else None


def select_smallest(cands):
    if not cands: return None
    def area(c):
        b = c["pred_bbox"]; return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return min(cands, key=area)


def select_min_vocab_entropy(cands):
    if not cands: return None
    return min(cands, key=lambda x: x["vocab_entropy_sum"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="model_response/.../bbox_attention_*_response_N dir")
    p.add_argument("--out_dir", required=True, help="output dir for selection CSVs")
    p.add_argument("--methods", nargs="+",
                   default=["first_valid", "smallest", "vocab_entropy"])
    p.add_argument("--tag", default="t1.0_top_p_0.9_response_10",
                   help="appended to output CSV filename")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_folders = sorted([d for d in os.listdir(args.data_root)
                            if os.path.isdir(os.path.join(args.data_root, d))])
    print(f"Found {len(image_folders)} image folders under {args.data_root}")

    method_results = {m: [] for m in args.methods}
    method_fns = {
        "first_valid": select_first_valid,
        "smallest": select_smallest,
        "vocab_entropy": select_min_vocab_entropy,
    }
    for i, ifolder in enumerate(image_folders):
        cands = collect_candidates(os.path.join(args.data_root, ifolder))
        for m in args.methods:
            sel = method_fns[m](cands)
            if sel is None:
                continue
            method_results[m].append({
                "subdir": ifolder,
                "pkl_path": f"response_{sel['response_id']}.pkl",
                "bbox_id": sel["bbox_id"],
            })
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(image_folders)} folders processed")

    # Write per-method CSVs
    for m, rows in method_results.items():
        out_csv = os.path.join(args.out_dir, f"{m}_{args.tag}.csv")
        pd.DataFrame(rows, columns=["subdir", "pkl_path", "bbox_id"]).to_csv(out_csv, index=False)
        print(f"  {m}: {len(rows)} rows → {out_csv}")


if __name__ == "__main__":
    main()
