"""Diff the new selection CSV against the original pipeline's selection CSV.

If logic is faithfully reproduced, the diff should be 0 rows.

Usage:
    python verify_reproduction.py [temp]
"""
import argparse
import os
import sys
import pandas as pd


ORIG_BASE = "/scratch/ty45972/MLLM_object_detection/object_detection_Qwen/regression_training/entropy_selection_results"
NEW_BASE = os.path.join(os.path.dirname(__file__), "entropy_selection_results", "COCO")


def diff_csvs(orig_path, new_path):
    if not os.path.exists(orig_path):
        sys.exit(f"ERROR: orig CSV missing: {orig_path}")
    if not os.path.exists(new_path):
        sys.exit(f"ERROR: new CSV missing: {new_path}")

    a = pd.read_csv(orig_path)
    b = pd.read_csv(new_path)
    a["pkl_path"] = a["pkl_path"].astype(str).str.replace(".gz", "", regex=False)
    b["pkl_path"] = b["pkl_path"].astype(str).str.replace(".gz", "", regex=False)

    print(f"orig rows: {len(a)}, new rows: {len(b)}")
    m = a.merge(b, on="subdir", how="outer", suffixes=("_orig", "_new"), indicator=True)
    print("merge:")
    print(m["_merge"].value_counts().to_string())

    both = m[m["_merge"] == "both"].copy()
    same = (both["pkl_path_orig"] == both["pkl_path_new"]) & (
        both["bbox_id_orig"] == both["bbox_id_new"]
    )
    print(f"\nOverlapping subdirs: {len(both)}")
    print(f"  Identical selection: {same.sum()}/{len(both)} ({100*same.mean():.2f}%)")
    diff = both[~same]
    print(f"  Differing selection: {len(diff)}")
    if len(diff):
        print(diff[["subdir", "pkl_path_orig", "bbox_id_orig", "pkl_path_new", "bbox_id_new"]].head(20).to_string(index=False))

    only_orig = m[m["_merge"] == "left_only"]
    only_new = m[m["_merge"] == "right_only"]
    if len(only_orig):
        print(f"\nIn orig only ({len(only_orig)}):")
        print(only_orig[["subdir"]].head(10).to_string(index=False))
    if len(only_new):
        print(f"\nIn new only ({len(only_new)}):")
        print(only_new[["subdir"]].head(10).to_string(index=False))

    n_mismatch = len(diff) + len(only_orig) + len(only_new)
    print("\n" + "=" * 60)
    if n_mismatch == 0:
        print("[OK] PERFECT REPRODUCTION (0 mismatches)")
    else:
        print(f"[WARN] Total mismatches: {n_mismatch}")
    print("=" * 60)
    return n_mismatch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("temp", nargs="?", default="0.5")
    args = p.parse_args()
    tdir = f"temp_{args.temp}_top_p_0.9_response_10"
    orig = os.path.join(ORIG_BASE, tdir, f"entropy_{tdir}_selection.csv")
    new = os.path.join(NEW_BASE, tdir, f"entropy_{tdir}_selection.csv")
    print(f"orig: {orig}\nnew : {new}\n")
    sys.exit(0 if diff_csvs(orig, new) == 0 else 1)


if __name__ == "__main__":
    main()
