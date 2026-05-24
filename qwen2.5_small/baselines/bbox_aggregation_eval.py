"""Per-coordinate bbox aggregation baselines (paper's MajVote + MeanBBox).

For each image:
  - Collect all candidate bboxes from all responses
  - Aggregate per coordinate (x1, y1, x2, y2) across candidates:
      * --agg basic: raw per-coord majority vote (= paper's MajVote)
                     ties -> first occurrence; invalid assemblies (x2<=x1
                     or y2<=y1) are counted as failures.
      * --agg mean : per-coord arithmetic mean         (= paper's MeanBBox)
  - Synthesize a bbox from the 4 aggregated values
  - Compare synthetic bbox vs GT (with model-specific coordinate processing)

Outputs:
  - selection CSV (with synthetic bbox columns, for record)
  - evaluation_summary_*.txt with Acc and Mean IoU
"""
import os, sys, gzip, pickle, glob, re, argparse
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np


def load_pkl(p):
    if p.endswith('.gz'):
        with gzip.open(p,'rb') as f: return pickle.load(f)
    with open(p,'rb') as f: return pickle.load(f)


def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2-x1); ih = max(0, y2-y1)
    inter = iw * ih
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def collect_per_folder(image_folder_path):
    """Return list of (response_id, bbox) and metadata."""
    cands = []
    metadata = None
    for f in sorted(glob.glob(os.path.join(image_folder_path, "response_*.pkl*"))):
        m = re.search(r"response_(\d+)", os.path.basename(f))
        if not m: continue
        try:
            d = load_pkl(f)
        except Exception:
            continue
        if not d.get('extraction_success'): continue
        if metadata is None:
            metadata = {
                'gt_boxes': d['metadata'].get('gt_boxes', []),
                'gt_boxes_xyxy': d['metadata'].get('gt_boxes_xyxy'),
                'token_grid': d.get('token_grid', {}),
            }
        for bk, bv in d.get('bboxes', {}).items():
            v = bv.get('values')
            if v and len(v) == 4:
                # Coord-order check: skip degenerate bboxes (matches old qwen2.5vl is_valid_bbox)
                if v[2] <= v[0] or v[3] <= v[1]:
                    continue
                cands.append(v)
    return cands, metadata


def majority_bbox(candidates, agg='basic'):
    """Per-coordinate synthetic bbox.

    agg='basic': raw per-coord majority vote (ties -> first occurrence). If the
                 assembled bbox is invalid (x2<=x1 or y2<=y1), returns None
                 -> counted as a failure.    [paper: MajVote]
    agg='mean' : per-coord arithmetic mean (no snapping).                  [paper: MeanBBox]
    """
    if not candidates:
        return None
    arr = np.array(candidates)
    if agg == 'mean':
        return arr.mean(axis=0).tolist()
    # basic: per-coord majority vote
    out = []
    for c in range(4):
        col = arr[:, c].tolist()
        mode_val = Counter(col).most_common(1)[0][0]
        out.append(float(mode_val))
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def process_pred_for_iou(pred_bbox, original_image_size, processed_image_size, model_type):
    """Convert pred bbox → pixel xyxy for IoU comparison.
    model_type: 'qwen2.5vl' (processed→original mapping) or 'qwen3vl'/'internvl' (RefCOCO normalize × image size)
    """
    if model_type == 'qwen2.5vl':
        # processed_size → original_size scaling
        if (original_image_size != (0,0) and processed_image_size != (0,0)
                and original_image_size != processed_image_size):
            ow, oh = original_image_size; pw, ph = processed_image_size
            sx, sy = ow/pw, oh/ph
            return [pred_bbox[0]*sx, pred_bbox[1]*sy, pred_bbox[2]*sx, pred_bbox[3]*sy]
        return pred_bbox
    else:  # qwen3vl, internvl: RefCOCO style
        if original_image_size == (0, 0):
            return None
        ow, oh = original_image_size
        if sum(pred_bbox) >= 4:
            n = [x/1000.0 for x in pred_bbox]
        else:
            n = pred_bbox
        return [n[0]*ow, n[1]*oh, n[2]*ow, n[3]*oh]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model_type', choices=['qwen2.5vl', 'qwen3vl', 'internvl'], default='qwen3vl')
    p.add_argument('--agg', choices=['basic', 'mean'], default='basic',
                   help='basic = per-coord majority vote (paper: MajVote); '
                        'mean  = per-coord arithmetic mean (paper: MeanBBox)')
    p.add_argument('--total_images', type=int, default=2225)
    p.add_argument('--label', default='majority_vote')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_folders = sorted([d for d in os.listdir(args.data_root)
                            if os.path.isdir(os.path.join(args.data_root, d))])
    print(f'{len(image_folders)} folders under {args.data_root}')

    rows = []
    detected = 0; total_iou = 0.0; processed = 0
    failed = 0

    for ifolder in image_folders:
        cands, meta = collect_per_folder(os.path.join(args.data_root, ifolder))
        if not cands or meta is None:
            failed += 1
            continue
        synth = majority_bbox(cands, agg=args.agg)
        if synth is None:
            failed += 1
            continue

        # GT prep
        gt = meta.get('gt_boxes_xyxy') or meta.get('gt_boxes', [])
        if not gt:
            continue

        tg = meta['token_grid']
        proc_pred = process_pred_for_iou(
            synth, tg.get('original_image_size', (0,0)),
            tg.get('processed_image_size', (0,0)),
            args.model_type
        )
        if proc_pred is None:
            failed += 1
            continue

        max_iou = max(calc_iou(proc_pred, g) for g in gt) if gt else 0
        total_iou += max_iou
        if max_iou > 0.5:
            detected += 1
        processed += 1
        rows.append({
            'subdir': ifolder,
            'synthetic_bbox': str(synth),
            'processed_pred': str(proc_pred),
            'gt_boxes': str(gt),
            'iou': max_iou,
        })

    total = args.total_images
    acc = detected / total if total > 0 else 0
    mean_iou = total_iou / total if total > 0 else 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sel_csv = os.path.join(args.out_dir, f'{args.label}_selection_{timestamp}.csv')
    pd.DataFrame(rows).to_csv(sel_csv, index=False)
    summary = os.path.join(args.out_dir, f'evaluation_summary_{timestamp}.txt')
    with open(summary, 'w') as f:
        f.write(f'Per-coordinate Majority Vote ({args.label}) Evaluation\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Model type: {args.model_type}\n')
        f.write(f'Aggregation: {args.agg}\n')
        f.write(f'Total images: {total}\n')
        f.write(f'Detected images (IoU > 0.5): {detected}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
        f.write(f'Mean IoU (after mapping): {mean_iou:.4f}\n')
        f.write(f'Mean IoU (before mapping): 0.0000\n')
        f.write(f'Synthesized: {processed}, Failed: {failed}\n')
    print(f'\nMajority vote results saved to {summary}')
    print(f'Accuracy: {acc:.4f}, Mean IoU: {mean_iou:.4f} (synth={processed}, fail={failed})')


if __name__ == '__main__':
    main()
