"""Print per-class IoU / accuracy / precision / mIoU / mAcc / confusion matrix
for a directory of evaluation .txt files (each line: gt_common pred_common).

Replicates eval/osm_bki_eval.ipynb in a single-shot CLI so we can quickly
verify smoke-test outputs without spinning up Jupyter. Numbers are
emitted in the same shape the notebook prints, and a JSON dump (the
raw_numbers.json artifact from the build plan) is written alongside.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

COMMON_CLASSES = [
    "unlabeled",
    "road",
    "sidewalk",
    "parking",
    "building",
    "fence",
    "vegetation",
    "vehicle",
    "terrain",
]
NUM_CLASSES = len(COMMON_CLASSES)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("eval_dir", help="Directory of <stem>.txt files")
    p.add_argument(
        "--out-json",
        default=None,
        help="Path to dump raw_numbers.json (default: <eval_dir>/raw_numbers.json)",
    )
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.eval_dir, "*.txt")))
    if not files:
        print(f"No .txt files in {args.eval_dir}")
        return 2

    print(f"Reading {len(files)} files from {args.eval_dir}")
    # Streaming accumulation: a full NxN confusion matrix over ALL points
    # (rows=gt, cols=pred, both in 0..N-1) plus a presence mask of which
    # classes ever appear. This keeps memory O(N^2) regardless of point count
    # (the prior np.loadtxt + sklearn-on-all-points path OOM'd at ~454M pts),
    # and every reported metric is recovered exactly from the confusion matrix.
    full_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    seen_gt = np.zeros(NUM_CLASSES, dtype=bool)
    seen_pred = np.zeros(NUM_CLASSES, dtype=bool)
    n_total = 0
    n_parsed = 0
    for fp in files:
        try:
            with open(fp) as fh:
                toks = fh.read().split()
        except Exception as e:
            print(f"WARN: failed to read {fp}: {e}")
            continue
        if not toks:
            continue
        a = np.asarray(toks, dtype=np.int64).reshape(-1, 2)
        gt = a[:, 0]
        pred = a[:, 1]
        # Guard against any stray out-of-range label.
        in_range = (gt >= 0) & (gt < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
        gt = gt[in_range]
        pred = pred[in_range]
        if gt.size == 0:
            continue
        idx = gt * NUM_CLASSES + pred
        full_cm += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
        seen_gt[np.unique(gt)] = True
        seen_pred[np.unique(pred)] = True
        n_total += int(a.shape[0])
        n_parsed += 1
    if n_parsed == 0:
        print("No usable .txt files parsed")
        return 2

    print(f"Total points: {n_total}")
    print(f"Unique GT classes:   {sorted(np.flatnonzero(seen_gt).tolist())}")
    print(f"Unique pred classes: {sorted(np.flatnonzero(seen_pred).tolist())}")

    # Drop unlabeled (gt class 0): metrics are computed over gt != 0 points,
    # i.e. confusion-matrix rows 1..N-1 (pred columns still include 0).
    eval_cm = full_cm[1:, :]                  # rows = gt 1..N-1, cols = pred 0..N-1
    n_eval = int(eval_cm.sum())
    print(f"Points after dropping unlabeled (class 0): {n_eval}")

    semantic_classes = np.arange(1, NUM_CLASSES)
    inter = np.array([full_cm[c, c] for c in semantic_classes], dtype=np.int64)
    gt_count = np.array([eval_cm[i, :].sum() for i in range(len(semantic_classes))],
                        dtype=np.int64)                      # |gt == c|
    pred_count = np.array([eval_cm[:, c].sum() for c in semantic_classes],
                          dtype=np.int64)                    # |pred == c| over gt!=0
    union = gt_count + pred_count - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    acc = np.where(gt_count > 0, inter / np.maximum(gt_count, 1), 0.0)
    prec = np.where(pred_count > 0, inter / np.maximum(pred_count, 1), 0.0)

    present = gt_count > 0
    miou = float(np.mean(iou[present]))
    macc = float(np.mean(acc[present]))

    print()
    header = f"{'cls':>3} {'name':<11} {'IoU':>7} {'Acc':>7} {'Prec':>7} {'GT_cnt':>10} {'Pred_cnt':>10}"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(semantic_classes):
        print(
            f"{c:>3} {COMMON_CLASSES[c]:<11} {iou[i]:>7.4f} {acc[i]:>7.4f} "
            f"{prec[i]:>7.4f} {gt_count[i]:>10d} {pred_count[i]:>10d}"
        )
    print()
    print(f"mIoU (over classes present in GT) = {miou:.4f}")
    print(f"mAcc (over classes present in GT) = {macc:.4f}")

    # confusion matrix over the semantic classes (gt 1..N-1 x pred 1..N-1),
    # matching the prior labels=semantic_classes output (pred 0 excluded).
    cm = full_cm[1:NUM_CLASSES, 1:NUM_CLASSES].astype(np.int64)

    out = {
        "eval_dir": os.path.abspath(args.eval_dir),
        "n_files": len(files),
        "n_points_total": int(n_total),
        "n_points_eval": int(n_eval),
        "semantic_classes": [int(c) for c in semantic_classes],
        "class_names": [COMMON_CLASSES[c] for c in semantic_classes],
        "per_class_iou": [float(x) for x in iou],
        "per_class_accuracy": [float(x) for x in acc],
        "per_class_precision": [float(x) for x in prec],
        "gt_count": [int(x) for x in gt_count],
        "pred_count": [int(x) for x in pred_count],
        "miou": miou,
        "macc": macc,
        "confusion_matrix": cm.tolist(),
        "unique_gt_classes": sorted(np.flatnonzero(seen_gt).tolist()),
        "unique_pred_classes": sorted(np.flatnonzero(seen_pred).tolist()),
    }
    out_path = args.out_json or os.path.join(args.eval_dir, "raw_numbers.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote raw numbers to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
