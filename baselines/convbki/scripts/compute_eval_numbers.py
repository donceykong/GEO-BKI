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
from sklearn.metrics import confusion_matrix, jaccard_score

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
    gt_all, pred_all = [], []
    for fp in files:
        try:
            arr = np.loadtxt(fp, dtype=np.int32)
        except Exception as e:
            print(f"WARN: failed to parse {fp}: {e}")
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            continue
        gt_all.append(arr[:, 0])
        pred_all.append(arr[:, 1])
    if not gt_all:
        print("No usable .txt files parsed")
        return 2

    gt_all = np.concatenate(gt_all)
    pred_all = np.concatenate(pred_all)
    print(f"Total points: {len(gt_all)}")
    print(f"Unique GT classes:   {sorted(np.unique(gt_all).tolist())}")
    print(f"Unique pred classes: {sorted(np.unique(pred_all).tolist())}")

    mask = gt_all != 0
    gt_eval = gt_all[mask]
    pred_eval = pred_all[mask]
    print(f"Points after dropping unlabeled (class 0): {len(gt_eval)}")

    semantic_classes = np.arange(1, NUM_CLASSES)
    iou = jaccard_score(
        gt_eval, pred_eval, labels=semantic_classes, average=None, zero_division=0
    )

    acc = np.zeros(len(semantic_classes))
    prec = np.zeros(len(semantic_classes))
    gt_count = np.zeros(len(semantic_classes), dtype=np.int64)
    pred_count = np.zeros(len(semantic_classes), dtype=np.int64)
    for i, c in enumerate(semantic_classes):
        gm = gt_eval == c
        pm = pred_eval == c
        gt_count[i] = int(gm.sum())
        pred_count[i] = int(pm.sum())
        acc[i] = (pred_eval[gm] == c).sum() / max(int(gm.sum()), 1)
        prec[i] = (gt_eval[pm] == c).sum() / max(int(pm.sum()), 1)

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

    cm = confusion_matrix(gt_eval, pred_eval, labels=semantic_classes).astype(np.int64)

    out = {
        "eval_dir": os.path.abspath(args.eval_dir),
        "n_files": len(files),
        "n_points_total": int(len(gt_all)),
        "n_points_eval": int(len(gt_eval)),
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
        "unique_gt_classes": sorted(np.unique(gt_all).tolist()),
        "unique_pred_classes": sorted(np.unique(pred_all).tolist()),
    }
    out_path = args.out_json or os.path.join(args.eval_dir, "raw_numbers.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote raw numbers to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
