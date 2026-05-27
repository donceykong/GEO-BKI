"""Aggregate the 8 per-combo raw_numbers.json into:

  baselines/convbki/results/raw_numbers.json       — full per-class numbers + CM
  baselines/convbki/results/in_domain.md           — Table-I-style ID rows
  baselines/convbki/results/cross_domain.md        — Table-II-style OOD rows

ID  = the dataset's own CENet  (KITTI-360 sequences with cenet_kitti360_softmax,
                                MCD with cenet_mcd_softmax).
OOD = the other dataset's CENet (KITTI-360 with cenet_mcd_softmax,
                                 MCD with cenet_kitti360_softmax).
"""

from __future__ import annotations

import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMBO_SEQ_TABLE = {
    "kitti360_seq0000_id":  ("kitti360", "2013_05_28_drive_0000_sync"),
    "kitti360_seq0000_ood": ("kitti360", "2013_05_28_drive_0000_sync"),
    "kitti360_seq0009_id":  ("kitti360", "2013_05_28_drive_0009_sync"),
    "kitti360_seq0009_ood": ("kitti360", "2013_05_28_drive_0009_sync"),
    "mcd_kth_day_09_id":    ("mcd", "kth_day_09"),
    "mcd_kth_day_09_ood":   ("mcd", "kth_day_09"),
    "mcd_kth_night_05_id":  ("mcd", "kth_night_05"),
    "mcd_kth_night_05_ood": ("mcd", "kth_night_05"),
}

CLASS_NAMES = ["road", "sidewalk", "parking", "building", "fence", "vegetation", "vehicle", "terrain"]

ID_COMBOS = ["kitti360_seq0000_id", "kitti360_seq0009_id", "mcd_kth_day_09_id", "mcd_kth_night_05_id"]
OOD_COMBOS = ["kitti360_seq0000_ood", "kitti360_seq0009_ood", "mcd_kth_day_09_ood", "mcd_kth_night_05_ood"]


def find_raw_numbers(combo: str, data_root: str) -> str | None:
    # Prefer the per-combo snapshot written by run_all.sh, since the
    # shared per-sequence eval_dir gets overwritten when the paired
    # ID/OOD combo runs.
    snap = os.path.join(REPO, "baselines", "convbki", "results", "per_combo", f"{combo}.json")
    if os.path.isfile(snap):
        return snap
    ds, seq = COMBO_SEQ_TABLE[combo]
    p = os.path.join(data_root, ds, seq, "evaluations", "convbki", "raw_numbers.json")
    return p if os.path.isfile(p) else None


def render_table(combos: list[str], all_data: dict, title: str) -> str:
    lines = [f"# {title}\n"]
    lines.append("Per-class IoU (and overall mIoU / mAcc) on the keyframes "
                 "S-BKI was evaluated on, common 9-class taxonomy with "
                 "class 0 (unlabeled) excluded from the average.\n")
    header = ["sequence"] + CLASS_NAMES + ["mIoU", "mAcc", "n_files", "n_points_eval"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for combo in combos:
        d = all_data.get(combo)
        if not d:
            lines.append(f"| {combo} | " + " | ".join(["-"] * (len(header) - 1)) + " |")
            continue
        iou = d["per_class_iou"]
        row = [combo]
        for v in iou:
            row.append(f"{v:.4f}")
        row.append(f"{d['miou']:.4f}")
        row.append(f"{d['macc']:.4f}")
        row.append(str(d["n_files"]))
        row.append(str(d["n_points_eval"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    data_root = os.environ.get(
        "OSM_BKI_DATA_ROOT",
        "/media/sgarimella34/hercules-collect3/datasets",
    )
    results_dir = os.path.join(REPO, "baselines", "convbki", "results")
    os.makedirs(results_dir, exist_ok=True)

    all_data: dict = {}
    for combo in COMBO_SEQ_TABLE:
        p = find_raw_numbers(combo, data_root)
        if p is None:
            print(f"missing: {combo}")
            continue
        with open(p) as f:
            all_data[combo] = json.load(f)
        print(f"loaded: {combo}  mIoU={all_data[combo]['miou']:.4f}")

    if not all_data:
        print("no combo data found; nothing to aggregate")
        return 2

    # Dump consolidated JSON
    with open(os.path.join(results_dir, "raw_numbers.json"), "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"wrote {results_dir}/raw_numbers.json")

    with open(os.path.join(results_dir, "in_domain.md"), "w") as f:
        f.write(render_table(ID_COMBOS, all_data, "Conv-BKI: in-domain (ID CENet)"))
    print(f"wrote {results_dir}/in_domain.md")

    with open(os.path.join(results_dir, "cross_domain.md"), "w") as f:
        f.write(render_table(OOD_COMBOS, all_data, "Conv-BKI: cross-domain (OOD CENet)"))
    print(f"wrote {results_dir}/cross_domain.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
