"""Convert Conv-BKI per-frame predictions into the eval-notebook .txt format.

Pipeline for each scan:
  1. Read Conv-BKI's per-point prediction file (uint32 SemKITTI training
     class, 0..19), produced by run_convbki.py.
  2. Map each prediction: SemKITTI training class -> raw SemKITTI label
     (via NeuralBKI Config/semantic_kitti.yaml's learning_map_inv) ->
     common 9-class index (via semkitti_to_common in labels_common.yaml).
  3. Read the original raw GT file, map raw -> common via the dataset's
     <gt_labels_key>_to_common.
  4. Write <gt_common> <pred_common> per point to
     <data_root>/<dataset>/<sequence>/evaluations/convbki/<stem>.txt.
  5. Optionally copy the raw .label file to
     baselines/convbki/raw_predictions/<dataset>/<sequence>/<stem>.label
     for archival (matches the artifact list in the build plan).

Keyframe gating: if --keyframe-stems-from <dir> is supplied, only stems
that have a sibling .txt in that directory (e.g. an existing S-BKI run)
are converted, so Conv-BKI is evaluated on the exact same keyframe set
as S-BKI.

Usage:
    python convert_outputs.py <experiment_config.yaml> [--keyframe-stems-from DIR] [--copy-raw]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import label_mappings as LM  # noqa: E402


def build_train_to_common_lut(train_to_common: dict[int, int]) -> np.ndarray:
    """LUT for the 20 SemKITTI training classes -> 9-class common."""
    lut = np.zeros(256, dtype=np.uint32)
    for t, c in train_to_common.items():
        if 0 <= t < lut.size:
            lut[t] = c
    return lut


def build_raw_gt_to_common_lut(
    gt_labels_key: str, common_cfg: dict, max_raw_label: int = 65536
) -> np.ndarray:
    """LUT for raw GT labels -> 9-class common."""
    map_key = gt_labels_key + "_to_common"
    if map_key not in common_cfg:
        raise KeyError(f"labels_common.yaml has no '{map_key}'")
    src_to_common = {int(k): int(v) for k, v in common_cfg[map_key].items()}
    lut = np.zeros(max_raw_label, dtype=np.uint32)
    for raw, common in src_to_common.items():
        if 0 <= raw < max_raw_label:
            lut[raw] = common
    return lut


def find_keyframe_stems(keyframe_dir: str) -> set[str]:
    if not keyframe_dir or not os.path.isdir(keyframe_dir):
        return set()
    return {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(keyframe_dir, "*.txt"))
    }


def convert(
    config_path: str,
    keyframe_stems_from: str | None,
    copy_raw: bool,
    data_root_override: str | None = None,
    eval_dir_override: str | None = None,
) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve data_root the same way stage_inputs.py does.
    if data_root_override:
        cfg["data_root"] = data_root_override
    elif not cfg.get("data_root"):
        cfg["data_root"] = os.environ.get("OSM_BKI_DATA_ROOT", "")
    if not cfg.get("data_root"):
        print("ERROR: data_root not set (use --data-root or OSM_BKI_DATA_ROOT)")
        return 2

    for k in ("data_root", "staging_root", "output_root", "nbki_root"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.expandvars(cfg[k])

    dataset = cfg["dataset"]
    sequence = cfg["sequence_name"]
    data_root = cfg["data_root"]
    output_root = os.path.abspath(cfg["output_root"])
    nbki_root = cfg.get(
        "nbki_root", os.path.join(REPO_ROOT, "baselines", "convbki", "NeuralBKI")
    )
    common_yaml = cfg.get(
        "labels_common_yaml",
        os.path.join(REPO_ROOT, "config", "datasets", "labels_common.yaml"),
    )
    gt_key = cfg["gt_labels_key"]
    gt_label_suffix = cfg["gt_label_suffix"]

    # native_common: Conv-BKI predictions are already 9-class common (0..8),
    # so map pred -> common is identity. Otherwise project SemKITTI 20-class
    # training -> common (Option A). Default False preserves Option A.
    native_common = bool(cfg.get("native_common", False))

    # -- LUTs ------------------------------------------------------------- #
    common_cfg = LM.load_common_config(common_yaml)
    if native_common:
        pred_lut = np.arange(256, dtype=np.uint32)  # identity: pred is common 0..8
    else:
        LM.load_semkitti_config(nbki_root)
        train_to_common = LM.build_semkitti_train_to_common(common_cfg)
        pred_lut = build_train_to_common_lut(train_to_common)
    gt_lut = build_raw_gt_to_common_lut(gt_key, common_cfg)

    # -- Locate inputs / outputs ----------------------------------------- #
    pred_dir = os.path.join(output_root, "sequences", "00", "predictions")
    gt_dir = os.path.join(data_root, dataset, sequence, gt_label_suffix)
    # eval_dir defaults under data_root, but --eval-dir / cfg eval_dir can
    # redirect it (required when data_root is a read-only / off-limits drive,
    # e.g. the native-9-class retrained eval writes under /home/sandilya).
    eval_dir = eval_dir_override or cfg.get("eval_dir") or os.path.join(
        data_root, dataset, sequence, "evaluations", "convbki"
    )
    eval_dir = os.path.expandvars(eval_dir)
    raw_archive_dir = os.path.join(
        REPO_ROOT, "baselines", "convbki", "raw_predictions", dataset, sequence
    )

    if not os.path.isdir(pred_dir):
        print(f"ERROR: prediction dir not found: {pred_dir}")
        return 2
    if not os.path.isdir(gt_dir):
        print(f"ERROR: GT dir not found: {gt_dir}")
        return 2

    os.makedirs(eval_dir, exist_ok=True)
    if copy_raw:
        os.makedirs(raw_archive_dir, exist_ok=True)

    keyframe_stems = find_keyframe_stems(keyframe_stems_from) if keyframe_stems_from else None
    if keyframe_stems is not None:
        print(f"Gating output to {len(keyframe_stems)} keyframe stems from {keyframe_stems_from}")

    # -- Conversion loop ------------------------------------------------- #
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.label")))
    written = 0
    skipped_missing_gt = 0
    skipped_not_keyframe = 0
    skipped_size = 0

    for pred_path in pred_files:
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        if keyframe_stems is not None and stem not in keyframe_stems:
            skipped_not_keyframe += 1
            continue
        gt_path = os.path.join(gt_dir, f"{stem}.bin")
        if not os.path.isfile(gt_path):
            skipped_missing_gt += 1
            continue

        pred_train = np.fromfile(pred_path, dtype=np.uint32)
        pred_common = pred_lut[pred_train.astype(np.uint16)]  # bound to LUT size
        raw_gt = np.fromfile(gt_path, dtype=np.uint32) & 0xFFFF
        gt_common = gt_lut[raw_gt]

        if pred_common.size != gt_common.size:
            skipped_size += 1
            continue

        out_path = os.path.join(eval_dir, f"{stem}.txt")
        rows = np.column_stack([gt_common, pred_common]).astype(np.int32)
        np.savetxt(out_path, rows, fmt="%d %d")
        written += 1

        if copy_raw:
            shutil.copy2(pred_path, os.path.join(raw_archive_dir, f"{stem}.label"))

    print(f"Wrote {written} eval files to {eval_dir}")
    if skipped_missing_gt:
        print(f"  skipped {skipped_missing_gt} scans with no matching GT")
    if skipped_not_keyframe:
        print(f"  skipped {skipped_not_keyframe} non-keyframe scans")
    if skipped_size:
        print(f"  skipped {skipped_size} scans with pred/gt size mismatch")
    if copy_raw:
        print(f"  archived raw .label files to {raw_archive_dir}")
    return 0 if written > 0 else 3


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument(
        "--keyframe-stems-from",
        default=None,
        help="If set, only convert stems that have a sibling .txt in this "
        "directory (e.g. <seq>/evaluations/osm_bki/) to gate Conv-BKI to "
        "the same keyframe set S-BKI was evaluated on.",
    )
    p.add_argument(
        "--copy-raw",
        action="store_true",
        help="Also copy Conv-BKI's per-frame .label predictions to "
        "baselines/convbki/raw_predictions/<dataset>/<sequence>/.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Overrides data_root from the YAML.",
    )
    p.add_argument(
        "--eval-dir",
        default=None,
        help="Overrides the output eval dir (default <data_root>/<dataset>/"
        "<sequence>/evaluations/convbki). Use to keep output off a read-only "
        "or off-limits data drive.",
    )
    args = p.parse_args()
    sys.exit(
        convert(
            args.config,
            keyframe_stems_from=args.keyframe_stems_from,
            copy_raw=args.copy_raw,
            data_root_override=args.data_root,
            eval_dir_override=args.eval_dir,
        )
    )
