"""Stage KITTI-360 GT as a Conv-BKI *training* tree (gt_as_predictions mode).

This is the training-data counterpart to stage_inputs.py. Where stage_inputs
builds an Option-A eval tree (CENet predictions -> SemKITTI 20-class), this
script builds a *native 9-class common* training tree where the per-scan
network "prediction" IS the ground-truth one-hot label. The Conv-BKI kernel
then learns the per-class spatial extent that best reproduces GT at the voxel
level — exactly the project lead's GT-based retraining design.

Key differences from stage_inputs.py:
  - num_classes = 9 (native common space), NOT SemKITTI 20-class. No
    learning_map / SemKITTI projection anywhere.
  - labels/<stem>.label  = raw KITTI-360 GT remapped to common (uint32 N x 1,
    values 0..8) via kitti360_to_common.
  - predictions_hard/<stem>.label = the SAME common GT. Written as a relative
    symlink to ../labels/<stem>.label to halve disk (the hard one-hot input
    and the target are byte-identical in GT-as-predictions mode).
  - No CENet softmax is read; no predictions_softmax/ is written.
  - Each KITTI-360 sequence is staged into its own sequences/<id>/ dir so the
    NeuralBKI loader can hold a multi-sequence train/val split.
  - Accumulates per-common-class point counts across all staged scans and
    writes class_counts.json (used to build inverse-log-frequency NLL weights
    at train time).

The pose / scan-pairing / calib machinery is reused verbatim from
stage_inputs.py (KITTI-360 poses, sparse scan IDs, Tr=identity).

MCD mode (--dataset mcd): same native-9-class GT-as-predictions tree, but the
source is MCD's tuhh/ntu sequences. Scans live in lidar_bin/data/, poses in
pose_inW.csv, GT in gt_labels_terrain/ (raw MCD labels 0..29, terrain at 29).
GT is remapped to common via mcd_to_common (terrain 29 -> common 8). Per-group
body->lidar extrinsics are auto-selected: tuhh -> mcd/tuhh/hhs_calib.yaml,
ntu -> mcd/ntu/atv_calib.yaml. Sequence ids are the full names (e.g.
tuhh_day_04, ntu_day_02); the group prefix selects the data path and calib.
The lidar->world transform is body_to_world @ inv(body_to_lidar), matching
stage_inputs.py / mcd_util.h.

Usage:
    # KITTI-360 (default dataset)
    python stage_train_gt.py \
        --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt \
        --sequences 0003 \
        [--data-root /media/.../datasets] [--every-nth 1]

    # MCD
    python stage_train_gt.py --dataset mcd \
        --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt_mcd \
        --sequences tuhh_day_04 ntu_day_02 \
        [--data-root /media/.../datasets] [--every-nth 1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

sys.path.insert(0, HERE)
import stage_inputs as SI  # noqa: E402  (pose loaders, scan_stem, pose_row_12, existing)
import label_mappings as LM  # noqa: E402

N_COMMON = 9


def kitti360_seq_dirname(seq_id: str) -> str:
    """'0003' -> '2013_05_28_drive_0003_sync'."""
    return f"2013_05_28_drive_{seq_id}_sync"


def build_raw_kitti360_to_common(common_cfg: dict, max_raw_label: int = 65536) -> np.ndarray:
    """LUT: raw KITTI-360 label (after &0xFFFF) -> common class 0..8.

    Unlike stage_inputs.build_raw_gt_to_semkitti_train, this maps straight to
    the 9-class common space (no SemKITTI hop), since the trained model is
    native 9-class. Labels with no entry collapse to 0 (unlabeled/ignore).
    """
    src_to_common = {int(k): int(v) for k, v in common_cfg["kitti360_to_common"].items()}
    lut = np.zeros(max_raw_label, dtype=np.uint32)
    for raw, common in src_to_common.items():
        if 0 <= raw < max_raw_label:
            lut[raw] = common
    return lut


def stage_one_sequence(
    seq_id: str,
    data_root: str,
    staging_root: str,
    raw_to_common_lut: np.ndarray,
    every_nth: int = 1,
) -> dict:
    """Stage a single KITTI-360 sequence into sequences/<seq_id>/.

    Returns a per-sequence report dict including the per-common-class point
    counts contributed by this sequence.
    """
    seq_dir = kitti360_seq_dirname(seq_id)
    seq_root = os.path.join(data_root, "kitti360", seq_dir)
    scan_dir = os.path.join(seq_root, "velodyne_points", "data")
    pose_path = os.path.join(seq_root, "velodyne_poses.txt")
    gt_dir = os.path.join(seq_root, "gt_labels")

    for p in (scan_dir, pose_path, gt_dir):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing input for seq {seq_id}: {p}")

    out_seq = os.path.join(staging_root, "sequences", seq_id)
    out_velo = os.path.join(out_seq, "velodyne")
    out_lbl = os.path.join(out_seq, "labels")
    out_hard = os.path.join(out_seq, "predictions_hard")
    for d in (out_velo, out_lbl, out_hard):
        os.makedirs(d, exist_ok=True)

    scan_indices, raw_poses = SI.load_poses_kitti360(pose_path)
    normalized = SI.normalize_poses_kitti360(raw_poses)

    class_counts = np.zeros(N_COMMON, dtype=np.int64)
    pose_rows: list[np.ndarray] = []
    staged_stems: list[str] = []
    skipped: list[dict] = []

    for i, scan_idx in enumerate(scan_indices):
        # Optional subsampling (every Nth *staged-eligible* scan).
        if every_nth > 1 and (i % every_nth != 0):
            continue
        stem = SI.scan_stem(scan_idx)
        in_scan = os.path.join(scan_dir, f"{stem}.bin")
        in_gt = os.path.join(gt_dir, f"{stem}.bin")
        if not SI.existing(in_scan, in_gt):
            skipped.append(
                {"stem": stem, "scan_exists": os.path.isfile(in_scan),
                 "gt_exists": os.path.isfile(in_gt)}
            )
            continue

        n_pts = os.path.getsize(in_scan) // (4 * 4)  # float32 * 4 channels

        # GT -> common (uint32 N x 1, values 0..8). Mask 0xFFFF like the C++.
        raw = np.fromfile(in_gt, dtype=np.uint32) & 0xFFFF
        if raw.size != n_pts:
            skipped.append({"stem": stem, "reason": f"gt_size={raw.size} vs scan_pts={n_pts}"})
            continue
        common_gt = raw_to_common_lut[raw].astype(np.uint32)

        # Symlink scan
        out_scan = os.path.join(out_velo, f"{stem}.bin")
        if os.path.islink(out_scan) or os.path.exists(out_scan):
            os.remove(out_scan)
        os.symlink(os.path.abspath(in_scan), out_scan)

        # labels/<stem>.label  (target == hard input source)
        lbl_path = os.path.join(out_lbl, f"{stem}.label")
        common_gt.tofile(lbl_path)

        # predictions_hard/<stem>.label -> ../labels/<stem>.label (relative symlink).
        # In GT-as-predictions mode the one-hot input and the target are the
        # same bytes, so we symlink instead of duplicating ~0.5 MB/scan.
        hard_path = os.path.join(out_hard, f"{stem}.label")
        if os.path.islink(hard_path) or os.path.exists(hard_path):
            os.remove(hard_path)
        os.symlink(os.path.join("..", "labels", f"{stem}.label"), hard_path)

        # Pose row (lidar->world; KITTI-360 pose row already is lidar->world).
        T_lw = SI.lidar_to_world_kitti360(normalized[i])
        pose_rows.append(SI.pose_row_12(T_lw))
        staged_stems.append(stem)

        binc = np.bincount(common_gt, minlength=N_COMMON)
        class_counts += binc[:N_COMMON].astype(np.int64)

    # Sort by stem so file order matches sorted(glob('velodyne/*.bin')).
    order = sorted(range(len(staged_stems)), key=lambda j: staged_stems[j])
    pose_rows = [pose_rows[j] for j in order]
    staged_stems = [staged_stems[j] for j in order]

    if pose_rows:
        np.savetxt(os.path.join(out_seq, "poses.txt"), np.asarray(pose_rows), fmt="%.10g")

    # calib.txt: Tr = identity (Conv-BKI's inv(Tr)*pose*Tr is then a no-op).
    with open(os.path.join(out_seq, "calib.txt"), "w") as f:
        ident_row = "1 0 0 0 0 1 0 0 0 0 1 0"
        for p in ("P0", "P1", "P2", "P3", "Tr"):
            f.write(f"{p}: {ident_row}\n")

    return {
        "seq_id": seq_id,
        "seq_dir": seq_dir,
        "n_staged": len(staged_stems),
        "n_skipped": len(skipped),
        "class_counts": class_counts.tolist(),
        "skipped": skipped,
    }


# --------------------------------------------------------------------------- #
# MCD staging (tuhh / ntu)
# --------------------------------------------------------------------------- #

# Per-group body->lidar extrinsics. tuhh and kth share the hhs platform; ntu
# uses the atv platform. kth is eval-only so not stageable for training here.
MCD_GROUP_CALIB = {"tuhh": "hhs_calib.yaml", "ntu": "atv_calib.yaml"}


def mcd_group_of(seq_name: str) -> str:
    """'tuhh_day_04' -> 'tuhh'."""
    return seq_name.split("_", 1)[0]


def mcd_seq_paths(seq_name: str, data_root: str) -> dict:
    """Resolve the on-disk paths + calib for one MCD sequence name."""
    group = mcd_group_of(seq_name)
    if group not in MCD_GROUP_CALIB:
        raise ValueError(
            f"MCD seq '{seq_name}': unknown group '{group}' "
            f"(expected one of {sorted(MCD_GROUP_CALIB)})"
        )
    group_root = os.path.join(data_root, "mcd", group)
    return {
        "group": group,
        "seq_root": os.path.join(group_root, seq_name),
        "scan_dir": os.path.join(group_root, seq_name, "lidar_bin", "data"),
        "pose_path": os.path.join(group_root, seq_name, "pose_inW.csv"),
        "gt_dir": os.path.join(group_root, seq_name, "gt_labels_terrain"),
        "calib_file": os.path.join(group_root, MCD_GROUP_CALIB[group]),
    }


def stage_one_sequence_mcd(
    seq_name: str,
    data_root: str,
    staging_root: str,
    raw_to_common_lut: np.ndarray,
    every_nth: int = 1,
) -> dict:
    """Stage a single MCD sequence into sequences/<seq_name>/ (native 9-class
    GT-as-predictions). Mirror of stage_one_sequence but with MCD scans, poses,
    per-group body->lidar calib, and gt_labels_terrain via mcd_to_common.
    """
    paths = mcd_seq_paths(seq_name, data_root)
    scan_dir, pose_path, gt_dir = paths["scan_dir"], paths["pose_path"], paths["gt_dir"]
    calib_file = paths["calib_file"]
    for pth in (scan_dir, pose_path, gt_dir, calib_file):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"missing input for seq {seq_name}: {pth}")

    out_seq = os.path.join(staging_root, "sequences", seq_name)
    out_velo = os.path.join(out_seq, "velodyne")
    out_lbl = os.path.join(out_seq, "labels")
    out_hard = os.path.join(out_seq, "predictions_hard")
    for d in (out_velo, out_lbl, out_hard):
        os.makedirs(d, exist_ok=True)

    scan_indices, raw_poses = SI.load_poses_mcd(pose_path)
    normalized = SI.normalize_poses_mcd(raw_poses)
    b2l = SI.load_body_to_lidar_mcd(calib_file)

    class_counts = np.zeros(N_COMMON, dtype=np.int64)
    pose_rows: list[np.ndarray] = []
    staged_stems: list[str] = []
    skipped: list[dict] = []

    for i, scan_idx in enumerate(scan_indices):
        if every_nth > 1 and (i % every_nth != 0):
            continue
        stem = SI.scan_stem(scan_idx)
        in_scan = os.path.join(scan_dir, f"{stem}.bin")
        in_gt = os.path.join(gt_dir, f"{stem}.bin")
        if not SI.existing(in_scan, in_gt):
            skipped.append(
                {"stem": stem, "scan_exists": os.path.isfile(in_scan),
                 "gt_exists": os.path.isfile(in_gt)}
            )
            continue

        n_pts = os.path.getsize(in_scan) // (4 * 4)  # float32 * 4 channels

        # MCD GT is direct uint32 labels 0..29 (no instance bits, no 0xFFFF mask).
        raw = np.fromfile(in_gt, dtype=np.uint32)
        if raw.size != n_pts:
            skipped.append({"stem": stem, "reason": f"gt_size={raw.size} vs scan_pts={n_pts}"})
            continue
        common_gt = raw_to_common_lut[raw].astype(np.uint32)

        out_scan = os.path.join(out_velo, f"{stem}.bin")
        if os.path.islink(out_scan) or os.path.exists(out_scan):
            os.remove(out_scan)
        os.symlink(os.path.abspath(in_scan), out_scan)

        lbl_path = os.path.join(out_lbl, f"{stem}.label")
        common_gt.tofile(lbl_path)

        hard_path = os.path.join(out_hard, f"{stem}.label")
        if os.path.islink(hard_path) or os.path.exists(hard_path):
            os.remove(hard_path)
        os.symlink(os.path.join("..", "labels", f"{stem}.label"), hard_path)

        # lidar->world = body_to_world @ inv(body_to_lidar)  (mcd_util.h query_scan).
        T_lw = SI.lidar_to_world_mcd(normalized[i], b2l)
        pose_rows.append(SI.pose_row_12(T_lw))
        staged_stems.append(stem)

        binc = np.bincount(common_gt, minlength=N_COMMON)
        class_counts += binc[:N_COMMON].astype(np.int64)

    order = sorted(range(len(staged_stems)), key=lambda j: staged_stems[j])
    pose_rows = [pose_rows[j] for j in order]
    staged_stems = [staged_stems[j] for j in order]

    if pose_rows:
        np.savetxt(os.path.join(out_seq, "poses.txt"), np.asarray(pose_rows), fmt="%.10g")

    with open(os.path.join(out_seq, "calib.txt"), "w") as f:
        ident_row = "1 0 0 0 0 1 0 0 0 0 1 0"
        for pname in ("P0", "P1", "P2", "P3", "Tr"):
            f.write(f"{pname}: {ident_row}\n")

    return {
        "seq_id": seq_name,
        "group": paths["group"],
        "calib_file": calib_file,
        "n_staged": len(staged_stems),
        "n_skipped": len(skipped),
        "class_counts": class_counts.tolist(),
        "skipped": skipped[:50],  # cap: MCD seqs start mid-sequence (leading scans lack GT)
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kitti360", "mcd"], default="kitti360",
                   help="Source dataset. kitti360 (4-digit seq ids) or mcd "
                        "(full names like tuhh_day_04, ntu_day_02).")
    p.add_argument("--staging-root", required=True,
                   help="Output staging tree (must live under /home/sandilya).")
    p.add_argument("--sequences", nargs="+", required=True,
                   help="Sequence ids: KITTI-360 4-digit (0002 0003 ...) or MCD "
                        "names (tuhh_day_04 ntu_day_02 ...).")
    p.add_argument("--data-root", default=None,
                   help="Dataset root containing kitti360/. Falls back to "
                        "OSM_BKI_DATA_ROOT.")
    p.add_argument("--every-nth", type=int, default=1,
                   help="Subsample: stage every Nth scan (1 = all). Logged in manifest.")
    args = p.parse_args()

    data_root = args.data_root or os.environ.get("OSM_BKI_DATA_ROOT", "")
    if not data_root:
        print("ERROR: data-root not set (pass --data-root or OSM_BKI_DATA_ROOT).")
        return 2

    staging_root = os.path.abspath(args.staging_root)
    if not staging_root.startswith("/home/sandilya/"):
        print(f"ERROR: staging-root must be under /home/sandilya/, got {staging_root}")
        return 2
    os.makedirs(staging_root, exist_ok=True)

    common_yaml = os.path.join(REPO_ROOT, "config", "datasets", "labels_common.yaml")
    common_cfg = LM.load_common_config(common_yaml)
    if args.dataset == "mcd":
        raw_to_common_lut = LM.build_raw_gt_to_common("mcd", common_cfg)
        stage_fn = stage_one_sequence_mcd
    else:
        raw_to_common_lut = build_raw_kitti360_to_common(common_cfg)
        stage_fn = stage_one_sequence

    manifest = {
        "mode": "gt_as_predictions",
        "dataset": args.dataset,
        "num_classes": N_COMMON,
        "data_root": data_root,
        "staging_root": staging_root,
        "every_nth": args.every_nth,
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sequences": [],
    }
    total_counts = np.zeros(N_COMMON, dtype=np.int64)
    total_staged = 0

    for seq_id in args.sequences:
        print(f"Staging seq {seq_id} ...")
        t0 = time.time()
        rep = stage_fn(seq_id, data_root, staging_root, raw_to_common_lut,
                       every_nth=args.every_nth)
        dt = time.time() - t0
        print(f"  seq {seq_id}: staged {rep['n_staged']}, skipped {rep['n_skipped']} "
              f"({dt:.1f}s)")
        manifest["sequences"].append(rep)
        total_counts += np.asarray(rep["class_counts"], dtype=np.int64)
        total_staged += rep["n_staged"]

    manifest["total_staged"] = total_staged
    manifest["total_class_counts"] = total_counts.tolist()

    with open(os.path.join(staging_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # class_counts.json: the canonical per-common-class point counts used to
    # build inverse-log-frequency NLL weights at train time.
    labels = common_cfg["labels"]
    class_counts_out = {
        "sequences_staged": list(args.sequences),
        "num_classes": N_COMMON,
        "class_names": {str(i): labels[i] for i in range(N_COMMON)},
        "class_counts": total_counts.tolist(),
        "total_points": int(total_counts.sum()),
    }
    with open(os.path.join(staging_root, "class_counts.json"), "w") as f:
        json.dump(class_counts_out, f, indent=2)

    print(f"\nTotal staged {total_staged} scans across {len(args.sequences)} sequences.")
    print(f"  staging_root = {staging_root}")
    print("  per-common-class point counts:")
    for i in range(N_COMMON):
        print(f"    {i} {labels[i]:<12} {total_counts[i]:>14,d}")
    return 0 if total_staged > 0 else 3


if __name__ == "__main__":
    sys.exit(main())
