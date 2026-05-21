"""Parity check between Conv-BKI's and S-BKI's per-point world transform.

The HARD STOP gate before running training/eval requires that the points
entering Conv-BKI's label_points() match the points entering S-BKI's
query_scan() on the same scan, in the same world frame, within float
precision.

This script reads:
  1. The original on-disk pose (KITTI-360 velodyne_poses.txt or MCD
     pose_inW.csv with hhs_calib.yaml) at the chosen scan index.
  2. The pose row our staging wrote for the same scan into
     <staging_root>/sequences/00/poses.txt.
  3. The raw lidar scan.

It then computes two transforms:
  - "S-BKI transform"   : body_to_world * inv(body_to_lidar)   [MCD]
                       or  pose - first_translation             [KITTI-360]
  - "Conv-BKI transform": the staged pose row (3x4, with Tr=I in
     calib.txt so Conv-BKI's inv(Tr) @ pose @ Tr = pose).

The two transforms must agree, and the world-frame points they produce
must agree within float precision. The script reports max abs delta on
both the transform matrices and on world-frame point coordinates.

Usage:
    python parity_check.py <experiment_config.yaml> --stem <10-digit-scan-id>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stage_inputs import (  # noqa: E402
    load_poses_kitti360,
    normalize_poses_kitti360,
    load_poses_mcd,
    normalize_poses_mcd,
    load_body_to_lidar_mcd,
    lidar_to_world_kitti360,
    lidar_to_world_mcd,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--stem", required=True, help="Scan stem to compare, e.g. 0000000123")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root
    elif not cfg.get("data_root"):
        cfg["data_root"] = os.environ.get("OSM_BKI_DATA_ROOT", "")
    for k in ("data_root", "calibration_file", "staging_root"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.expandvars(cfg[k])

    dataset = cfg["dataset"]
    sequence = cfg["sequence_name"]
    data_root = cfg["data_root"]
    seq_root = os.path.join(data_root, dataset, sequence)
    scan_dir = os.path.join(seq_root, cfg["input_data_suffix"])
    pose_path = os.path.join(seq_root, cfg["lidar_pose_suffix"])
    staging_root = cfg["staging_root"]
    staged_poses = os.path.join(staging_root, "sequences", "00", "poses.txt")

    print(f"Parity check on scan stem {args.stem}")
    print(f"  raw pose path  = {pose_path}")
    print(f"  staged poses   = {staged_poses}")
    print(f"  raw scan path  = {scan_dir}/{args.stem}.bin")

    # ------------------------------------------------------------------ #
    # Recompute S-BKI's lidar-to-world from raw inputs (no staging).
    # ------------------------------------------------------------------ #
    if dataset == "kitti360":
        scan_indices, raw_poses = load_poses_kitti360(pose_path)
        normalized = normalize_poses_kitti360(raw_poses)
        b2l = None
    elif dataset == "mcd":
        scan_indices, raw_poses = load_poses_mcd(pose_path)
        normalized = normalize_poses_mcd(raw_poses)
        b2l = load_body_to_lidar_mcd(cfg["calibration_file"])
    else:
        print(f"unknown dataset {dataset!r}")
        return 2

    target_scan_idx = int(args.stem)
    try:
        i = scan_indices.index(target_scan_idx)
    except ValueError:
        print(f"scan index {target_scan_idx} not in pose file")
        return 2

    if dataset == "kitti360":
        T_sbki = lidar_to_world_kitti360(normalized[i])
    else:
        T_sbki = lidar_to_world_mcd(normalized[i], b2l)

    # ------------------------------------------------------------------ #
    # Read the row our staging wrote for the same stem.
    # ------------------------------------------------------------------ #
    import json
    with open(os.path.join(staging_root, "manifest.json")) as f:
        manifest = json.load(f)
    staged_order = manifest["staged_order"]
    try:
        row_idx = staged_order.index(args.stem)
    except ValueError:
        print(f"stem {args.stem} not in staged_order (was it skipped?)")
        return 2

    rows = np.loadtxt(staged_poses)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    T_convbki = np.eye(4)
    T_convbki[:3, :4] = rows[row_idx].reshape(3, 4)

    # ------------------------------------------------------------------ #
    # Compare matrices
    # ------------------------------------------------------------------ #
    d_mat = np.abs(T_sbki - T_convbki)
    print(f"\nTransform matrix delta (S-BKI - Conv-BKI):")
    print(f"  max abs delta = {d_mat.max():.6e}")
    print(f"  per-row max:")
    for r in range(4):
        print(f"    row {r}: {d_mat[r].max():.6e}")

    # ------------------------------------------------------------------ #
    # Compare world-frame points
    # ------------------------------------------------------------------ #
    scan_file = os.path.join(scan_dir, f"{args.stem}.bin")
    pts = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    hom = np.column_stack([pts, np.ones(len(pts))])

    w_sbki = (T_sbki @ hom.T).T[:, :3]
    w_conv = (T_convbki @ hom.T).T[:, :3]
    d_pts = np.abs(w_sbki - w_conv)
    max_abs = d_pts.max()
    p99 = np.percentile(d_pts, 99)
    p999 = np.percentile(d_pts, 99.9)
    print(f"\nWorld-frame point delta (N={len(pts)}):")
    print(f"  max abs = {max_abs:.6e}")
    print(f"  p99     = {p99:.6e}")
    print(f"  p99.9   = {p999:.6e}")
    print(f"  mean    = {d_pts.mean():.6e}")

    tol = 1e-9
    if max_abs <= tol:
        print(f"\nPARITY OK: max delta {max_abs:.2e} <= tol {tol:.0e}")
        return 0
    print(f"\nPARITY FAIL: max delta {max_abs:.2e} > tol {tol:.0e}")
    return 4


if __name__ == "__main__":
    sys.exit(main())
