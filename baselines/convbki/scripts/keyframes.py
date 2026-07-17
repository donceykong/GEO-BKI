"""Pose-based keyframe derivation for Conv-BKI eval.

Keyframe definition (per the project lead): walk the scans of a sequence in
stem-sorted order and select a keyframe whenever the Euclidean translation
distance from the last selected keyframe reaches the spacing threshold
(default 5.0 m). The first scan is always a keyframe.

The pose translation is loaded exactly as stage_inputs.py loads it (so the
world-frame translation used here is the same one the staging pipeline uses):
  - kitti360: velodyne_poses.txt, lidar->world, translation = row[:3,3].
  - mcd:      pose_inW.csv, body->world, translation = (x,y,z).
Euclidean distances between poses are invariant to the first-pose
normalization stage_inputs applies (a rigid left-multiply preserves distances),
so we use the raw loaded translations directly.

CLI:
    python keyframes.py <experiment_config.yaml> [--spacing 5.0]
        [--dist-from last_kf|prev] [--compare 0]
        [--stems-out <file>]

Writes one stem per line to --stems-out (default:
<staging_root>/keyframes_<spacing>m.txt), and prints the count.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage_inputs as SI  # noqa: E402


def scan_stem(scan_index: int) -> str:
    return f"{scan_index:010d}"


def load_sequence_translations(
    cfg: dict, data_root: str
) -> tuple[list[str], np.ndarray]:
    """Return (stems, translations Nx3) in stem-sorted order for a config.

    Mirrors stage_inputs.py pose loading. translations[i] is the world-frame
    lidar (kitti360) / body (mcd) position for stems[i].
    """
    dataset = cfg["dataset"]
    sequence = cfg["sequence_name"]
    seq_root = os.path.join(data_root, dataset, sequence)
    pose_path = os.path.join(seq_root, cfg["lidar_pose_suffix"])

    if dataset == "kitti360":
        scan_indices, raw_poses = SI.load_poses_kitti360(pose_path)
    elif dataset == "mcd":
        scan_indices, raw_poses = SI.load_poses_mcd(pose_path)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    stems = [scan_stem(i) for i in scan_indices]
    trans = np.array([T[:3, 3] for T in raw_poses], dtype=np.float64)
    # Sort by stem so the walk matches the dataset's sorted(glob) order.
    order = np.argsort(stems)
    stems = [stems[i] for i in order]
    trans = trans[order]
    return stems, trans


def derive_keyframes(
    stems: list[str],
    trans: np.ndarray,
    spacing: float = 5.0,
    dist_from: str = "last_kf",
    strict_gt: bool = False,
    include_first: bool = True,
) -> list[str]:
    """Select keyframe stems by Euclidean pose spacing.

    dist_from="last_kf": distance measured from the last SELECTED keyframe
                         (cumulative-anchored, the lead's stated definition).
    dist_from="prev":    distance measured from the immediately previous scan
                         (path-length accumulation).
    strict_gt=False: select when distance >= spacing; True: strictly > spacing.
    include_first:   always select the first scan as a keyframe.
    """
    n = len(stems)
    if n == 0:
        return []
    ge = (lambda d: d > spacing) if strict_gt else (lambda d: d >= spacing)
    kf: list[str] = []
    if include_first:
        kf.append(stems[0])
        anchor = trans[0]
    else:
        anchor = None
    if dist_from == "prev":
        accum = 0.0
        prev = trans[0]
        start = 1
        for i in range(start, n):
            accum += float(np.linalg.norm(trans[i] - prev))
            prev = trans[i]
            if ge(accum):
                kf.append(stems[i])
                accum = 0.0
        return kf
    # dist_from == "last_kf"
    start = 1 if include_first else 0
    for i in range(start, n):
        if anchor is None:
            kf.append(stems[i])
            anchor = trans[i]
            continue
        d = float(np.linalg.norm(trans[i] - anchor))
        if ge(d):
            kf.append(stems[i])
            anchor = trans[i]
    return kf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--spacing", type=float, default=5.0)
    p.add_argument("--dist-from", choices=["last_kf", "prev"], default="last_kf")
    p.add_argument("--strict-gt", action="store_true")
    p.add_argument("--no-first", action="store_true")
    p.add_argument("--data-root", default=None)
    p.add_argument("--stems-out", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_root = args.data_root or cfg.get("data_root") or os.environ.get(
        "OSM_BKI_DATA_ROOT", ""
    )
    data_root = os.path.expandvars(data_root)
    if not data_root:
        print("ERROR: data_root not set")
        return 2

    stems, trans = load_sequence_translations(cfg, data_root)
    kf = derive_keyframes(
        stems, trans, spacing=args.spacing, dist_from=args.dist_from,
        strict_gt=args.strict_gt, include_first=not args.no_first,
    )
    out = args.stems_out or os.path.join(
        cfg["staging_root"], f"keyframes_{args.spacing:g}m.txt"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(kf) + "\n")
    print(f"{len(stems)} scans -> {len(kf)} keyframes @ {args.spacing}m "
          f"(dist_from={args.dist_from}, strict_gt={args.strict_gt}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
