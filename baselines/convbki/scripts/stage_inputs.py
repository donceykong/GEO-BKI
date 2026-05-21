"""Build a Conv-BKI staging tree for one (dataset, sequence, CENet variant) combo.

Per the Option A baseline plan:
  - Conv-BKI is run with its pretrained 20-class SemanticKITTI weights.
  - Input CENet softmax (in MCD or KITTI-360 channels) is projected to a
    20-class soft distribution by routing each source channel to a
    canonical SemKITTI training class via the common taxonomy.
  - GT is converted the same way so the labels/ files Conv-BKI's loader
    reads carry sensible class indices.
  - Poses are emitted as the lidar->world transform per scan, with
    calib.txt's Tr = identity so Conv-BKI's `inv(Tr) @ pose @ Tr` is a
    no-op and the pose row is taken directly as the global transform.

Staging tree layout:

    <staging_root>/sequences/00/
        velodyne/<stem>.bin              symlinks to original scans
        labels/<stem>.label              uint32, SemKITTI training-class GT
        predictions_softmax/<stem>.label float32 N x 20 SemKITTI soft probs
        poses.txt                        12 floats per scan, in stem order
        calib.txt                        Tr = identity
    <staging_root>/manifest.json         per-scan staging report
    <staging_root>/routing.json          channel -> training-class table

Usage:
    python stage_inputs.py <staging_config.yaml>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # OSM-BKI-ROS/

sys.path.insert(0, HERE)
import label_mappings as LM  # noqa: E402


# --------------------------------------------------------------------------- #
# Pose loading
# --------------------------------------------------------------------------- #


def load_poses_kitti360(pose_path: str) -> tuple[list[int], list[np.ndarray]]:
    """Read velodyne_poses.txt. Each line: frame_idx 12-or-16 floats (3x4 or 4x4 row-major).

    Returns (frame_indices, poses_4x4_lidar_to_world) BEFORE first-pose normalization.
    """
    frames: list[int] = []
    poses: list[np.ndarray] = []
    with open(pose_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            frame_idx = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            if len(vals) == 12:
                T = np.eye(4)
                T[:3, :4] = np.asarray(vals).reshape(3, 4)
            elif len(vals) == 16:
                T = np.asarray(vals).reshape(4, 4)
            else:
                continue
            frames.append(frame_idx)
            poses.append(T)
    return frames, poses


def normalize_poses_kitti360(poses: list[np.ndarray]) -> list[np.ndarray]:
    """Subtract the first pose's translation from all poses; rotations unchanged.

    Matches read_lidar_poses_kitti360() in include/osm_bki/dataset_devkit/mcd_util.h.
    """
    if not poses:
        return poses
    t0 = poses[0][:3, 3].copy()
    out = []
    for T in poses:
        T2 = T.copy()
        T2[:3, 3] = T2[:3, 3] - t0
        out.append(T2)
    return out


def load_poses_mcd(pose_path: str) -> tuple[list[int], list[np.ndarray]]:
    """Read MCD pose_inW.csv (header: num,timestamp,x,y,z,qx,qy,qz,qw).

    Returns (scan_indices, poses_4x4_body_to_world) BEFORE first-pose normalization.
    """
    frames: list[int] = []
    poses: list[np.ndarray] = []
    with open(pose_path) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Skip header line if first row has non-numeric first field
            if i == 0 and ("num" in s.lower() or "timestamp" in s.lower()):
                continue
            delim = "," if "," in s else None
            tokens = [t.strip() for t in (s.split(delim) if delim else s.split())]
            try:
                vals = [float(t) for t in tokens if t]
            except ValueError:
                continue
            if len(vals) < 8:
                continue
            scan_idx = int(vals[0])
            x, y, z = vals[2], vals[3], vals[4]
            qx, qy, qz = vals[5], vals[6], vals[7]
            qw = vals[8] if len(vals) > 8 else 1.0
            n = np.linalg.norm([qx, qy, qz, qw])
            if n < 1e-12:
                continue
            qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
            R = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            frames.append(scan_idx)
            poses.append(T)
    return frames, poses


def normalize_poses_mcd(poses: list[np.ndarray]) -> list[np.ndarray]:
    """Left-multiply by inv(first pose) so first pose = identity. Matches
    read_lidar_poses() in include/osm_bki/dataset_devkit/mcd_util.h."""
    if not poses:
        return poses
    inv0 = np.linalg.inv(poses[0])
    return [inv0 @ T for T in poses]


def load_body_to_lidar_mcd(yaml_path: str) -> np.ndarray:
    """Read body/os_sensor/T from hhs_calib.yaml; returns the 4x4 body_to_lidar."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    T = np.asarray(cfg["body"]["os_sensor"]["T"], dtype=np.float64)
    assert T.shape == (4, 4), f"body_to_lidar shape {T.shape}, expected (4,4)"
    return T


# --------------------------------------------------------------------------- #
# File enumeration
# --------------------------------------------------------------------------- #


def scan_stem(scan_index: int) -> str:
    """10-digit zero-padded scan id, matching KITTI-360 and MCD on-disk naming."""
    return f"{scan_index:010d}"


def existing(*paths: str) -> bool:
    return all(os.path.isfile(p) for p in paths)


# --------------------------------------------------------------------------- #
# Per-scan processing
# --------------------------------------------------------------------------- #


def convert_gt(
    gt_path: str,
    raw_to_train_lut: np.ndarray,
) -> np.ndarray:
    """Read uint32 raw GT and return uint32 SemKITTI training-class labels.

    Mirrors the C++ side's `& 0xFFFF` mask on raw labels before lookup, so
    KITTI-360's 65535 sentinel resolves to whatever 65535 maps to in
    kitti360_to_common (i.e. 0)."""
    raw = np.fromfile(gt_path, dtype=np.uint32) & 0xFFFF
    return raw_to_train_lut[raw].astype(np.uint32)


def convert_softmax(
    softmax_path: str,
    n_points: int,
    channel_to_common: np.ndarray,
    num_train_classes: int = 20,
) -> np.ndarray:
    """Read CENet uint16 (float16) softmax buffer and return (N, num_train_classes) float32.

    Pipeline: float16 logits -> softmax -> 9-class common (C++-faithful sum
    + average-by-count + renormalize) -> 20-class SemKITTI training
    distribution (weighted split per COMMON_TO_SEMKITTI_TRAIN_WEIGHTED).
    Buffer length must be n_points * K with K inferred from file size.
    """
    raw = np.fromfile(softmax_path, dtype=np.uint16)
    if raw.size == 0:
        raise ValueError(f"empty softmax file: {softmax_path}")
    if raw.size % n_points != 0:
        raise ValueError(
            f"softmax size {raw.size} not divisible by n_points={n_points} for {softmax_path}"
        )
    K = raw.size // n_points
    if channel_to_common.size != K:
        raise ValueError(
            f"channel_to_common has {channel_to_common.size} entries but softmax has {K} channels "
            f"({softmax_path}); rebuild the routing table with the correct n_channels"
        )
    logits = raw.reshape(n_points, K)
    softmax = LM.softmax_from_float16_raw(logits)
    return LM.aggregate_channels_to_semkitti_train(softmax, channel_to_common, num_train_classes)


def lidar_to_world_kitti360(
    normalized_pose: np.ndarray,
) -> np.ndarray:
    """For KITTI-360, the pose row in velodyne_poses.txt is already the
    lidar->world transform; body_to_lidar is identity."""
    return normalized_pose


def lidar_to_world_mcd(
    normalized_body_to_world: np.ndarray,
    body_to_lidar: np.ndarray,
) -> np.ndarray:
    """Matches query_scan() in mcd_util.h: lidar->world = body_to_world * inv(body_to_lidar)."""
    return normalized_body_to_world @ np.linalg.inv(body_to_lidar)


def pose_row_12(T_4x4: np.ndarray) -> np.ndarray:
    return T_4x4[:3, :4].astype(np.float64).reshape(-1)


# --------------------------------------------------------------------------- #
# Main staging routine
# --------------------------------------------------------------------------- #


def stage(config_path: str, data_root_override: str | None = None) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI / env override -> YAML field -> environment variable. The YAML may
    # leave data_root null/empty and require a CLI flag or env var, so a single
    # source-of-truth path covers all eight per-experiment configs.
    if data_root_override:
        cfg["data_root"] = data_root_override
    elif not cfg.get("data_root"):
        cfg["data_root"] = os.environ.get("OSM_BKI_DATA_ROOT", "")
    # Expand ${VAR} / $VAR in any path-like field after the override is applied.
    for k in ("data_root", "calibration_file", "staging_root", "nbki_root", "labels_common_yaml"):
        if isinstance(cfg.get(k), str):
            cfg[k] = os.path.expandvars(cfg[k])
    if not cfg.get("data_root"):
        print(
            "ERROR: data_root not set. Provide it via the YAML, --data-root, or "
            "the OSM_BKI_DATA_ROOT env var."
        )
        return 2

    dataset = cfg["dataset"]                    # "kitti360" or "mcd"
    sequence = cfg["sequence_name"]
    data_root = cfg["data_root"]
    input_data_suffix = cfg["input_data_suffix"]
    lidar_pose_suffix = cfg["lidar_pose_suffix"]
    input_label_suffix = cfg["input_label_suffix"]
    gt_label_suffix = cfg["gt_label_suffix"]
    inferred_key = cfg["inferred_labels_key"]   # "mcd" or "kitti360"
    gt_key = cfg["gt_labels_key"]
    staging_root = cfg["staging_root"]
    calib_file = cfg.get("calibration_file") or ""
    nbki_root = cfg.get(
        "nbki_root",
        os.path.join(REPO_ROOT, "baselines", "convbki", "NeuralBKI"),
    )
    common_yaml = cfg.get(
        "labels_common_yaml",
        os.path.join(REPO_ROOT, "config", "datasets", "labels_common.yaml"),
    )

    # ------------------------------------------------------------------ #
    # Path setup
    # ------------------------------------------------------------------ #
    seq_root = os.path.join(data_root, dataset, sequence)
    scan_dir = os.path.join(seq_root, input_data_suffix)
    pose_path = os.path.join(seq_root, lidar_pose_suffix)
    cenet_dir = os.path.join(seq_root, input_label_suffix)
    gt_dir = os.path.join(seq_root, gt_label_suffix)

    missing_inputs = [
        p for p in [scan_dir, pose_path, cenet_dir, gt_dir] if not os.path.exists(p)
    ]
    if missing_inputs:
        print("MISSING INPUTS — cannot stage:")
        for p in missing_inputs:
            print(f"  {p}")
        return 2

    out_seq = os.path.join(staging_root, "sequences", "00")
    out_velo = os.path.join(out_seq, "velodyne")
    out_lbl = os.path.join(out_seq, "labels")
    out_pred = os.path.join(out_seq, "predictions_softmax")
    out_hard = os.path.join(out_seq, "predictions_hard")
    # predictions_softmax is ~60GB per KITTI-360 seq and unused when the
    # experiment runs in hard-input mode; skip the write to keep disk
    # usage in check across all 8 combos.
    write_softmax = not bool(cfg.get("hard_input", False))
    dirs_to_make = [out_velo, out_lbl, out_hard]
    if write_softmax:
        dirs_to_make.append(out_pred)
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)

    # Canonical 1-to-1 routing for hard labels: common -> single SemKITTI training class.
    # No weighted split (the split was causing argmax to land on common classes whose
    # mass was split, never on vegetation/vehicle whose mass was halved/thirded).
    common_to_train_canon = np.zeros(9, dtype=np.uint32)
    for c, t in LM.COMMON_TO_SEMKITTI_TRAIN.items():
        common_to_train_canon[int(c)] = int(t)

    # ------------------------------------------------------------------ #
    # Label mappings
    # ------------------------------------------------------------------ #
    LM.load_semkitti_config(nbki_root)
    common_cfg = LM.load_common_config(common_yaml)
    LM.build_semkitti_train_to_common(common_cfg)
    raw_to_train_lut = LM.build_raw_gt_to_semkitti_train(gt_key, common_cfg)

    # Channel count differs per CENet variant; learn it from the first
    # softmax file we encounter. We carry the channel->common routing
    # rather than channel->training-class because the staging-time split
    # to 20 training columns is now a separate weighted step.
    channel_to_common: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Poses
    # ------------------------------------------------------------------ #
    if dataset == "kitti360":
        scan_indices, raw_poses = load_poses_kitti360(pose_path)
        normalized = normalize_poses_kitti360(raw_poses)
        b2l = None
    elif dataset == "mcd":
        scan_indices, raw_poses = load_poses_mcd(pose_path)
        normalized = normalize_poses_mcd(raw_poses)
        if not calib_file:
            print(f"ERROR: dataset=mcd requires calibration_file in config")
            return 2
        b2l = load_body_to_lidar_mcd(calib_file)
    else:
        print(f"ERROR: unknown dataset {dataset!r}")
        return 2

    print(f"Loaded {len(normalized)} poses from {pose_path}")
    if not normalized:
        return 2

    # ------------------------------------------------------------------ #
    # Per-scan staging
    # ------------------------------------------------------------------ #
    manifest: dict = {
        "config": cfg,
        "dataset": dataset,
        "sequence": sequence,
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_poses": len(normalized),
        "staged": [],
        "skipped": [],
    }

    pose_rows: list[np.ndarray] = []
    staged_stems: list[str] = []

    for i, scan_idx in enumerate(scan_indices):
        stem = scan_stem(scan_idx)
        in_scan = os.path.join(scan_dir, f"{stem}.bin")
        in_gt = os.path.join(gt_dir, f"{stem}.bin")
        in_pred = os.path.join(cenet_dir, f"{stem}.bin")
        if not existing(in_scan, in_gt, in_pred):
            manifest["skipped"].append(
                {
                    "stem": stem,
                    "scan_exists": os.path.isfile(in_scan),
                    "gt_exists": os.path.isfile(in_gt),
                    "cenet_exists": os.path.isfile(in_pred),
                }
            )
            continue

        # Lidar->world pose
        if dataset == "kitti360":
            T_lw = lidar_to_world_kitti360(normalized[i])
        else:
            T_lw = lidar_to_world_mcd(normalized[i], b2l)
        pose_rows.append(pose_row_12(T_lw))
        staged_stems.append(stem)

        # Symlink scan
        out_scan = os.path.join(out_velo, f"{stem}.bin")
        if os.path.islink(out_scan) or os.path.exists(out_scan):
            os.remove(out_scan)
        os.symlink(os.path.abspath(in_scan), out_scan)

        # Read raw scan only to know point count for softmax/labels validation
        scan_bytes = os.path.getsize(in_scan)
        n_pts = scan_bytes // (4 * 4)   # float32 * 4 channels

        # GT -> SemKITTI training class
        gt_train = convert_gt(in_gt, raw_to_train_lut)
        if gt_train.size != n_pts:
            manifest["skipped"].append(
                {"stem": stem, "reason": f"gt_size={gt_train.size} vs scan_pts={n_pts}"}
            )
            # roll back this scan
            pose_rows.pop()
            staged_stems.pop()
            os.remove(out_scan)
            continue
        gt_train.astype(np.uint32).tofile(os.path.join(out_lbl, f"{stem}.label"))

        # CENet softmax -> SemKITTI 20-class soft probs
        if channel_to_common is None:
            raw_pred = np.fromfile(in_pred, dtype=np.uint16)
            if raw_pred.size == 0 or raw_pred.size % n_pts != 0:
                manifest["skipped"].append(
                    {"stem": stem, "reason": f"pred_size={raw_pred.size}, n_pts={n_pts}"}
                )
                pose_rows.pop()
                staged_stems.pop()
                os.remove(out_scan)
                os.remove(os.path.join(out_lbl, f"{stem}.label"))
                continue
            K = raw_pred.size // n_pts
            channel_to_common = LM.build_source_channel_to_common(
                inferred_key, common_cfg, K
            )
            manifest["n_channels"] = int(K)
            print(f"Inferred K={K} channels from first CENet file; built routing table")

        if write_softmax:
            try:
                pred20 = convert_softmax(in_pred, n_pts, channel_to_common, num_train_classes=20)
            except ValueError as e:
                manifest["skipped"].append({"stem": stem, "reason": str(e)})
                pose_rows.pop()
                staged_stems.pop()
                os.remove(out_scan)
                os.remove(os.path.join(out_lbl, f"{stem}.label"))
                continue
            pred20.astype(np.float32).tofile(os.path.join(out_pred, f"{stem}.label"))

        # Hard labels: 9-class common argmax routed via canonical (no split).
        # Re-derive from CENet softmax instead of pred20 because pred20 has the
        # weighted split applied, which makes vegetation/vehicle never win argmax.
        raw_u16 = np.fromfile(in_pred, dtype=np.uint16).reshape(n_pts, -1)
        soft = LM.softmax_from_float16_raw(raw_u16)
        common = LM.aggregate_source_to_common(soft, channel_to_common, n_common=9)
        common_argmax = common.argmax(axis=1).astype(np.uint32)
        hard_train = common_to_train_canon[common_argmax]
        hard_train.astype(np.uint32).tofile(os.path.join(out_hard, f"{stem}.label"))

        manifest["staged"].append({"stem": stem, "n_points": int(n_pts)})

    # Sort by stem so file order matches sorted(glob('velodyne/*.bin'))
    order = sorted(range(len(staged_stems)), key=lambda j: staged_stems[j])
    pose_rows = [pose_rows[j] for j in order]
    staged_stems = [staged_stems[j] for j in order]
    manifest["staged_order"] = staged_stems

    # poses.txt: one 12-float row per staged scan, in stem-sorted order.
    if pose_rows:
        np.savetxt(
            os.path.join(out_seq, "poses.txt"),
            np.asarray(pose_rows),
            fmt="%.10g",
        )

    # calib.txt: Tr = identity (so Conv-BKI's inv(Tr)*pose*Tr is a no-op).
    with open(os.path.join(out_seq, "calib.txt"), "w") as f:
        # P0..P3 (unused by Conv-BKI but kept for layout parity) + Tr identity row.
        ident_row = "1 0 0 0 0 1 0 0 0 0 1 0"
        f.write(f"P0: {ident_row}\n")
        f.write(f"P1: {ident_row}\n")
        f.write(f"P2: {ident_row}\n")
        f.write(f"P3: {ident_row}\n")
        f.write(f"Tr: {ident_row}\n")

    # Manifest and routing table
    with open(os.path.join(staging_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if channel_to_common is not None:
        routing = {
            "inferred_labels_key": inferred_key,
            "n_channels": int(channel_to_common.size),
            "channel_to_common": [int(x) for x in channel_to_common],
            "common_to_semkitti_train_canonical": LM.COMMON_TO_SEMKITTI_TRAIN,
            "common_to_semkitti_train_weighted": {
                str(c): [(int(t), float(w)) for t, w in splits]
                for c, splits in LM.COMMON_TO_SEMKITTI_TRAIN_WEIGHTED.items()
            },
        }
        with open(os.path.join(staging_root, "routing.json"), "w") as f:
            json.dump(routing, f, indent=2)

    # Round-trip sanity check on one pose (Tr=I so global_pose = pose).
    if pose_rows:
        row = pose_rows[0]
        T34 = row.reshape(3, 4)
        T44 = np.vstack([T34, [[0.0, 0.0, 0.0, 1.0]]])
        recon = T44[:3, :4].reshape(-1)
        assert np.allclose(recon, row), "pose row round-trip failed"

    n_staged = len(staged_stems)
    n_skipped = len(manifest["skipped"])
    print(f"\nStaged {n_staged} scans, skipped {n_skipped} (see {staging_root}/manifest.json)")
    print(f"  staging_root = {staging_root}")
    return 0 if n_staged > 0 else 3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Per-sequence staging YAML")
    p.add_argument(
        "--data-root",
        default=None,
        help="Overrides data_root from the YAML (otherwise falls back to "
        "OSM_BKI_DATA_ROOT env var).",
    )
    args = p.parse_args()
    return stage(args.config, data_root_override=args.data_root)


if __name__ == "__main__":
    sys.exit(main())
