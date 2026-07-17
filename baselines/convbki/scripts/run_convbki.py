"""Drive Conv-BKI inference over a staged sequence and write per-point predictions.

This is a slimmed-down reimplementation of NeuralBKI's generate_results.py
that:
  - Loads our StagingKittiDataset overlay instead of the upstream
    KittiDataset (so non-6-digit scan stems and sparse frame IDs work).
  - Skips the rospy / visualization_msgs imports so we don't need ROS
    installed in the run environment.
  - Writes per-frame predictions/<stem>.label using the actual scan stem
    rather than a sequential counter, so convert_outputs.py can pair
    each prediction file back with the on-disk scan/GT.
  - Honors the plan's "no-observation -> class 0 (unlabeled)" rule via
    ignore_labels=[0] (label_points uses this to fill OOB points).

Usage:
    python run_convbki.py <experiment_config.yaml>

Logs stdout+stderr to baselines/convbki/logs/<dataset>_<sequence>_<timestamp>.log
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
NBKI_DEFAULT = os.path.join(REPO_ROOT, "baselines", "convbki", "NeuralBKI")


class TeeLogger:
    """Mirror writes to stdout AND a file. Picked up by both Python's print
    and any C-level writes via dup2 of fd 1/2 (we set sys.stdout/stderr)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fp = open(path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, msg: str) -> None:
        self.stdout.write(msg)
        self.fp.write(msg)

    def flush(self) -> None:
        self.stdout.flush()
        self.fp.flush()


def _setup_paths(nbki_root: str) -> None:
    if nbki_root not in sys.path:
        sys.path.insert(0, nbki_root)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)


def main(config_path: str, hard: bool = False, limit: int | None = None) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    nbki_root = cfg.get("nbki_root", NBKI_DEFAULT)
    experiment = cfg["experiment_name"]
    staging_root = os.path.abspath(cfg["staging_root"])
    output_root = os.path.abspath(cfg["output_root"])
    log_dir = os.path.abspath(cfg.get("log_dir", os.path.join(REPO_ROOT, "baselines", "convbki", "logs")))
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{experiment}_{ts}.log")
    sys.stdout = TeeLogger(log_path)

    print(f"Conv-BKI inference run")
    print(f"  experiment   = {experiment}")
    print(f"  staging_root = {staging_root}")
    print(f"  output_root  = {output_root}")
    print(f"  log_path     = {log_path}")
    print(f"  config       = {config_path}")
    print(f"  start_time   = {ts}")

    # Upstream Data/SemanticKitti.py reads Config/semantic_kitti.yaml at
    # import time, relative to CWD. Chdir there for the import.
    orig_cwd = os.getcwd()
    os.chdir(nbki_root)
    _setup_paths(nbki_root)
    try:
        from Models.mapping_utils import GlobalMap
        from staging_dataset import StagingKittiDataset
    finally:
        os.chdir(orig_cwd)

    # ------------------------------------------------------------------ #
    # Device + seed
    # ------------------------------------------------------------------ #
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"  device       = {device}")

    # ------------------------------------------------------------------ #
    # Grid + weights
    # ------------------------------------------------------------------ #
    grid = cfg["grid"]
    grid_size = [int(p) for p in grid["grid_size"]]
    min_bound = [float(x) for x in grid["min_bound"]]
    max_bound = [float(x) for x in grid["max_bound"]]
    vx = (max_bound[0] - min_bound[0]) / grid_size[0]
    print(f"  grid_size    = {grid_size}")
    print(f"  bounds       = {min_bound} -> {max_bound}")
    print(f"  voxel_size   ~ {vx:.4f} m")

    # weights.path (absolute) takes precedence over dir/epoch — lets the
    # retrained native-9-class checkpoints live outside NeuralBKI/Models/Weights
    # (e.g. baselines/convbki/Weights/kitti360_9class_gt/filtersN.pt).
    wcfg = cfg["weights"]
    if wcfg.get("path"):
        wp = os.path.expandvars(wcfg["path"])
        weights_path = wp if os.path.isabs(wp) else os.path.join(REPO_ROOT, wp)
    else:
        weights_path = os.path.join(
            nbki_root, "Models", "Weights", wcfg["dir"], f"filters{int(wcfg['epoch'])}.pt"
        )
    print(f"  weights      = {weights_path}")
    weights = torch.load(weights_path, map_location=device)

    num_classes = int(cfg["num_classes"])
    filter_size = int(cfg["filter_size"])

    # Allow YAML or CLI to switch between soft (continuous) and hard (one-hot)
    # input. The pretrained ConvBKI_PC_02_V kernel was trained with hard labels
    # (from_continuous: False in the upstream ConvBKI_PerClass.yaml); soft input
    # with our routing was causing the parking kernel (sum=26.7, vs sidewalk
    # 5.6 / building 4.8) to dominate every voxel from the uniform-ish floor mass.
    yaml_hard = bool(cfg.get("hard_input", False))
    use_hard = hard or yaml_hard
    pred_path = "predictions_hard" if use_hard else "predictions_softmax"
    from_continuous = not use_hard
    to_continuous = not use_hard
    print(f"  input_format = {'hard (uint32 argmax)' if use_hard else 'soft (float32 N x 20)'}")
    print(f"  pred_path    = {pred_path}")

    grid_params = {"grid_size": grid_size, "min_bound": min_bound, "max_bound": max_bound}
    ds = StagingKittiDataset(
        grid_params=grid_params,
        staging_root=staging_root,
        sequences=["00"],
        device=device_str,
        num_frames=1,
        use_aug=False,
        apply_transform=True,
        from_continuous=from_continuous,
        to_continuous=to_continuous,
        pred_path=pred_path,
        num_classes=num_classes,
        remove_zero=False,
    )
    print(f"  scans staged = {len(ds)}")

    if len(ds) == 0:
        print("ERROR: no scans found in staging_root", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ #
    # Keyframe / query-at-end mode
    # ------------------------------------------------------------------ #
    # keyframe_query_at_end: build the map over the keyframe set (propagate +
    # update_map only, no per-scan labeling), then query ONCE at the end at the
    # same keyframe set. This mirrors the offline global-map eval the other
    # baselines use, and matches S-BKI. It is a SEMANTIC change from the default
    # per-scan online path: each keyframe is labeled against the FULL finished
    # map (including scans that come AFTER it), not just the map-so-far.
    kf_query_at_end = bool(cfg.get("keyframe_query_at_end", False))
    kf_spacing = float(cfg.get("keyframe_spacing", 5.0))
    kf_indices: list[int] | None = None
    if kf_query_at_end:
        kf_file = cfg.get("keyframe_stems_file") or os.path.join(
            staging_root, f"keyframes_{kf_spacing:g}m.txt"
        )
        kf_file = os.path.expandvars(kf_file)
        if not os.path.isfile(kf_file):
            print(f"ERROR: keyframe_query_at_end set but keyframe file missing: {kf_file}",
                  file=sys.stderr)
            return 2
        with open(kf_file) as f:
            kf_stems = {ln.strip() for ln in f if ln.strip()}
        # Intersect with the staged dataset, preserving dataset order (which is
        # stem-sorted); keyframes on scans that were not staged (missing GT/pred)
        # are silently dropped.
        kf_indices = [i for i in range(len(ds)) if ds.stem_for_index(i) in kf_stems]
        print(f"  keyframe mode = ON (query-at-end, {kf_spacing:g} m spacing)")
        print(f"  keyframe file = {kf_file}")
        print(f"  keyframes     = {len(kf_stems)} in file, {len(kf_indices)} staged (build==query set)")
        if len(kf_indices) == 0:
            print("ERROR: no keyframe stems intersect the staged set", file=sys.stderr)
            return 2

    # Disable garbage collection in keyframe/query-at-end mode so the map
    # persists over the whole sequence (default delete_time=10 would prune the
    # map to a rolling ~10-frame window, and end-querying an early keyframe would
    # return all-prior/ignore). A per-scan run keeps the default rolling window.
    delete_time = (len(ds) + 10) if kf_query_at_end else 10

    map_object = GlobalMap(
        torch.tensor(grid_size, dtype=torch.long).to(device),
        torch.tensor(min_bound).to(device),
        torch.tensor(max_bound).to(device),
        weights,
        filter_size,
        num_classes=num_classes,
        ignore_labels=[0],
        device=device,
        delete_time=delete_time,
    )
    print(f"  delete_time  = {delete_time}{' (GC disabled for persistent map)' if kf_query_at_end else ''}")

    out_preds = os.path.join(output_root, "sequences", "00", "predictions")
    os.makedirs(out_preds, exist_ok=True)

    def peak_rss_gb() -> float:
        # ru_maxrss is KiB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

    t0 = time.time()
    written = 0

    if kf_query_at_end:
        # -------------------------------------------------------------- #
        # Two-pass keyframe / query-at-end
        # -------------------------------------------------------------- #
        # BUILD pass: fold every keyframe into ONE persistent map (GC disabled),
        # no labeling. reset_grid ONCE up front; all keyframes are one sequence,
        # and we must NOT reset between build and query so the finished map
        # survives into the query pass.
        map_object.reset_grid()
        n_kf = len(kf_indices)
        for j, idx in enumerate(kf_indices):
            with torch.no_grad():
                pose, points, pred_labels, _gt, _sid, _fid = ds.get_test_item(idx, get_gt=False)
                map_object.propagate(pose)
                labeled_pc = np.hstack((points, pred_labels))
                labeled_pc_t = torch.from_numpy(labeled_pc).to(device=device, non_blocking=True)
                map_object.update_map(labeled_pc_t)
            if (j + 1) % 50 == 0 or j + 1 == n_kf:
                el = time.time() - t0
                gm = 0 if map_object.global_map is None else map_object.global_map.shape[0]
                print(f"  [build {j + 1}/{n_kf}] rate={(j + 1) / max(el, 1e-9):.2f} kf/s "
                      f"map_cells={gm} rss={peak_rss_gb():.2f}GB elapsed={el:.1f}s")

        gm = 0 if map_object.global_map is None else map_object.global_map.shape[0]
        print(f"  BUILD done: {n_kf} keyframes, map_cells={gm}, "
              f"peak_rss={peak_rss_gb():.2f}GB, build_time={time.time() - t0:.1f}s")

        # QUERY pass: label each keyframe against the FINISHED map. No reset, no
        # further update_map — propagate re-windows the persistent map to each
        # keyframe's pose.
        tq = time.time()
        for j, idx in enumerate(kf_indices):
            with torch.no_grad():
                pose, points, _pred, _gt, _sid, _fid = ds.get_test_item(idx, get_gt=False)
                map_object.propagate(pose)
                predictions, _local_mask = map_object.label_points(points)
                preds_np = predictions.detach().cpu().numpy().astype(np.uint32)
                stem = ds.stem_for_index(idx)
                preds_np.tofile(os.path.join(out_preds, f"{stem}.label"))
                written += 1
            if (j + 1) % 50 == 0 or j + 1 == n_kf:
                el = time.time() - tq
                print(f"  [query {j + 1}/{n_kf}] rate={(j + 1) / max(el, 1e-9):.2f} kf/s elapsed={el:.1f}s")

        print(f"Done: wrote {written} prediction files to {out_preds}")
        print(f"Total time: {time.time() - t0:.1f}s  peak_rss={peak_rss_gb():.2f}GB")
        return 0

    # ------------------------------------------------------------------ #
    # Default per-scan online inference loop (unchanged)
    # ------------------------------------------------------------------ #
    map_object.reset_grid()
    current_scene = None
    current_frame_id = None

    end = len(ds) if limit is None else min(len(ds), int(limit))
    for idx in range(end):
        with torch.no_grad():
            pose, points, pred_labels, _gt, scene_id, frame_id = ds.get_test_item(idx, get_gt=False)

            # Same reset rule as generate_results.py — but in our staging,
            # frame_id is the contiguous dataset index, so (frame_id-1) ==
            # current_frame_id holds whenever we're inside one sequence.
            if scene_id != current_scene or (frame_id - 1) != current_frame_id:
                map_object.reset_grid()

            map_object.propagate(pose)

            # pred_labels: (N, num_classes) float32 soft probs (from
            # from_continuous=True + to_continuous=True). Conv-BKI's
            # update_map switches to the continuous Dirichlet path when
            # input width is num_classes + 3.
            labeled_pc = np.hstack((points, pred_labels))
            labeled_pc_t = torch.from_numpy(labeled_pc).to(device=device, non_blocking=True)
            map_object.update_map(labeled_pc_t)

            predictions, local_mask = map_object.label_points(points)
            # label_points already sets predictions[~local_mask] = ignore_labels[0]
            preds_np = predictions.detach().cpu().numpy().astype(np.uint32)

            stem = ds.stem_for_index(idx)
            preds_np.tofile(os.path.join(out_preds, f"{stem}.label"))
            written += 1

            current_scene = scene_id
            current_frame_id = frame_id

        if (idx + 1) % 100 == 0 or idx + 1 == end:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx + 1}/{len(ds)}]  rate={rate:.2f} scans/s  elapsed={elapsed:.1f}s")

    print(f"Done: wrote {written} prediction files to {out_preds}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Per-experiment run YAML")
    p.add_argument("--hard", action="store_true",
                   help="Use hard (one-hot) inputs from predictions_hard/ instead of "
                        "the soft N x 20 predictions_softmax/. Pretrained ConvBKI_PC_02_V "
                        "was trained with hard labels, so this matches the model's "
                        "training distribution.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many scans (smoke testing).")
    args = p.parse_args()
    sys.exit(main(args.config, hard=args.hard, limit=args.limit))
