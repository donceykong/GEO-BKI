"""Train a native 9-class Conv-BKI per-class kernel on KITTI-360 GT.

GT-based retraining (the project lead's design): the per-scan network
"prediction" is the GROUND-TRUTH one-hot label (staged by stage_train_gt.py
into predictions_hard/), and the target is the same GT. The only learnable
parameters are the per-class kernel length-scales `ell` (9 values); the loop
learns the spatial extent per class that best reproduces GT at the voxel level
after multi-frame accumulation.

This adapts NeuralBKI/train.py + Models/ConvBKI.py to our setup:
  - native 9-class common space (num_classes=9), NOT SemKITTI 20-class;
  - StagingKittiDataset overlay (sparse KITTI-360 ids, glob pairing);
  - hard one-hot input (from_continuous=False), NLL loss ignore_index=0;
  - inverse-log-frequency class weights from the staged class_counts.json.

The two tiny tensor helpers from NeuralBKI/Data/utils.py are reimplemented
here because that module imports rospy at top level (no ROS in this env) —
the same dodge run_convbki.py uses.

Recipe (upstream Config/ConvBKI_PerClass.yaml): Adam lr 0.007, betas
(0.9,0.999), ExponentialLR gamma 0.96, 5 epochs, seed 42, per-class sparse
filters, filter_size 5, ell init 0.5 (learnable).

Grid: we train on the upstream ±40 m grid (400x400x26, voxel 0.2 m). The eval
grid is ±50 m (501x501x27) but ALSO 0.2 m/voxel, and the sparse kernel depends
only on voxel size (via voxel_sizes in kernel_dists), not grid extent — so a
kernel learned at ±40 m transfers exactly to the ±50 m eval grid while using
less VRAM and time per scan. Pass --grid eval50 to train on the eval grid
instead.

Usage (smoke, one sequence, capped iters):
    python train_convbki_9class.py \
        --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt \
        --train-seqs 0003 --val-seqs 0003 \
        --out-dir /home/sandilya/convbki_train_workspace/smoke_ckpt \
        --epochs 1 --max-train-iters 60 --max-val-iters 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
NBKI_DEFAULT = os.path.join(REPO_ROOT, "baselines", "convbki", "NeuralBKI")

N_COMMON = 9

GRIDS = {
    # upstream ConvBKI_PerClass training grid: ±40 m, 0.2 m voxel
    "train40": {"grid_size": [400, 400, 26],
                "min_bound": [-40.0, -40.0, -2.6],
                "max_bound": [40.0, 40.0, 2.6]},
    # our Option-A eval grid: ±50 m, 0.2 m voxel (501x501x27)
    "eval50": {"grid_size": [501, 501, 27],
               "min_bound": [-50.1, -50.1, -2.7],
               "max_bound": [50.1, 50.1, 2.7]},
}


# --------------------------------------------------------------------------- #
# Reimplemented from NeuralBKI/Data/utils.py (that module imports rospy).
# --------------------------------------------------------------------------- #
def iou_one_frame(pred, target, n_classes):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = np.zeros(n_classes)
    union = np.zeros(n_classes)
    for cls in range(n_classes):
        pi = pred == cls
        ti = target == cls
        inter[cls] = (pi[ti]).long().sum().item()
        union[cls] = pi.long().sum().item() + ti.long().sum().item() - inter[cls]
    return inter, union


def points_to_voxels_torch(voxel_grid, points, min_bound, grid_dims, voxel_sizes):
    voxels = torch.floor((points - min_bound) / voxel_sizes).to(dtype=torch.int)
    maxes = (grid_dims - 1).reshape(1, 3)
    mins = torch.zeros_like(maxes)
    voxels = torch.clip(voxels, mins, maxes).to(dtype=torch.long)
    return voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# Class weights
# --------------------------------------------------------------------------- #
def inverse_log_freq_weights(class_counts, device, eps=1e-5):
    """weights[0]=0 (ignore); weights[c]=1/log(count_c+eps) for c>0.

    Matches NeuralBKI/train.py. Guards classes with zero/near-zero count (which
    would yield a negative weight from 1/log(<1)) by zeroing them.
    """
    freq = np.asarray(class_counts, dtype=np.float64)
    w = np.zeros_like(freq)
    pos = freq[1:] > 1.0
    w[1:][pos] = 1.0 / np.log(freq[1:][pos] + eps)
    return torch.from_numpy(w).to(dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# One pass over a loader
# --------------------------------------------------------------------------- #
def run_loop(model, dataloader, criterion, optimizer, device, num_classes,
             min_bound_t, grid_dims_t, voxel_sizes_t, training, max_iters=None,
             desc=""):
    all_inter = np.zeros(num_classes)
    all_union = np.zeros(num_classes) + 1e-6
    num_correct = 0
    num_total = 0
    losses = []

    it = 0
    for points, points_labels, gt_labels in tqdm(dataloader, desc=desc):
        if max_iters is not None and it >= max_iters:
            break
        it += 1

        batch_gt = torch.zeros((0, 1), device=device, dtype=torch.uint8)
        batch_preds = torch.zeros((0, num_classes), device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad()

        for b in range(len(points)):
            current_map = model.initialize_grid()

            pc_np = np.vstack(np.array(points[b], dtype=object))
            labels_np = np.vstack(np.array(points_labels[b], dtype=object))
            labeled_pc = np.hstack((pc_np, labels_np)).astype(np.float32)
            if labeled_pc.shape[0] == 0:
                continue

            labeled_pc_t = torch.from_numpy(labeled_pc).to(device=device, non_blocking=True)
            preds = model(current_map, labeled_pc_t)
            gt_sem = torch.from_numpy(gt_labels[b]).to(device=device, non_blocking=True)

            last_pc_t = torch.from_numpy(points[b][-1].astype(np.float32)).to(device=device,
                                                                              non_blocking=True)
            sem_preds = points_to_voxels_torch(preds, last_pc_t, min_bound_t,
                                               grid_dims_t, voxel_sizes_t)
            sem_preds = sem_preds / torch.sum(sem_preds, dim=-1, keepdim=True)

            non_void = gt_sem[:, 0] != 0
            batch_gt = torch.vstack((batch_gt, gt_sem[non_void, :]))
            batch_preds = torch.vstack((batch_preds, sem_preds[non_void, :]))

        batch_gt = batch_gt.reshape(-1)
        not_ignore = batch_gt != 0
        batch_preds = batch_preds[not_ignore, :]
        batch_gt = batch_gt[not_ignore]
        if batch_gt.numel() == 0:
            continue

        loss = criterion(torch.log(batch_preds), batch_gt.long())
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            max_preds = torch.argmax(batch_preds, dim=-1)
            preds_np = max_preds.cpu().numpy()
            gt_np = batch_gt.detach().cpu().numpy()
            num_correct += int(np.sum(preds_np == gt_np))
            num_total += int(gt_np.shape[0])
            inter, union = iou_one_frame(max_preds, batch_gt, n_classes=num_classes)
            all_inter += inter
            all_union += union + 1e-6

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    acc = (num_correct / num_total) if num_total else float("nan")
    valid = all_union > 1e-3
    miou = float(np.mean(all_inter[valid] / all_union[valid])) if valid.any() else float("nan")
    per_class_iou = (all_inter / all_union).tolist()
    return {"loss": mean_loss, "acc": acc, "miou": miou,
            "per_class_iou": per_class_iou, "iters": len(losses)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging-root", required=True)
    p.add_argument("--train-seqs", nargs="+", required=True)
    p.add_argument("--val-seqs", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.007)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--decay", type=float, default=0.96)
    p.add_argument("--ell", type=float, default=0.5)
    p.add_argument("--filter-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", choices=list(GRIDS.keys()), default="train40")
    p.add_argument("--class-counts", default=None,
                   help="class_counts.json (default <staging-root>/class_counts.json)")
    p.add_argument("--max-train-iters", type=int, default=None, help="cap (smoke)")
    p.add_argument("--max-val-iters", type=int, default=None, help="cap (smoke)")
    p.add_argument("--nbki-root", default=NBKI_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-aug", action="store_true", help="random rotation/flip aug on train")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if not out_dir.startswith("/home/sandilya/") and "baselines/convbki/Weights" not in out_dir:
        print(f"WARNING: out-dir {out_dir} is outside /home/sandilya and the repo Weights dir")
    os.makedirs(out_dir, exist_ok=True)

    staging_root = os.path.abspath(args.staging_root)
    cc_path = args.class_counts or os.path.join(staging_root, "class_counts.json")
    with open(cc_path) as f:
        class_counts = json.load(f)["class_counts"]

    device = torch.device(args.device)
    setup_seed(args.seed)

    # Import ConvBKI + the staging overlay. The overlay imports
    # Data/SemanticKitti.py, which reads Config/semantic_kitti.yaml relative to
    # CWD at import time, so chdir into nbki_root for the import (run_convbki
    # does the same).
    orig_cwd = os.getcwd()
    os.chdir(args.nbki_root)
    if args.nbki_root not in sys.path:
        sys.path.insert(0, args.nbki_root)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        from Models.ConvBKI import ConvBKI
        from staging_dataset import StagingKittiDataset
    finally:
        os.chdir(orig_cwd)

    grid = GRIDS[args.grid]
    grid_size = grid["grid_size"]
    min_bound = grid["min_bound"]
    max_bound = grid["max_bound"]
    vx = (max_bound[0] - min_bound[0]) / grid_size[0]
    print(f"Grid {args.grid}: size={grid_size} bounds={min_bound}->{max_bound} "
          f"voxel~{vx:.3f}m")

    model = ConvBKI(
        torch.tensor(grid_size, dtype=torch.long).to(device),
        torch.tensor(min_bound, dtype=torch.float32).to(device),
        torch.tensor(max_bound, dtype=torch.float32).to(device),
        filter_size=args.filter_size,
        num_classes=N_COMMON,
        device=device,
        datatype=torch.float32,
        max_dist=args.ell,
        kernel="sparse",
        per_class=True,
        compound=False,
    )

    weights = inverse_log_freq_weights(class_counts, device)
    print("class_counts:", class_counts)
    print("NLL weights:", [round(w, 5) for w in weights.cpu().tolist()])
    criterion = nn.NLLLoss(weight=weights, ignore_index=0)

    grid_params = {"grid_size": grid_size, "min_bound": min_bound, "max_bound": max_bound}

    def make_ds(seqs, use_aug):
        return StagingKittiDataset(
            grid_params=grid_params, staging_root=staging_root, sequences=seqs,
            device=args.device, num_frames=args.num_frames, use_aug=use_aug,
            apply_transform=True, from_continuous=False, to_continuous=False,
            pred_path="predictions_hard", num_classes=N_COMMON, remove_zero=False,
        )

    train_ds = make_ds(args.train_seqs, use_aug=args.use_aug)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=train_ds.collate_fn, num_workers=args.num_workers,
                          pin_memory=False)
    print(f"train scans: {len(train_ds)} across seqs {args.train_seqs}")

    val_dl = None
    if args.val_seqs:
        val_ds = make_ds(args.val_seqs, use_aug=False)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=val_ds.collate_fn, num_workers=args.num_workers,
                            pin_memory=False)
        print(f"val scans: {len(val_ds)} across seqs {args.val_seqs}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay)

    min_bound_t = torch.from_numpy(train_ds.min_bound).to(device=device)
    grid_dims_t = torch.from_numpy(train_ds.grid_dims).to(dtype=torch.int, device=device)
    voxel_sizes_t = torch.from_numpy(train_ds.voxel_sizes).to(device=device)

    history = {"args": vars(args), "grid": grid, "class_counts": class_counts,
               "nll_weights": weights.cpu().tolist(), "epochs": []}

    def save_filters(tag):
        torch.save(model.get_filters(), os.path.join(out_dir, f"filters{tag}.pt"))

    t_start = time.time()
    for epoch in range(args.epochs):
        # Save the pre-training-epoch kernel (epoch 0 = init), matching upstream.
        save_filters(epoch)

        ep = {"epoch": epoch, "ell_before": model.ell.detach().cpu().tolist()}

        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                vstats = run_loop(model, val_dl, criterion, optimizer, device, N_COMMON,
                                  min_bound_t, grid_dims_t, voxel_sizes_t, training=False,
                                  max_iters=args.max_val_iters, desc=f"val e{epoch}")
            ep["val"] = vstats
            print(f"[epoch {epoch}] VAL loss={vstats['loss']:.4f} acc={vstats['acc']:.4f} "
                  f"miou={vstats['miou']:.4f}")

        model.train()
        tstats = run_loop(model, train_dl, criterion, optimizer, device, N_COMMON,
                          min_bound_t, grid_dims_t, voxel_sizes_t, training=True,
                          max_iters=args.max_train_iters, desc=f"train e{epoch}")
        scheduler.step()
        ep["train"] = tstats
        ep["ell_after"] = model.ell.detach().cpu().tolist()
        ep["lr_after"] = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch}] TRAIN loss={tstats['loss']:.4f} acc={tstats['acc']:.4f} "
              f"miou={tstats['miou']:.4f} iters={tstats['iters']}")
        print(f"           ell={[round(e,4) for e in ep['ell_after']]}")
        history["epochs"].append(ep)

    # Final validation + final filters (filters{epochs}.pt, like upstream).
    if val_dl is not None:
        model.eval()
        with torch.no_grad():
            vstats = run_loop(model, val_dl, criterion, optimizer, device, N_COMMON,
                              min_bound_t, grid_dims_t, voxel_sizes_t, training=False,
                              max_iters=args.max_val_iters, desc="val final")
        history["final_val"] = vstats
        print(f"[final] VAL loss={vstats['loss']:.4f} acc={vstats['acc']:.4f} "
              f"miou={vstats['miou']:.4f}")
    save_filters(args.epochs)

    history["total_seconds"] = time.time() - t_start
    history["final_ell"] = model.ell.detach().cpu().tolist()
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone in {history['total_seconds']:.1f}s. Checkpoints + history -> {out_dir}")
    print(f"final ell = {[round(e,4) for e in history['final_ell']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
