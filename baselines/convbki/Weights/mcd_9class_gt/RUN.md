# Conv-BKI 9-class GT retraining — mcd_9class_gt

GT-based retraining (project lead's design): the per-scan network input is the **ground-truth one-hot** common label (not CENet predictions); the target is the same GT. The per-class sparse kernel learns the spatial extent per class that best reproduces GT at the voxel level after multi-frame accumulation. Native **9-class common** space — no SemKITTI 20-class projection. Same recipe as `kitti360_9class_gt`, trained on MCD (NTU + TUHH) GT.

## Configuration

- **Seed:** 42
- **Grid:** train40 (±40 m, 400×400×26, 0.2 m voxel; transfers exactly to ±50 m eval grid — same voxel size)
- **Recipe:** num_frames=10, batch=2, epochs=5, lr=0.007 (Adam betas 0.9/0.999), ExponentialLR γ=0.96, ell init=0.5 (learnable), filter_size=5, per-class sparse kernel, NLL ignore_index=0
- **Train split:** 11 seqs (5 tuhh + 6 ntu) `['tuhh_day_02', 'tuhh_day_03', 'tuhh_day_04', 'tuhh_night_07', 'tuhh_night_08', 'ntu_day_01', 'ntu_day_02', 'ntu_day_10', 'ntu_night_04', 'ntu_night_08', 'ntu_night_13']` — 24,095 scans
- **Val split:** `['tuhh_night_09']` — 1,833 scans (monitoring only; true held-out test is the kth eval seqs, not yet on this machine)
- **Subsampling:** --every-nth 2 — every 2nd scan kept at STAGING time on the 11 training seqs (24,095 scans). Val seq tuhh_night_09 staged at FULL density (1,833).
- **GT source:** `gt_labels_terrain` remapped via `mcd_to_common` (raw 0..29 → common 0..8; terrain 29→8). Per-group calib auto-selected at staging: tuhh→`hhs_calib.yaml`, ntu→`atv_calib.yaml`.
- **Data root:** `/media/donceykong/donceys_data_ssd/datasets`
- **Staging root:** `/home/sandilya/convbki_train_workspace/staging_train_gt_mcd`
- **Started:** 2026-07-08 14:37:43 MDT

### Command

```
python train_convbki_9class.py \
  --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt_mcd \
  --train-seqs tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 \
               ntu_day_01 ntu_day_02 ntu_day_10 ntu_night_04 ntu_night_08 ntu_night_13 \
  --val-seqs tuhh_night_09 \
  --out-dir /home/sandilya/GEO-BKI/baselines/convbki/Weights/mcd_9class_gt \
  --epochs 5 --batch 2 --num-frames 10 --num-workers 4 \
  --grid train40 --seed 42
```

### Class weights (inverse log frequency from training GT)

| common class | point count | NLL weight |
|---|---|---|
| 0 unlabeled (ignored) | 319,072,812 | 0.00000 |
| 1 road | 345,277,711 | 0.05087 |
| 2 sidewalk | 212,509,283 | 0.05215 |
| 3 parking | 28,056,708 | 0.05831 |
| 4 building | 474,873,272 | 0.05005 |
| 5 fence | 84,927,565 | 0.05477 |
| 6 vegetation | 537,394,089 | 0.04975 |
| 7 vehicle | 87,438,382 | 0.05469 |
| 8 terrain | 192,282,626 | 0.05243 |

> Note: unlike KITTI-360 (where class 0 GT was a masked count), MCD class 0
> here holds 319 M real points — raw MCD labels not covered by `mcd_to_common`
> collapse to unlabeled. It is still `ignore_index=0` in the loss and dropped
> from all metrics, so it does not affect the learned kernel or reported mIoU.

## Per-epoch history

VAL is computed on the kernel entering that epoch (pre-train); TRAIN is that epoch's mean. Both are GT-reproduction metrics, not held-out.

| epoch | val loss | val mIoU | train loss | train mIoU | train iters |
|---|---|---|---|---|---|
| 0 | 0.1097 | 0.7677 | 0.1306 | 0.7897 | 12048 |
| 1 | 0.1004 | 0.7773 | 0.1443 | 0.7815 | 12048 |
| 2 | 0.1000 | 0.7771 | 0.1444 | 0.7816 | 12048 |
| 3 | 0.1009 | 0.7771 | 0.1440 | 0.7817 | 12048 |
| 4 | 0.1004 | 0.7771 | 0.1443 | 0.7817 | 12048 |
| final | 0.1001 | 0.7771 | — | — | — |

Val loss drops from 0.1097 (pre-training, ell=0.5) to ~0.100 and converges after epoch 1; the kernel is effectively stable from epoch 2 on.

> **Val per-class note.** On val seq `tuhh_night_09`, road (class 1) and
> parking (class 3) IoU are 0.0 across every epoch — that val sequence
> carries no road/parking GT to reproduce. Training itself learned both
> (train road IoU ≈0.92, parking ≈0.85), so this is a property of the val
> seq, not a training failure. The final-val mIoU (0.7771) is averaged over
> the classes present in that seq.

### Learned per-class `ell` (final)

| class | ell |
|---|---|
| 0 unlabeled | 0.1694 |
| 1 road | 0.1381 |
| 2 sidewalk | 0.2024 |
| 3 parking | 0.2399 |
| 4 building | 0.1104 |
| 5 fence | 0.0152 |
| 6 vegetation | 0.1216 |
| 7 vehicle | 0.0554 |
| 8 terrain | 0.2172 |

> The MCD kernel learns **narrower** length-scales than KITTI-360 for the
> tall/sparse classes (fence 0.015, vehicle 0.055, building 0.110, vegetation
> 0.122), and its widest scales are on the ground/planar classes (parking
> 0.240, terrain 0.217, sidewalk 0.202). This is the opposite emphasis from
> KITTI-360 (where building/fence/veg/vehicle widened to ~0.33–0.38) — a
> genuine dataset difference (MCD's hhs/atv sweeps and denser near-field
> ground coverage), not a config change.

**Total training time:** 8787s (2.44 h) on the RTX 4070 Ti SUPER.

**Final checkpoint:** `filters5.pt` (referenced by future MCD Option-B eval configs). Snapshots `filters0.pt`..`filters4.pt` are the pre-epoch states.

_Status: training COMPLETE. Eval NOT yet run — awaiting Invascal inferred labels (`invascal_<dataset>`) and the kth eval sequences._
