# Conv-BKI 9-class GT retraining — kitti360_9class_gt

GT-based retraining (project lead's design): the per-scan network input is the **ground-truth one-hot** common label (not CENet predictions); the target is the same GT. The per-class sparse kernel learns the spatial extent per class that best reproduces GT at the voxel level after multi-frame accumulation. Native **9-class common** space — no SemKITTI 20-class projection.

## Configuration

- **Seed:** 42
- **Grid:** train40 (±40 m, 400×400×26, 0.2 m voxel; transfers exactly to ±50 m eval grid — same voxel size)
- **Recipe:** num_frames=10, batch=2, epochs=5, lr=0.007 (Adam betas 0.9/0.999), ExponentialLR γ=0.96, ell init=0.5 (learnable), filter_size=5, per-class sparse kernel, NLL ignore_index=0
- **Train split:** seqs ['0002', '0004', '0005', '0006', '0007', '0010'] — 20001 scans
- **Val split:** seqs ['0003'] — 988 scans (monitoring only; true held-out test is 0000/0009)
- **Subsampling:** --every-nth 2 — every 2nd scan kept at STAGING time on the 6 training seqs (20,001 of ~40,005 staged-eligible scans). Val seq 0003 staged at FULL density (988).
- **Data root:** `/media/donceykong/donceys_data_ssd/datasets`
- **Staging root:** `/home/sandilya/convbki_train_workspace/staging_train_gt`
- **Started:** 2026-06-12 14:17:50 MDT

### Command

```
python train_convbki_9class.py \
  --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt \
  --train-seqs 0002 0004 0005 0006 0007 0010 \
  --val-seqs 0003 \
  --out-dir /home/sandilya/GEO-BKI/baselines/convbki/Weights/kitti360_9class_gt \
  --epochs 5 --batch 2 --num-frames 10 --num-workers 4 \
  --grid train40 --seed 42
```

### Class weights (inverse log frequency from training GT)

| common class | point count | NLL weight |
|---|---|---|
| 0 unlabeled (ignored) | 273,385,737 | 0.00000 |
| 1 road | 465,457,364 | 0.05010 |
| 2 sidewalk | 265,407,367 | 0.05155 |
| 3 parking | 76,341,682 | 0.05509 |
| 4 building | 348,773,904 | 0.05084 |
| 5 fence | 37,094,475 | 0.05738 |
| 6 vegetation | 705,891,578 | 0.04908 |
| 7 vehicle | 70,441,204 | 0.05534 |
| 8 terrain | 145,119,651 | 0.05321 |

## Per-epoch history

VAL is computed on the kernel entering that epoch (pre-train); TRAIN is that epoch's mean. Both are GT-reproduction metrics, not held-out.

| epoch | val loss | val mIoU | train loss | train mIoU | train iters |
|---|---|---|---|---|---|
| 0 | 0.0609 | 0.8305 | 0.0752 | 0.8213 | 10001 |
| 1 | 0.0439 | 0.8383 | 0.0751 | 0.8212 | 10001 |
| 2 | 0.0439 | 0.8382 | 0.0751 | 0.8212 | 10001 |
| 3 | 0.0439 | 0.8383 | 0.0751 | 0.8212 | 10001 |
| 4 | 0.0442 | 0.8379 | 0.0751 | 0.8212 | 10001 |
| final | 0.0440 | 0.8381 | — | — | — |

### Learned per-class `ell` (final)

| class | ell |
|---|---|
| 0 unlabeled | 0.2000 |
| 1 road | 0.2000 |
| 2 sidewalk | 0.2000 |
| 3 parking | 0.2000 |
| 4 building | 0.3339 |
| 5 fence | 0.3364 |
| 6 vegetation | 0.3419 |
| 7 vehicle | 0.3818 |
| 8 terrain | 0.1954 |

**Total training time:** 8327s (2.31 h)

_Status: training COMPLETE._
