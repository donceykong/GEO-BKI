# Conv-BKI baseline

This directory drives the Conv-BKI baseline using the authors' pretrained
20-class SemanticKITTI weights (`ConvBKI_PC_02_V`, `filters1.pt`),
projected to our 9-class common taxonomy. No retraining yet — this is
Option A of the baseline plan. See:

- `convbki_baseline_plan.md` at the repo root for the design.
- `DEBUG_REPORT.md` in this directory for the running incident log
  (29-channel CENet handling, hard vs. soft input, smoke results,
  per-combo numbers).

## Setup

1. Initialize the NeuralBKI submodule:
   ```
   git submodule update --init baselines/convbki/NeuralBKI
   ```
2. Export the data root pointing at the dataset tree
   (`<root>/kitti360/...` and `<root>/mcd/...`):
   ```
   export OSM_BKI_DATA_ROOT=/path/to/datasets
   ```
3. The `staging/`, `nbki_runs/`, and `raw_predictions/` directories are
   bulk artifacts (tens of GB per combo) and live outside git. On this
   machine they are symlinks to an external drive; reproduce by either
   creating equivalent symlinks or letting the scripts create local
   directories. `.gitignore` keeps them out of the repo regardless.

## Running a single sequence

The three scripts pipe together via per-experiment YAMLs in `configs/`:

```
PY=python  # or your env-specific python (Where2comm on this machine)

# 1. Stage: build the Conv-BKI input tree (velodyne symlinks, GT in
#    SemKITTI training-class space, hard-label predictions, poses,
#    calib).
$PY scripts/stage_inputs.py configs/<combo>.yaml

# 2. Inference: run Conv-BKI over the staging tree and write per-frame
#    .label predictions.
$PY scripts/run_convbki.py configs/<combo>.yaml

# 3. Convert: produce <gt> <pred> .txt files in 9-class common space
#    under <data_root>/<dataset>/<sequence>/evaluations/convbki/, gated
#    to the S-BKI keyframe set, and copy raw .label files into
#    raw_predictions/ for archival.
$PY scripts/convert_outputs.py configs/<combo>.yaml \
    --keyframe-stems-from <data_root>/<dataset>/<sequence>/evaluations/<sbki_dir> \
    --copy-raw

# 4. Numbers: per-class IoU / accuracy / precision and mIoU/mAcc.
$PY scripts/compute_eval_numbers.py \
    <data_root>/<dataset>/<sequence>/evaluations/convbki
```

## Full sweep

`scripts/run_all.sh` runs all 8 combos sequentially:

- Skips staging when `<staging>/manifest.json` already exists.
- After each combo passes the 3% mIoU floor it deletes
  `staging/<combo>/sequences/` and `nbki_runs/<combo>/sequences/` to
  keep the drive uncluttered; `raw_predictions/` is preserved as the
  per-frame archive.
- Hard-aborts the rest of the sweep if any combo mIoU < 3% and appends
  the abort details to `DEBUG_REPORT.md`.
- After the loop, calls `aggregate_results.py` to write
  `results/{raw_numbers.json, in_domain.md, cross_domain.md}`.

Launch it with `nohup` (it runs for hours):

```
nohup bash scripts/run_all.sh > /dev/null 2>&1 &
```

The orchestrator log goes to `logs/run_all_<TS>.log`; per-combo logs to
`logs/<combo>_<TS>.log`.

## The 8 configs

| combo                  | dataset  | sequence                       | ID CENet                   | OOD CENet                   |
| ---------------------- | -------- | ------------------------------ | -------------------------- | --------------------------- |
| kitti360_seq0000_id    | kitti360 | 2013_05_28_drive_0000_sync     | cenet_kitti360_softmax     |                             |
| kitti360_seq0000_ood   | kitti360 | 2013_05_28_drive_0000_sync     |                            | cenet_mcd_softmax           |
| kitti360_seq0009_id    | kitti360 | 2013_05_28_drive_0009_sync     | cenet_kitti360_softmax     |                             |
| kitti360_seq0009_ood   | kitti360 | 2013_05_28_drive_0009_sync     |                            | cenet_mcd_softmax           |
| mcd_kth_day_09_id      | mcd      | kth_day_09                     | cenet_mcd_softmax          |                             |
| mcd_kth_day_09_ood     | mcd      | kth_day_09                     |                            | cenet_kitti360_softmax      |
| mcd_kth_night_05_id    | mcd      | kth_night_05                   | cenet_mcd_softmax          |                             |
| mcd_kth_night_05_ood   | mcd      | kth_night_05                   |                            | cenet_kitti360_softmax      |

ID = the dataset's own CENet variant. OOD = the other dataset's CENet
(cross-domain). The grid is ±50 m at 0.2 m voxel (501 × 501 × 27,
paper-native).

## Hard input

All 8 configs set `hard_input: true`. The pretrained `ConvBKI_PC_02_V`
kernel was trained with one-hot inputs; feeding it the soft 20-class
distribution from our routing caused the parking column (per-class
kernel sum ~26.7 vs. building ~4.8) to dominate every voxel from the
~uniform 11% floor mass the C++ aggregation produces. Hard labels
remove the floor mass entirely. See `DEBUG_REPORT.md` for the full
investigation.

## Status

- Option A (pretrained, no retraining) — pipeline working. Numbers
  land in `results/` after `run_all.sh` finishes.
- Option B (retrain in 9-class common) — not attempted yet. Pending
  access to KITTI-360 sequences 0002–0010 and the MCD `tuhh/` and
  `ntu/` sequences for a proper train split.
