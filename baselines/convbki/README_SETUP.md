# Conv-BKI baseline — reproduction guide

This document is the hand-off for the Conv-BKI baseline. It records the
exact environment, data layout, commands, and run artifacts used to
produce every Conv-BKI number we have, so the work can be re-run from
scratch on this machine (or rebuilt on another).

It is written against what is actually in the repo and on disk as of the
KITTI-360 runs, not generic instructions. Where a path or value is
specific to this machine (`donceykong`'s box), that is called out.

Companion docs:

- `README.md` — short Option-A overview (the original pretrained path).
- `DEBUG_REPORT.md` — the running incident log: the 29-channel mismatch,
  the hard-vs-soft input root cause, per-combo pretrained numbers, and
  the MCD staging blocker.
- `Weights/kitti360_9class_gt/RUN.md` + `run_meta.json` — the exact
  GT-retraining recipe, splits, class weights, per-epoch history, timings.
- `results/retrained_vs_pretrained_kitti360.md` — the headline results
  table (retrained Option B vs pretrained Option A) with confound notes.

---

## 0. TL;DR of what exists

- **Two baseline modes.** Option A = the authors' pretrained 20-class
  SemanticKITTI kernel (`ConvBKI_PC_02_V`, `filters1.pt`) projected to our
  9-class common taxonomy. Option B = a native 9-class kernel retrained on
  KITTI-360 GT (`Weights/kitti360_9class_gt/filters5.pt`).
- **Done:** all four KITTI-360 combos (seq 0000 / 0009 × ID / OOD), in
  both modes. Numbers in `results/retrained_vs_pretrained_kitti360.md` and
  `results/per_combo/*.json`.
- **Staged but not trained:** the MCD GT-retraining tree (24,095 scans,
  11 tuhh/ntu sequences) under
  `/home/sandilya/convbki_train_workspace/staging_train_gt_mcd`. Blocked
  on shared-GPU availability.
- **Not yet runnable as-is:** the four MCD eval combos. See
  [§7 Current state](#7-current-state) for the specific blockers.

---

## 1. Environment setup

### 1.1 Conda env

The runtime is a standalone miniforge env named `convbki`. It does **not**
use the submodule's own `NeuralBKI/environment.yml` (that file is old and
pulls in ROS/rospy, which we deliberately avoid — the scripts reimplement
the few rospy-tainted helpers; see `train_convbki_9class.py` and
`run_convbki.py` header comments).

Miniforge lives at `~/miniforge3`. The env was built as:

```bash
# install/init miniforge if not present, then:
conda create -n convbki python=3.10
conda activate convbki

# PyTorch 2.5.1 + CUDA 12.1 wheels (the CUDA runtime ships in the wheels;
# no system CUDA toolkit is required, only an NVIDIA driver):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# the rest of the runtime:
pip install numpy==2.2.6 PyYAML==6.0.3 scipy==1.15.3 scikit-learn==1.7.2 tqdm==4.68.1
```

Verified versions in the live env (`pip freeze`, key packages):

| package      | version       |
| ------------ | ------------- |
| python       | 3.10.20       |
| torch        | 2.5.1+cu121   |
| torchvision  | 0.20.1+cu121  |
| numpy        | 2.2.6         |
| scipy        | 1.15.3        |
| scikit-learn | 1.7.2         |
| PyYAML       | 6.0.3         |
| tqdm         | 4.68.1        |
| triton       | 3.1.0         |

The CUDA stack is pulled in transitively as `nvidia-*-cu12` wheels
(cuBLAS 12.1, cuDNN 9.1, etc.). Sanity check:

```bash
conda activate convbki
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# -> 2.5.1+cu121 12.1 True
```

**GPU on this machine:** single NVIDIA RTX 4070 Ti SUPER (16 GB),
driver 535.309.01. The ±50 m eval grid (501×501×27) and the ±40 m train
grid (400×400×26) both fit; 16 GB is the working assumption for the
streaming eval rewrite (see [§6](#6-known-confounds--gotchas)).

> This is a **shared GPU** — the project lead's EDL job and these
> Conv-BKI jobs cannot run concurrently. The MCD-eval/retrain helper
> scripts under `~/convbki_train_workspace/` (e.g. `gpu_wait_*.sh`) exist
> to block until the GPU frees up before launching.

### 1.2 NeuralBKI submodule

The model code (`Models/ConvBKI.py`, `Models/mapping_utils.py`) is the
upstream NeuralBKI repo, vendored as a git submodule:

```bash
cd /home/sandilya/GEO-BKI
git submodule update --init baselines/convbki/NeuralBKI
```

- URL: `https://github.com/UMich-CURLY/NeuralBKI.git`
- Pinned commit: `47b42ff` (on `main`).
- The pretrained Option-A weights ship inside the submodule at
  `NeuralBKI/Models/Weights/ConvBKI_PC_02_V/filters{0,1,2}.pt`. Option A
  uses `filters1.pt` (epoch 1), matching upstream's `LOAD_EPOCH`.

Our scripts `chdir` into `NeuralBKI/` before importing model classes,
because `NeuralBKI/Data/SemanticKitti.py` reads
`Config/semantic_kitti.yaml` relative to CWD at import time. This is
handled automatically by the scripts — just run them from anywhere with
the env active.

---

## 2. Data layout

### 2.1 Data root

All scripts resolve the dataset tree from `OSM_BKI_DATA_ROOT` (a YAML
`data_root: null` falls back to this env var, and `--data-root` overrides
both):

```bash
export OSM_BKI_DATA_ROOT=/media/donceykong/donceys_data_ssd/datasets
```

> **Machine-specific.** On this box the data lives on
> `donceys_data_ssd`. On the earlier machine it was
> `/media/sgarimella34/hercules-collect3/datasets` (that path appears
> throughout `DEBUG_REPORT.md`). Only the prefix changes; the tree
> structure below is identical.

> **`/media` is off-limits for writes.** The data drive is treated as
> read-only. All bulk staging / inference / eval output is redirected to
> `/home/sandilya/convbki_train_workspace/...` via the configs and the
> `--eval-dir` override (see [§5](#5-evaluation)). `stage_train_gt.py`
> hard-enforces this (it refuses a `--staging-root` outside
> `/home/sandilya/`).

### 2.2 KITTI-360

```
$OSM_BKI_DATA_ROOT/kitti360/2013_05_28_drive_<SEQ>_sync/
    velodyne_points/data/<stem>.bin     # raw scans, float32 x4 (x,y,z,intensity)
    velodyne_poses.txt                  # per-frame lidar->world (frame_idx + 12/16 floats)
    gt_labels/<stem>.bin                # raw KITTI-360 GT, uint32 (low 16 bits = class)
    inferred_labels/<variant>/<stem>.bin  # CENet per-point softmax, uint16 (float16) x K channels
```

`<stem>` is a 10-digit zero-padded scan id. Sequences present on this
machine: `0000, 0002, 0003, 0004, 0005, 0006, 0007, 0009, 0010`
(no `0008`).

CENet `inferred_labels` variants used:

| variant                     | channels | trained on | used for                              |
| --------------------------- | -------- | ---------- | ------------------------------------- |
| `cenet_kitti360_softmax`    | 45       | KITTI-360  | **ID** input (both modes)             |
| `cenet_mcd_terrain_softmax` | 30       | MCD (full) | **OOD** input, Option B + reruns      |
| `cenet_mcd_softmax`         | 29       | MCD        | original Option-A OOD (see note below)|

> The 29-channel `cenet_mcd_softmax` was used for the *original* Option-A
> OOD numbers (`DEBUG_REPORT.md` combos 2 & 4, mIoU 0.1184 / 0.1069). It
> drops MCD class 29 (terrain), so OOD terrain IoU was forced to 0. **It
> is not present on this machine** — only the 30-channel terrain variant
> is. The Option B OOD evals and the controlled pretrained reruns all use
> `cenet_mcd_terrain_softmax` (terrain channel present → terrain can score
> > 0). This is one of the confounds in [§6](#6-known-confounds--gotchas).

### 2.3 MCD

```
$OSM_BKI_DATA_ROOT/mcd/
    tuhh/hhs_calib.yaml                 # body->lidar extrinsic, hhs platform (tuhh + kth)
    ntu/atv_calib.yaml                  # body->lidar extrinsic, atv platform (ntu)
    kth/hhs_calib.yaml                  # copy of the hhs extrinsic for kth eval seqs
    <group>/<seq>/                      # group ∈ {tuhh, ntu, kth}; seq e.g. tuhh_day_04, kth_day_09
        lidar_bin/data/<stem>.bin       # raw scans
        pose_inW.csv                    # body->world (num,timestamp,x,y,z,qx,qy,qz,qw)
        gt_labels_terrain/<stem>.bin    # raw MCD GT, uint32 0..29 (terrain at 29, no 0xFFFF mask)
        inferred_labels/<variant>/...   # CENet softmax (eval seqs only)
```

- MCD scans are `lidar_bin/data` (not `velodyne_points/data`); poses are a
  quaternion CSV; GT is `gt_labels_terrain` (the terrain-bearing variant,
  raw labels `0..29`). The lidar→world transform is
  `body_to_world @ inv(body_to_lidar)`, matching `mcd_util.h`.
- **Calib paths differ by platform.** tuhh and kth share the *hhs*
  extrinsic; ntu uses *atv*. `stage_train_gt.py` auto-selects by group
  prefix (`tuhh`/`kth` → `hhs_calib.yaml`, `ntu` → `atv_calib.yaml`).
- Train groups for retraining: `tuhh/`, `ntu/`. Eval groups: `kth/`
  (`kth_day_09`, `kth_night_05`).

### 2.4 Common 9-class taxonomy and collapse maps

The shared taxonomy is `config/datasets/labels_common.yaml`:

```
0 unlabeled (ignore)   1 road    2 sidewalk   3 parking   4 building
5 fence                6 vegetation          7 vehicle    8 terrain
```

Three raw→common collapse maps live in that file and are consumed by
`scripts/label_mappings.py`:

- `kitti360_to_common` — KITTI-360 raw (0..38) → common. e.g. road 7→1,
  sidewalk 8→2, parking 9→3, building/wall/bridge/tunnel/garage→4,
  fence/guardrail/gate→5, vegetation 21→6, all vehicles→7, terrain 22→8.
- `mcd_to_common` — MCD raw (0..29) → common. e.g. building 2→4, fence
  0/7→5, parkinglot 13→3, road 16→1, sidewalk 18→2, treetrunk 24 +
  vegetation 25→6, all vehicles 26/27/28→7, terrain 29→8.
- `semkitti_to_common` — SemanticKITTI raw → common, used on the
  **prediction** side of Option A (the pretrained kernel outputs SemKITTI
  training classes, which are projected back to common at convert time).

For Option A, `label_mappings.py` also builds the SemKITTI 20-class
training space and the channel→common routing tables (`K` channels
inferred from the first CENet file: 45 for KITTI-360, 30/29 for MCD).

---

## 3. The two baseline modes

| | **Option A — pretrained** | **Option B — GT-retrained** |
| --- | --- | --- |
| kernel | authors' `ConvBKI_PC_02_V` `filters1.pt` (20-class SemKITTI) | `Weights/kitti360_9class_gt/filters5.pt` (native 9-class) |
| label space | 20-class SemKITTI, projected to 9 at convert time | native 9-class common throughout |
| config flag | `num_classes: 20`, `native_common` unset | `num_classes: 9`, `native_common: true` |
| input | hard one-hot (see below) | hard one-hot |
| when used | zero-shot baseline; no training needed | the stronger model; trained on KITTI-360 GT |

**Both modes use hard (one-hot) input** (`hard_input: true` in every
config). This is not cosmetic — feeding the pretrained kernel *soft*
20-class input made parking dominate 99.4% of voxels (per-class kernel
sums are wildly asymmetric: parking 26.7 vs building 4.8, multiplied
against the ~uniform 11% floor mass the aggregation leaks). The full
root-cause is in `DEBUG_REPORT.md` Tasks 1–4. Staging writes
`predictions_hard/<stem>.label` (uint32 argmax) and `run_convbki.py`
reads it.

Option A is the no-training reference. Option B is the headline model and
is what goes next to OSM-BKI in the paper's table. The single number for
Table I (retrained Conv-BKI OOD): **seq0000 0.1509 / seq0009 0.1661**.

---

## 4. GT-based retraining (Option B)

This is the path that produced `Weights/kitti360_9class_gt/`. The
per-scan network "prediction" is the **ground-truth one-hot** common
label; the target is the same GT. The only learnable parameters are the
9 per-class kernel length-scales (`ell`); the kernel learns the spatial
extent per class that best reproduces GT at the voxel level after
multi-frame accumulation. Native 9-class — no SemKITTI projection.

### 4.1 Stage the GT training tree

`stage_train_gt.py` builds a native-9-class GT-as-predictions tree:
`labels/<stem>.label` = GT remapped to common (uint32 0..8);
`predictions_hard/<stem>.label` = a relative symlink to the same label
file (input and target are byte-identical in this mode, halving disk).

**KITTI-360 (what was actually run):**

```bash
conda activate convbki
cd /home/sandilya/GEO-BKI/baselines/convbki/scripts

python stage_train_gt.py \
  --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt \
  --sequences 0002 0004 0005 0006 0007 0010 \
  --every-nth 2 \
  --data-root /media/donceykong/donceys_data_ssd/datasets

# val sequence, staged at FULL density (no --every-nth):
python stage_train_gt.py \
  --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt \
  --sequences 0003 \
  --data-root /media/donceykong/donceys_data_ssd/datasets
```

- `--every-nth 2`: keep every 2nd scan on the 6 training sequences
  (20,001 of ~40,005 staged-eligible scans). Val seq 0003 is staged at
  full density (988 scans).
- The script writes `manifest.json` and `class_counts.json` (the
  per-common-class point counts used to build the NLL weights at train
  time) into the staging root.
- It refuses any `--staging-root` outside `/home/sandilya/`.

**MCD (`--dataset mcd`, staged but not yet trained):**

```bash
python stage_train_gt.py --dataset mcd \
  --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt_mcd \
  --sequences tuhh_day_02 tuhh_day_03 tuhh_day_04 tuhh_night_07 tuhh_night_08 \
              ntu_day_01 ntu_day_02 ntu_day_10 ntu_night_04 ntu_night_08 ntu_night_13 \
  --data-root /media/donceykong/donceys_data_ssd/datasets
```

This is already staged: 24,095 scans across the 11 sequences above (GT via
`mcd_to_common`, per-group calib auto-selected). It has **not** been
trained yet (see [§7](#7-current-state)).

### 4.2 Train the native 9-class kernel

The exact command from `Weights/kitti360_9class_gt/RUN.md`:

```bash
conda activate convbki
cd /home/sandilya/GEO-BKI/baselines/convbki/scripts

python train_convbki_9class.py \
  --staging-root /home/sandilya/convbki_train_workspace/staging_train_gt \
  --train-seqs 0002 0004 0005 0006 0007 0010 \
  --val-seqs 0003 \
  --out-dir /home/sandilya/GEO-BKI/baselines/convbki/Weights/kitti360_9class_gt \
  --epochs 5 --batch 2 --num-frames 10 --num-workers 4 \
  --grid train40 --seed 42
```

Recipe (all defaults baked into the script, matching upstream
`Config/ConvBKI_PerClass.yaml`):

- **Loss:** `NLLLoss(weight=inverse_log_freq, ignore_index=0)` — class 0
  (unlabeled) is ignored; per-class weights are `1/log(count+eps)` from
  the staged `class_counts.json`.
- **Optimizer:** Adam, lr `0.007`, betas `(0.9, 0.999)`.
- **Scheduler:** `ExponentialLR(gamma=0.96)`.
- **Epochs:** 5. **Seed:** 42. **Batch:** 2. **num_frames:** 10.
- **Kernel:** per-class sparse, `filter_size=5`, `ell` init `0.5`
  (learnable). `from_continuous=False` (hard one-hot input read from
  `predictions_hard/`).
- **Grid:** `train40` (±40 m, 400×400×26, 0.2 m voxel). The eval grid is
  ±50 m but **also** 0.2 m/voxel, and the sparse kernel depends only on
  voxel size — so a kernel trained at ±40 m transfers exactly to the ±50 m
  eval grid while using less VRAM/time. (`--grid eval50` trains on the
  eval grid directly if ever needed.)

Class weights actually used (KITTI-360 training GT):

| class | point count | NLL weight |
| --- | --- | --- |
| 0 unlabeled (ignored) | 273,385,737 | 0.00000 |
| 1 road | 465,457,364 | 0.05010 |
| 2 sidewalk | 265,407,367 | 0.05155 |
| 3 parking | 76,341,682 | 0.05509 |
| 4 building | 348,773,904 | 0.05084 |
| 5 fence | 37,094,475 | 0.05738 |
| 6 vegetation | 705,891,578 | 0.04908 |
| 7 vehicle | 70,441,204 | 0.05534 |
| 8 terrain | 145,119,651 | 0.05321 |

**Outputs** land in `--out-dir`:
`filters0.pt`..`filters5.pt` (pre-epoch snapshots + final), plus
`train_history.json`, `run_meta.json`, `RUN.md`. **`filters5.pt` is the
final checkpoint** referenced by the Option-B eval configs.

**Run facts:** total training time 8,327 s (2.31 h) on the 4070 Ti SUPER;
train mIoU ≈ 0.821, val mIoU (seq 0003, GT-reproduction) ≈ 0.838. Learned
`ell` widens for the tall/sparse classes (building 0.33, fence 0.34,
vegetation 0.34, vehicle 0.38) and stays at the 0.20 floor for the
ground classes. Full per-epoch table in `RUN.md`.

**Monitoring (optional).** `monitor_train.py` watches a running train log,
refreshes `RUN.md` each epoch, and enforces a divergence guard (SIGTERM +
`DIVERGED.flag` if loss goes NaN or val loss rises 2+ consecutive
epochs). It is meant to be launched in the background alongside training.

---

## 5. Evaluation

A single combo runs as a 4-stage pipeline. The retrained (native 9-class)
runs are wrapped by `scripts/run_eval_native.sh`:

```bash
# run_eval_native.sh <config.yaml> <eval_out_dir> <per_combo_json> [keyframe_dir]
conda activate convbki
export OSM_BKI_DATA_ROOT=/media/donceykong/donceys_data_ssd/datasets
cd /home/sandilya/GEO-BKI/baselines/convbki

bash scripts/run_eval_native.sh \
  configs/kitti360_seq0000_id_retrained.yaml \
  /home/sandilya/convbki_train_workspace/eval/kitti360_seq0000_id_retrained/evaluations_convbki \
  results/per_combo/kitti360_seq0000_id_retrained.json
```

That wrapper runs, in order:

1. **`stage_inputs.py <config>`** — builds the staging tree under the
   config's `staging_root` (velodyne symlinks, `labels/` = GT→common,
   `predictions_hard/` = CENet argmax→common, `poses.txt`, `calib.txt`
   with `Tr=identity`). With `native_common: true`, labels and hard
   predictions are written in 9-class space; the `predictions_softmax/`
   write is skipped whenever `hard_input: true` (saves ~60 GB/combo).
   `K` (channel count) is inferred from the first CENet file.
2. **`run_convbki.py <config>`** — runs Conv-BKI inference over the
   staging tree, writing per-frame `predictions/<stem>.label` (uint32).
   Loads `weights.path` (the retrained checkpoint) or `weights.{dir,epoch}`
   (the pretrained kernel). Logs to `logs/<experiment>_<ts>.log`.
   Throughput ≈ 2.5–2.6 scans/s on the 4070 Ti SUPER.
3. **`convert_outputs.py <config> --eval-dir <dir> [--keyframe-stems-from D]`**
   — pairs each prediction with its raw GT and writes
   `<eval_dir>/<stem>.txt` (two columns: `gt_common pred_common`). With
   `native_common: true`, pred→common is identity; otherwise it projects
   SemKITTI 20-class→common. `--eval-dir` redirects output off the
   read-only data drive. `--keyframe-stems-from` gates to an existing
   keyframe set (omitted for the all-GT-scan basis we used — see below).
   `--copy-raw` also archives `.label` files to `raw_predictions/`.
4. **`compute_eval_numbers.py <eval_dir> --out-json <json>`** — streams
   the `.txt` files into an N×N confusion matrix and prints per-class
   IoU / Acc / Prec and mIoU / mAcc (over classes present in GT, class 0
   dropped), and writes the per-combo JSON.

### 5.1 The four KITTI-360 combos

Configs (Option B, retrained):
`configs/kitti360_seq{0000,0009}_{id,ood}_retrained.yaml`. The companion
pretrained Option-A configs are `kitti360_seq{0000,0009}_{id,ood}.yaml`
(and the controlled same-input reruns
`kitti360_seq0000_{id,ood}_pretrained_rerun.yaml`).

- **ID** combos feed `cenet_kitti360_softmax` (45-channel, in-domain).
- **OOD** combos feed `cenet_mcd_terrain_softmax` (30-channel,
  cross-domain, terrain present).
- All four retrained evals use the **all-GT-scan basis** (every scan with
  GT: 10,483 for seq0000, 13,164 for seq0009) — `convert_outputs.py` is
  run *without* `--keyframe-stems-from` because the S-BKI keyframe sets
  are not on this machine (the `evaluations/` dirs are empty).

The seq0000 same-basis reruns (pretrained ID, retrained OOD, pretrained
OOD on identical 30-channel input) were driven together by
`scripts/run_seq0000_reruns.sh`, since they share the one GPU.

### 5.2 The OOD terrain-variant note

The retrained (and rerun) OOD configs route the **30-channel**
`cenet_mcd_terrain_softmax` through `mcd_to_common`
(`build_source_channel_to_common('mcd', cfg, K=30)`). Channel index =
MCD raw label 0..29, with terrain at channel 29 — so unlike the original
Option-A 29-channel OOD, terrain mass enters the pipeline and OOD terrain
IoU can be > 0 (it lands ~0.05–0.07). Empirically ~24/30 channels fire on
KITTI-360 sweeps; all 8 semantic common classes are reachable.

### 5.3 How per-combo JSONs map to the paper table

Each `results/per_combo/<combo>.json` holds the arrays
`per_class_iou`, `per_class_accuracy`, `per_class_precision`,
`gt_count`, `pred_count`, plus `miou` / `macc` and the confusion matrix.
The table cells in `results/retrained_vs_pretrained_kitti360.md` are read
directly from these:

- `miou` → the headline mIoU cell for that combo.
- `per_class_iou[i]` → the per-class IoU rows.

> **Index offset (gotcha):** the per-class arrays are length **8** and
> drop class 0, so position `i` is common class `i+1`. `per_class_iou[0]`
> is **road**, `per_class_iou[7]` is **terrain**. Always align via the
> JSON's `class_names` / `semantic_classes` fields — do **not** assume
> array position equals the common class id.

Current KITTI-360 results (from
`results/retrained_vs_pretrained_kitti360.md`):

| combo | retrained (Opt B) | pretrained (Opt A) | basis re / pre |
| --- | --- | --- | --- |
| seq0000 ID  | 0.5593 | 0.5631 | 10483 / 4433 |
| seq0009 ID  | 0.6852 | 0.5958 | 13164 / 5566 |
| seq0000 OOD | 0.1509 | 0.1184 | 10483 / 4433 † |
| seq0009 OOD | 0.1661 | 0.1069 | 13164 / 5566 † |

† OOD rows also differ in input channels (30-ch terrain-present retrained
vs 29-ch terrain-absent original pretrained) — see [§6](#6-known-confounds--gotchas).

---

## 6. Known confounds / gotchas

1. **Hard vs soft input.** The pretrained kernel must be fed hard
   one-hot labels. Soft 20-class input collapses to ~99% parking because
   the per-class kernel sums are asymmetric (parking 26.7 vs building 4.8)
   and the C++-faithful aggregation leaks ~uniform floor mass into every
   common class. Every config sets `hard_input: true`; the soft path is
   preserved in code but unused. (`DEBUG_REPORT.md` Tasks 1–4.)

2. **OOD basis + input-channel confound.** The original Option-A OOD
   numbers (`DEBUG_REPORT.md`: 0.1184 / 0.1069) used the **29-channel**
   `cenet_mcd_softmax` (terrain absent) on the **S-BKI keyframe basis**.
   The Option-B OOD numbers use the **30-channel**
   `cenet_mcd_terrain_softmax` (terrain present) on the **all-GT-scan
   basis**. So an Option-B-vs-original-Option-A OOD delta differs in
   *both* basis and input — it is indicative, not a controlled model-only
   A/B. The honest decomposition (terrain gain is an input artifact, the
   vegetation/building gain is the real model improvement) is in
   `results/retrained_vs_pretrained_kitti360.md`. The
   `*_pretrained_rerun` configs exist precisely to give a same-input,
   same-basis A/B where a clean model-only delta is wanted. **ID rows are
   clean** (same `cenet_kitti360_softmax` input; differ only in basis).

3. **Streaming eval for OOM.** `compute_eval_numbers.py` accumulates an
   N×N confusion matrix incrementally instead of loading all points into
   memory. The earlier `np.loadtxt` + sklearn-over-all-points approach
   OOM'd at ~454M points. Every reported metric is recovered exactly from
   the confusion matrix, so numbers are unchanged — but if you revert to
   a load-everything approach it will blow up on the full-basis evals.

4. **Per-class array column order.** As in [§5.3](#53-how-per-combo-jsons-map-to-the-paper-table):
   the JSON per-class arrays are 0-indexed over common classes **1..8**
   (class 0 dropped). Position 0 = road … position 7 = terrain. Reading
   `per_class_iou[6]` expecting vegetation, or `[8]` for terrain, is the
   easy mistake — vegetation is `[5]`, vehicle is `[6]`, terrain is `[7]`.
   Use `class_names` to be safe.

5. **Off-limits data drive.** `/media/.../datasets` is read-only here.
   Configs write staging/inference under `/home/sandilya/...` and eval is
   redirected with `--eval-dir`; `aggregate_results.py` had a latent bug
   (`REPO` one `dirname` too shallow) that misfiled output under
   `baselines/baselines/...` and silently read OOD numbers into ID rows —
   fixed (four `dirname` levels). If you re-run aggregation, sanity-check
   that ID/OOD snapshots resolve correctly.

---

## 7. Current state

**Done — KITTI-360, all 4 combos, both modes:**

| combo | Option A (pretrained) | Option B (retrained) |
| --- | --- | --- |
| seq0000 ID  | 0.5631 | 0.5593 |
| seq0000 OOD | 0.1184 | 0.1509 |
| seq0009 ID  | 0.5958 | 0.6852 |
| seq0009 OOD | 0.1069 | 0.1661 |

Artifacts: `results/retrained_vs_pretrained_kitti360.md`,
`results/per_combo/kitti360_seq{0000,0009}_{id,ood}_retrained.json`,
trained kernel `Weights/kitti360_9class_gt/filters5.pt`.

**Staged, not trained — MCD GT retraining.** The MCD training tree is
fully staged at
`/home/sandilya/convbki_train_workspace/staging_train_gt_mcd` (24,095
scans, 11 tuhh/ntu sequences; `manifest.json` + `class_counts.json`
present). Training has **not** been run — blocked on shared-GPU
availability. To train, point `train_convbki_9class.py` at that staging
root with the same recipe as [§4.2](#42-train-the-native-9-class-kernel),
choosing train/val splits from the 11 staged sequences and a fresh
`--out-dir` (e.g. `Weights/mcd_9class_gt`).

**Blocked — MCD evaluation.** The four MCD eval combos cannot run as-is.
The kth eval sequences themselves (`kth_day_09`, `kth_night_05`) *are*
present on this machine with `gt_labels_terrain` and `inferred_labels`,
but the existing MCD configs are still in their original Option-A form and
do not match what's on disk:

- `mcd_kth_day_09_id.yaml` / `mcd_kth_night_05_id.yaml` point at
  `inferred_labels/cenet_mcd_softmax`, which is **absent** — only
  `cenet_mcd_terrain_softmax` is present (same situation as KITTI-360
  OOD). The ID configs would need re-pointing to the terrain variant.
- All four MCD configs set `calibration_file:
  ${OSM_BKI_DATA_ROOT}/mcd/hhs_calib.yaml`, which **does not exist** at
  that path — the calib files are at `mcd/kth/hhs_calib.yaml`,
  `mcd/tuhh/hhs_calib.yaml`, `mcd/ntu/atv_calib.yaml`. This is the
  `FileNotFoundError` recorded in `DEBUG_REPORT.md` ("5-8 STAGING
  FAILURE"); on this machine the fix is to point the config at
  `mcd/kth/hhs_calib.yaml`.
- There are **no** retrained (native-9-class) MCD eval configs yet, and
  no trained MCD kernel — so even once the above are fixed, MCD can only
  run Option A until an MCD kernel is trained.

In short: re-running KITTI-360 from scratch is fully reproducible with the
commands above; bringing MCD online needs (a) the GPU to train the staged
MCD kernel, and (b) updated MCD eval configs (terrain CENet variant +
correct calib path, plus `*_retrained` variants once a kernel exists).
