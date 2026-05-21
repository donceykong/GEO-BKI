# Conv-BKI Baseline Integration Plan (OSM-BKI RA-L)

**Purpose.** Add Conv-BKI (Wilson et al., T-RO 2024; repo *NeuralBKI*) as a third comparison row alongside S-BKI and the raw-softmax baseline in Tables I and II of the RA-L submission. The mapping stage is what's being compared, so Conv-BKI must consume the **same CENet predictions** we feed into S-BKI/OSM-BKI and be evaluated against the **same per-point GT** in the same **9-class common taxonomy** (8 classes + unlabeled).

This document is a plan only — no code is to be written until the open questions at the bottom are answered.

---

## 1. What I found in our repo

### 1.1 Evaluation interface

Each mapping run writes one `.txt` file per evaluated scan:

- Path: `<DATA_ROOT>/<dataset>/<sequence>/evaluations/<EVAL_PREFIX>/<scan_id>.txt`
- Scan ID: 10-digit zero-padded (`0000000123.txt`).
- File content: one line per query point, format `<gt_label> <pred_label>` — both already in the **9-class common taxonomy** (0..8, where 0 = unlabeled, 1..8 = road / sidewalk / parking / building / fence / vegetation / vehicle / terrain — order per [config/datasets/labels_common.yaml](config/datasets/labels_common.yaml)).
- Writer: [include/osm_bki/dataset_devkit/mcd_util.h](include/osm_bki/dataset_devkit/mcd_util.h#L1616-L1697) `query_scan()`. For each pose, it loads the corresponding scan + GT label file, transforms scan points into the map frame, looks up each point in the BKI octree, and writes `(gt_in_common, pred_in_common)`.
- Reader (the evaluator that produces the table numbers): [eval/osm_bki_eval.ipynb](eval/osm_bki_eval.ipynb). It globs `*.txt` under `EVALUATION_DIR`, concatenates into `gt_all`, `pred_all`, masks out class 0, computes per-class IoU via `jaccard_score(labels=1..8)`, per-class accuracy (recall) and precision, mIoU, mAcc, and a 8×8 normalized confusion matrix.
- The baseline (no-mapping) evaluator [eval/baseline_evaluation.ipynb](eval/baseline_evaluation.ipynb) does the same thing but constructs `(gt, pred)` per point on-the-fly from raw CENet softmax — useful as a reference for the channel→common LUT we need to mirror.

**This is the only interface Conv-BKI's output has to match.** If we produce per-scan `.txt` files with `(gt_common, pred_common)` rows in the right directory, the existing notebook produces Table I/II numbers verbatim.

### 1.2 What CENet predictions look like on disk

Per-sequence multiclass file: `<sequence>/inferred_labels/cenet_<TAXONOMY>_softmax/<scan_id>.bin`

- **Binary format:** float16, length = `n_points * n_classes`, row-major (point-major), no header. Loaded in [mcd_util.h L1885-L2050](include/osm_bki/dataset_devkit/mcd_util.h#L1885-L2050).
- **Channels:** indexed by the *network output index* (0..n_classes-1). The mapping back to a raw dataset label is in `<TAXONOMY>.yaml`'s `learning_map_inv`. Then `<TAXONOMY>_to_common` (in [labels_common.yaml](config/datasets/labels_common.yaml)) takes the raw label to the common 0..8 index.
- **Per-dataset assignment** (from [config/methods/](config/methods)):
  - KITTI-360 (`kitti360.yaml`): `inferred_labels/cenet_mcd_softmax`, `inferred_labels_key: mcd`. So on KITTI-360 sequences, CENet was trained on MCD taxonomy and outputs MCD channels; we then apply `mcd_to_common`.
  - MCD (`mcd.yaml`): `inferred_labels/cenet_kitti360[_softmax]`, `inferred_labels_key: kitti360`. CENet on MCD sequences predicts kitti360-taxonomy channels; we apply `kitti360_to_common`.
- Note: the file path suffix sometimes lacks the `_softmax` tail when `inferred_use_multiclass=false`. For Conv-BKI we will always use multiclass.

### 1.3 Where scans, poses, and GT live

Common per-sequence layout:

```
<DATA_ROOT>/<dataset>/<sequence>/
  <scan_dir>/                 # *.bin scans, float32, N×4 (x,y,z,intensity), 10-digit IDs
  <pose_file>                 # see below
  inferred_labels/cenet_*_softmax/<scan_id>.bin   # CENet softmax (input to BKI)
  gt_labels[_terrain]/<scan_id>.bin               # GT labels (uint32 raw IDs)
  evaluations/<EVAL_PREFIX>/<scan_id>.txt         # output: gt_common pred_common
```

- **KITTI-360 (`2013_05_28_drive_0000_sync`, `..._0009_sync`):**
  - Scans: `velodyne_points/data/*.bin`
  - Poses: `velodyne_poses.txt` — per line: `frame_idx [12 or 16 floats]` (3×4 or 4×4 row-major). UTM-derived world-frame poses. Loaded in [mcd_util.h L291-L337](include/osm_bki/dataset_devkit/mcd_util.h#L291-L337). The code subtracts the first pose's translation from all poses (rotation untouched).
  - GT: `gt_labels/*.bin` (uint32 per point).
  - Pose frame index ≠ scan filename; `scan_indices_` carries the actual filename int.
- **MCD (`kth_day_09`, `kth_night_05`):**
  - Scans: `lidar_bin/data/*.bin`
  - Poses: `pose_inW.csv` — CSV with header `num,timestamp,x,y,z,qx,qy,qz,qw`. Body-frame poses in world; pose_inW = body→world. Loaded in [mcd_util.h L159-L277](include/osm_bki/dataset_devkit/mcd_util.h#L159-L277). Code aligns to first pose by left-multiplying `first_pose^-1`.
  - Body-to-lidar calibration: `hhs_calib.yaml`, key `body/os_sensor/T` (4×4). The full lidar→world is `pose_inW * lidar_to_body`.
  - GT: `gt_labels_terrain/*.bin` (uint32 raw MCD IDs).

### 1.4 Label remapping in code

The "M matrix" equivalent is just two dictionary lookups composed:

1. **Channel → raw label**: `learning_map_inv` in `labels_<TAXONOMY>.yaml` (network channel index → raw dataset label).
2. **Raw label → common 0..8**: `<TAXONOMY>_to_common` in [config/datasets/labels_common.yaml](config/datasets/labels_common.yaml). Three mappings are defined: `semkitti_to_common`, `mcd_to_common`, `kitti360_to_common`.

The C++ side composes (1)+(2) point-wise; the Python baseline notebook composes them into a flat LUT `channel_to_common`. Either works.

### 1.5 Where S-BKI / OSM-BKI results currently land

Driven by config:

- KITTI-360: `evaluation_result_prefix: evaluations/osm_bki` → on-disk dir is `<seq>/evaluations/osm_bki/`.
- MCD: `evaluation_result_prefix: evaluations/osm_bki_onehot` (and variants like `evaluations/vanilla` for the no-OSM baseline run).

So S-BKI/OSM-BKI and the raw-softmax baseline already coexist as sibling sub-directories under `<seq>/evaluations/`. Conv-BKI should follow the same convention: write to `<seq>/evaluations/convbki/`.

---

## 2. What I found in the Conv-BKI repo

**Repo:** https://github.com/UMich-CURLY/NeuralBKI — verified live; this is the codebase backing Wilson et al., "ConvBKI: Real-Time Probabilistic Semantic Mapping Network with Quantifiable Uncertainty" (T-RO 2024). Top level: `Config/`, `Data/`, `Models/`, `train.py`, `generate_results.py`, `VisualizeKernel.ipynb`, `environment.yml`.

### 2.1 What Conv-BKI consumes

It expects a **SemanticKITTI-style sequence directory**:

```
<conv_bki_data_dir>/sequences/<XX>/
  velodyne/<frame:06d>.bin        # float32 N×4
  labels/<frame:06d>.label        # uint32, lower 16 bits = class
  predictions_darknet/<frame:06d>.label   # the per-point predictions to be fused
  poses.txt                       # each row = 12 floats = 3×4 row-major
  calib.txt                       # must contain a Tr row (vel→cam)
```

- The data loader is [`Data/SemanticKitti.py`](https://github.com/UMich-CURLY/NeuralBKI/blob/main/Data/SemanticKitti.py).
- The path to the predictions dir is controlled by the `pred_path` key in the config YAML (default `"predictions_darknet"`).
- Predictions can be either:
  - **Hard labels** (`from_continuous: False`, `to_continuous: False`): `.label` files of `uint32`, lower 16 bits = class index. Inside the loader these are turned into one-hot inside Conv-BKI (`current_map` increments by 1 in the predicted class).
  - **Continuous (per-class probabilities)** (`from_continuous: True`): `.label` files of `float32`, length `N × num_classes`, treated as soft Dirichlet observations.
- Poses are applied as `global_pose = inv(Tr) @ pose @ Tr` — the standard SemanticKITTI calib trick. If we set `Tr = I` we can put lidar→world directly into `poses.txt` and the math collapses to identity.

### 2.2 What Conv-BKI produces

[`generate_results.py`](https://github.com/UMich-CURLY/NeuralBKI/blob/main/generate_results.py):

- Maintains a `GlobalMap` (`Models/BKINet.py`) with shape `(X, Y, Z, num_classes)` storing per-voxel Dirichlet counts.
- After each scan, calls `map.update_map(...)` with the (pose, scan, predictions) tuple, then a `label_points()` step that — for any point cloud — returns the per-point argmax read out of the voxel grid.
- If `gen_preds: True` is set in the config, it writes per-frame `.label` files (uint32) under `<MODEL_NAME>/sequences/<seq:02d>/predictions/<frame:06d>.label`. These per-point labels are in Conv-BKI's *training* taxonomy (typically SemanticKITTI's 20-class space after `learning_map`).
- If `meas_result: True`, it also computes per-class IoU internally using SemanticKITTI's `learning_ignore`.

For our purposes we want `gen_preds: True` and we **want the per-point labels to align with the GT scan we evaluate against** — i.e. we want the per-frame output corresponding to *each scan's own points* (which is what `label_points` already does when called with the current frame's points).

### 2.3 Taxonomy and training

- Default is **SemanticKITTI's 20-class taxonomy** (19 valid + 1 ignore). Pretrained filter weights ship in `Models/Weights/ConvBKI_PC_02_V/{filters0,filters1,filters2}.pt` (per-class, validation-tuned, 0.2 m voxel).
- The trainable parameters are very small: per-class kernel filter weights and length-scale `ell`. Loss is `NLLLoss(weight=class_weights, ignore_index=0)` over per-voxel labels derived from the final frame's GT point cloud (`points_to_voxels_torch`). Five epochs is the default in `ConvBKI_PerClass.yaml`.
- Training requires per-point GT (already what we have); no extra voxel labels are needed beyond GT scans.

### 2.4 Grid params

From `ConvBKI_PerClass.yaml`:
- Train: 80 m × 80 m × 5.2 m grid, 0.2 m voxel (400×400×26).
- Test: 100.2 m × 100.2 m × 5.4 m grid, 0.2 m voxel (501×501×27).
- `filter_size: 5`, `kernel: sparse`, `ell: 0.5` (learnable), `per_class: True`.

These are SemanticKITTI-tuned. For KITTI-360 the LiDAR is the same Velodyne HDL-64E and ranges are comparable, so the same grid should work. For MCD (Ouster) ranges are similar but mounting is different; either grid should still cover the scan.

### 2.5 Hardware / deps

PyTorch + CUDA. `environment.yml` pins torch ≥ 1.10 and a few standard scientific Python packages. Inference (per-scan) on a single GPU is real-time (~100 ms/scan reported in the paper). Per-sequence inference for our four target sequences will fit easily on one GPU.

---

## 3. Integration architecture (recommended)

**Run Conv-BKI as a completely separate offline Python pipeline.** Do *not* try to embed it in our ROS2 process. The two pipelines share only:

1. **Inputs**: the same CENet softmax `.bin` files and the same GT label `.bin` files on disk.
2. **Outputs**: per-scan `<scan_id>.txt` files of `(gt_common, pred_common)` in `<seq>/evaluations/convbki/`, which our existing Jupyter notebook reads.

```
                  ┌─────────────────────────────────────────┐
   on-disk inputs │ <seq>/velodyne_points/data/*.bin         │  (or lidar_bin/data for MCD)
   shared by both │ <seq>/inferred_labels/cenet_*_softmax/   │  CENet per-point softmax
   methods        │ <seq>/<pose_file>                        │  velodyne_poses.txt or pose_inW.csv
                  │ <seq>/gt_labels[_terrain]/*.bin          │  uint32 raw labels
                  └─────────────────────────────────────────┘
                                  │
                ┌─────────────────┴────────────────┐
                ▼                                   ▼
   ┌─────────────────────────┐         ┌──────────────────────────────────┐
   │ OSM-BKI / S-BKI ROS2    │         │ Conv-BKI offline pipeline (new)  │
   │ kitti360_node /         │         │  1. stage_inputs.py              │
   │ mcd_node                │         │  2. generate_results.py (forked) │
   │                         │         │  3. convert_outputs.py           │
   └─────────────┬───────────┘         └──────────────────┬───────────────┘
                 │                                         │
                 ▼                                         ▼
        <seq>/evaluations/osm_bki/*.txt          <seq>/evaluations/convbki/*.txt
                                  │
                                  ▼
                ┌──────────────────────────────────┐
                │ eval/osm_bki_eval.ipynb          │
                │ (existing — just change          │
                │  EVAL_PREFIX to "convbki")       │
                └──────────────────────────────────┘
```

The new code lives entirely under a new top-level directory (e.g. `baselines/convbki/`) on a branch off `main`, **not** mixed into the ROS2 package.

### 3.1 Directory layout for the new code

```
<repo>/baselines/convbki/
  NeuralBKI/                   # git submodule of UMich-CURLY/NeuralBKI
  configs/
    kitti360_seq0000.yaml      # one per (dataset, sequence, mode) combo
    kitti360_seq0009.yaml
    mcd_kth_day_09.yaml
    mcd_kth_night_05.yaml
  scripts/
    stage_inputs.py            # build per-sequence SemanticKITTI-style staging dir
    run_convbki.py             # thin wrapper that calls NeuralBKI/generate_results.py
    convert_outputs.py         # ConvBKI predictions → <seq>/evaluations/convbki/*.txt
    train_convbki_9class.py    # only if we decide to retrain (see §4)
  staging/                     # not committed; produced by stage_inputs.py
    kitti360/seq0000/
      sequences/00/
        velodyne/             (symlinks to <seq>/velodyne_points/data/)
        labels/               (uint32 9-class GT — converted from raw)
        predictions_softmax/  (float32 N×9 from CENet softmax)
        poses.txt             (lidar→world, 3×4 row-major)
        calib.txt             (Tr: identity)
  README.md                    # how to reproduce the table rows
```

### 3.2 Directory for results

Mirroring S-BKI:

- KITTI-360 (`2013_05_28_drive_0000_sync`, `..._0009_sync`):
  `<DATA_ROOT>/kitti360/<sequence>/evaluations/convbki/<scan_id>.txt`
- MCD (`kth_day_09`, `kth_night_05`):
  `<DATA_ROOT>/mcd/<sequence>/evaluations/convbki/<scan_id>.txt`

Then `osm_bki_eval.ipynb` just runs with `EVAL_PREFIX = "convbki"` and produces the per-class table numbers in the same shape as for S-BKI.

---

## 4. Taxonomy strategy (the biggest decision)

Conv-BKI's pretrained weights are **SemanticKITTI 20-class**. We have three options.

### Option A — Use pretrained 20-class Conv-BKI, project at I/O boundaries

- Inputs: convert CENet's predicted taxonomy (MCD or KITTI-360) into a 20-class **soft** distribution by mass-splitting each source class into its corresponding SemanticKITTI classes — but several of our common classes (parking, terrain, sidewalk) have direct SemanticKITTI counterparts only when the source taxonomy already encodes them. From `mcd_to_common`, MCD has `parkinglot→parking`, `road→road`, `lanemarking→road`, `sidewalk→sidewalk`, `terrain→terrain`, `building/shelter→building`, etc., so a clean MCD→SemanticKITTI mapping exists for the classes we care about; same for KITTI-360.
- Outputs: at the end, map ConvBKI's per-voxel argmax (in SemanticKITTI 20-class) back through `semkitti_to_common` (already defined in [labels_common.yaml](config/datasets/labels_common.yaml#L33)).
- Pros: zero retraining; uses the model exactly as the authors intended.
- Cons: information loss / asymmetry from taxonomy bouncing. CENet was never trained to output SemanticKITTI classes; the source→SemanticKITTI projection is approximate and may artificially under-represent classes ConvBKI was good at (e.g. parking, terrain).

### Option B — Retrain Conv-BKI per dataset in the 9-class common space (RECOMMENDED)

- Set `num_classes: 9` in the ConvBKI config. The model is tiny (one ell + per-class filter; ~thousands of params), trains in minutes per sequence on a single GPU.
- Train ConvBKI on **CENet softmax inputs + GT-derived voxel labels** on **held-out KITTI-360 sequences** (and **held-out MCD sequences**), then evaluate on the four target sequences without parameter leakage.
- Loss: same NLL with `ignore_index=0` against per-voxel argmax of GT mapped to common.
- Pros: a fair comparison — every method consumes identical input and predicts in identical taxonomy.
- Cons: requires choosing a train/eval split that doesn't already overlap with what S-BKI/OSM-BKI are evaluated on. KITTI-360 has plenty of other sequences; MCD has other `kth_*` and `ntu_*` sequences.

### Option C — Use Conv-BKI's filter shape with random init (no training)

- Set `num_classes: 9`, skip training, run with the initial (uniform/random) kernel.
- Pros: cheapest. Cons: doesn't really test Conv-BKI's contribution (the kernel weights *are* the contribution); reviewers will say we crippled it. **Do not do this.**

**Recommendation: Option B** for the headline table, optionally Option A in an appendix to show robustness. The retraining cost is small (Conv-BKI's whole point is that the model is lightweight) and the resulting comparison is unambiguous.

---

## 5. Input adapter (`stage_inputs.py`)

For each target sequence, produce a `staging/<dataset>/<seq>/sequences/00/` tree that Conv-BKI's SemanticKITTI loader will accept verbatim.

### 5.1 `velodyne/`
- KITTI-360: symlink each `<scan_id>.bin` in `<seq>/velodyne_points/data/` to `velodyne/<scan_id>.bin`. (Conv-BKI expects 6-digit naming `000123.bin`; KITTI-360 already uses 10-digit, **and the loader uses `f"{frame_num:06d}"` style**, so we need to either (a) rename to `000123.bin` and keep a sidecar mapping, or (b) patch the loader's `f"{i:06d}"` to read whatever stem the scan list returns. Patching is safer because frame indices in KITTI-360 are not contiguous — many scans are missing.)
- MCD: same approach; lidar_bin/data IDs may not be contiguous either.

### 5.2 `labels/` (per-point GT in 9-class common)
- Read `<seq>/gt_labels[_terrain]/<scan_id>.bin` (uint32 raw labels).
- Apply `<gt_labels_key>_to_common` (kitti360_to_common or mcd_to_common) to each point → uint32 9-class label.
- Write `labels/<scan_id>.label` as uint32 (the lower 16 bits are what Conv-BKI reads — class fits in 4 bits, so any encoding is fine).

### 5.3 `predictions_softmax/` (CENet softmax → 9-class continuous predictions)
- Read `<seq>/inferred_labels/cenet_<TAX>_softmax/<scan_id>.bin` (float16, N×K, K = network output dim).
- Apply `learning_map_inv` to figure out which raw label each channel corresponds to, then `<inferred_labels_key>_to_common` to collapse into 9 channels by summing source-channel mass into each common class. (This is exactly what the C++ side does today during BKI insertion — see `mcd2pcl_multiclass` aggregating into `common_probs` in [mcd_util.h L2026-L2046](include/osm_bki/dataset_devkit/mcd_util.h#L2026-L2046). Reuse that exact aggregation logic in Python so input is identical.)
- Re-normalize each row to sum to 1.
- Cast to float32 and write `predictions_softmax/<scan_id>.label` as `(N × 9) float32`.
- Set `pred_path: predictions_softmax`, `from_continuous: True` in the Conv-BKI YAML.

### 5.4 `poses.txt` and `calib.txt`
- Goal: have `global_pose = inv(Tr) @ pose @ Tr` come out as **lidar-frame → world**, identical to what `kitti360_node`/`mcd_node` use internally before BKI insertion.
- Set `calib.txt` Tr row to identity (`1 0 0 0 0 1 0 0 0 0 1 0`).
- For KITTI-360: emit each pose row as the 12 row-major entries of the 3×4 top of the existing 4×4 (after the same "subtract first-pose translation" step our C++ code does, so that map frame conventions match — though the absolute frame doesn't actually matter for IoU, only consistency of pose↔scan↔GT does).
- For MCD: build `lidar_to_world = body_to_world @ inverse(body_to_lidar)`, take the top 3×4, emit row-major; loop through the same first-pose normalization the C++ does.
- Pose rows must align by row with the scan IDs the loader iterates (the loader builds the scan list from `velodyne/*.bin` sorted lexically). If we want to keep filenames as the original 10-digit scan IDs, we need to patch the loader to enumerate files rather than computing them from a frame index. **This is the cleanest spot to make a one-line patch upstream.**

### 5.5 What about scans without CENet predictions or without GT?
Skip them at staging time — same filtering S-BKI uses (`scan_and_label_exist` in [mcd_util.h L779](include/osm_bki/dataset_devkit/mcd_util.h#L779-L797)). All three pipelines must see the same scan set; record the surviving scan list to a manifest so we can confirm.

---

## 6. Output adapter (`convert_outputs.py`)

Conv-BKI's `generate_results.py` with `gen_preds: True` writes:

```
<MODEL_NAME>/sequences/00/predictions/<frame:06d>.label   # uint32 per point
```

These are predicted per-point labels in **whatever taxonomy Conv-BKI was trained in** — under Option B that's already common 0..8, so the conversion is trivial.

For each frame:
- Load the matching GT label file (already 9-class common) from `staging/.../labels/<scan_id>.label`.
- Load Conv-BKI's prediction `predictions/<frame_or_scan_id>.label`.
- Write `<seq>/evaluations/convbki/<scan_id>.txt` with one line per point: `<gt_common> <pred_common>`.

Per-point ordering must match between `labels` and `predictions` files for this to be valid. Conv-BKI's `label_points()` returns predictions in the same order as the input scan's points, so as long as we pass the original scan in, this holds.

**Subtle point on point-set parity:**
- S-BKI's `query_scan` looks up each point of the (transformed) scan in the octree. Points where the octree voxel is unoccupied get `pred_label = 0`.
- Conv-BKI's `label_points` looks up each point in the dense voxel grid; out-of-bounds points get the prior (uniform). To match S-BKI, we should map "out-of-bounds" or "no observation" in Conv-BKI to class 0 (unlabeled) too, so the eval notebook's `mask = gt_all != 0` discards the same set on both sides. This needs to be enforced in the adapter (or a tiny patch to `label_points`).

---

## 7. Per-sequence pre-processing checklist

For each of `2013_05_28_drive_0000_sync`, `2013_05_28_drive_0009_sync`, `kth_day_09`, `kth_night_05`:

1. Enumerate scans for which all four files exist: `velodyne/.bin`, `labels/.bin` GT, `inferred_labels/cenet_*_softmax/.bin`, pose row.
2. Write `staging/<dataset>/<seq>/sequences/00/`:
   - symlinked `velodyne/`
   - converted `labels/` (9-class)
   - converted `predictions_softmax/` (N×9 float32)
   - `poses.txt` (3×4 rows, lidar→world)
   - `calib.txt` (Tr = I)
3. Run `generate_results.py` with the per-sequence config (Option B weights from §4) and `gen_preds: True`.
4. Run `convert_outputs.py` to write `<seq>/evaluations/convbki/`.
5. Run `eval/osm_bki_eval.ipynb` with `EVAL_PREFIX = "convbki"` for that sequence; record per-class IoU/mIoU/Accuracy.

---

## 8. Training strategy (Option B specifics)

If we retrain:

- **Train sequences (KITTI-360):** any combination of `..._0002`, `_0003`, `_0004`, `_0005`, `_0006`, `_0007`, `_0010` (sequences that have GT released). Hold out **`_0000` and `_0009`** which are our eval set.
- **Train sequences (MCD):** other `kth_day_*` and `kth_night_*` sequences, plus `ntu_*` if we want generalisation. Hold out `kth_day_09` and `kth_night_05`.
- **Loss:** NLL with `ignore_index=0` (matches the published recipe; common class 0 == unlabeled).
- **Class weights:** inverse log frequency on the train split, computed once and stored in the config.
- **Schedule:** 5 epochs as per `ConvBKI_PerClass.yaml`, Adam lr=0.007 → exponential decay 0.96.
- **Train one model per dataset (KITTI-360 ConvBKI vs MCD ConvBKI) or one joint model.** Joint is cleaner if we believe ConvBKI generalises; per-dataset is fairer to *Conv-BKI* since the authors trained per-dataset. Default: **per-dataset.**
- **Save weights** to `baselines/convbki/Weights/<dataset>_9class/` and record the exact training command in a `RUN.md` next to them.

---

## 9. Open questions / risks (for the author to decide)

### 9.1 Hard decisions
1. **Taxonomy strategy: Option A (pretrained) or Option B (retrain in 9-class)?** Recommended: B. Reviewers care about a fair comparison; the cost is hours of GPU time.
2. **Per-dataset ConvBKI weights, or one joint set?** Recommended: per-dataset.
3. **Where do retrained weights live?** Tracked in git LFS under `baselines/convbki/Weights/`? Or kept off-repo and downloaded by a script? Recommended: LFS — these are small (KB range).
4. **CENet softmax → 9-class aggregation: sum-and-renormalize, or weighted by an `M` matrix?** S-BKI today uses sum-and-renormalize ([mcd_util.h L2026-L2046](include/osm_bki/dataset_devkit/mcd_util.h#L2026-L2046)). To keep parity ConvBKI's input must use the same recipe — confirm before coding.
5. **Frame-skipping policy:** S-BKI/OSM-BKI uses `keyframe_dist` (10 m on KITTI-360, 5 m on MCD) — only the keyframe pose insertion accumulates into the map. ConvBKI's `update_map` is called per frame. Question: should we feed ConvBKI every frame (its native cadence) or only the same keyframes as S-BKI (for parity)? Recommended: feed ConvBKI every frame (its native, advantaged cadence), but **query only on the same keyframe scan set** that S-BKI evaluates. That way the comparison is "given identical CENet inputs and an identical evaluation point set, who's right more often?"
6. **Hard-label vs soft-label input to ConvBKI:** `from_continuous: True` (per-class probabilities) more closely matches what we feed S-BKI today, and the ConvBKI paper supports it. Recommended: soft.
7. **Should we patch Conv-BKI in-tree (fork) or as a vendored submodule with overlay scripts?** Recommended: submodule + a few small monkey-patches in our wrapper, so we can pull upstream fixes.

### 9.2 Smaller risks worth naming
- **File-naming mismatch.** Conv-BKI's loader uses 6-digit `f"{i:06d}.bin"` in code paths derived from KITTI Odometry. KITTI-360 uses 10-digit IDs and they're sparse. We'll have to either rename (carries a risk of breaking the pose↔scan correspondence) or patch the loader to glob actual filenames. The latter is cleaner.
- **Pose-frame alignment.** Conv-BKI applies `inv(Tr) @ pose @ Tr`. With `Tr = I` this collapses, but if upstream changes assume a real KITTI calib, our staging will silently break. Add a sanity check in `stage_inputs.py` that round-trips one pose.
- **Voxel-grid radius vs scan range.** KITTI-360 keeps points up to 200 m (`max_range` in [kitti360.yaml L19](config/methods/kitti360.yaml#L19)). Conv-BKI's default test grid is ±50.1 m. Points beyond the grid get the prior, which would make far-field IoU worse for ConvBKI than for S-BKI. Mitigation: extend ConvBKI's grid to ±100 m (still cheap at 0.2 m) or clip both methods' evaluation to ≤ 50 m so the comparison is on the region ConvBKI is configured for. Decide before publishing numbers.
- **MCD vs SemanticKITTI sensor differences.** MCD uses an Ouster (not Velodyne). The point density and ring pattern differ. ConvBKI's continuous kernel is sensor-agnostic in theory, but if the ell that worked for HDL-64E doesn't transfer well, MCD numbers may suffer. Mitigation: retrain ell per dataset (Option B already covers this).
- **GT-point set drift.** S-BKI's `query_scan` transforms scan points with `pose * lidar_to_body^-1`. We need to verify ConvBKI sees points in the same frame before they hit `label_points`. Add a parity test: pick one scan, compare the transformed point coordinates between the two pipelines.
- **Determinism.** Set `seed: 42` (already in the YAML) and verify ConvBKI's `meas_result: True` IoU agrees with what our notebook computes from the `.txt` files — they should match exactly if the conversion is correct; a discrepancy will surface bugs in `convert_outputs.py`.
- **Resource:** training is cheap; inference on four sequences is ~minutes per sequence on a single GPU. We don't need a cluster.

---

## 10. Decisions needed before any code is written

Please tick / answer:

- [ ] **Taxonomy strategy** — Option A (pretrained 20-class with projection), **Option B (retrain in 9-class)**, or both (B headline, A appendix)?
- [ ] **Train/val split for Option B** — confirm we can hold out KITTI-360 sequences {0000, 0009} and MCD sequences {kth_day_09, kth_night_05}, and that we have at least 2–3 other GT-labelled sequences per dataset to train on.
- [ ] **ConvBKI grid radius for evaluation** — extend to match S-BKI's 200 m range, or clip both methods to ConvBKI's native 50 m radius?
- [ ] **Frame cadence** — feed ConvBKI every scan and only query at keyframes (parity with S-BKI), or feed only keyframes (parity at update too)?
- [ ] **Input format to ConvBKI** — `from_continuous: True` with soft 9-class probs (recommended), or `False` with hard argmax?
- [ ] **Vendoring** — Conv-BKI as a git submodule under `baselines/convbki/NeuralBKI/`, or as a vendored copy?
- [ ] **Weights storage** — git LFS under `baselines/convbki/Weights/`, or out-of-repo with a downloader script?
- [ ] **Branch name** — proposed: `convbki-baseline` off `main`.
- [ ] **Out-of-bounds / no-observation semantics** — predict class 0 (drops from IoU), or predict prior argmax (counts against ConvBKI)? Recommended: class 0, to match S-BKI's "unoccupied → unlabeled" behavior.
- [ ] **Output directory naming** — `evaluations/convbki/` (recommended), or something more specific like `evaluations/convbki_9class_softmax/`?

Once these are settled I'll start coding in this order:
1. `stage_inputs.py` for one KITTI-360 sequence end-to-end (smoke test on a handful of scans).
2. Patch ConvBKI loader for variable scan filenames (one small upstream-friendly diff).
3. `run_convbki.py` wrapper + per-sequence configs.
4. `convert_outputs.py` and a parity check vs S-BKI on one scan.
5. (If Option B) `train_convbki_9class.py` and weights.
6. Roll out to all four sequences; produce Table I/II rows.
