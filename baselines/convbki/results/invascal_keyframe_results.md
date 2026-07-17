# Conv-BKI eval — Invascal (lidarrv) preds, KEYFRAME query-at-end (5.0 m)

Keyframe-gated, query-at-end variant of the Invascal Conv-BKI eval, per the lead's
two changes: (1) query the map ONCE at the end of the build, not per-scan; (2) build
and query only at keyframes. Keyframes are defined by Euclidean pose spacing (5.0 m)
for both datasets.

Companion to `invascal_results.md` (the ungated, per-scan online eval). Same inputs,
kernels, and staging; only the map-eval procedure changed. Per-combo JSONs:
`results/per_combo/*_invascal_kf.json` (the ungated ones are `*_invascal.json`).

## Keyframe definition + validation

Keyframe walk (`scripts/keyframes.py`): in stem-sorted order, select a keyframe when
the Euclidean translation distance from the **last selected keyframe** reaches 5.0 m;
the first scan is always a keyframe.

**Validated against the lead's ground truth.** The lead generated `multiclass_alpha`
predictions only at 5.0 m keyframes; `kth_day_09` has 208 such stems. Our derivation
reproduces them **exactly: 208/208, zero extra, zero missing** (symmetric-difference 0).
The alternative "distance from previous scan" (path-length) variant does NOT match
(212/211 stems, ~4 overlap), so the `last_kf` definition is the correct one.

> Derivation detail: keyframes are taken from the **body pose** (`pose_inW.csv` x,y,z
> for MCD; `velodyne_poses.txt` lidar→world for KITTI-360), i.e. the raw config poses —
> NOT the staged `poses.txt`, which stores lidar-in-world (`body_to_world @
> inv(body_to_lidar)`); the lever arm shifts boundary selections by one scan and does
> not match the 208.

Per-sequence keyframe counts (staged intersection = build==query set):

| sequence | scans (poses) | 5.0 m keyframes | staged (build/query) |
| --- | --- | --- | --- |
| kitti360 seq0000 | 10514 | 1533 | 1529 |
| kitti360 seq0009 | 13247 | 1937 | 1921 |
| mcd kth_day_09   | 7656  | 208  | 208 |
| mcd kth_night_05 | 6637  | 178  | 178 |

(A few kitti360 keyframes land on scans lacking GT/pred and are dropped, hence
1529<1533, 1921<1937. MCD keyframes are all staged.)

## Pipeline change (run_convbki.py)

`keyframe_query_at_end: true` triggers a two-pass path:
- **Disable GC:** `delete_time = len(dataset)+10` so the map persists over the whole
  sequence (the default `delete_time=10` prunes to a rolling ~10-frame window, which
  would make end-querying an early keyframe return all-prior/ignore).
- **BUILD pass:** over the keyframe set, `propagate` + `update_map` only (no labeling).
  `reset_grid` once up front; no reset between build and query.
- **QUERY pass:** over the same keyframe set, `propagate(pose)` + `label_points` +
  write `<stem>.label` against the finished map.

The default per-scan online path is unchanged (used by the `*_invascal` runs).

> **SEMANTIC CHANGE — online/causal → offline/batch.** In the per-scan path each scan
> is labeled against the map built *so far*. In this keyframe path each keyframe is
> labeled against the **full finished map, including scans that come AFTER it**. This
> matches the offline global-map eval the other baselines (S-BKI) use, and it shifts
> the numbers (here, marginally — see deltas).

## Per-combo results (keyframe query-at-end vs ungated per-scan)

mIoU over classes present in GT (class 0 dropped).

| combo | KF mIoU | KF files | ungated mIoU | ungated files | Δ (KF−ung) |
| --- | --- | --- | --- | --- | --- |
| KITTI-360 seq0000 ID  | **0.6548** | 1529 | 0.6599 | 10483 | −0.0051 |
| KITTI-360 seq0000 OOD | **0.2825** | 1529 | 0.2913 | 10483 | −0.0088 |
| KITTI-360 seq0009 ID  | **0.6863** | 1921 | 0.6843 | 13164 | +0.0020 |
| KITTI-360 seq0009 OOD | **0.2940** | 1921 | 0.2971 | 13164 | −0.0030 |
| MCD kth_day_09 ID     | **0.2919** | 208  | 0.2994 | 7656  | −0.0075 |
| MCD kth_day_09 OOD    | **0.1811** | 208  | 0.1863 | 7656  | −0.0052 |
| MCD kth_night_05 ID   | **0.3301** | 178  | 0.3271 | 6637  | +0.0030 |
| MCD kth_night_05 OOD  | **0.1729** | 178  | 0.1782 | 6637  | −0.0053 |

**All 8 combos completed OK; none failed or OOM'd.** The keyframe query-at-end numbers
track the ungated per-scan numbers within ±0.009 everywhere — the offline global-map
keyframe eval faithfully reproduces the per-scan online result on this data, which
validates the two-pass implementation.

### Per-class IoU (KF), JSON class order

Order: road, sidewalk, parking, building, fence, vegetation, vehicle, terrain.

| combo | road | sidewalk | parking | building | fence | vegetation | vehicle | terrain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-360 seq0000 ID  | 0.764 | 0.723 | 0.457 | 0.816 | 0.370 | 0.725 | 0.845 | 0.538 |
| KITTI-360 seq0000 OOD | 0.175 | 0.106 | 0.023 | 0.641 | 0.025 | 0.563 | 0.499 | 0.229 |
| KITTI-360 seq0009 ID  | 0.894 | 0.754 | 0.513 | 0.819 | 0.368 | 0.785 | 0.811 | 0.545 |
| KITTI-360 seq0009 OOD | 0.182 | 0.141 | 0.033 | 0.625 | 0.036 | 0.675 | 0.433 | 0.227 |
| MCD kth_day_09 ID     | 0.109 | 0.298 | 0.022 | 0.544 | 0.031 | 0.500 | 0.231 | 0.601 |
| MCD kth_day_09 OOD    | 0.224 | 0.006 | 0.003 | 0.483 | 0.068 | 0.362 | 0.179 | 0.124 |
| MCD kth_night_05 ID   | 0.160 | 0.449 | 0.005 | 0.612 | 0.007 | 0.617 | 0.238 | 0.552 |
| MCD kth_night_05 OOD  | 0.128 | 0.005 | 0.016 | 0.516 | 0.013 | 0.430 | 0.075 | 0.200 |

## Averaged rows for the paper tables

**Paper column order: road, sidewalk, parking, building, fence, terrain, vegetation,
vehicle** (note terrain precedes vegetation/vehicle here — differs from the JSON order).

| row | mIoU | road | sidewalk | parking | building | fence | terrain | vegetation | vehicle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **KITTI-360 in-domain** (avg 0000/0009 ID)     | **0.6706** | 0.829 | 0.739 | 0.485 | 0.817 | 0.369 | 0.541 | 0.755 | 0.828 |
| **KITTI-360 cross-domain** (avg 0000/0009 OOD) | **0.2882** | 0.178 | 0.123 | 0.028 | 0.633 | 0.030 | 0.228 | 0.619 | 0.466 |
| **MCD in-domain** (avg day09/night05 ID)        | **0.3110** | 0.135 | 0.374 | 0.014 | 0.578 | 0.019 | 0.576 | 0.558 | 0.234 |
| **MCD cross-domain** (avg day09/night05 OOD)    | **0.1770** | 0.176 | 0.006 | 0.010 | 0.500 | 0.040 | 0.162 | 0.396 | 0.127 |

## Run facts / cost

Build+query both scan the persistent global map per step (GC disabled), so cost grows
with map size — the kitti360 builds decelerate as the map accumulates. Peak RSS is CPU
(the global map is a CPU numpy array); GPU holds only the fixed ±50 m local grid, so
the whole batch ran within ~2.3 GB free GPU alongside the lead's job (no OOM).

| combo | keyframes | final map cells | peak RSS | build time |
| --- | --- | --- | --- | --- |
| mcd kth_night_05 ID  | 178  | 1.95 M | 1.17 GB | 34 s |
| mcd kth_night_05 OOD | 178  | 1.89 M | 1.16 GB | 31 s |
| mcd kth_day_09 ID    | 208  | 2.74 M | 1.31 GB | 57 s |
| mcd kth_day_09 OOD   | 208  | 2.32 M | 1.28 GB | 48 s |
| kitti360 seq0000 ID  | 1529 | 34.8 M | 5.80 GB | 96 min |
| kitti360 seq0000 OOD | 1529 | 37.1 M | 6.13 GB | 103 min |
| kitti360 seq0009 ID  | 1921 | 39.7 M | 6.49 GB | 143 min |
| kitti360 seq0009 OOD | 1921 | 41.3 M | 6.72 GB | 157 min |

> GPU note: the lead's `run_mcd_train.py` job held ~10 GB of the 16 GB GPU for the
> entire run; the batch was launched at the lead's instruction with the free-GPU
> threshold lowered to 2000 MiB and ran alongside it without OOM.
