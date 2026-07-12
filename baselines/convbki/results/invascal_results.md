# Conv-BKI eval — Invascal (lidarrv) predictions, native 9-class kernels

Eval of the GT-retrained native-9-class Conv-BKI kernels
(`Weights/kitti360_9class_gt`, `Weights/mcd_9class_gt`) on **Invascal (lidarrv)**
hard predictions. The old CENet inferred labels were deleted by the lead, so
these runs are Invascal-only.

## Setup

- **Input:** `<seq>/inferred_labels/lidarrv_<model>/predicted_labels/*.bin`, one
  `uint32` **hard** RAW dataset label per point (not softmax, not collapsed).
  Staged via a new `pred_format: hard_labels` path in `stage_inputs.py`:
  raw → common by the per-dataset collapse keyed on `inferred_labels_key`
  (`lidarrv_kitti360` → `kitti360_to_common`; `lidarrv_mcd` → `mcd_to_common`).
- **ID vs OOD:** ID = input from the eval domain's own model; OOD = cross.
  - KITTI-360 seq: ID `lidarrv_kitti360`, OOD `lidarrv_mcd`.
  - MCD kth seq:   ID `lidarrv_mcd`,      OOD `lidarrv_kitti360`.
- **Kernel per eval dataset:** KITTI-360 seqs use `kitti360_9class_gt/filters5.pt`;
  MCD kth seqs use `mcd_9class_gt/filters5.pt`.
- **Grid:** ±50 m (501×501×27, 0.2 m voxel). `hard_input: true`, `native_common: true`.
- **Basis:** UNGATED — all GT scans. No keyframe-gating reference exists (KITTI-360
  has no `evaluations/` dir; MCD kth has only an EBS-baseline keyframe set, not the
  S-BKI set, and there is no prior MCD retrained run to match). Consistent with the
  earlier `*_retrained` all-GT-scan basis.
- Configs: `configs/*_invascal.yaml`. Per-combo JSON: `results/per_combo/*_invascal.json`.

## Per-combo results

Per-class IoU (common classes 1–8). mIoU over classes present in GT (class 0 dropped).

| combo | files | mIoU | road | sidewalk | parking | building | fence | vegetation | vehicle | terrain |
|---|---|---|---|---|---|---|---|---|---|---|
| kitti360 seq0000 ID  | 10483 | **0.6599** | 0.8035 | 0.7211 | 0.4644 | 0.8039 | 0.3651 | 0.7134 | 0.8438 | 0.5644 |
| kitti360 seq0000 OOD | 10483 | **0.2913** | 0.2047 | 0.1378 | 0.0261 | 0.6408 | 0.0190 | 0.5535 | 0.5144 | 0.2341 |
| kitti360 seq0009 ID  | 13164 | **0.6843** | 0.9019 | 0.7594 | 0.5120 | 0.8033 | 0.3557 | 0.7723 | 0.8065 | 0.5636 |
| kitti360 seq0009 OOD | 13164 | **0.2971** | 0.1846 | 0.1758 | 0.0268 | 0.6204 | 0.0275 | 0.6640 | 0.4552 | 0.2220 |
| mcd kth_day_09 ID    | 7656  | **0.2994** | 0.1074 | 0.3034 | 0.0168 | 0.5448 | 0.0310 | 0.5121 | 0.2944 | 0.5852 |
| mcd kth_day_09 OOD   | 7656  | **0.1863** | 0.2188 | 0.0036 | 0.0031 | 0.4950 | 0.0757 | 0.3791 | 0.1938 | 0.1212 |
| mcd kth_night_05 ID  | 6637  | **0.3271** | 0.1307 | 0.4297 | 0.0024 | 0.6008 | 0.0108 | 0.6233 | 0.2668 | 0.5522 |
| mcd kth_night_05 OOD | 6637  | **0.1782** | 0.1173 | 0.0052 | 0.0113 | 0.5187 | 0.0103 | 0.4685 | 0.0755 | 0.2187 |

All 8 combos ran (seq0009 OOD was added once the lead's `lidarrv_mcd` generation for
seq0009 completed — 14056 predictions matching the scan count).

## Comparison to the earlier CENet-based retrained numbers

CENet numbers exist only for KITTI-360 (the four `*_retrained` runs). MCD combos
never ran under CENet (they were blocked on the missing calib / no MCD kernel), so
there is no CENet baseline for MCD — these Invascal MCD numbers are the first.

| combo | Invascal mIoU | CENet (retrained) mIoU | Δ |
|---|---|---|---|
| kitti360 seq0000 ID  | 0.6599 | 0.559 | +0.101 |
| kitti360 seq0009 ID  | 0.6843 | 0.685 |  0.000 |
| kitti360 seq0000 OOD | 0.2913 | 0.151 | +0.140 |
| kitti360 seq0009 OOD | 0.2971 | 0.166 | +0.131 |
| mcd kth_day_09 ID    | 0.2994 | — (no CENet baseline) | — |
| mcd kth_day_09 OOD   | 0.1863 | — | — |
| mcd kth_night_05 ID  | 0.3271 | — | — |
| mcd kth_night_05 OOD | 0.1782 | — | — |

### Notes

- **Same kernel, same basis, only the input source changed** (Invascal hard labels
  vs CENet-argmax hard labels), so the KITTI-360 ID/OOD deltas are a clean
  input-quality A/B on identical model + basis.
- KITTI-360 ID gains on seq0000 (+0.10) and holds on seq0009; the Invascal
  in-domain predictor is at least as good as CENet at the point level.
- KITTI-360 OOD nearly doubles on both seqs (seq0000 0.151 → 0.291, seq0009
  0.166 → 0.297): the MCD-trained Invascal model transfers to KITTI-360 scans much
  better than the MCD-trained CENet did, with large gains on building/vegetation/
  vehicle and a non-zero terrain (~0.22–0.23).
- **MCD kth is a held-out scene domain for both stages** — the Invascal models and
  both kernels were trained on tuhh+ntu (KITTI-360) / seqs 0002–0010, never on kth.
  So even MCD "ID" carries a scene-domain gap, hence ~0.30 vs KITTI-360 ID ~0.66.
  Parking/fence are near-zero on kth (sparse/absent GT), matching the mcd kernel's
  val behavior. OOD (KITTI-360→MCD input) drops further, as expected.
