# Conv-BKI baseline debug — Option A on KITTI-360 0000 OOD

## TL;DR (read this first)

The smoke test was broken because **the pretrained `ConvBKI_PC_02_V`
kernel was trained with hard (one-hot) labels**, not the soft 20-class
softmax we were feeding it. The per-class kernel weights have very
asymmetric sums (parking sum = 26.7, sidewalk = 5.6, vegetation = 4.5),
so any uniform-ish "floor" mass that we leaked into the parking column
got multiplied by ~5–6× more than the same mass in any other class —
and the floor mass we leaked was large, because the C++ aggregation
recipe gives each common class a ~uniform 11% share when CENet's
softmax is approximately uniform over its 29 channels (which it
mostly is, even for confident scans).

The compounding (soft input × per-class kernel asymmetry × ~uniform
floor mass) made Conv-BKI predict SemKITTI class 10 (parking) for
**99.4% of all voxels**, regardless of input — that's the 51M/52M
parking story from the first smoke run.

**Two changes applied:**

1. Switched Conv-BKI input to **hard labels** (`from_continuous: False`,
   `hard_input: true` in the per-experiment YAML). This is also what
   the upstream `Config/ConvBKI_PerClass.yaml` defaults to. The dataset
   now reads `predictions_hard/<stem>.label` (uint32 N×1) instead of
   `predictions_softmax/<stem>.label` (float32 N×20).

2. Built the hard labels via the **canonical 1-to-1** SemKITTI mapping
   (no weighted split). The weighted split for soft input (vegetation
   50/50 across t15+t16, vehicle 1/3 each across t1+t4+t5) systematically
   blocks vegetation and vehicle from ever being the per-point argmax
   when other common classes carry full mass — so vegetation/vehicle
   never appeared in Conv-BKI's output at all (0 predictions). Hard
   labels are the argmax of the 9-class common distribution, routed
   through `COMMON_TO_SEMKITTI_TRAIN[c]` (single target per common
   class: road→9, sidewalk→11, parking→10, building→13, fence→14,
   vegetation→15, vehicle→1, terrain→17).

**Smoke result after fix:** mIoU **0.1144** on 209 keyframes from
KITTI-360 0000 OOD vs. mIoU **0.0043** before. Conv-BKI now predicts
common classes {0..7} including vegetation. Terrain (common 8) is
still absent because the MCD CENet emits only 29 channels and the one
missing class is MCD class 29 (terrain) — no terrain mass ever enters
the pipeline, identical to what S-BKI sees. (S-BKI's
`static_gaussian_crossdomain/` keyframe set reaches mIoU 0.20 in our
common taxonomy, so we're roughly half S-BKI's score on the smoke
keyframes — plausible for pretrained Option A.)

Both `stage_inputs.py` and the eight per-experiment YAMLs were updated;
grid was also shrunk to ±50 m (paper-native 501×501×27 at 0.2 m voxel)
for the headline run. See commits on `convbki-baseline`.

---

## Task 1: the 29-channel mismatch

`labels_mcd.yaml`'s `learning_map_inv` has 30 entries (raw labels
0..29 mapped identity). The CENet softmax file for KITTI-360 scans
through the MCD-trained CENet contains exactly **29 uint16 channels
per point** (verified on `0000000009.bin`: 5843326 bytes ÷ 2 ÷ 100747 pts
= 29). The KITTI-360-trained CENet emits 45 channels (matches
`labels_kitti360.yaml`).

**Empirical class identification:** For scan `0000000100`, per-channel
mean softmax across all points:
- ch 25 = 0.054, argmax pick 41.5% — matches MCD class 25 (vegetation)
- ch 2  = 0.050, argmax pick 32.7% — matches MCD class 2 (building)
- ch 18 = 0.042, argmax pick 18.9% — matches MCD class 18 (sidewalk)

That ordering pins channel 0 = MCD raw label 0 (barrier). The 29
channels are MCD raw labels 0..28; MCD class 29 (terrain) is simply
absent from the network output. This is consistent with the CENet
training apparently treating terrain as out-of-vocabulary on KITTI-360
scenes.

**S-BKI behavior on 29 channels:** `mcd_util.h:1925` derives
`n_classes = n_mc_values / n_points` and trusts it. The aggregation
loop at `mcd_util.h:2022-2034` iterates `k in [0, n_classes)`, looks
up `learning_map_inv_[k]`, and routes to common via
`inferred_to_common_`. With 29 channels and an identity learning_map_inv
through key 29, the loop only ever touches k=0..28 — MCD class 29 is
never accessed. **No terrain signal enters S-BKI either.** Both
pipelines silently lose terrain when the source is MCD CENet.

**Our pipeline:** `build_source_channel_to_common('mcd', cfg, 29)`
returns a 29-element route. Same effect. No code change needed.

---

## Task 2: input projection sanity check

On scan `0000000100` (n=109355 pts), feeding the CENet softmax through
`aggregate_source_to_common` → 9-class common produced:

```
c1 road       mean=0.114  argmax_pick=0.9%
c2 sidewalk   mean=0.142  argmax_pick=21.2%
c3 parking    mean=0.113  argmax_pick=0.04%
c4 building   mean=0.142  argmax_pick=32.8%
c5 fence      mean=0.113  argmax_pick=0.5%
c6 vegetation mean=0.149  argmax_pick=42.3%
c7 vehicle    mean=0.113  argmax_pick=0.5%
c8 terrain    mean=0.000  argmax_pick=0%
c0 unlabeled  mean=0.113  argmax_pick=1.9%
```

Argmax distribution is sensible (vegetation 42%, building 33%,
sidewalk 21% — matches the scene). **The argmax-per-point projection
is not the problem.**

But `mean` values are nearly flat at 0.113 across c0, c3, c5, c7. That's
the C++ `average-by-count + row-renormalize` recipe producing a
~uniform "floor" share for any common class that has any source
contribution. Since CENet's softmax is itself approximately uniform
(every channel ≈ 1/29 = 0.034 plus a small bump for the winning class),
averaging gives every common class ~0.034 raw, and row-renormalizing
brings each to ~1/N_present ≈ 0.11. **Parking (c3, 1 source channel)
gets the same per-row mass as fence (c5, 2 channels) as road (c1, 2
channels).**

After the weighted split to 20-class:
```
t11 sidewalk    mean=0.142  argmax_pick=32.6%
t13 building    mean=0.142  argmax_pick=57.4%
t14 fence       mean=0.113  argmax_pick=3.2%
t10 parking     mean=0.113  argmax_pick=0.2%
t 0 unlabeled   mean=0.113  argmax_pick=5.9%
t 9 road        mean=0.114  argmax_pick=0.9%
t15 vegetation  mean=0.075  argmax_pick=0%      ← split: c6*0.5
t16 trunk       mean=0.075  argmax_pick=0%      ← split: c6*0.5
t 1 car         mean=0.038  argmax_pick=0%      ← split: c7*1/3
t 4 truck       mean=0.038  argmax_pick=0%      ← split: c7*1/3
t 5 other-veh   mean=0.038  argmax_pick=0%      ← split: c7*1/3
```

The split halves vegetation mass into two columns that each lose every
argmax to building/sidewalk. Same for vehicle thirded across three
columns. **The weighted split is harmful for argmax-based downstream
behavior.**

Soft 20-class input row sums to 1.0 (verified). Per-point parking mass
is 0.113 — looks small. But Conv-BKI doesn't argmax per point; it
accumulates per-voxel Dirichlet counts and applies a per-class kernel
with these per-class sums (from `filters1.pt`):

```
t10 parking          sum=26.7
t19 traffic-sign     sum=34.2
t 5 other-veh        sum=43.6
t 9 road             sum=6.96
t11 sidewalk         sum=5.61
t13 building         sum=4.80
t15 vegetation       sum=4.50
```

So soft mass of 0.113 into parking propagates as 0.113 × 26.7 = 3.02
per point, while 0.142 into building only propagates as 0.142 × 4.80
= 0.68. **Parking wins 4× over building in every voxel.** That is
where the 99% parking output comes from.

The conclusion: **the pretrained model's per-class kernel weights only
make sense for hard (one-hot) inputs where the floor mass is zero**.
With one-hot inputs, only the "winning" class for each point contributes
to update[v,c], so the parking kernel is only activated when CENet
actually predicts parking — which it almost never does.

---

## Task 3: output back-projection check

Conv-BKI's raw `.label` files (uint32 SemKITTI training class) from
the soft-input smoke run, distribution across 495 files / 58.5M points:

```
t10 parking      58208265   0.9941
t11 sidewalk       166431   0.0028
t13 building       101905   0.0017
t 0 unlab           73067   0.0012
t14 fence            3515   0.0001
t 9 road             2359   0.0000
t 5 other-veh         634   0.0000
```

**Conv-BKI itself is the source of 99% parking** — not the back-projection.
The `learning_map_inv` composition with `semkitti_to_common` only ever
sends t10 → raw 44 → common 3 (parking), which is correct.

---

## Task 4: root-cause & fix

Three contributing factors compounded:

1. **CENet softmax is approximately uniform** (each of 29 channels at
   ~1/29 = 0.034, the winning class only ~0.05). Not the bug, just a
   property of the input.

2. **The C++ aggregation washes the signal further:** averaging across
   source channels and then row-renormalizing produces ~uniform mass
   (~0.11) across every common class with any source feed. This makes
   parking's per-row mass equal to road's even when CENet has zero
   parking prediction.

3. **The pretrained per-class kernel multiplies mass by very different
   factors** — parking gets ~5–6× more spatial propagation than
   sidewalk or building because it's a rare class in SemKITTI training
   data. The kernel was learned assuming sparse one-hot inputs where
   only the genuine parking observations get amplified.

Together: soft uniform parking mass × big parking kernel = dominant
parking output regardless of scene content.

**Fix.** Two changes:

a. **Use hard labels.** `from_continuous: False` in the model, write
   `predictions_hard/<stem>.label` as uint32 N×1 at staging time.
   Pretrained ConvBKI_PC_02_V was trained with hard labels (the
   upstream config's default), and hard labels eliminate the floor-mass
   problem — only the argmax class contributes 1 per point.

b. **Canonical 1-to-1 routing for hard.** Hard labels = argmax of the
   9-class common distribution routed via `COMMON_TO_SEMKITTI_TRAIN`
   (single target per common class). No weighted split. Vegetation
   becomes a viable argmax winner.

Both changes are now in `stage_inputs.py` (writes both
`predictions_softmax/` and `predictions_hard/`) and `run_convbki.py`
(reads `hard_input: true` from YAML). The soft path is preserved in
case someone wants to revisit it later.

**Smoke result after fix** on 209 of the same 4433-keyframe set
(KITTI-360 0000 OOD, partial inference, ±100 m grid):

```
cls name           IoU    Acc    Prec     GT_cnt   Pred_cnt
  1 road        0.1604 0.1623 0.9307    4492222     783332
  2 sidewalk    0.1650 0.3946 0.2209    2368640    4232255
  3 parking     0.0055 0.0057 0.1604     680859      24188
  4 building    0.3914 0.9539 0.3989    3508112    8388816
  5 fence       0.0280 0.0314 0.2013     500834      78239
  6 vegetation  0.1586 0.3527 0.2237    4539242    7157718
  7 vehicle     0.0060 0.0061 0.4606     989544      13044
  8 terrain     0.0000 0.0000 0.0000    3765747          0
mIoU = 0.1144   mAcc = 0.2383
```

Compare to S-BKI (`static_gaussian_crossdomain/`, full 4433 keyframes,
**different taxonomy** — that dir is in a 14-class space, not our
9-class one, so the column labels there are different — but the
mIoU/mAcc number under the same eval pipeline is mIoU 0.2003).

For Option A pretrained-only on cross-domain CENet, mIoU 0.11 is
plausible. Terrain stays 0 because no terrain channel reaches us
from MCD CENet (true for S-BKI too).

---

## Status of the full run

- Code fix committed on `convbki-baseline`.
- KITTI-360 0000 OOD has been re-staged with `predictions_hard/`.
- All 8 per-experiment YAMLs updated: grid → ±50 m at 0.2 m
  (paper-native 501×501×27), `hard_input: true`.
- About to (re-)verify on 1000 scans at ±50 m grid; if mIoU holds
  above ~5% there, launch the full 8-combo run sequentially overnight
  with stdout/stderr logged per combo, plus auto-convert + auto-eval
  after each combo so partial results survive a mid-run abort.
- Hard constraint: if any combo's mIoU drops below 3%, the launcher
  aborts the remaining combos and appends to this report.

---

## Resume after disk-full crash — 2026-05-21

**Disk / symlinks** (task 1):
- `/home` (rpool): 53 GB free.
- `/media/sgarimella34/hercules-collect3`: 704 GB free.
- `baselines/convbki/{staging,nbki_runs,raw_predictions}` all symlinks
  to `/media/.../convbki_workspace/<same name>/`. Verified.

**29-channel handling** (task 2):
- `stage_inputs.py:433` infers `K = raw_pred.size // n_pts` from the
  first CENet file (gets K=29 for MCD-trained CENet).
- `build_source_channel_to_common(inferred_key, common_cfg, K=29)` only
  ever builds a 29-entry routing table — channel 29 (terrain) is never
  created or accessed.
- This mirrors S-BKI's `for k in [0, n_classes)` loop at
  `mcd_util.h:2020-2034`, which derives `n_classes = n_mc_values / n_points`
  and silently stops at k=28. Both pipelines lose terrain identically
  on MCD-trained CENet.
- **No code change needed.**

**Re-staging KITTI-360 0000 OOD with hard labels** (task 3):
- Wiped the partial staging (6985-scan fragment, 67 GB) and re-ran.
- Tweaked `stage_inputs.py` to skip the `predictions_softmax/` write
  when `hard_input: true` is set in the experiment YAML — saves ~60 GB
  per combo and is necessary to fit all 8 combos on the drive.
- Result: 10,483 scans staged (31 skipped — missing GT/pred files for
  scans that have a pose line). 9.2 GB on disk (vs 67 GB if softmax
  were written).
- Channel routing: K=29 confirmed.

**Smoke test on 1000 scans, ±50 m grid** (tasks 4–5):
- Inference: 1000 scans in 446s (≈2.24 scans/s).
- Convert: 418 of the 1000 scans matched the
  `static_gaussian_crossdomain/` keyframe set (4433 total keyframes).
- Eval on 8 classes (terrain present in GT but no MCD source channel):
  ```
  cls name           IoU    Acc    Prec     GT_cnt   Pred_cnt
    1 road        0.1672 0.1696 0.9225    8473875    1557605
    2 sidewalk    0.1550 0.3653 0.2121    4711202    8113323
    3 parking     0.0031 0.0032 0.1236    1415130      36463
    4 building    0.4119 0.9430 0.4224    8055917   17983631
    5 fence       0.0255 0.0291 0.1676     702524     122083
    6 vegetation  0.1585 0.3503 0.2245    9706076   15144114
    7 vehicle     0.0030 0.0030 0.4098    2156959      15838
    8 terrain     0.0000 0.0000 0.0000    8278596          0
  mIoU = 0.1155   mAcc = 0.2329
  ```
- Gate: mIoU 0.1155 >= 0.05. **PASSED.**
- Numbers track the pre-crash ±100 m smoke (mIoU 0.1144 on 209
  keyframes) — the grid shrink from ±100 m to ±50 m is essentially
  free here, as expected since Conv-BKI's spatial signal saturates
  well inside 50 m.

**Full 8-combo run launched** (task 6, 2026-05-21 11:04 EDT):
- Orchestrator: `bash baselines/convbki/scripts/run_all.sh` (nohup, PID 394896).
- Orchestrator log: `baselines/convbki/logs/run_all_20260521-110457.log`.
- Per-combo logs: `baselines/convbki/logs/<combo>_<TS>.log`.
- Cleanup hooks added: after each combo passes the 3% floor, the
  staging tree (`baselines/convbki/staging/<combo>/sequences/`) and
  the raw `.label` predictions under `nbki_runs/<combo>/sequences/`
  are deleted. `raw_predictions/` is preserved as the per-frame
  archive.
- Aggregation runs automatically after all 8 combos:
  `baselines/convbki/scripts/aggregate_results.py` →
  `results/{raw_numbers.json, in_domain.md, cross_domain.md}`.
- Per-combo mIoU appears here as combos complete.

### Per-combo results

**1/8 kitti360_seq0000_id** (in-domain, kitti360-trained CENet) —
finished 2026-05-21 12:37 EDT (1h33m end-to-end; inference 4046s @
~2.6 scans/s). Eval on all 4433 `static_gaussian_indomain` keyframes,
454.6M points after dropping class 0.

```
cls name           IoU    Acc    Prec     GT_cnt    Pred_cnt
  1 road        0.7430 0.7703 0.9544 109007289    87984032
  2 sidewalk    0.6457 0.7283 0.8505  58758977    50319509
  3 parking     0.2643 0.7768 0.2860  10927791    29682203
  4 building    0.7176 0.7479 0.9465 111688763    88258428
  5 fence       0.2535 0.6337 0.2970  10381934    22150564
  6 vegetation  0.6363 0.7274 0.8355  82138524    71509332
  7 vehicle     0.7552 0.9187 0.8093  31770610    36066802
  8 terrain     0.4896 0.8594  0.5322 39912866    64442563
mIoU = 0.5631   mAcc = 0.7703
```

Notable: terrain is alive (IoU 0.49) — this CENet has its own terrain
channel, unlike MCD's 29-channel network. Strong road/building/vehicle
results (all > 0.71). Cleanup ran (staging + nbki_runs/ dropped, raw
.label files archived to `raw_predictions/`).

**2/8 kitti360_seq0000_ood** (cross-domain, mcd-trained CENet) —
finished 2026-05-27 15:31 EDT (1h32m end-to-end; inference 4214s @
~2.5 scans/s). Eval on all 4433 `static_gaussian_crossdomain`
keyframes, 454.6M points after dropping class 0.

```
cls name            IoU    Acc    Prec     GT_cnt    Pred_cnt
  1 road         0.0885 0.0892 0.9098 109007289    10692305
  2 sidewalk     0.2279 0.5487 0.2804  58758977   114978648
  3 parking      0.0129 0.0134 0.2501  10927791      587497
  4 building     0.4749 0.9636 0.4835 111688763   222580804
  5 fence        0.0216 0.0241 0.1693  10381934     1480704
  6 vegetation   0.1157 0.2263 0.1914  82138524    97080182
  7 vehicle      0.0059 0.0059 0.6129  31770610      305744
  8 terrain      0.0000 0.0000 0.0000  39912866           0
mIoU = 0.1184   mAcc = 0.2339
```

Notable: building hangs on under cross-domain (IoU 0.47, recall 0.96 —
MCD CENet aggressively labels everything tall as building, hence the
2x over-prediction). Road collapses to 0.09 (CENet predicts only 10M
of 109M road points, prec 0.91 but recall 0.09). Vehicle effectively
absent (0.006) — MCD CENet's vehicle channel doesn't fire on KITTI-360
sweeps. Terrain at 0.00 because MCD CENet has no terrain class.
Cleanup ran.

**3/8 kitti360_seq0009_id** (in-domain, kitti360-trained CENet) —
finished 2026-05-27 17:29 EDT (1h58m end-to-end; inference 5096s @
~2.58 scans/s, 13164 scans). Eval on 5566 `static_gaussian_indomain`
keyframes, 577.5M points after dropping class 0.

```
cls name            IoU    Acc    Prec     GT_cnt    Pred_cnt
  1 road         0.8564 0.8969 0.9499 115882269   109423095
  2 sidewalk     0.6819 0.7634 0.8646  88510302    78150726
  3 parking      0.3140 0.8604 0.3308  15721096    40889183
  4 building     0.7059 0.7524 0.9196 112828029    92313539
  5 fence        0.2615 0.6613 0.3019  10769711    23593247
  6 vegetation   0.7048 0.7735 0.8880 167155289   145617105
  7 vehicle      0.7344 0.8875 0.8097  34104412    37381615
  8 terrain      0.5079 0.7926 0.5858  32513044    43991951
mIoU = 0.5958   mAcc = 0.7985
```

Notable: ~3pt higher mIoU than seq 0000 (0.5958 vs 0.5631), driven by
much stronger road (0.86 vs 0.74), sidewalk (0.68 vs 0.65), and
vegetation (0.70 vs 0.64). Vehicle drops slightly (0.73 vs 0.76).
Parking/fence remain the weak classes (~0.26-0.31), consistent across
both KITTI-360 sequences. Cleanup ran.

**4/8 kitti360_seq0009_ood** (cross-domain, mcd-trained CENet) —
finished 2026-05-27 19:19 EDT (1h51m end-to-end; inference 5017s @
~2.62 scans/s, 13164 scans). Eval on 5566 `indomain_with_height`
keyframes, 577.5M points after dropping class 0.

```
cls name            IoU    Acc    Prec     GT_cnt    Pred_cnt
  1 road         0.0404 0.0408 0.8138 115882269     5804881
  2 sidewalk     0.2439 0.5712 0.2986  88510302   169302565
  3 parking      0.0005 0.0005 0.0639  15721096      112614
  4 building     0.3725 0.9746 0.3762 112828029   292313147
  5 fence        0.0124 0.0136 0.1251  10769711     1170263
  6 vegetation   0.1836 0.2485 0.4127 167155289   100625848
  7 vehicle      0.0020 0.0021 0.5181  34104412      135016
  8 terrain      0.0000 0.0000 0.0000  32513044           0
mIoU = 0.1069   mAcc = 0.2314
```

Notable: same cross-domain failure pattern as seq 0000 OOD — building
dominates (recall 0.97, prec 0.38, over-predicts 2.6x), road collapses
to 0.04, vehicle ~0.002, terrain 0. Slightly worse mIoU than seq 0000
OOD (0.1069 vs 0.1184), primarily from road dropping further (0.04 vs
0.09). Cleanup ran. All 4 KITTI-360 combos now complete.

**5-8 STAGING FAILURE — MCD combos blocked.** At 2026-05-27 19:19 EDT
all four MCD combos (mcd_kth_day_09_id/_ood, mcd_kth_night_05_id/_ood)
failed in <1s during `stage_inputs.py`:

```
FileNotFoundError: '/media/sgarimella34/hercules-collect3/datasets/
                    mcd/hhs_calib.yaml'
```

The file is simply absent from this machine — not a wrong-name issue.
All four MCD configs point at the same `${OSM_BKI_DATA_ROOT}/mcd/
hhs_calib.yaml`, which is correct: MCD ships a single body->lidar
extrinsic shared across every MCD sequence (same sensor platform; KTH
day vs night use identical calibration). `launch/mcd_launch.py:126`
resolves the same path, and S-BKI eval outputs already exist on disk
for these KTH sequences — so the file was present when those were
generated, but no copy survives anywhere under `/` now (searched).

`stage_inputs.py:150-156` `load_body_to_lidar_mcd` reads
`cfg["body"]["os_sensor"]["T"]` and asserts a 4x4. On a missing file it
raises an uncaught `FileNotFoundError` and crashes hard — the
`if not calib_file` guard at line 353 only catches an *empty* config
value, not a configured-but-missing path. The C++ side
(`mcd_node.cpp` / `mcd_util.h:1525`) only falls back to identity when
the param is the empty string (the `cu_north_campus` path); for
`dataset='mcd'` a missing file is FATAL there too.

Resolution per maintainer: Doncey has the original `hhs_calib.yaml`;
once dropped at `/media/sgarimella34/hercules-collect3/datasets/mcd/
hhs_calib.yaml` the 4 MCD combos resume unchanged. The fix is "provide
the file," NOT patch the loader — `stage_inputs.py` left untouched.

### Current state — 2026-05-28

Done (4/8), results in `baselines/convbki/results/per_combo/`:
  - 1 kitti360_seq0000_id   mIoU 0.5631  (ID)
  - 2 kitti360_seq0000_ood  mIoU 0.1184  (OOD)
  - 3 kitti360_seq0009_id   mIoU 0.5958  (ID)
  - 4 kitti360_seq0009_ood  mIoU 0.1069  (OOD)

Blocked (4/8) on missing `hhs_calib.yaml`:
  - 5 mcd_kth_day_09_id    6 mcd_kth_day_09_ood
  - 7 mcd_kth_night_05_id  8 mcd_kth_night_05_ood

Partial aggregate written to `results/{raw_numbers.json, in_domain.md,
cross_domain.md}` — KITTI-360 rows populated, MCD rows blank (`-`).

Also fixed a latent bug in `aggregate_results.py`: `REPO` was computed
one `dirname()` too shallow, resolving to `<repo>/baselines` instead of
`<repo>`. That (a) misfiled all aggregate output under
`baselines/baselines/convbki/results/`, and (b) defeated the per-combo
snapshot lookup added earlier — so the aggregator silently fell back to
the shared per-sequence eval_dir and reported the OOD numbers in the ID
rows. Corrected to four `dirname()` levels; ID/OOD rows now read from
the correct snapshots. The stray `baselines/baselines/` tree was
removed.


