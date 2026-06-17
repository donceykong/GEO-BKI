# Conv-BKI retrained (9-class GT) vs pretrained (Option A) — all KITTI-360 combos

Retrained = native 9-class kernel trained on KITTI-360 GT one-hot
(`Weights/kitti360_9class_gt/filters5.pt`), evaluated on **all GT scans**.
Pretrained = authors' 20-class SemKITTI `ConvBKI_PC_02_V` kernel projected to
the 9-class common space; numbers from `DEBUG_REPORT.md`, evaluated on the
**S-BKI keyframe sets**.

> ⚠️ **Two confounds — read before quoting deltas.**
> 1. **Basis differs.** Retrained = all GT scans (10,483 for seq0000; 13,164 for
>    seq0009). Pretrained = S-BKI keyframes (4,433 / 5,566). That keyframe set is
>    not on this machine, so a confound-free same-basis A/B is not possible here.
> 2. **OOD input differs.** Retrained OOD feeds the **30-channel**
>    `cenet_mcd_terrain_softmax` (terrain at ch29 present → terrain can score > 0).
>    Pretrained OOD used the **29-channel** `cenet_mcd_softmax` (terrain channel
>    absent → terrain IoU forced to 0). So the OOD rows differ in *both* basis and
>    input — the OOD deltas are indicative, not a controlled model-only A/B.
> ID rows differ only in basis (same `cenet_kitti360_softmax` input).

## Headline mIoU

| combo | retrained | pretrained | Δ (re − pre) | basis (re / pre) |
|---|---|---|---|---|
| seq0000 **ID**  | 0.5593 | 0.5631 | **−0.0038** | 10483 / 4433 |
| seq0009 **ID**  | **0.6852** | 0.5958 | **+0.0894** | 13164 / 5566 |
| seq0000 **OOD** | **0.1509** | 0.1184 | **+0.0325** | 10483 / 4433 † |
| seq0009 **OOD** | **0.1661** | 0.1069 | **+0.0592** | 13164 / 5566 † |

† OOD also differs in input channels (30-ch terrain-present vs 29-ch terrain-absent).

## Per-class IoU

### In-domain (kitti360-trained CENet)

| class | s0000 re | s0000 pre | s0009 re | s0009 pre |
|---|---|---|---|---|
| road       | 0.7465 | 0.7430 | 0.8846 | 0.8564 |
| sidewalk   | 0.6602 | 0.6457 | 0.7678 | 0.6819 |
| parking    | 0.3069 | 0.2643 | 0.5309 | 0.3140 |
| building   | 0.7138 | 0.7176 | 0.8025 | 0.7059 |
| fence      | 0.2462 | 0.2535 | 0.3944 | 0.2615 |
| vegetation | 0.6426 | 0.6363 | 0.7865 | 0.7048 |
| vehicle    | 0.6489 | 0.7552 | 0.7517 | 0.7344 |
| terrain    | 0.5093 | 0.4896 | 0.5628 | 0.5079 |
| **mIoU**   | **0.5593** | **0.5631** | **0.6852** | **0.5958** |
| mAcc       | 0.7560 | 0.7703 | 0.8179 | 0.7985 |

### Cross-domain (mcd-trained CENet) — see confound note †

| class | s0000 re | s0000 pre | s0009 re | s0009 pre |
|---|---|---|---|---|
| road       | 0.0046 | 0.0885 | 0.0005 | 0.0404 |
| sidewalk   | 0.1188 | 0.2279 | 0.1359 | 0.2439 |
| parking    | 0.0003 | 0.0129 | 0.0001 | 0.0005 |
| building   | 0.5741 | 0.4749 | 0.5254 | 0.3725 |
| fence      | 0.0061 | 0.0216 | 0.0351 | 0.0124 |
| vegetation | 0.4021 | 0.1157 | 0.5415 | 0.1836 |
| vehicle    | 0.0531 | 0.0059 | 0.0214 | 0.0020 |
| terrain    | 0.0485 | 0.0000 | 0.0685 | 0.0000 |
| **mIoU**   | **0.1509** | **0.1184** | **0.1661** | **0.1069** |
| mAcc       | 0.2485 | 0.2339 | 0.2771 | 0.2314 |

## Read

- **In-domain: retrained clearly wins on seq0009 (+0.089), flat on seq0000
  (−0.004).** The earlier seq0000-only "tied in-domain" call does *not*
  generalise — on seq0009 the GT kernel beats pretrained on 7/8 classes, led by
  parking (+0.22), sidewalk (+0.086), fence (+0.13), vegetation (+0.082). seq0000
  is the outlier, dragged to flat almost entirely by **vehicle (−0.106)**: the
  GT-trained vehicle kernel learned the widest `ell` (0.38) and over-spreads
  vehicle on noisy CENet input. That vehicle regression does not appear on
  seq0009 (+0.017), so it's a seq0000-specific interaction, not a kernel-wide
  defect.

- **Cross-domain: retrained mIoU is higher on both (+0.033, +0.059), but the
  comparison is doubly confounded** (basis + 29-vs-30-ch input). The honest
  decomposition:
  - **Terrain** goes 0.00 → ~0.05–0.07 purely because the retrained run is *fed*
    a terrain channel the pretrained run never had. This is an input difference,
    not a model win — discount it.
  - **Vegetation** is the real story: 0.12 → 0.40 (seq0000), 0.18 → 0.54
    (seq0009). Building also climbs (0.47 → 0.57; 0.37 → 0.53). The native
    9-class kernel handles the cross-domain MCD distribution far better on the
    tall-vegetation classes.
  - **Road collapses further** under retrained cross-domain (0.09 → 0.005;
    0.04 → 0.0005) — MCD CENet barely fires road on KITTI-360 sweeps, and the
    GT kernel doesn't recover it. Parking/fence stay near-zero in both.

- **Net:** the retrained native-9-class kernel is the stronger model in-domain
  (decisively on seq0009) and scores higher cross-domain, though the OOD gain is
  partly an input-channel artifact (terrain) layered on a genuine vegetation/
  building improvement. The single number for Table I — **retrained Conv-BKI OOD,
  next to OSM-BKI** — is **seq0000 0.1509 / seq0009 0.1661**.

## Provenance

- Retrained jsons: `results/per_combo/kitti360_seq{0000,0009}_{id,ood}_retrained.json`
  (seq0000_id done 2026-06-12; the other three 2026-06-16).
- Pretrained numbers: `DEBUG_REPORT.md` combos 1–4.
- Configs: `configs/kitti360_seq{0000,0009}_{id,ood}_retrained.yaml`.
