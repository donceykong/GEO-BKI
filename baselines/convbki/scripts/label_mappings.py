"""Label mapping helpers shared by stage_inputs.py and convert_outputs.py.

Conv-BKI runs in SemanticKITTI's 20-class training space (post-learning_map).
Our common taxonomy is the 9-class space from config/datasets/labels_common.yaml.

Transforms used in the pipeline:

  1. Source CENet channel -> 9-class common (via labels_common.yaml's
     <source>_to_common). Aggregated per-point with the C++ recipe from
     include/osm_bki/dataset_devkit/mcd_util.h:2020-2043: sum source
     channels into each common column, divide by per-common-class
     contributor count, then row-normalize.

  2. 9-class common -> 20-class SemKITTI training distribution (via
     COMMON_TO_SEMKITTI_TRAIN_WEIGHTED). Each common class's row mass
     is spread over one or more training columns by per-target weights
     that sum to 1.0, so row sums are preserved exactly.

  3. Raw GT label -> SemKITTI training class (via COMMON_TO_SEMKITTI_TRAIN,
     the canonical single-target mapping). Used for the staging
     labels/<stem>.label files (which Conv-BKI's loader reads even
     though we don't train).

  4. SemKITTI training class -> 9-class common (via the SemKITTI
     learning_map_inv composed with semkitti_to_common). Used at
     convert-outputs time to map Conv-BKI's per-point argmax back to
     the eval-notebook's common space.

Source layout per CENet:
  inferred_labels_key == "mcd"      -> raw labels 0..29 (30 channels)
  inferred_labels_key == "kitti360" -> raw labels 0..44 (45 channels)
Both source taxonomies use an identity learning_map_inv, so channel
index equals raw source label index.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

# Canonical single-target SemKITTI training class (0..19) per common class
# (0..8). Used for GT labels and as a stable name for "the SemKITTI class
# that maps to this common class" anywhere we need a single value.
COMMON_TO_SEMKITTI_TRAIN = {
    0: 0,   # unlabeled
    1: 9,   # road
    2: 11,  # sidewalk
    3: 10,  # parking
    4: 13,  # building
    5: 14,  # fence
    6: 15,  # vegetation
    7: 1,   # vehicle -> car
    8: 17,  # terrain
}

# Weighted projection from common -> {training class: weight}. Per-common
# weights MUST sum to 1.0 so that row-normalized common distributions
# project to row-normalized 20-class training distributions.
#
# Vegetation common: split evenly over training 15 (vegetation) and 16
# (trunk). Both feed back to common-vegetation on the output side via
# semkitti_to_common, so the BKI prediction is consistent either way.
#
# Vehicle common: split evenly over training 1 (car), 4 (truck), and 5
# (other-vehicle). All three feed back to common-vehicle on the output.
COMMON_TO_SEMKITTI_TRAIN_WEIGHTED: dict[int, list[tuple[int, float]]] = {
    0: [(0, 1.0)],
    1: [(9, 1.0)],
    2: [(11, 1.0)],
    3: [(10, 1.0)],
    4: [(13, 1.0)],
    5: [(14, 1.0)],
    6: [(15, 0.5), (16, 0.5)],
    7: [(1, 1.0 / 3.0), (4, 1.0 / 3.0), (5, 1.0 / 3.0)],
    8: [(17, 1.0)],
}

# Sanity-check: weights per common class must sum to 1.0.
for _c, _splits in COMMON_TO_SEMKITTI_TRAIN_WEIGHTED.items():
    _total = sum(w for _, w in _splits)
    assert abs(_total - 1.0) < 1e-9, (
        f"COMMON_TO_SEMKITTI_TRAIN_WEIGHTED[{_c}] weights sum to {_total}, not 1.0"
    )


SEMKITTI_LEARNING_MAP_INV: dict[int, int] = {}
SEMKITTI_LEARNING_MAP: dict[int, int] = {}
SEMKITTI_TRAIN_TO_COMMON: dict[int, int] = {}


def load_semkitti_config(nbki_root: str) -> None:
    """Populate module-level SemKITTI learning_map / learning_map_inv from the
    NeuralBKI config (so we use the same definitions Conv-BKI was trained on).
    """
    cfg_path = os.path.join(nbki_root, "Config", "semantic_kitti.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    SEMKITTI_LEARNING_MAP_INV.clear()
    SEMKITTI_LEARNING_MAP.clear()
    for k, v in cfg["learning_map_inv"].items():
        SEMKITTI_LEARNING_MAP_INV[int(k)] = int(v)
    for k, v in cfg["learning_map"].items():
        SEMKITTI_LEARNING_MAP[int(k)] = int(v)


def load_common_config(common_yaml_path: str) -> dict:
    """Return the parsed labels_common.yaml dict."""
    with open(common_yaml_path) as f:
        return yaml.safe_load(f)


def build_semkitti_train_to_common(common_cfg: dict) -> dict[int, int]:
    """Compose SemKITTI learning_map_inv (training -> raw) with
    semkitti_to_common (raw -> common) to produce training -> common.
    """
    semk_to_common = {int(k): int(v) for k, v in common_cfg["semkitti_to_common"].items()}
    if not SEMKITTI_LEARNING_MAP_INV:
        raise RuntimeError("Call load_semkitti_config() before build_semkitti_train_to_common().")
    out: dict[int, int] = {}
    for train_cls, raw_label in SEMKITTI_LEARNING_MAP_INV.items():
        out[train_cls] = semk_to_common.get(raw_label, 0)
    out[0] = 0
    SEMKITTI_TRAIN_TO_COMMON.clear()
    SEMKITTI_TRAIN_TO_COMMON.update(out)
    return out


def build_source_channel_to_common(
    inferred_labels_key: str,
    common_cfg: dict,
    n_channels: int,
) -> np.ndarray:
    """For each CENet network channel (0..n_channels-1), return the 9-class
    common label that channel maps to via <inferred_labels_key>_to_common.

    Channels with no entry collapse to common 0 (unlabeled), so their mass
    is later either averaged into the unlabeled column (and ignored by the
    eval notebook) or dropped.
    """
    map_key = inferred_labels_key + "_to_common"
    if map_key not in common_cfg:
        raise KeyError(f"labels_common.yaml has no mapping '{map_key}'")
    src_to_common = {int(k): int(v) for k, v in common_cfg[map_key].items()}
    out = np.zeros(n_channels, dtype=np.int32)
    for ch in range(n_channels):
        out[ch] = src_to_common.get(ch, 0)
    return out


def build_raw_gt_to_semkitti_train(
    gt_labels_key: str,
    common_cfg: dict,
    max_raw_label: int = 65536,
) -> np.ndarray:
    """Lookup table: raw GT label -> canonical SemKITTI training class.

    Uses the single-target COMMON_TO_SEMKITTI_TRAIN; GT has one class per
    point so there's nothing to split. KITTI-360's 65535 sentinel resolves
    to whatever 65535 maps to in kitti360_to_common (i.e. 0).
    """
    map_key = gt_labels_key + "_to_common"
    if map_key not in common_cfg:
        raise KeyError(f"labels_common.yaml has no mapping '{map_key}'")
    src_to_common = {int(k): int(v) for k, v in common_cfg[map_key].items()}
    lut = np.zeros(max_raw_label, dtype=np.uint32)
    for raw, common in src_to_common.items():
        if 0 <= raw < max_raw_label:
            lut[raw] = COMMON_TO_SEMKITTI_TRAIN.get(common, 0)
    return lut


def aggregate_source_to_common(
    softmax_NK: np.ndarray,
    channel_to_common: np.ndarray,
    n_common: int = 9,
) -> np.ndarray:
    """Mirror mcd_util.h:2020-2043 exactly. Sum source-channel softmax into
    each common column, divide by per-column contributor count, then
    row-normalize.
    """
    N, K = softmax_NK.shape
    assert channel_to_common.shape == (K,)
    out = np.zeros((N, n_common), dtype=np.float32)
    count = np.zeros(n_common, dtype=np.int32)
    for ch in range(K):
        c = int(channel_to_common[ch])
        if c < 0 or c >= n_common:
            continue
        out[:, c] += softmax_NK[:, ch]
        count[c] += 1
    nz_cols = count > 0
    out[:, nz_cols] = out[:, nz_cols] / count[nz_cols].astype(np.float32)
    row_sum = out.sum(axis=1, keepdims=True)
    nz_rows = row_sum.squeeze(-1) > 1e-10
    out[nz_rows] = out[nz_rows] / row_sum[nz_rows]
    return out


def project_common_to_semkitti_train(
    common_N9: np.ndarray,
    num_train_classes: int = 20,
    weighted_table: dict[int, list[tuple[int, float]]] | None = None,
) -> np.ndarray:
    """Spread the 9-class common per-point distribution into a 20-class
    SemKITTI training distribution via COMMON_TO_SEMKITTI_TRAIN_WEIGHTED.

    Per-common weights sum to 1.0 (asserted at module load time), so a
    row-normalized common distribution projects to a row-normalized
    20-class distribution: no additional renormalization is needed and
    none is performed.
    """
    if weighted_table is None:
        weighted_table = COMMON_TO_SEMKITTI_TRAIN_WEIGHTED
    N, n_common = common_N9.shape
    out = np.zeros((N, num_train_classes), dtype=np.float32)
    for c, splits in weighted_table.items():
        if c < 0 or c >= n_common:
            continue
        for t, w in splits:
            if t < 0 or t >= num_train_classes:
                continue
            out[:, t] += float(w) * common_N9[:, c]
    return out


def aggregate_channels_to_semkitti_train(
    softmax_NK: np.ndarray,
    channel_to_common: np.ndarray,
    num_train_classes: int = 20,
    weighted_table: dict[int, list[tuple[int, float]]] | None = None,
) -> np.ndarray:
    """End-to-end staging projection: CENet softmax -> common -> SemKITTI
    20-class soft probs.

    Composed from aggregate_source_to_common (C++-faithful 9-class aggregation)
    and project_common_to_semkitti_train (weighted split per
    COMMON_TO_SEMKITTI_TRAIN_WEIGHTED).
    """
    common_N9 = aggregate_source_to_common(softmax_NK, channel_to_common, n_common=9)
    return project_common_to_semkitti_train(
        common_N9, num_train_classes=num_train_classes, weighted_table=weighted_table
    )


def softmax_from_float16_raw(raw_uint16: np.ndarray) -> np.ndarray:
    """Convert a uint16 buffer of float16 logits (as stored in CENet output
    files) into a row-softmax float32 array, matching mcd_util.h:1996-2006.

    Caller reshapes raw_uint16 to (N, K) first.
    """
    logits = raw_uint16.view(np.float16).astype(np.float32)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-30)
