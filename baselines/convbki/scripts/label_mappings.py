"""Label mapping helpers shared by stage_inputs.py and convert_outputs.py.

Conv-BKI runs in SemanticKITTI's 20-class training space (post-learning_map).
Our common taxonomy is the 9-class space from config/datasets/labels_common.yaml.
We need three transforms:

  1. CENet network channel -> SemKITTI training class (0..19).
     Used at staging time to project per-point softmax into the soft
     20-class input Conv-BKI expects.

  2. Raw GT label -> SemKITTI training class (0..19).
     Used at staging time so the labels/ files Conv-BKI's loader reads
     line up with the same training-class indexing.

  3. SemKITTI training class -> 9-class common (0..8).
     Used at convert-outputs time to map Conv-BKI's per-point argmax
     back to the common space the eval notebook scores against.

Source layout per CENet:
  inferred_labels_key == "mcd"      -> 30 channels, raw labels 0..29
  inferred_labels_key == "kitti360" -> 45 channels, raw labels 0..44

Both source taxonomies use an identity learning_map_inv (channel index
equals raw label index) so we can skip that hop.

The common-class -> canonical SemKITTI training class routing pins each
common class to a single training class:
  road       -> 9   (road)
  sidewalk   -> 11  (sidewalk)
  parking    -> 10  (parking)
  building   -> 13  (building)
  fence      -> 14  (fence)
  vegetation -> 15  (vegetation)  [not 16 (trunk); 15 is the dominant class]
  vehicle    -> 1   (car)         [most populous trained vehicle class]
  terrain    -> 17  (terrain)
  unlabeled  -> 0   (unlabeled)
"""

from __future__ import annotations

import os

import numpy as np
import yaml

# Canonical SemKITTI training class (0..19) per common class (0..8).
COMMON_TO_SEMKITTI_TRAIN = {
    0: 0,   # unlabeled -> unlabeled (ignored)
    1: 9,   # road
    2: 11,  # sidewalk
    3: 10,  # parking
    4: 13,  # building
    5: 14,  # fence
    6: 15,  # vegetation (not 16 trunk)
    7: 1,   # vehicle -> car
    8: 17,  # terrain
}

# Inverse: SemKITTI training class -> common class id, computed at module
# load time from semkitti_to_common composed with the SemKITTI learning_map_inv.
# Populated by load_semkitti_train_to_common().


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

    Training class 0 is always unlabeled. Any training class whose canonical
    raw label isn't in semkitti_to_common falls through to 0.
    """
    semk_to_common = {int(k): int(v) for k, v in common_cfg["semkitti_to_common"].items()}
    out: dict[int, int] = {}
    if not SEMKITTI_LEARNING_MAP_INV:
        raise RuntimeError("Call load_semkitti_config() before build_semkitti_train_to_common().")
    for train_cls, raw_label in SEMKITTI_LEARNING_MAP_INV.items():
        out[train_cls] = semk_to_common.get(raw_label, 0)
    # Make absolutely sure training 0 -> common 0
    out[0] = 0
    SEMKITTI_TRAIN_TO_COMMON.clear()
    SEMKITTI_TRAIN_TO_COMMON.update(out)
    return out


def build_source_channel_to_semkitti_train(
    inferred_labels_key: str,
    common_cfg: dict,
    n_channels: int,
) -> np.ndarray:
    """For each CENet network channel (0..n_channels-1), return the SemKITTI
    training class (0..19) that channel should be routed to.

    Path: channel -> raw source label (identity, since source learning_map_inv
    is identity for MCD and KITTI-360) -> common (via source_to_common) ->
    canonical SemKITTI training class (via COMMON_TO_SEMKITTI_TRAIN).

    Channels with no entry in source_to_common map to training class 0
    (unlabeled), so their mass is discarded.
    """
    map_key = inferred_labels_key + "_to_common"
    if map_key not in common_cfg:
        raise KeyError(f"labels_common.yaml has no mapping '{map_key}'")
    src_to_common = {int(k): int(v) for k, v in common_cfg[map_key].items()}
    route = np.zeros(n_channels, dtype=np.int32)
    for ch in range(n_channels):
        common_id = src_to_common.get(ch, 0)
        route[ch] = COMMON_TO_SEMKITTI_TRAIN.get(common_id, 0)
    return route


def build_raw_gt_to_semkitti_train(
    gt_labels_key: str,
    common_cfg: dict,
    max_raw_label: int = 65536,
) -> np.ndarray:
    """Lookup table: raw GT label -> SemKITTI training class.

    Path: raw GT -> common (via gt_to_common) -> canonical SemKITTI training
    class. Raw labels with no entry in gt_to_common collapse to training 0.
    The LUT is sized for uint16 GT (lower 16 bits of uint32 .bin), so KITTI-360's
    65535 sentinel resolves to whatever 65535 maps to in kitti360_to_common (0).
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


def aggregate_channels_to_semkitti_train(
    softmax_NK: np.ndarray,
    channel_to_train: np.ndarray,
    num_train_classes: int = 20,
) -> np.ndarray:
    """Project an (N, K) per-point softmax over CENet network channels into
    an (N, num_train_classes) soft distribution over SemKITTI training
    classes.

    Mirrors mcd_util.h:2020-2043 exactly: sum source channels into the
    target column, then divide by the per-column count of contributing
    source channels, then row-normalize. Multiple source channels feeding
    the same training class (e.g. MCD "shelter" and "building" both ->
    train 13 building, or MCD "vehicle-{dynamic,other,static}" all ->
    train 1 car) get averaged so the training-class belief equals the
    "average source-channel belief for the common class that routes
    there" -- bit-identical to the C++ 9-class aggregation up to column
    permutation.
    """
    N, K = softmax_NK.shape
    assert channel_to_train.shape == (K,)
    out = np.zeros((N, num_train_classes), dtype=np.float32)
    count = np.zeros(num_train_classes, dtype=np.int32)
    for ch in range(K):
        t = int(channel_to_train[ch])
        if t < 0 or t >= num_train_classes:
            continue
        out[:, t] += softmax_NK[:, ch]
        count[t] += 1
    nz_cols = count > 0
    out[:, nz_cols] = out[:, nz_cols] / count[nz_cols].astype(np.float32)
    row_sum = out.sum(axis=1, keepdims=True)
    nz_rows = (row_sum.squeeze(-1) > 1e-10)
    out[nz_rows] = out[nz_rows] / row_sum[nz_rows]
    return out


def softmax_from_float16_raw(raw_uint16: np.ndarray) -> np.ndarray:
    """Convert a uint16 buffer of float16 logits (as stored in CENet output
    files) into a row-softmax float32 array of shape (N, K), matching the
    C++ path in mcd_util.h:1996-2006.

    Caller is responsible for reshaping to (N, K) before calling.
    """
    logits = raw_uint16.view(np.float16).astype(np.float32)
    # Row-wise stable softmax
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    return e / np.maximum(s, 1e-30)
