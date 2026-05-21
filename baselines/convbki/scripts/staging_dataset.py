"""StagingKittiDataset: glob-based overlay of NeuralBKI's KittiDataset.

The upstream Data/SemanticKitti.py loader makes two assumptions that
fail for our datasets:

  1. Scan IDs are pure integers padded to 6 digits. KITTI-360 uses
     10-digit IDs and MCD uses long timestamps.
  2. find_horizon() measures temporal adjacency via raw frame-ID
     differences. KITTI-360 / MCD IDs are sparse (many gaps), so this
     would mark every history frame as zero-padded.

It also reads SPLIT_SEQUENCES out of Config/semantic_kitti.yaml at
module-import time, which is awkward for per-sequence staging trees.

This subclass:
  - Bypasses SPLIT_SEQUENCES; the caller passes the staging root and
    sequence list directly.
  - Pairs velodyne / labels / predictions by exact filename stem from
    sorted(glob('velodyne/*.bin')) — no f"{i:06d}" recomputation.
  - Disables the inherited remap LUT (label/channel remapping is done
    once at staging time, not at __getitem__ time).
  - Overrides find_horizon to use list-index distance, so num_frames>1
    works on sparse IDs.

Working-directory contract: the importing script must chdir to
baselines/convbki/NeuralBKI before `from Data.SemanticKitti import ...`,
because upstream reads Config/semantic_kitti.yaml at import time.
"""

import glob
import os

import numpy as np

from Data.SemanticKitti import KittiDataset


class StagingKittiDataset(KittiDataset):
    def __init__(
        self,
        grid_params,
        staging_root,
        sequences,
        device="cuda",
        num_frames=1,
        voxelize_input=True,
        binary_counts=False,
        random_flips=False,
        use_aug=False,
        apply_transform=True,
        from_continuous=True,
        to_continuous=False,
        pred_path="predictions_softmax",
        num_classes=9,
        remove_zero=False,
    ):
        # Intentionally not calling super().__init__(): it reads
        # SPLIT_SEQUENCES, expects a single shared directory layout, and
        # would recompute filenames via zfill(6).
        self.remove_zero = remove_zero
        self.from_continuous = from_continuous
        self.to_continuous = to_continuous
        self.use_aug = use_aug
        self.apply_transform = apply_transform
        self.num_classes = num_classes

        self._grid_size = grid_params["grid_size"]
        self.grid_dims = np.asarray(self._grid_size)
        self._eval_size = list(np.uint32(self._grid_size))
        self.coor_ranges = grid_params["min_bound"] + grid_params["max_bound"]
        self.voxel_sizes = np.asarray(
            [
                abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_size[0],
                abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_size[1],
                abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_size[2],
            ]
        )
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])

        self.voxelize_input = voxelize_input
        self.binary_counts = binary_counts
        self._directory = os.path.join(staging_root, "sequences")
        self._num_frames = num_frames
        self.device = device
        self.random_flips = random_flips

        # All taxonomy work is done at staging time; the loader should
        # treat label values as already-final 9-class indices.
        self.remap = False
        self.split = "all"
        # Identity LUT large enough for any uint8 value, used when
        # callers flip self.remap back on.
        self._remap_lut = np.arange(256, dtype=np.int32)

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._eval_labels = []
        self._eval_valid = []
        self._frames_list = []
        self._timestamps = []
        self._poses = np.empty((0, 12))
        self._Tr = np.empty((0, 12))
        self._num_frames_scene = []
        self._seqs = list(sequences)
        self._scene_id = []

        for seq in self._seqs:
            velodyne_dir = os.path.join(self._directory, seq, "velodyne")
            label_dir = os.path.join(self._directory, seq, "labels")
            preds_dir = os.path.join(self._directory, seq, pred_path)
            poses_path = os.path.join(self._directory, seq, "poses.txt")
            calib_path = os.path.join(self._directory, seq, "calib.txt")

            scan_paths = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
            stems = [os.path.splitext(os.path.basename(p))[0] for p in scan_paths]
            n_scans = len(scan_paths)

            self._num_frames_scene.append(n_scans)
            self._scene_id += [seq] * n_scans

            pose = np.loadtxt(poses_path)
            if pose.ndim == 1:
                pose = pose.reshape(1, -1)
            assert pose.shape == (n_scans, 12), (
                f"poses.txt for {seq} has shape {pose.shape}, expected "
                f"({n_scans}, 12); one 3x4 row per scan in stem-sorted order"
            )
            Tr_row = np.genfromtxt(calib_path)[-1, 1:]
            Tr = np.repeat(Tr_row[None, :], pose.shape[0], axis=0)
            self._Tr = np.vstack((self._Tr, Tr))
            self._poses = np.vstack((self._poses, pose))

            self._frames_list.extend(stems)
            self._velodyne_list.extend(scan_paths)
            self._label_list.extend(
                [os.path.join(label_dir, stem + ".label") for stem in stems]
            )
            self._pred_list.extend(
                [os.path.join(preds_dir, stem + ".label") for stem in stems]
            )

        assert len(self._velodyne_list) == sum(
            self._num_frames_scene
        ), "inconsistent number of frames"

        n = sum(self._num_frames_scene)
        self._poses = self._poses.reshape(n, 12)
        self._Tr = self._Tr.reshape(n, 12)

    def find_horizon(self, idx):
        """Use list-index distance instead of raw frame-ID distance.

        Upstream version compares int(frames_list[end]) - int(frames_list[i])
        against the expected 1,2,3,... — assumes contiguous integer IDs.
        For sparse IDs this fails and every history frame becomes -1
        (zero-padded), which silently degrades training when num_frames>1.
        We instead treat consecutive list indices as adjacent and only
        mark -1 for out-of-range or cross-sequence indices.
        """
        idx_range = np.arange(idx - self._num_frames, idx) + 1
        idx_range = np.where(idx_range >= 0, idx_range, -1)
        if idx < len(self._scene_id):
            end_scene = self._scene_id[idx]
            for k, i in enumerate(idx_range):
                if i == -1:
                    continue
                if self._scene_id[i] != end_scene:
                    idx_range[k] = -1
        return idx_range

    def stem_for_index(self, idx):
        """Original scan stem for index `idx`.

        Driver uses this to write predictions/<stem>.label instead of
        predictions/<frame_counter:06d>.label, so outputs match the
        on-disk scan IDs the eval notebook expects.
        """
        return self._frames_list[idx]
