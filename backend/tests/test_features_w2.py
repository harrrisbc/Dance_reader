"""Wave 2 feature-correctness checks (masks, padding, energy joints)."""

from __future__ import annotations

import unittest

import numpy as np

from app.pipeline.metrics import (
    KEY_INDICES,
    limb_energy_and_speed,
    pose_feature,
    pose_feature_rich,
)
from app.pipeline.motifs import (
    MIN_SHARED_VISIBLE_JOINTS,
    _dtw_distance,
    _masked_dist_matrix,
    _shared_visible_joints,
)


def _fake_pose(*, occlude_wrists: bool = False) -> np.ndarray:
    """Minimal COCO-17 pose in normalized coords with usable torso."""
    k = np.zeros((17, 3), dtype=np.float32)
    # shoulders / hips — visible
    k[5] = [0.40, 0.30, 0.9]
    k[6] = [0.60, 0.30, 0.9]
    k[11] = [0.42, 0.55, 0.9]
    k[12] = [0.58, 0.55, 0.9]
    # elbows / wrists / knees / ankles
    for i, xy in {
        7: (0.35, 0.40),
        8: (0.65, 0.40),
        9: (0.30, 0.50),
        10: (0.70, 0.50),
        13: (0.42, 0.70),
        14: (0.58, 0.70),
        15: (0.42, 0.85),
        16: (0.58, 0.85),
        0: (0.50, 0.20),
    }.items():
        k[i] = [xy[0], xy[1], 0.9]
    if occlude_wrists:
        k[9, 2] = 0.05
        k[10, 2] = 0.05
    return k


class TestOcclusionMask(unittest.TestCase):
    def test_occluded_joint_not_marked_visible(self) -> None:
        packed = pose_feature(_fake_pose(occlude_wrists=True))
        self.assertIsNotNone(packed)
        feat, mask = packed  # type: ignore[misc]
        # wrists are KEY_INDICES positions 9,10 → indices 4,5 in KEY_INDICES list
        wrist_l = KEY_INDICES.index(9)
        wrist_r = KEY_INDICES.index(10)
        self.assertFalse(bool(mask[2 * wrist_l]))
        self.assertFalse(bool(mask[2 * wrist_r]))
        # Visible shoulder still True
        sh_l = KEY_INDICES.index(5)
        self.assertTrue(bool(mask[2 * sh_l]))

    def test_low_vis_frame_not_cheaper_than_median(self) -> None:
        a = np.random.default_rng(1).standard_normal((6, 40)).astype(np.float64)
        b = a.copy() + 0.1
        mask_a = np.ones((6, 40), dtype=bool)
        mask_b = np.ones((6, 40), dtype=bool)
        # Frame 2: almost no shared joints
        mask_a[2, :] = False
        mask_a[2, :2] = True  # one joint only
        dists, median = _masked_dist_matrix(a, b, mask_a, mask_b)
        # Low-vis cell must be ≥ median of well-observed pairs
        self.assertGreaterEqual(dists[2, 2], median - 1e-12)
        self.assertGreaterEqual(
            _shared_visible_joints(mask_a[0], mask_b[0]), MIN_SHARED_VISIBLE_JOINTS
        )

    def test_padded_zero_mask_excluded(self) -> None:
        seq = np.ones((4, 8), dtype=np.float64)
        mask = np.ones((4, 8), dtype=bool)
        mask[-1] = False  # padded frame
        other = np.zeros((4, 8), dtype=np.float64)
        other_m = np.ones((4, 8), dtype=bool)
        d_full = _dtw_distance(seq, other, mask, other_m)
        # If pad were treated as visible ones-vs-zeros it would inflate; with mask
        # the pad contributes imputed ≥ median, never a free zero-cost match to zeros.
        self.assertGreater(d_full, 0.0)

    def test_energy_uses_key_indices_only(self) -> None:
        prev = _fake_pose()
        cur = _fake_pose()
        # Move only the nose (not in KEY_INDICES) a lot
        cur[0, 0] += 0.5
        e_nose, s_nose = limb_energy_and_speed(
            prev, cur, 1 / 30, aspect=16 / 9, scale=0.2
        )
        # Move a wrist the same amount instead
        cur2 = _fake_pose()
        cur2[9, 0] += 0.5
        e_wrist, s_wrist = limb_energy_and_speed(
            prev, cur2, 1 / 30, aspect=16 / 9, scale=0.2
        )
        self.assertEqual(e_nose, 0.0)
        self.assertEqual(s_nose, 0.0)
        self.assertGreater(e_wrist, 0.0)
        self.assertGreater(s_wrist, 0.0)

    def test_rich_occluded_bone_masked(self) -> None:
        packed = pose_feature_rich(_fake_pose(occlude_wrists=True))
        self.assertIsNotNone(packed)
        _feat, mask = packed  # type: ignore[misc]
        # Bone (7,9) left elbow→wrist: bone index 1 in BONE_PAIRS → dims after XY
        bone_offset = len(KEY_INDICES) * 2
        # second bone pair is (7,9) → index 1 → dims bone_offset+2, +3
        self.assertFalse(bool(mask[bone_offset + 2]))
        self.assertFalse(bool(mask[bone_offset + 3]))


if __name__ == "__main__":
    unittest.main()
