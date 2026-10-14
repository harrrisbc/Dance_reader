import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.metrics import (  # noqa: E402
    SMPL_L_ANKLE,
    SMPL_L_HIP,
    SMPL_L_SHOULDER,
    SMPL_PELVIS,
    SMPL_R_ANKLE,
    SMPL_R_HIP,
    SMPL_R_SHOULDER,
    compute_metrics_smpl3d,
    enrich_metrics,
    level_stance_ratio_2d,
    level_stance_ratio_3d,
)
from app.pipeline.pose_track import TrackBuffer, TrackedPose  # noqa: E402
from app.schemas import Actor, AnalysisResult, MetricPoint, PoseFrame  # noqa: E402


def _standing_smpl(*, x: float = 0.0, z: float = 0.0, crouch: float = 0.0) -> np.ndarray:
    """Minimal Y-up SMPL-like pose: torso ~0.5m, ankles near y=0."""
    j = np.zeros((24, 3), dtype=np.float64)
    j[SMPL_PELVIS] = [x, 0.95 - crouch, z]
    j[SMPL_L_HIP] = [x - 0.1, 0.9 - crouch, z]
    j[SMPL_R_HIP] = [x + 0.1, 0.9 - crouch, z]
    j[SMPL_L_SHOULDER] = [x - 0.15, 1.4 - crouch, z]
    j[SMPL_R_SHOULDER] = [x + 0.15, 1.4 - crouch, z]
    j[SMPL_L_ANKLE] = [x - 0.1, 0.05, z]
    j[SMPL_R_ANKLE] = [x + 0.1, 0.05, z]
    # fill other joints near pelvis so energy has finite limbs
    for i in range(24):
        if np.allclose(j[i], 0.0):
            j[i] = j[SMPL_PELVIS] + np.array([0.01 * i, 0.0, 0.0])
    return j


class TestMetricsPhase2Schema(unittest.TestCase):
    def test_defaults(self) -> None:
        r = AnalysisResult(duration=1.0, fps=30.0)
        self.assertEqual(r.metricsSpace, "coco2d")
        p = MetricPoint(time=0.0, energy=0.0, speed=0.0)
        self.assertEqual(p.pos_foot_x, 0.0)
        self.assertEqual(p.pos_foot_y, 0.0)

    def test_roundtrip_smpl3d(self) -> None:
        r = AnalysisResult(duration=1.0, fps=30.0, metricsSpace="smpl3d")
        d = r.model_dump()
        self.assertEqual(AnalysisResult.model_validate(d).metricsSpace, "smpl3d")


class TestLevel2d(unittest.TestCase):
    def test_supporting_ankle_not_lifted_leg(self) -> None:
        sq = np.zeros((17, 3), np.float32)
        sq[:, 2] = 1.0
        sq[5] = [50, 40, 1]
        sq[6] = [70, 40, 1]
        sq[11] = [50, 100, 1]
        sq[12] = [70, 100, 1]
        sq[15] = [55, 180, 1]
        sq[16] = [65, 120, 1]
        S = 60.0
        lv = level_stance_ratio_2d(sq, S)
        self.assertIsNotNone(lv)
        self.assertAlmostEqual(float(lv), abs(180 - 100) / 60.0, places=3)


class TestLevel3d(unittest.TestCase):
    def test_lifted_leg_ignored(self) -> None:
        j = _standing_smpl()
        j[SMPL_R_ANKLE, 1] = 0.6  # lifted
        s = 0.5
        lv = level_stance_ratio_3d(j, s)
        self.assertIsNotNone(lv)
        # uses left ankle at 0.05
        hip_y = 0.5 * (j[SMPL_L_HIP, 1] + j[SMPL_R_HIP, 1])
        self.assertAlmostEqual(float(lv), (hip_y - 0.05) / s, places=3)

    def test_level_flat_on_turn(self) -> None:
        a = _standing_smpl(x=0.0, z=0.0)
        b = _standing_smpl(x=0.3, z=0.2)  # translate on floor, same heights
        s = 0.5
        self.assertAlmostEqual(
            float(level_stance_ratio_3d(a, s)),
            float(level_stance_ratio_3d(b, s)),
            places=5,
        )


class TestMetricsSmpl3d(unittest.TestCase):
    def test_travel_uses_pelvis_xz(self) -> None:
        buf = TrackBuffer(track_id=0)
        for i, x in enumerate((0.0, 0.5, 1.0)):
            buf.poses.append(
                TrackedPose(
                    time=float(i) * 0.1,
                    track_id=0,
                    keypoints=np.zeros((24, 3), np.float32),
                    xyxy=np.zeros(4, np.float32),
                    joints3d=_standing_smpl(x=x, z=0.0).astype(np.float32),
                )
            )
        pts = compute_metrics_smpl3d(buf)
        self.assertGreaterEqual(len(pts), 2)
        # Origin is the first plotted pelvis (second pose, x=0.5) → last X = 0.5
        self.assertAlmostEqual(pts[0].pos_x, 0.0, places=3)
        self.assertAlmostEqual(pts[0].pos_y, 0.0, places=3)
        self.assertAlmostEqual(pts[-1].pos_x, 0.5, places=3)
        self.assertGreater(pts[-1].travel, 0.0)
        self.assertGreater(max(p.energy_abs for p in pts), 0.0)
        self.assertAlmostEqual(max(p.energy for p in pts), 1.0, places=3)

    def test_enrich_sets_metrics_space(self) -> None:
        poses = []
        for i in range(3):
            j = _standing_smpl(x=0.1 * i).tolist()
            poses.append(
                PoseFrame(
                    time=0.1 * i,
                    keypoints=[[0.5, 0.5, 1.0]] * 24,
                    joints3d=j,
                )
            )
        result = AnalysisResult(
            duration=1.0,
            fps=30.0,
            actors=[Actor(id="track_0", label="Dancer 1", poses=poses)],
        )
        enrich_metrics(result)
        self.assertEqual(result.metricsSpace, "smpl3d")
        self.assertGreater(len(result.actors[0].metrics), 0)
        # World metres should not be clipped to 0–1 screen space for a 1m walk
        self.assertGreater(abs(result.actors[0].metrics[-1].pos_x), 0.05)


if __name__ == "__main__":
    unittest.main()
