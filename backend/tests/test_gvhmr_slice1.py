import unittest

import numpy as np

from app.pipeline.gvhmr_io import parse_gvhmr_document
from app.pipeline.metrics import enrich_metrics
from app.pipeline.pose_track import COCO_EDGES
from app.pipeline.smpl_skeleton import SMPL_PARENTS, skeleton_edges, smpl_keypoints_to_coco17
from app.schemas import Actor, AnalysisResult, PoseFrame


class TestSkeletonEdges(unittest.TestCase):
    def test_coco_17_uses_coco_edges(self) -> None:
        self.assertEqual(skeleton_edges(17), list(COCO_EDGES))

    def test_smpl_24_uses_parent_tree(self) -> None:
        edges = skeleton_edges(24)
        self.assertEqual(len(SMPL_PARENTS), 24)
        self.assertEqual(SMPL_PARENTS[0], -1)
        self.assertIn((1, 0), edges)  # left hip → pelvis
        self.assertNotIn((0, -1), edges)
        self.assertEqual(len(edges), 23)

    def test_other_counts_draw_nothing(self) -> None:
        self.assertEqual(skeleton_edges(0), [])
        self.assertEqual(skeleton_edges(10), [])


class TestCocoBridge(unittest.TestCase):
    def test_maps_body_and_zeros_face_confidence(self) -> None:
        smpl = [[float(i), float(i) + 0.1, 1.0] for i in range(24)]
        coco = smpl_keypoints_to_coco17(smpl)
        self.assertEqual(len(coco), 17)
        self.assertEqual(coco[0], [15.0, 15.1, 1.0])  # nose ← head
        self.assertEqual(coco[1][2], 0.0)  # eye confidence
        self.assertEqual(coco[5], [16.0, 16.1, 1.0])  # L shoulder
        self.assertEqual(coco[6], [17.0, 17.1, 1.0])  # R shoulder
        self.assertEqual(coco[11], [1.0, 1.1, 1.0])  # L hip
        self.assertEqual(coco[15], [7.0, 7.1, 1.0])  # L ankle
        self.assertEqual(coco[16], [8.0, 8.1, 1.0])  # R ankle

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            smpl_keypoints_to_coco17([[0.0, 0.0, 1.0]])


def _frame(t: float, n: int = 24) -> dict:
    return {
        "time": t,
        "joints3d": [[0.1, 0.2, 0.3] for _ in range(n)],
        "keypoints": [[0.4, 0.5, 1.0] for _ in range(n)],
    }


class TestParseGvhmr(unittest.TestCase):
    def test_sorts_longest_then_earlier_and_keeps_gap(self) -> None:
        doc = {
            "staticCamera": True,
            "coordFrame": "gvhmr_world",
            "ignoredTracks": 1,
            "tracks": [
                {"frames": [_frame(1.0), _frame(1.1)]},
                {"frames": [_frame(0.0), _frame(0.5), _frame(1.0)]},
            ],
        }
        tracks, ignored = parse_gvhmr_document(doc)
        self.assertEqual(ignored, 1)
        self.assertEqual(len(tracks), 2)
        self.assertEqual([p.time for p in tracks[0]], [0.0, 0.5, 1.0])
        self.assertEqual(len(tracks[0][0].joints3d), 24)
        self.assertEqual(len(tracks[0][0].keypoints), 24)
        self.assertNotIn(0.25, [p.time for p in tracks[0]])

    def test_tie_uses_earlier_first_sample(self) -> None:
        doc = {
            "ignoredTracks": 0,
            "tracks": [
                {"frames": [_frame(2.0), _frame(3.0)]},
                {"frames": [_frame(0.5), _frame(1.5)]},
            ],
        }
        tracks, _ = parse_gvhmr_document(doc)
        self.assertEqual(tracks[0][0].time, 0.5)

    def test_rejects_short_joints(self) -> None:
        doc = {"ignoredTracks": 0, "tracks": [{"frames": [_frame(0.0, n=3)]}]}
        with self.assertRaises(ValueError):
            parse_gvhmr_document(doc)

    def test_old_analysis_json_still_loads(self) -> None:
        old = {
            "version": 1,
            "duration": 1.0,
            "fps": 30.0,
            "actors": [{"id": "track_0", "label": "Dancer 1", "poses": [{"time": 0.0, "keypoints": [[0, 0, 1]]}]}],
        }
        result = AnalysisResult.model_validate(old)
        self.assertEqual(result.poseBackend, "yolo")
        self.assertIsNone(result.actors[0].poses[0].joints3d)
        self.assertFalse(result.staticCamera)


class TestEnrichKeepsSmplKeypoints(unittest.TestCase):
    def test_metrics_compute_but_stored_pose_stays_24(self) -> None:
        kpts = [[0.5, 0.5, 1.0] for _ in range(24)]
        # Give the head and ankles distinct image positions so level/reach are finite.
        kpts[15] = [0.5, 0.2, 1.0]
        kpts[7] = [0.45, 0.9, 1.0]
        kpts[8] = [0.55, 0.9, 1.0]
        poses = [
            PoseFrame(time=0.0, keypoints=kpts, joints3d=[[0.0, 0.0, 0.0] for _ in range(24)]),
            PoseFrame(time=0.1, keypoints=kpts, joints3d=[[0.0, 0.0, 0.0] for _ in range(24)]),
        ]
        result = AnalysisResult(
            duration=0.1,
            fps=10.0,
            frame_width=100,
            frame_height=100,
            poseBackend="gvhmr",
            actors=[Actor(id="track_0", label="Dancer 1", poses=poses)],
        )
        enrich_metrics(result)
        self.assertEqual(len(result.actors[0].poses[0].keypoints), 24)
        self.assertGreaterEqual(len(result.actors[0].metrics), 1)
        self.assertTrue(np.isfinite(result.actors[0].metrics[0].energy))
