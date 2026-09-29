import unittest

from app.pipeline.pose_track import COCO_EDGES
from app.pipeline.smpl_skeleton import SMPL_PARENTS, skeleton_edges, smpl_keypoints_to_coco17


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
