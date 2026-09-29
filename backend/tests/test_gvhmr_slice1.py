import unittest

from app.pipeline.pose_track import COCO_EDGES
from app.pipeline.smpl_skeleton import SMPL_PARENTS, skeleton_edges


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
