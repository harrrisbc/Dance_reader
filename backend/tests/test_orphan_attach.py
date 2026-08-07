"""Orphan phrase attach: ~8s joins the 10/12 type without loosening cluster cuts."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.pipeline.metrics import enrich_metrics, frame_aspect
from app.pipeline.motifs import (
    CLUSTER_LINKAGE,
    CLUSTER_MAX_DIST,
    INTRA_MAX_DIST,
    LAST_DETECT_STATS,
    ORPHAN_ATTACH_MAX_DIST,
    SUBSEQ_STRONG_DIST,
    detect_motif_types,
    find_similar_motif,
    query_segment_at_time,
)
from app.project_io import buffers_from_analysis, load_project

DOWNLOADS = Path(r"C:\Users\USER\Downloads")
POS_CANDIDATES = [
    DOWNLOADS / "07 走路+武拜廟.dance 2.dance.json",
    DOWNLOADS / "07 走路+武拜廟.dance.json",
]
POS = next((p for p in POS_CANDIDATES if p.is_file()), POS_CANDIDATES[-1])
NEG = DOWNLOADS / "hi.dance.json"


class TestGatesUnchanged(unittest.TestCase):
    def test_cluster_untouched(self) -> None:
        self.assertEqual(CLUSTER_MAX_DIST, 0.24)
        self.assertEqual(INTRA_MAX_DIST, 0.26)
        self.assertEqual(CLUSTER_LINKAGE, "complete")
        self.assertEqual(ORPHAN_ATTACH_MAX_DIST, SUBSEQ_STRONG_DIST)


@unittest.skipUnless(POS.is_file(), f"positive fixture missing: {POS}")
class TestOrphanAttachPositive(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        proj = load_project(POS)
        enrich_metrics(proj.analysis)
        cls.asp = frame_aspect(proj.analysis.frame_width, proj.analysis.frame_height)
        cls.actors = buffers_from_analysis(proj.analysis)
        cls.aid = next(iter(cls.actors))
        cls.types = detect_motif_types(cls.actors, aspect=cls.asp)

    def test_covers_eight_and_ten_twelve(self) -> None:
        self.assertGreaterEqual(len(self.types), 1)

        def covers(t: float, tol: float = 0.55) -> bool:
            for mt in self.types:
                for o in mt.occurrences:
                    mid = 0.5 * (o.start + o.end)
                    if abs(mid - t) <= tol or (o.start - 0.2 <= t <= o.end + 0.2):
                        return True
            return False

        self.assertTrue(covers(8.0), "expected an occurrence near ~8s after orphan attach")
        self.assertTrue(covers(10.8), "expected ~10.6 region")
        self.assertTrue(covers(12.1), "expected ~12 region")

        # Same type should own 10 + 12 and the ~8 attach
        owner_10 = None
        owner_12 = None
        owner_8 = None
        for mt in self.types:
            for o in mt.occurrences:
                mid = 0.5 * (o.start + o.end)
                if abs(mid - 10.8) <= 0.6 or (o.start <= 10.8 <= o.end):
                    owner_10 = mt.id
                if abs(mid - 12.1) <= 0.6 or (o.start <= 12.1 <= o.end):
                    owner_12 = mt.id
                if abs(mid - 8.0) <= 0.55 or (o.start <= 8.0 <= o.end):
                    owner_8 = mt.id
        self.assertIsNotNone(owner_10)
        self.assertEqual(owner_10, owner_12)
        self.assertEqual(owner_10, owner_8)

    def test_no_mega_type(self) -> None:
        for mt in self.types:
            self.assertLess(
                len(mt.occurrences),
                6,
                f"{mt.name} looks like a MEGA_TYPE ({len(mt.occurrences)} occ)",
            )

    def test_diag_mentions_attaches(self) -> None:
        self.assertGreaterEqual(LAST_DETECT_STATS.orphan_attaches, 1)
        self.assertIn("orphan", LAST_DETECT_STATS.summary())

    def test_find_similar_still_works(self) -> None:
        q = query_segment_at_time(
            self.actors, self.aid, 8.2, start=7.95, end=8.50, aspect=self.asp
        )
        self.assertIsNotNone(q)
        found = find_similar_motif(self.actors, q, aspect=self.asp)
        self.assertIsNotNone(found)
        assert found is not None
        others = [o for o in found.occurrences if not o.is_query]

        def covers(t: float, tol: float = 0.6) -> bool:
            return any(o.start - tol <= t <= o.end + tol for o in others)

        self.assertTrue(covers(10.3))
        self.assertTrue(covers(12.0))


@unittest.skipUnless(NEG.is_file(), "negative fixture missing")
class TestOrphanAttachNegative(unittest.TestCase):
    def test_hi_still_zero(self) -> None:
        proj = load_project(NEG)
        enrich_metrics(proj.analysis)
        asp = frame_aspect(proj.analysis.frame_width, proj.analysis.frame_height)
        actors = buffers_from_analysis(proj.analysis)
        types = detect_motif_types(actors, aspect=asp)
        self.assertEqual(len(types), 0)
        self.assertEqual(LAST_DETECT_STATS.orphan_attaches, 0)


if __name__ == "__main__":
    unittest.main()
