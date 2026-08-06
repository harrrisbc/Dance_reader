"""Over-merge regression: auto discovery prefers precision over a mega-type."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.pipeline.metrics import enrich_metrics, frame_aspect
from app.pipeline.motifs import (
    CLUSTER_LINKAGE,
    CLUSTER_MAX_DIST,
    INTRA_MAX_DIST,
    detect_motif_types,
    find_similar_motif,
    query_segment_at_time,
)
from app.project_io import buffers_from_analysis, load_project

POS = Path(r"C:\Users\USER\Downloads") / "07 走路+武拜廟.dance.json"
NEG = Path(r"C:\Users\USER\Downloads\hi.dance.json")


class TestGateDefaults(unittest.TestCase):
    def test_precision_band(self) -> None:
        self.assertLessEqual(CLUSTER_MAX_DIST, 0.24)
        self.assertGreaterEqual(CLUSTER_MAX_DIST, 0.22)
        self.assertAlmostEqual(INTRA_MAX_DIST, CLUSTER_MAX_DIST + 0.02, places=5)
        self.assertEqual(CLUSTER_LINKAGE, "complete")


@unittest.skipUnless(POS.is_file(), "positive fixture missing")
class TestNoMegaType(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        proj = load_project(POS)
        enrich_metrics(proj.analysis)
        cls.asp = frame_aspect(proj.analysis.frame_width, proj.analysis.frame_height)
        cls.actors = buffers_from_analysis(proj.analysis)
        cls.aid = next(iter(cls.actors))
        cls.types = detect_motif_types(cls.actors, aspect=cls.asp)

    def test_not_single_mega_type(self) -> None:
        # Reject W3 plateau: 1 type with ≥6 occ on this ~16 s clip
        if len(self.types) == 1:
            self.assertLess(
                len(self.types[0].occurrences),
                6,
                "single mega-type still present",
            )
        for t in self.types:
            self.assertIsNotNone(t.meanIntraDist)
            self.assertLessEqual(t.meanIntraDist or 0.0, INTRA_MAX_DIST + 1e-6)

    def test_ten_twelve_co_cluster_or_find_similar(self) -> None:
        """10/12 should share a type, or Find Similar from ~8 still recovers both."""
        def type_covering(t: float) -> str | None:
            for mt in self.types:
                for o in mt.occurrences:
                    if o.start - 0.4 <= t <= o.end + 0.4:
                        return mt.id
            return None

        t10 = type_covering(10.8)
        t12 = type_covering(12.1)
        if t10 and t12 and t10 == t12:
            return  # preferred: true partners together
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
class TestNegativeZeroTypes(unittest.TestCase):
    def test_hi_zero_types(self) -> None:
        proj = load_project(NEG)
        enrich_metrics(proj.analysis)
        asp = frame_aspect(proj.analysis.frame_width, proj.analysis.frame_height)
        actors = buffers_from_analysis(proj.analysis)
        types = detect_motif_types(actors, aspect=asp)
        self.assertEqual(len(types), 0)


if __name__ == "__main__":
    unittest.main()
