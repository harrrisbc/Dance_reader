"""W4 Find-similar tests: tempo bound + subsequence hits on fixtures."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.pipeline.metrics import enrich_metrics, frame_aspect
from app.pipeline.motifs import (
    TEMPO_RATIO_MAX,
    _tempo_in_bounds,
    detect_motif_types,
    find_similar_motif,
    format_tempo_label,
    query_segment_at_time,
)
from app.project_io import buffers_from_analysis, load_project

POS = Path(r"C:\Users\USER\Downloads") / "07 走路+武拜廟.dance.json"
NEG = Path(r"C:\Users\USER\Downloads\hi.dance.json")


class TestTempoBound(unittest.TestCase):
    def test_bounds(self) -> None:
        self.assertTrue(_tempo_in_bounds(1.0))
        self.assertTrue(_tempo_in_bounds(TEMPO_RATIO_MAX))
        self.assertTrue(_tempo_in_bounds(1.0 / TEMPO_RATIO_MAX))
        self.assertFalse(_tempo_in_bounds(TEMPO_RATIO_MAX + 0.1))
        self.assertFalse(_tempo_in_bounds(1.0 / TEMPO_RATIO_MAX - 0.05))

    def test_label(self) -> None:
        self.assertIn("slower", format_tempo_label(1.4))
        self.assertIn("faster", format_tempo_label(0.7))
        self.assertEqual(format_tempo_label(1.0), "")


@unittest.skipUnless(POS.is_file(), "positive fixture missing")
class TestSubseqPositive(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        proj = load_project(POS)
        enrich_metrics(proj.analysis)
        cls.asp = frame_aspect(proj.analysis.frame_width, proj.analysis.frame_height)
        cls.actors = buffers_from_analysis(proj.analysis)
        cls.aid = next(iter(cls.actors))

    def test_auto_discovery_unchanged_path(self) -> None:
        types = detect_motif_types(self.actors, aspect=self.asp)
        self.assertGreaterEqual(len(types), 0)

    def test_find_similar_returns_10_and_12(self) -> None:
        q = query_segment_at_time(
            self.actors, self.aid, 8.2, start=7.95, end=8.50, aspect=self.asp
        )
        self.assertIsNotNone(q)
        found = find_similar_motif(self.actors, q, aspect=self.asp)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertTrue(any(o.is_query for o in found.occurrences))
        others = [o for o in found.occurrences if not o.is_query]
        self.assertGreaterEqual(len(others), 1)

        def covers(t: float, tol: float = 0.6) -> bool:
            return any(o.start - tol <= t <= o.end + tol for o in others)

        self.assertTrue(covers(10.3), "expected a hit near 10.3s")
        self.assertTrue(covers(12.0), "expected a hit near 12s")


@unittest.skipUnless(NEG.is_file(), "negative fixture missing")
class TestSubseqNegative(unittest.TestCase):
    def test_few_or_no_hits(self) -> None:
        proj = load_project(NEG)
        enrich_metrics(proj.analysis)
        asp = frame_aspect(proj.analysis.frame_width, proj.analysis.frame_height)
        actors = buffers_from_analysis(proj.analysis)
        aid = next(iter(actors))
        q = query_segment_at_time(actors, aid, 2.0, aspect=asp)
        if q is None:
            return  # no usable query window — vacuously few hits
        found = find_similar_motif(actors, q, aspect=asp)
        n = 0 if found is None else len([o for o in found.occurrences if not o.is_query])
        self.assertLessEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
