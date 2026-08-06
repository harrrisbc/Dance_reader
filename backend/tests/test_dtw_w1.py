"""Identity + smoke tests for vectorised DTW (Wave 1)."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import numpy as np

from app.pipeline.motifs import (
    _dtw_distance,
    _dtw_distance_naive,
    build_segments,
    detect_motif_types,
)
from app.pipeline.metrics import frame_aspect, enrich_metrics
from app.project_io import buffers_from_analysis, load_project


FIXTURE_CANDIDATES = [
    Path(r"C:\Users\USER\Downloads") / "07 走路+武拜廟.dance.json",
    Path(__file__).resolve().parents[1]
    / "uploads"
    / "review_project_07 走路+武拜廟.dance.json",
]


def _find_positive_fixture() -> Path | None:
    for p in FIXTURE_CANDIDATES:
        if p.is_file():
            return p
    return None


class TestDtwIdentity(unittest.TestCase):
    def test_random_pairs_match_naive(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(40):
            n, m, d = int(rng.integers(3, 16)), int(rng.integers(3, 16)), int(rng.integers(4, 24))
            a = rng.standard_normal((n, d)).astype(np.float32)
            b = rng.standard_normal((m, d)).astype(np.float32)
            got = _dtw_distance(a, b)
            ref = _dtw_distance_naive(a, b)
            self.assertTrue(
                np.isclose(got, ref, rtol=1e-9, atol=1e-7),
                msg=f"got={got} ref={ref} shape=({n},{m},{d})",
            )

    def test_empty_and_identical(self) -> None:
        a = np.zeros((0, 4), dtype=np.float64)
        b = np.ones((5, 4), dtype=np.float64)
        self.assertEqual(_dtw_distance(a, b), _dtw_distance_naive(a, b))
        seq = np.linspace(0, 1, 28).reshape(14, 2).astype(np.float64)
        self.assertTrue(np.isclose(_dtw_distance(seq, seq), _dtw_distance_naive(seq, seq)))


class TestFixtureDtwIdentity(unittest.TestCase):
    """Full pairwise identity + timing on the positive regression fixture."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _find_positive_fixture()
        if cls.fixture is None:
            raise unittest.SkipTest("Positive fixture .dance.json not found")
        project = load_project(cls.fixture)
        enrich_metrics(project.analysis)
        cls.aspect = frame_aspect(
            int(project.analysis.frame_width or 0),
            int(project.analysis.frame_height or 0),
        )
        cls.actors = buffers_from_analysis(project.analysis)
        cls.segs = build_segments(cls.actors, aspect=cls.aspect)

    def test_every_pair_matches_naive(self) -> None:
        segs = self.segs
        n = len(segs)
        self.assertGreaterEqual(n, 2)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = segs[i], segs[j]
                for seq_a, seq_b in (
                    (a.seq, b.seq),
                    (a.seq, b.seq_mirror),
                ):
                    got = _dtw_distance(seq_a, seq_b)
                    ref = _dtw_distance_naive(seq_a, seq_b)
                    self.assertTrue(
                        np.isclose(got, ref, rtol=1e-9, atol=1e-7),
                        msg=f"pair ({i},{j}) got={got} ref={ref}",
                    )

    def test_pairwise_speed_budget(self) -> None:
        segs = self.segs
        n = len(segs)
        t0 = time.perf_counter()
        for i in range(n):
            for j in range(i + 1, n):
                _dtw_distance(segs[i].seq, segs[j].seq)
                _dtw_distance(segs[i].seq, segs[j].seq_mirror)
        elapsed = time.perf_counter() - t0
        # Acceptance: full pairwise under ~2 s (was ~20 s). Allow headroom on slow CPUs.
        self.assertLess(elapsed, 5.0, msg=f"pairwise DTW took {elapsed:.2f}s for {n} segs")

    def test_detect_motif_types_runs(self) -> None:
        types = detect_motif_types(self.actors, aspect=self.aspect)
        self.assertIsInstance(types, list)
        # Exact type count moves with Wave 2 features; W3 sets thresholds.


if __name__ == "__main__":
    unittest.main()
