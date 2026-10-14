import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.pipeline.mesh_cache import (
    load_mesh_cache,
    load_smpl_sidecar,
    save_smpl_sidecar,
    sha256_file,
)
from app.schemas import AnalysisResult


class TestMeshCacheSchema(unittest.TestCase):
    def test_analysis_result_defaults_none(self) -> None:
        r = AnalysisResult(duration=1.0, fps=30.0)
        self.assertIsNone(r.smplSidecar)
        self.assertIsNone(r.meshCache)

    def test_load_smpl_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.dance.smpl.npz"
            np.savez_compressed(
                p,
                n_tracks=np.int32(1),
                n_frames_0=np.int32(2),
                g0_body_pose=np.zeros((2, 63), np.float32),
                g0_betas=np.zeros((2, 10), np.float32),
                g0_global_orient=np.zeros((2, 3), np.float32),
                g0_transl=np.zeros((2, 3), np.float32),
                c0_body_pose=np.zeros((2, 63), np.float32),
                c0_betas=np.zeros((2, 10), np.float32),
                c0_global_orient=np.zeros((2, 3), np.float32),
                c0_transl=np.zeros((2, 3), np.float32),
                K_0=np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(2, axis=0),
            )
            data = load_smpl_sidecar(p)
            self.assertEqual(int(data["n_tracks"]), 1)

    def test_load_mesh_cache_requires_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.dance.mesh.npz"
            np.savez_compressed(p, n_tracks=np.int32(0))
            with self.assertRaises(ValueError):
                load_mesh_cache(p)


class TestSaveSmplSidecar(unittest.TestCase):
    def test_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "clip.dance.smpl.npz"
            tracks = [
                {
                    "global": {
                        "body_pose": np.zeros((2, 63), np.float32),
                        "betas": np.zeros((2, 10), np.float32),
                        "global_orient": np.zeros((2, 3), np.float32),
                        "transl": np.zeros((2, 3), np.float32),
                    },
                    "incam": {
                        "body_pose": np.zeros((2, 63), np.float32),
                        "betas": np.zeros((2, 10), np.float32),
                        "global_orient": np.zeros((2, 3), np.float32),
                        "transl": np.zeros((2, 3), np.float32),
                    },
                    "K": np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(2, 0),
                }
            ]
            save_smpl_sidecar(p, tracks)
            self.assertTrue(p.is_file())
            self.assertEqual(len(sha256_file(p)), 64)
            load_smpl_sidecar(p)

    def test_mismatched_frame_count_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "clip.dance.smpl.npz"
            tracks = [
                {
                    "global": {
                        "body_pose": np.zeros((2, 63), np.float32),
                        "betas": np.zeros((2, 10), np.float32),
                        "global_orient": np.zeros((2, 3), np.float32),
                        "transl": np.zeros((2, 3), np.float32),
                    },
                    "incam": {
                        "body_pose": np.zeros((2, 63), np.float32),
                        "betas": np.zeros((2, 10), np.float32),
                        "global_orient": np.zeros((2, 3), np.float32),
                        "transl": np.zeros((2, 3), np.float32),
                    },
                    "K": np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(3, 0),
                }
            ]
            with self.assertRaises(ValueError) as ctx:
                save_smpl_sidecar(p, tracks)
            self.assertIn("K has T=3", str(ctx.exception))
            self.assertIn("expected T=2", str(ctx.exception))


class TestResolveSidecarBesideProject(unittest.TestCase):
    def test_missing_rel_or_file_returns_none(self) -> None:
        from app.project_io import resolve_smpl_sidecar_beside_project

        with tempfile.TemporaryDirectory() as tmp:
            dance = Path(tmp) / "proj.dance.json"
            dance.write_text("{}", encoding="utf-8")
            self.assertIsNone(resolve_smpl_sidecar_beside_project(dance, None))
            self.assertIsNone(
                resolve_smpl_sidecar_beside_project(dance, "missing.dance.smpl.npz")
            )

    def test_returns_path_when_sidecar_beside_json(self) -> None:
        from app.project_io import resolve_smpl_sidecar_beside_project

        with tempfile.TemporaryDirectory() as tmp:
            dance = Path(tmp) / "proj.dance.json"
            dance.write_text("{}", encoding="utf-8")
            sidecar = Path(tmp) / "proj.dance.smpl.npz"
            np.savez_compressed(sidecar, n_tracks=np.int32(0))
            got = resolve_smpl_sidecar_beside_project(dance, sidecar.name)
            self.assertEqual(got, sidecar)


class TestPersistSidecar(unittest.TestCase):
    def test_copy_sidecar_next_to_dance_json(self) -> None:
        from app.pipeline.mesh_cache import persist_sidecar_with_project

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "work" / "gvhmr.dance.smpl.npz"
            src.parent.mkdir()
            np.savez_compressed(src, n_tracks=np.int32(0))
            dance = Path(tmp) / "proj.dance.json"
            dance.write_text("{}", encoding="utf-8")
            rel = persist_sidecar_with_project(dance, src, sha256_file(src))
            self.assertEqual(rel, "proj.dance.smpl.npz")
            self.assertTrue((dance.parent / rel).is_file())
