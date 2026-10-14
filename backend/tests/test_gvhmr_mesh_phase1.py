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
