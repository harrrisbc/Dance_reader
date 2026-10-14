import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.pipeline.mesh_cache import (
    load_mesh_cache,
    load_smpl_sidecar,
    save_mesh_cache,
    save_smpl_sidecar,
    sha256_file,
)
from app.schemas import AnalysisResult


class TestProjectVerts(unittest.TestCase):
    def test_shape(self) -> None:
        from app.pipeline.mesh_project import project_verts_to_pixels

        v = np.array([[0.0, 0.0, 2.0]], np.float32)
        k = np.array(
            [[1000.0, 0, 500.0], [0, 1000.0, 500.0], [0, 0, 1]],
            np.float32,
        )
        uv = project_verts_to_pixels(v, k, width=1000, height=1000)
        self.assertEqual(uv.shape, (1, 2))


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


class TestMeshBakeRunner(unittest.TestCase):
    def _gvhmr_env(self, tmp: str) -> dict[str, str]:
        root = Path(tmp) / "gvhmr"
        weight = root / "inputs" / "checkpoints" / "yolo" / "yolov8x.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"x")
        return {
            "DANCE_GVHMR_PYTHON": sys.executable,
            "DANCE_GVHMR_ROOT": str(root),
        }

    def test_missing_python_raises_gvhmr_not_configured(self) -> None:
        from app.pipeline.gvhmr_run import GvhmrNotConfigured, run_mesh_bake

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GvhmrNotConfigured) as ctx:
                run_mesh_bake(
                    smpl_sidecar=Path("a.dance.smpl.npz"),
                    mesh_out=Path("a.dance.mesh.npz"),
                    video=None,
                )
        self.assertIn("DANCE_GVHMR_PYTHON", str(ctx.exception))

    def test_success_subprocess_pythonpath_and_args(self) -> None:
        from app.pipeline.gvhmr_run import run_mesh_bake

        backend_dir = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "clip.dance.smpl.npz"
            sidecar.write_bytes(b"s")
            mesh_out = Path(tmp) / "clip.dance.mesh.npz"
            mesh_out.write_bytes(b"m")
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"v")
            env = self._gvhmr_env(tmp)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("app.pipeline.gvhmr_run.subprocess.run") as run_mock:
                    run_mock.return_value = mock.Mock(returncode=0, stderr="")
                    got = run_mesh_bake(
                        smpl_sidecar=sidecar,
                        mesh_out=mesh_out,
                        video=video,
                    )
            self.assertEqual(got, mesh_out)
            run_mock.assert_called_once()
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["cwd"], env["DANCE_GVHMR_ROOT"])
            py_path = kwargs["env"]["PYTHONPATH"]
            self.assertIn(str(backend_dir), py_path.split(os.pathsep))
            argv = run_mock.call_args.args[0]
            self.assertEqual(argv[0], sys.executable)
            self.assertIn("--smpl", argv)
            self.assertIn(str(sidecar), argv)
            self.assertIn("--out", argv)
            self.assertIn(str(mesh_out), argv)
            self.assertIn("--video", argv)
            self.assertIn(str(video), argv)

    def test_nonzero_exit_raises_mesh_bake_error_with_stderr(self) -> None:
        from app.pipeline.gvhmr_run import MeshBakeError, run_mesh_bake

        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "clip.dance.smpl.npz"
            sidecar.write_bytes(b"s")
            mesh_out = Path(tmp) / "clip.dance.mesh.npz"
            env = self._gvhmr_env(tmp)
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("app.pipeline.gvhmr_run.subprocess.run") as run_mock:
                    run_mock.return_value = mock.Mock(
                        returncode=2,
                        stderr="ignored\nboom-from-bake\n",
                    )
                    with self.assertRaises(MeshBakeError) as ctx:
                        run_mesh_bake(
                            smpl_sidecar=sidecar,
                            mesh_out=mesh_out,
                            video=None,
                        )
        self.assertIn("boom-from-bake", str(ctx.exception))


class TestOverlayPaths(unittest.TestCase):
    def test_default_overlay_beside_mesh(self) -> None:
        import sys

        tools_dir = Path(__file__).resolve().parents[1] / "tools"
        sys.path.insert(0, str(tools_dir))
        from gvhmr_mesh_bake import resolve_overlay_output

        mesh = Path("C:/proj/clip.dance.mesh.npz")
        path, rel = resolve_overlay_output(mesh, None)
        self.assertEqual(path.name, "clip.dance.mesh.overlay.mp4")
        self.assertEqual(rel, "clip.dance.mesh.overlay.mp4")


class TestSaveMeshCache(unittest.TestCase):
    def test_roundtrip_faces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.dance.mesh.npz"
            faces = np.array([[0, 1, 2]], np.int32)
            save_mesh_cache(
                p,
                faces=faces,
                tracks=[
                    {
                        "verts_world": np.zeros((2, 3, 3), np.float32),
                        "verts_incam": np.zeros((2, 3, 3), np.float32),
                        "times": np.array([0.0, 0.1], np.float64),
                        "mask": np.array([True, True]),
                    }
                ],
            )
            d = load_mesh_cache(p)
            np.testing.assert_array_equal(d["faces"], faces)


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
