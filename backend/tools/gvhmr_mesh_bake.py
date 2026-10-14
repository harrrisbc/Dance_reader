"""Bake SMPL mesh verts from a params sidecar. Runs in the GVHMR venv; cwd is that repo.

``--help`` is parsed before any hmr4d or torch import.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _dance_backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_overlay_output(mesh_out: Path, overlay_cli: str | None) -> tuple[Path, str]:
    """Return absolute overlay path and relpath from mesh_out.parent."""
    if overlay_cli:
        overlay_path = Path(overlay_cli)
        if overlay_path.is_dir():
            overlay_path = overlay_path / f"{mesh_out.stem}.overlay.mp4"
    else:
        overlay_path = mesh_out.with_name(f"{mesh_out.stem}.overlay.mp4")
    try:
        rel = overlay_path.relative_to(mesh_out.parent).as_posix()
    except ValueError:
        rel = overlay_path.name
    return overlay_path, rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--overlay", type=str, default=None)
    args = parser.parse_args()
    write_bake(
        Path(args.smpl),
        Path(args.out),
        video=Path(args.video) if args.video else None,
        overlay_cli=args.overlay,
    )


def write_bake(
    smpl_path: Path,
    mesh_out: Path,
    *,
    video: Path | None,
    overlay_cli: str | None,
) -> None:
    import numpy as np
    import torch

    if not getattr(torch.load, "_dance_full_load", False):
        _torch_load = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _torch_load(*args, **kwargs)

        _load._dance_full_load = True  # type: ignore[attr-defined]
        torch.load = _load  # type: ignore[assignment]

    from hmr4d.utils.net_utils import to_cuda
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer

    backend = _dance_backend_dir()
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.pipeline.mesh_cache import SMPL_PARAM_KEYS, load_smpl_sidecar, save_mesh_cache

    sidecar = load_smpl_sidecar(smpl_path)
    n_tracks = int(sidecar["n_tracks"])
    if n_tracks == 0:
        raise ValueError("smpl sidecar has n_tracks=0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(device)

    faces_smpl: np.ndarray | None = None
    try:
        faces_smpl = np.asarray(make_smplx("smpl").faces, dtype=np.int32)
    except Exception as exc:
        print(f"warning: could not load SMPL faces: {exc}", file=sys.stderr)
        faces_smpl = np.zeros((0, 3), dtype=np.int32)

    fps = 30.0
    width = height = 0
    if video is not None and video.is_file():
        import cv2

        cap = cv2.VideoCapture(str(video))
        try:
            got = float(cap.get(cv2.CAP_PROP_FPS))
            if got > 0:
                fps = got
        finally:
            cap.release()
        _, width, height = get_video_lwh(video)

    cache_tracks: list[dict] = []
    for i in range(n_tracks):
        n_frames = int(sidecar[f"n_frames_{i}"])
        global_params = {
            key: torch.as_tensor(sidecar[f"g{i}_{key}"], device=device) for key in SMPL_PARAM_KEYS
        }
        incam_params = {
            key: torch.as_tensor(sidecar[f"c{i}_{key}"], device=device) for key in SMPL_PARAM_KEYS
        }
        verts_world = _verts(smplx, smplx2smpl, global_params)
        verts_incam = _verts(smplx, smplx2smpl, incam_params)
        times = (np.arange(n_frames, dtype=np.float64) / fps)
        mask = np.ones(n_frames, dtype=bool)
        cache_tracks.append(
            {
                "verts_world": verts_world,
                "verts_incam": verts_incam,
                "times": times,
                "mask": mask,
            }
        )

    overlay_relpath: str | None = None
    if video is not None and video.is_file() and width > 0 and faces_smpl.size > 0:
        overlay_path, overlay_relpath = resolve_overlay_output(mesh_out, overlay_cli)
        try:
            _render_incam_overlay(
                video=video,
                verts_incam=cache_tracks[0]["verts_incam"],
                k_fullimg=np.asarray(sidecar["K_0"], dtype=np.float32),
                faces=faces_smpl,
                overlay_path=overlay_path,
            )
        except Exception as exc:
            print(f"warning: overlay render failed: {exc}", file=sys.stderr)
            overlay_relpath = None

    save_mesh_cache(
        mesh_out,
        faces=faces_smpl,
        tracks=cache_tracks,
        overlay_relpath=overlay_relpath,
    )


def _verts(smplx, smplx2smpl, params: dict):
    import torch
    from hmr4d.utils.net_utils import to_cuda

    smplx_out = smplx(**to_cuda(params))
    verts = torch.stack([torch.matmul(smplx2smpl, vertex) for vertex in smplx_out.vertices])
    return verts.detach().cpu().numpy().astype(np.float32)


def _render_incam_overlay(
    *,
    video: Path,
    verts_incam: "np.ndarray",
    k_fullimg: "np.ndarray",
    faces: "np.ndarray",
    overlay_path: Path,
    crf: int = 23,
) -> None:
    import torch
    from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer
    from hmr4d.utils.vis.renderer import Renderer

    length, width, height = get_video_lwh(video)
    k0 = torch.as_tensor(k_fullimg[0], dtype=torch.float32)
    renderer = Renderer(width, height, device="cuda", faces=faces, K=k0)
    reader = get_video_reader(video)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    writer = get_writer(overlay_path, fps=30, crf=crf)
    try:
        for i, img_raw in enumerate(reader):
            if i >= verts_incam.shape[0]:
                break
            v = torch.as_tensor(verts_incam[i], device="cuda")
            img = renderer.render_mesh(v, img_raw, [0.8, 0.8, 0.8])
            writer.write_frame(img)
    finally:
        writer.close()
        reader.close()


if __name__ == "__main__":
    main()
