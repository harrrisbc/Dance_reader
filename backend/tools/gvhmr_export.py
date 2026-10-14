"""Export SMPL tracks with GVHMR. Runs in the GVHMR venv; cwd is that repo.

``--help`` is parsed before any hmr4d, torch, or hydra import so the app
interpreter can check the CLI without those packages.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


def _dance_backend_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-people", type=int, default=2)
    args = parser.parse_args()
    # imports of hmr4d, torch, hydra stay below this line
    write_export(Path(args.video), Path(args.out), max_people=args.max_people)


def write_export(video: Path, out: Path, *, max_people: int) -> None:
    import cv2
    import hydra
    import torch
    from hydra import compose, initialize_config_module

    # PyTorch 2.6+ defaults weights_only=True; YOLO / Lightning ckpts need False.
    if not getattr(torch.load, "_dance_full_load", False):
        _torch_load = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _torch_load(*args, **kwargs)

        _load._dance_full_load = True  # type: ignore[attr-defined]
        torch.load = _load  # type: ignore[assignment]

    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.preproc import Extractor, Tracker, VitPoseExtractor
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.video_io_utils import get_video_lwh

    backend = _dance_backend_dir()
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.pipeline.mesh_cache import save_smpl_sidecar, sha256_file

    # 1. history = Tracker().track(str(video))
    history = Tracker().track(str(video))
    # 2. Group ids. Rank by (-len(frame_ids), min(frame_ids)).
    id_to_frame_ids, id_to_boxes = _group_ids(history)
    ranked = sorted(
        id_to_frame_ids,
        key=lambda track_id: (
            -len(id_to_frame_ids[track_id]),
            min(id_to_frame_ids[track_id]),
        ),
    )
    kept = ranked[:max_people]
    ignored = max(0, len(ranked) - max_people)

    tracks: list[dict] = []
    smpl_tracks: list[dict] = []
    if kept:
        # 7. time = frame_index / cv2 fps of the original video.
        fps = _original_fps(cv2, video)
        length, width, height = get_video_lwh(video)
        run_root = f"outputs/dance_reader/{uuid.uuid4().hex}"
        bbx_paths: list[str] = []
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            model = None
            smplx = None
            smplx2smpl = None
            regressor = None
            vitpose_extractor = VitPoseExtractor()
            feature_extractor = Extractor()
            for track_id in kept:
                # 4. Per track, new preprocess dir so paths.bbx is not shared.
                cfg = compose(
                    config_name="demo",
                    overrides=[
                        f"video_name=track{track_id}",
                        f"output_root={run_root}",
                        "static_cam=True",
                        "verbose=False",
                        "use_dpvo=False",
                    ],
                )
                bbx_path = str(Path(cfg.paths.bbx))
                if bbx_path in bbx_paths:
                    raise RuntimeError(f"two tracks share one bbx file: {bbx_path}")
                bbx_paths.append(bbx_path)
                Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

                if model is None:
                    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
                    model.load_pretrained_model(cfg.ckpt_path)
                    model = model.eval().cuda()
                    smplx = make_smplx("supermotion").cuda()
                    smplx2smpl = torch.load(
                        "hmr4d/utils/body_model/smplx2smpl_sparse.pt"
                    ).cuda()
                    regressor = torch.load(
                        "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
                    ).cuda()

                # 3. Dense box via the same interpolate+smooth as get_one_track.
                bbx_xyxy, mask = _dense_box(
                    id_to_frame_ids[track_id], id_to_boxes[track_id], length
                )
                bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
                torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, cfg.paths.bbx)

                vitpose = vitpose_extractor.extract(str(video), bbx_xys)
                torch.save(vitpose, cfg.paths.vitpose)
                vit_features = feature_extractor.extract_video_features(str(video), bbx_xys)
                torch.save(vit_features, cfg.paths.vit_features)

                # static_cam=True → R_w2c = eye(3) repeated, matching load_data_dict.
                r_w2c = torch.eye(3).repeat(length, 1, 1)
                k_fullimg = estimate_K(width, height).repeat(length, 1, 1)
                data = {
                    "length": torch.tensor(length),
                    "bbx_xys": bbx_xys,
                    "kp2d": vitpose,
                    "K_fullimg": k_fullimg,
                    "cam_angvel": compute_cam_angvel(r_w2c),
                    "f_imgseq": vit_features,
                }
                pred = detach_to_cpu(model.predict(data, static_cam=True))
                # 5. Global joints from smpl_params_global, no recenter.
                joints_world = _joints(smplx, smplx2smpl, regressor, pred["smpl_params_global"])
                # 6. In-camera joints, project with pred["K_fullimg"].
                joints_incam = _joints(smplx, smplx2smpl, regressor, pred["smpl_params_incam"])
                # 8. Append a frame only when the detection mask is True.
                frames = _frames_where_detected(
                    joints_world,
                    joints_incam,
                    pred["K_fullimg"],
                    mask,
                    fps,
                    width,
                    height,
                )
                tracks.append({"frames": frames})
                smpl_tracks.append(_sidecar_track(pred))

    sidecar_path = out.with_name(out.stem + ".dance.smpl.npz")
    save_smpl_sidecar(sidecar_path, smpl_tracks)
    payload = {
        "staticCamera": True,
        "coordFrame": "gvhmr_world",
        "ignoredTracks": ignored,
        "tracks": tracks,
        "smplSidecar": sidecar_path.name,
        "smplSidecarSha256": sha256_file(sidecar_path),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")


def _sidecar_track(pred: dict) -> dict:
    import numpy as np

    from app.pipeline.mesh_cache import SMPL_PARAM_KEYS

    def params_numpy(params: dict) -> dict[str, np.ndarray]:
        return {
            key: params[key].detach().cpu().numpy().astype(np.float32) for key in SMPL_PARAM_KEYS
        }

    return {
        "global": params_numpy(pred["smpl_params_global"]),
        "incam": params_numpy(pred["smpl_params_incam"]),
        "K": pred["K_fullimg"].detach().cpu().numpy().astype(np.float32),
    }


def _group_ids(history: list) -> tuple[dict[int, list[int]], dict[int, list]]:
    id_to_frame_ids: dict[int, list[int]] = {}
    id_to_boxes: dict[int, list] = {}
    for frame_id, frame in enumerate(history):
        for det in frame:
            track_id = int(det["id"])
            id_to_frame_ids.setdefault(track_id, []).append(frame_id)
            id_to_boxes.setdefault(track_id, []).append(det["bbx_xyxy"])
    return id_to_frame_ids, id_to_boxes


def _original_fps(cv2, video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError(f"could not read fps from {video}")
    return fps


def _dense_box(frame_ids: list[int], boxes: list, length: int):
    """Interpolate and smooth like Tracker.get_one_track. Mask is pre-interp."""
    import numpy as np
    import torch

    from hmr4d.utils.net_utils import moving_average_smooth
    from hmr4d.utils.seq_utils import (
        frame_id_to_mask,
        get_frame_id_list_from_mask,
        linear_interpolate_frame_ids,
        rearrange_by_mask,
    )

    ids = torch.tensor(frame_ids)
    bbx_xyxys = torch.tensor(np.asarray(boxes)).float()
    mask = frame_id_to_mask(ids, length)
    dense = rearrange_by_mask(bbx_xyxys, mask)
    missing = get_frame_id_list_from_mask(~mask)
    dense = linear_interpolate_frame_ids(dense, missing)
    assert (dense.sum(1) != 0).all()
    dense = moving_average_smooth(dense, window_size=5, dim=0)
    dense = moving_average_smooth(dense, window_size=5, dim=0)
    return dense.float(), mask


def _joints(smplx, smplx2smpl, regressor, params: dict):
    """SMPL-24 joints. Does not call move_to_start_point_face_z."""
    import torch
    from einops import einsum

    from hmr4d.utils.net_utils import to_cuda

    smplx_out = smplx(**to_cuda(params))
    verts = torch.stack([torch.matmul(smplx2smpl, vertex) for vertex in smplx_out.vertices])
    return einsum(regressor, verts, "j v, l v i -> l j i").detach().cpu()


def _frames_where_detected(
    joints_world,
    joints_incam,
    k_fullimg,
    mask,
    fps: float,
    width: int,
    height: int,
) -> list[dict]:
    if joints_world.shape[0] != mask.shape[0]:
        raise RuntimeError(
            f"joint length {joints_world.shape[0]} != detection mask {int(mask.shape[0])}"
        )
    frames: list[dict] = []
    for index in range(int(mask.shape[0])):
        if not bool(mask[index]):
            continue
        projected = (k_fullimg[index] @ joints_incam[index].T).T
        z = projected[:, 2]
        u = projected[:, 0] / z
        v = projected[:, 1] / z
        keypoints = [
            [float(u[joint] / width), float(v[joint] / height), 1.0]
            for joint in range(projected.shape[0])
        ]
        frames.append(
            {
                "time": index / fps,
                "joints3d": joints_world[index].tolist(),
                "keypoints": keypoints,
            }
        )
    return frames


if __name__ == "__main__":
    main()
