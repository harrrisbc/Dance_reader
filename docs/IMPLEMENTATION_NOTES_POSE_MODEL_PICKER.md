# Implementation notes — pose model picker (S / M)

- `POSE_MODELS` + `resolve_pose_model()` in `pose_track.py` (unknown → S).
- Qt **Model** combo next to Max dancers; default S (`yolo11s-pose.pt`).
- Analyze passes resolved weight into `AnalyzeWorker` / `analyze_video`.
- `DanceProject.pose_model` saved/restored; old projects → S.
- Done status includes model stem, e.g. `(yolo11m-pose)`.
- First M run may download weights via Ultralytics.
