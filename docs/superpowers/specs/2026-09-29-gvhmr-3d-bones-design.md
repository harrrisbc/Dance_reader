# GVHMR 3D bones — slice 1 design

**Date:** 2026-09-29  
**Status:** approved in chat, pending written-spec review  
**Slice:** 1 of 3. Later slices are not this spec.

## Goal

After Analyze, up to two dancers each have an SMPL 3D skeleton from [GVHMR](https://github.com/zju3dv/GVHMR/), stored in `.dance.json`. Playback draws those bones on the recorded video in sync with the playhead.

Analysis (energy, motifs, and the rest) does not need to be real time. This slice does not recompute those formulas from 3D.

## Decisions already made

| Topic | Choice |
|-------|--------|
| When | Recorded video, offline. Not a live camera. |
| Unreal, later | Playback Live Link **and** a saved animation file. Not this slice. |
| Unreal target, later | UE5 Mannequin first. MetaHuman later. |
| Pose source | Replace YOLO Pose as the skeleton. Do not fall back to it. |
| Model | GVHMR, once per person, static-camera mode (`-s`). |
| License | Personal / research. SMPL registration is acceptable. Weights stay off git. |
| People | At most 2 tracks (the longest two). |

## Out of scope

- Rewriting energy, speed, reach, level, travel, directness, or motifs onto 3D joints
- Live Link, FBX, BVH, MetaHuman, UE axis conversion
- Moving cameras, crowds above 2 people, fingers/face beyond SMPL-24
- Vendoring GVHMR or SMPL weights

## Architecture

GVHMR runs in a **separate Python environment**, not the app venv.

```text
Qt Analyze
  → app subprocess
      DANCE_GVHMR_PYTHON  backend/tools/gvhmr_export.py  <video>
  → adapter (inside the GVHMR env)
      GVHMR tracker → keep the 2 tracks with the most frames
      static-camera inference per track
      write --out result.json (logs on stderr only)
  → app reads that file, writes .dance.json, builds the temporary COCO-17 bridge
  → playback draws stored 2D projections (no second GVHMR run)
```

Environment variables:

| Name | Meaning |
|------|---------|
| `DANCE_GVHMR_PYTHON` | Python executable of the GVHMR venv |
| `DANCE_GVHMR_ROOT` | Clone of the GVHMR repo (checkpoints live in that clone) |

If either variable is missing, or the process exits non-zero, Analyze **stops** and the status line says what is missing or shows the tail of stderr. It must not run YOLO Pose and report success.

The Qt **Model** S/M combo is not the pose source anymore. Hide it. `analyze` stops calling `track_video` / `yolo11*-pose`.

More than two people: keep the two longest tracks and say in the status line that others were ignored. A person who is absent for part of the clip simply has no sample at those times. Never write zeros for a missing pose.

## Data

`poseBackend` on the analysis result is `"gvhmr"`. `coordFrame` is `"gvhmr_world"`. Store a `staticCamera: true` flag. Joint positions are GVHMR’s global joints **as returned**, in meters. This slice does not convert axes for Unreal.

Each stored pose sample, when the person is tracked at that time:

| Field | Content |
|-------|---------|
| `time` | Seconds, same meaning as today |
| `joints3d` | Exactly 24 joints, SMPL regressor order, meters, `gvhmr_world` |
| `keypoints` | The same 24 joints projected to normalized image `x, y` plus confidence. Confidence is `1` when the sample exists. |

SMPL order is the standard 24-joint regressor: pelvis, L/R hip, spine1, L/R knee, spine2, L/R ankle, spine3, L/R foot, neck, L/R collar, head, L/R shoulder, L/R elbow, L/R wrist, L/R hand.

Parent index of each joint (`-1` is the root). Overlay lines are these parent→child edges:

```text
[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
```

Dancer 1 is the track with the most frames. Dancer 2 is the next. A tie goes to the track whose first sample is earlier.

If the person is not in frame, omit that sample. Do not emit a 24-length array of zeros, and do not emit a sparse joint list. A present sample is always 24 joints.

Old projects have no `joints3d` and 17 keypoints. They still load. Overlay uses COCO edges when a pose has 17 points, and the SMPL parent→child tree when it has 24.

## Playback

Projections are computed at Analyze time and stored in `keypoints`. Playback does not call GVHMR.

`desktop/playback.py` currently draws `COCO_EDGES`. For 24-point poses, draw the SMPL kinematic tree instead. Skeleton / Joints only / Off stay as they are. Only the selected dancer is drawn.

## Temporary metric bridge

Existing metric and motif code still expects COCO-17 2D points. This slice builds that 17-point list **in memory** from the projected SMPL points and feeds the current formulas. The stored skeleton remains SMPL-24. The bridge is not the 3D source of truth and is removed in the next slice, when metrics are recomputed from 3D.

COCO index ← SMPL index:

| COCO | SMPL | Note |
|------|------|------|
| 0 nose | 15 head | SMPL has no nose |
| 1–4 eyes, ears | 15 head | confidence `0` (current limb metrics ignore these) |
| 5, 6 shoulders | 16, 17 | |
| 7, 8 elbows | 18, 19 | |
| 9, 10 wrists | 20, 21 | |
| 11, 12 hips | 1, 2 | |
| 13, 14 knees | 4, 5 | |
| 15, 16 ankles | 7, 8 | |

## Adapter JSON

The app calls `gvhmr_export.py --video <path> --out <result.json>`. The file is the only machine-readable output. Logs go to stderr, never mixed into the JSON.

Top-level fields: `staticCamera` (true), `coordFrame` (`"gvhmr_world"`), `ignoredTracks` (integer), `tracks` (array). Each track has `frames`. Each frame has `time` (seconds), `joints3d` (24×3 meters), and `keypoints` (24×3: `x`, `y`, `conf`). `ignoredTracks` is how many detected tracks were dropped because of the cap of 2.

## Errors

| Case | Behavior |
|------|----------|
| Env var or weights missing | Analyze fails. Status names the missing piece. |
| Non-zero exit | Analyze fails. Status includes the stderr tail. |
| No person | Empty actors. Status says no person was tracked. |
| More than 2 people | Two longest tracks kept. Status reports `ignoredTracks`. |
| Person enters late | Samples only while tracked. |

Progress text while the subprocess runs: `Running GVHMR…`.

## Testing

CI does not need a GPU or weights.

- Round-trip the adapter JSON into `PoseFrame` / analysis fields.
- 17-point poses select COCO edges; 24-point poses select the SMPL tree.
- A fixture with a gap omits that time instead of writing zeros.
- Parser rejects a frame whose `joints3d` length is not 24.

Manual, on a machine with GVHMR installed:

- One-dancer clip: bones stay on the body during playback.
- Two-dancer clip: two tracks, extras ignored and reported.

## Later slices (not designed here)

1. Recompute charts and motifs from 3D joints. Delete the COCO-17 bridge.
2. During playback, send the skeleton to a UE5 Mannequin over Live Link, and write a BVH or FBX file.
