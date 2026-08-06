# Implementation notes — English UI + timeline click + skeleton overlay

## Bone overlap (clarification)

Motif accuracy embeddings do **not** change preview drawing. Self-crossing limbs in 2D pose are expected. Overlay polish added:

- Web + Qt: **Skeleton / Joints only / Off**
- Tooltip: “Lines may cross when limbs overlap in the camera view.”
- Skeleton lines drawn at ~0.65 alpha (web canvas + Qt OpenCV blend)

## English UI

User-visible strings switched to English in:

- `backend/app/static/review/{index.html,review.js}`
- `desktop/{window.py,__main__.py,playback.py}`
- Actor / motif defaults: `Dancer n`, `Type A`, `Similar 1` (`run.py`, `motifs.py`, `annotate.py`)
- README notes UI language English

## Timeline click fix

**Web:** Removed CSS `.uplot { width: 100% }` stretch; `setSize` + `ResizeObserver`; seek via `plot.over` + `posToVal` on full panel click.

**Qt:** Sync figure inches to canvas widget pixels on resize; `subplots_adjust` instead of `tight_layout`; map clicks with `ax.transData.inverted()` across each row’s Y band.

## Smoke-test

1. Web + Qt: click left / centre / right of a chart → seek matches playhead (±1 frame).
2. Resize window, click again.
3. Overlay Off / Joints / Skeleton; tooltip visible on hover.
