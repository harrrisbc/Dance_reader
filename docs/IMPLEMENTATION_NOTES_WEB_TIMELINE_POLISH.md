# Implementation notes — Web timeline polish

## Skeleton lag + registration

- Overlay canvas no longer `inset:0` on the wrap; JS positions it to the **letterboxed video content** (`object-fit` math from `videoWidth/Height` vs display size).
- While playing: `requestAnimationFrame` loop samples `video.currentTime` every frame; cheap single-stroke draw (no glow); selected dancer only.
- Pause / seek: full style (glow + optional all dancers).
- `timeupdate` no longer drives the skeleton.

## Timeline alignment

- Seek and wheel bind to **`plot.over` only** (same coordinate space as the playhead CSS line).
- Clicks outside the plot area (title / Y gutter) are ignored.
- After `setSize` / brush-zoom / `setScale`, playheads redraw.
- Y ranges locked to design table; expand only when data exceeds (reach never expands below floor).

## Chart kinds (match Qt)

| Series | kind |
|--------|------|
| Energy, Travel | area (fill + stroke) |
| Speed, L–R, F–B | line / pos |
| Reach | line + closed/open guides at 1.5 / 3.0 |
| Level | three background bands + line |
| Directness | colour ribbon; NaN gaps; no Y axis |

## Cache bust

`index.html` loads `review.js?v=20260806c` and `review.css?v=20260806c`.
