# Implementation notes (space / position metrics)

See `docs/AGENT_HANDOFF_SPACE_POSITION.md`.

## Shipped

- Absolute `pos_x` / `pos_y` (never per-actor normalized); feet for `pos_y`
- `level` = posture (leg / 90th-pct standing), carry-forward on drops
- `travel` / `directness` on foot track; directness = NaN when still
- Units: speed & travel in 肩寬／秒; reach in 肩寬倍數; energy relative
- `frame_width` / `frame_height` on `AnalysisResult`
- Desktop: stacked small-multiples + 走位圖（畫面座標）

## Verify

Open an old `.dance.json` — `enrich_metrics` on load rebuilds curves without re-running YOLO.
