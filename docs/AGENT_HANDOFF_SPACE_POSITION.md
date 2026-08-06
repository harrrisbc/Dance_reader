# WORK ORDER — space metrics: record position, not just change

**To:** programming agent
**From:** Design agent
**Follows:** the shipped `space_analysis_metrics` stage (reach / level / travel / directness)

## Why

`travel` and `directness` are *derivatives* of the dancer's position, but we never store the
position itself. A choreographer marking 走位 wants "he is stage-left at 0:10, crosses to
downstage-right by 0:20" — the current curves cannot answer that, and cannot answer
"did he use the whole stage" or "does he keep returning to centre".

Position is the primitive. Store it, and travel/directness stay available as cheap derived values.

## Two bugs to fix first

1. **False 高低 spikes on dropped detections.** In `compute_metrics`, when `_torso_frame`
   returns `None` we push `level_raw = 0.0`. Real values are `-hip_y` ≈ `-0.4 … -0.6`, so
   after min–max normalization those missing frames become **1.0 — the highest level in the
   track**. Carry forward the previous value (or drop the frame) instead of writing `0.0`.
   Same treatment for `reach`, which currently reads as "fully contracted" on dropped frames.

2. **直進 lies when the dancer stands still.** `directness = 1.0` when `path ≈ 0`, so a static
   dancer plots a solid "perfectly direct" line. Gate it: when `travel_raw` is below a small
   threshold, emit `NaN` (matplotlib leaves a gap) or `0.0`. Do not plot a confident value for
   a movement that did not happen.

## Change 1 — new absolute position fields

Add to `MetricPoint` (defaults `0.0`, backward compatible):

```python
pos_x: float = 0.0   # COM x in frame, 0 = 畫面左, 1 = 畫面右
pos_y: float = 0.0   # foot y in frame, 0 = 畫面上（遠）, 1 = 畫面下（近）
```

**These two must NOT be per-actor normalized.** Every other series is divided by the actor's
own maximum; that is exactly wrong here. Absolute frame coordinates are the point — they are
what makes two dancers comparable on the same map, and what makes "he stays stage-left the
whole piece" visible.

Use the **lower ankle** (larger y, whichever of indices 15/16 is more confident) for `pos_y`,
not the COM. COM y drops when the dancer crouches even though they have not moved on the
floor; the feet are the actual floor contact. Fall back to COM y only when both ankles are
missing.

## Change 2 — redefine 高低 as posture, not screen height

`level` currently stores `-hip_y` min–max stretched over the track. Two problems: it duplicates
`pos_y` (screen position), and the min–max stretch means a dancer who never leaves standing
gets their hip jitter blown up to a full 0–1 swing that reads as dramatic level changes.

Redefine as **body-relative posture height**:

```
leg      = |ankle_y - hip_y|            # current frame
leg_ref  = 90th percentile of leg over the actor's whole track   # ≈ upright
level    = clamp(leg / leg_ref, 0, 1)
```

Standing ≈ 1.0, plié ≈ 0.75, deep crouch ≈ 0.4, on the floor ≈ 0.1. Still calibrated per actor,
but against a *physical* reference (their own standing height) rather than a stretched range,
so the number means something and holds across videos.

## Change 3 — derive travel / directness from the feet

Keep both metrics and the ~0.8 s window, but run them on the **foot track** `(pos_x, pos_y)`
instead of the torso COM, so crouching and arm work no longer register as floor travel.

## Change 4 — store the frame size

`run.py` calls `track_video(...)` and discards width/height as `_w, _h`. Add to `AnalysisResult`:

```python
frame_width: int = 0
frame_height: int = 0
```

The floor map needs the real aspect ratio. Defaults `0` keep old `.dance.json` loading; fall
back to 16:9 when absent.

## Change 5 — desktop UI

**Timeline:** two more toggles, both default off.

| Series | Label | Color |
|--------|-------|-------|
| `pos_x` | 左右 | `#8f7cc0` |
| `pos_y` | 前後 | `#c07fa8` |

**New panel — 走位圖 (floor map).** This is the real payoff and the thing a timeline curve
cannot show. Bird's-eye scatter/line of `(pos_x, pos_y)` for the selected dancer:

- Draw only the currently zoomed time window, so it stays readable
- Color the path by time, light `#d8d8d8` → dark `#333333`, so direction of travel is visible
- Mark the playhead position with a dot in `#c45c4a`
- Axes off; frame the plot to `frame_width : frame_height`
- Label it 「走位圖（畫面座標）」 — see the caveat below

Put it under the timeline chart, collapsible or in a tab, so it does not crowd the motif lists.

## Backward compatibility

`enrich_metrics(result)` already recomputes every curve from `actors[].poses` on load, so old
projects pick up `pos_x` / `pos_y` and the redefined `level` with no re-run of YOLO. Nothing
extra needed — just make sure the new fields go through `enrich_metrics`, not only through
`analyze_video`.

## Known limitation — say it in the UI, don't hide it

These are **screen** coordinates, not stage coordinates. They equal stage position only for a
fixed, non-zooming camera, which is the normal rehearsal-video case. A pan or zoom will smear
the floor map. Hence the 「畫面座標」 label.

Also, a frontal camera projects the floor as a trapezoid, so `pos_y` is not linear in stage
depth. Out of scope now; the eventual fix is a one-time homography — user clicks the four floor
corners, we map screen → true top-down stage metres. Worth designing for later, do not build it
in this pass.

## Change 6 — stop normalizing away real units

The single shared 0–1 y-axis is the main source of user confusion: `0.8 能量` and `0.8 直進`
appear equally precise and comparable, but one is "80% of this dancer's own peak" and the other
is a true ratio. Worse, three series already carry a real, scale-invariant unit that
`compute_metrics` currently destroys by dividing by the track maximum.

| Series | Now | Should be | Why |
|--------|-----|-----------|-----|
| 能量 `energy` | ÷ max | keep relative, label 「相對自身峰值」 | Σv² has no unit a human reads; be explicit that it is relative |
| 速度 `speed` | ÷ max | **肩寬 / 秒** — divide by shoulder width, not by max | scale-invariant *and* absolute; comparable across videos |
| 伸展 `reach` | ÷ max | **肩寬倍數** — it is already this before the ÷ max | ~1.0 curled, ~3.0+ fully extended; the ÷ max throws away the meaning |
| 高低 `level` | min–max stretch | **站高比例 0–1** (Change 2) | 1 = 企直, 0.4 = 蹲, 0.1 = 地面 |
| 走位 `travel` | ÷ max | **肩寬 / 秒** (window path ÷ window seconds) | same argument as 速度 |
| 直進 `directness` | ratio | ratio, unchanged | genuinely dimensionless 0–1 |
| 左右／前後 `pos_x` `pos_y` | n/a | **畫面比例 0–1**, never per-actor normalized | Change 1 |

Note for the record: `energy` (Σv²) and `speed` (mean v) are strongly correlated and plot as
near-duplicate shapes. That is another reason to give them different visual treatments below
rather than two similar lines on one axis.

## Change 7 — one row per metric, one shared time axis

Replace the single overlaid chart with **stacked small multiples**: one short row per enabled
series, each with its own y-axis, unit in the axis label, all sharing the x (time) axis via
`fig.subplots(n, 1, sharex=True)`. Rebuild the figure when toggles change. This gives every
metric an honest scale while keeping vertical alignment, so correlations are still readable —
that is why it should be stacked rows, **not** six separate windows.

Row height ~70 px. Default rows: 能量、速度、伸展 (same defaults as now). Only the bottom row
draws the 時間（秒） tick labels.

| Row | Chart type | Y label | Notes |
|-----|-----------|---------|-------|
| 能量 | filled area | 相對強度 | area reads as accumulated effort; no false precision |
| 速度 | line | 肩寬／秒 | |
| 伸展 | line | 肩寬倍數 | faint reference lines at 1.5「收」and 3.0「全開」 |
| 高低 | line over 3 shaded bands | 站高比例 | bands 低 (0–0.45) / 中 (0.45–0.8) / 高 (0.8–1.0), Laban-style discrete levels |
| 走位 | filled area | 肩寬／秒 | "how busy is the blocking"; the *where* is the floor map |
| 直進 | ribbon, not a curve | — (no y-axis) | draw only where 走位 is above the threshold; color 直線 → 迂迴; avoids the meaningless-when-static problem from bug 2 |
| 左右／前後 | line | 畫面比例 | keep, but the floor map is the primary read |

Draw the playhead line and the motif `axvspan` shading on **every** row, so a marked phrase
lines up across all metrics.

**Lock the y-ranges — do not autoscale.** Autoscaling on zoom makes an unchanged movement look
different at a different zoom level, which is its own misreading. Suggested fixed ranges,
expanding only if data exceeds them: 能量 0–1, 速度 0–4, 伸展 1–3.5, 高低 0–1.05, 走位 0–2,
左右／前後 0–1.

## Acceptance

- Standing still: 走位 near zero, 直進 shows a gap (not a confident 1.0)
- Dropped detections no longer produce 高低 spikes to 1.0
- Crouching changes 高低 but not 走位
- Floor map traces the dancer's path across the frame and follows the playhead
- Old `.dance.json` opens and gains position curves and the floor map after load
- Every chart row shows its unit in the y-axis label; no two metrics share an axis
- 伸展 reads in 肩寬倍數 and 速度 in 肩寬／秒 — same values for the same movement filmed at a
  different distance or resolution
- Zooming the timeline does not change any curve's vertical scale
- Still no emotion labels in this stage
