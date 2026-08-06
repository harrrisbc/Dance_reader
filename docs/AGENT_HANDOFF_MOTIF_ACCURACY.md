# WORK ORDER — motif / 搵相似 accuracy + left–right as same gesture

**To:** programming agent  
**From:** Design agent  
**User feedback:** auto 重複動作類型 and 搵相似 are not accurate — different gestures get grouped / found as the same. Also: **left and right versions of a move should count as the same** (choreography mirror). User said “vertical flip”; treat that as **mirror across the vertical axis** (swap stage-left / stage-right), not upside-down.

---

## Current behaviour (why it fails)

In `backend/app/pipeline/motifs.py`:

1. Phrases cut at **energy valleys** (or sliding 1.2s windows).
2. Each phrase → `_trajectory_embedding`: 10 samples of `pose_feature` (torso-normalized limb XY) + frame-to-frame Δ, then **L2-normalize**.
3. Auto types: agglomerative clustering, `distance_threshold=0.48` on those vectors.
4. 搵相似: cosine ≥ `SIMILARITY_MIN` (0.82) on the same embedding.

**Failure modes:**

| Problem | Effect |
|---------|--------|
| Embedding is short + over-smoothed | Many different moves look “kind of moving limbs” → same cluster |
| Thresholds too loose (`0.48` / `0.82`) | False merges |
| No temporal alignment (no DTW) | Same gesture at different speed/phase mismatches; unrelated phrases can still get mid cosine |
| Signed X in `pose_feature` | Raise-left ≠ raise-right → user wants the **opposite** for choreography |
| Valley cuts misaligned | Half of gesture A compared to half of gesture B |

Do **not** only tweak `SIMILARITY_MIN` / `CLUSTER_DISTANCE` without changing the similarity definition — that will trade one error for the other.

---

## Product rules

1. **Mirror invariance (required):** reflecting the body left↔right is the **same** motif for clustering and 搵相似.  
2. Prefer **fewer, tighter** types over many polluted ones. Missing a repeat is better than merging two different gestures.  
3. Keep Traditional Chinese UI; no emotion labels in this work.

---

## Change 1 — mirror-aware pose features

COCO left/right pairs to swap when mirroring (x → −x after torso normalize, then swap L/R joint indices):

- Shoulders 5↔6, elbows 7↔8, wrists 9↔10  
- Hips 11↔12, knees 13↔14, ankles 15↔16  
- Nose / centre joints: negate x only  

Implement helpers:

```text
pose_feature(kpts)           # existing
pose_feature_mirrored(kpts)  # swap pairs + negate x
```

For a trajectory embedding `E(seg)` built as today (or improved below), also build `E_mirror(seg)` from mirrored keypoints (and mirrored velocities).

**Similarity between two segments A, B:**

```text
sim(A,B) = max( cosine(E(A), E(B)), cosine(E(A), E_mirror(B)) )
```

Use this everywhere: clustering affinity (convert to distance `1 - sim`), and 搵相似.  
Do **not** average the two cosines — that blurs discrimination; take the **max**.

---

## Change 2 — stronger trajectory match (accuracy)

Minimum viable upgrade (do all of these):

### 2a — Align before compare

Before embedding or as part of distance:

- Optionally align windows so **peak motion** (or max energy frame) lines up within ±0.25s, or  
- Prefer **DTW / soft-DTW** on the sequence of `pose_feature` frames (downsample to ~12–16 frames), then mirror-aware:

```text
dist(A,B) = min( dtw(seq(A), seq(B)), dtw(seq(A), seq_mirror(B)) )
sim = exp(-dist) or 1 / (1+dist)   # document the chosen mapping
```

If full DTW is too slow on large segment sets, use: DTW only for **pairwise verify** after a cheap embedding pre-filter (top-K neighbors), then cluster / 搵相似 on verified scores.

### 2b — Richer local features (cheap)

Extend per-frame features beyond raw limb XY, still torso-normalized:

- Include **bone angles** or unit bone vectors (shoulder→elbow, elbow→wrist, hip→knee, knee→ankle), mirrored consistently.  
- Optional: append **relative** metric snapshot over the phrase (mean reach, mean level) — light, helps separate “big kick” vs “small arm”.

Keep vectors L2-normalized only if using cosine; for DTW use unnormalized frame features.

### 2c — Stricter gating for auto types

- Raise motion gate slightly so standing/idle phrases do not enter clustering.  
- After clustering, **drop types with &lt; 2 occurrences** (already likely) and drop a type if mean intra-sim is weak (e.g. mean pairwise sim &lt; 0.88 under new metric).  
- Tighten distance threshold after switching to mirror-aware sim (re-calibrate on 1–2 rehearsal clips; start stricter than today).

### 2d — 搵相似 UX honesty

- Return matches sorted by score; show score in the occurrence list or status (e.g. `相似 0.91`) so the user can see weak hits.  
- Default threshold on the **new** sim space; expose a constant at top of `motifs.py` for tuning.  
- Same mirror rule as clustering.

---

## Change 3 — phrase quality (optional same PR if small)

Energy-valley cuts often bisect gestures. Incremental fix:

- Prefer valleys that are both local minima **and** below a fraction of nearby peak (true rest), not every tiny dip.  
- Soften `PHRASE_MAX_S` splits so a single continuous accent is not chopped into two weak half-gestures that then falsely match other halves.

Do not switch back to sliding windows as the primary path when valleys exist — sliding windows inflate false merges.

---

## Files

- Primary: `backend/app/pipeline/motifs.py`  
- Likely: `backend/app/pipeline/metrics.py` (`pose_feature` + mirrored variant)  
- Desktop / web UI later: show similarity scores when listing 搵相似 results (`desktop/window.py` or fluent web UI)  
- Notes: `docs/IMPLEMENTATION_NOTES.md` — document mirror-invariant matching + new thresholds  

Re-analyze (or rebuild segments from saved poses/buffers) required to refresh motif types; 搵相似 after load already uses `track_buffers` — ensure new embedding path is used there too.

---

## Acceptance

- Same phrase performed to the **left** and to the **right** merges into one type / is found by 搵相似.  
- Clearly different gestures (e.g. arm circle vs kick) are **not** the same type under default settings.  
- 搵相似 on a selected occurrence returns mostly true repeats; weak mismatches are rarer than today (or visible via lower scores).  
- Still offline / local; no new cloud model required for this pass.  
- Traditional Chinese UI strings unchanged except optional score text.

## Out of scope

- Learned video embeddings / ActionCLIP (possible later research stage).  
- Emotion labels.  
- Fluent Web UI (separate handoff) — but keep motif API stable for it.

## Suggested test clips for the implementer

1. Two mirrored port de bras (L then R) → one type.  
2. Kick vs unrelated upper-body phrase → two types.  
3. Same jump twice at different timings → one type / 搵相似 hits both.
