# Motif accuracy notes

See `docs/AGENT_HANDOFF_MOTIF_ACCURACY.md`.

## Shipped

- Mirror L↔R invariance via mirrored keypoints + `max(sim, sim_mirror)`
- Rich frame features (limb XY + bone unit vectors) + DTW alignment
- Stricter motion / intra-cluster gates; valley cuts require true rests
- 搵相似 shows per-occurrence scores in Qt (and best score in web motif list)

## Synthetic check

- Left raise vs right raise → sim ≈ 1.0  
- Left raise vs kick → sim ≈ 0.29 (below default threshold)
