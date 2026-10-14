# SDD progress — GVHMR mesh viewer Phase 1

Branch: cursor/gvhmr-mesh-viewer-phase1
Plan: docs/superpowers/plans/2026-10-14-gvhmr-mesh-viewer-phase1.md
Base: (record after first commit)

Task 1: complete (commits 8e19cad..23458d5, review clean)
Minor for final: sha256 defaults untested; faces dtype not enforced; missing per-track negative loader tests.
Task 2: complete (commits 23458d5..c4fd97b, review clean after T-consistency fix)
Minor for final: no live GPU pred smoke; export JSON fields untested; round-trip value asserts weak.
Task 3: complete (commits c4fd97b..e5bddc6, review clean after open_project sidecar src fix)
Minor for final: mkstemp without artifacts; pre-save basename semantics.
Task 4: complete (commits e5bddc6..425101f, review clean after np import + fps fix)
Minor for final: mask all-True; overlay track 0 only; no manual bake smoke.
Task 5: complete (commits 425101f..23dfc5f, review clean)
Task 6: complete (commits 23dfc5f..e85dc98, review clean after alpha-composite fix)
Note: old full-composite overlay mp4s need re-bake.
Task 7: complete (commits e85dc98..6d767a1, review clean)
Minor for final: pyvista not installed in venv yet; import guard only ImportError.
Task 8: complete (commits 6d767a1..e6b1270, review clean after status+stale-worker fix)
Minor for final: multi-dancer overlay track 0 only; pyvista may need pip install in app venv.
Final review: mesh cache rel name unified (`foo.dance.mesh.npz`); overlay toggle triggers bake when cache missing.
