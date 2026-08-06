# WORK ORDER — Qt is the product; retire the web review UI

**To:** programming agent
**From:** Design agent
**Date:** 2026-08-06

## Decision

User reports the **PySide6 desktop app now runs smooth and fluent**. The web review UI existed
only to buy fluency and interactive charts. Qt now delivers both, and Qt already implements the
agreed chart kinds (area / reach reference lines / level bands / directness ribbon) in
`desktop/window.py`. Maintaining two review UIs is no longer worth it.

**Qt is the product. The web review UI is retired.**

## Changes

### 1. Flip the default entry point

`python -m desktop` should open the **Qt window** directly — no browser, no FastAPI needed for
normal use. Remove the `--qt` flag (or keep it as a no-op alias for one release).

### 2. Deprecate, do not delete yet

Keep the files for one release behind an explicit opt-in flag such as `--web`:

- `backend/app/static/review/` (`index.html`, `review.js`, `review.css`)
- `backend/app/review_session.py`
- the `/review` and `/api/review/*` routes in `backend/app/main.py`

Mark them clearly as legacy in a header comment. Delete in a follow-up once the user has run on
Qt only for a while. Rationale: cheap to keep, and it preserves the uPlot interaction work in
case we ever want a shareable/read-only viewer.

The analyze-oriented FastAPI endpoints stay — they are independent of the review UI.

### 3. Cancel the in-flight web work order

`docs/AGENT_HANDOFF_WEB_TIMELINE_POLISH.md` is **superseded — do not implement**. Its three
items resolve as:

| Item | Resolution |
|------|-----------|
| Web skeleton lag | Moot — Qt overlay is smooth |
| Web timeline click misalignment | Moot |
| Web chart kinds | Already implemented in Qt |

### 4. Docs

Update `README.md` so the documented launch path is the Qt window, and note the web viewer as
legacy/opt-in. Same in `docs/IMPLEMENTATION_PLAN.md` if it still calls web the primary UI.

## Acceptance

- `python -m desktop` opens the Qt window with no browser involved.
- Qt covers everything the user did in the web UI: video + skeleton overlay, stacked metric
  charts with the agreed kinds, click-to-seek, zoom, motif list, 搵相似, save/load project.
- `--web` still serves the legacy page (or is documented as removed if you drop it).
- README reflects the Qt-first flow.

## Out of scope

- Deleting the web files outright this pass
- Removing analyze API endpoints
- Any new Qt features beyond parity
