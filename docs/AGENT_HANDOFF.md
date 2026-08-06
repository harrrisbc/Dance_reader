# WORK ORDER — Python desktop pivot

**To:** programming agent `fc656603-33e0-4a37-9426-8743dd68bcf5`  
**From:** Design agent  

## Decision

User chose **option 1**: local **single Python app**. Theatre/HTML beauty is **not** required. Stop treating Vite as primary UI.

## Do

1. Read `docs/IMPLEMENTATION_PLAN.md` (Python-first).
2. Prefer extending `desktop/` (if present) + existing `backend/app/pipeline`.
3. Do not invest in new React features unless asked.
4. Keep Traditional Chinese UI strings.
5. FastAPI optional; desktop must call pipeline in-process.

## Done when

`python -m desktop` (or documented equivalent) opens a local window and runs analyze end-to-end without the browser.
