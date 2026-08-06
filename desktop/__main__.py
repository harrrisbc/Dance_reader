"""Dance Analyser entry: python -m desktop

Default: PySide6 Qt window (product UI).
Legacy: python -m desktop --web  → FastAPI + browser /review.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Allow `import app` from backend/
_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _run_web(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Legacy web review UI — prefer Qt for normal use."""
    import uvicorn

    url = f"http://{host}:{port}/review"

    def _open() -> None:
        time.sleep(0.8)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if open_browser:
        threading.Thread(target=_open, daemon=True).start()

    print(f"Dance Analyser legacy web review: {url}")
    print("Prefer Qt: python -m desktop")
    print("Analysis runs locally (Python / CUDA). Press Ctrl+C to stop.")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


def _run_qt() -> None:
    from desktop.window import run_app

    run_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dance Analyser")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Legacy FastAPI + browser review UI (Qt is the default)",
    )
    parser.add_argument(
        "--qt",
        action="store_true",
        help="Open Qt window (default; kept as alias)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.web:
        _run_web(host=args.host, port=args.port, open_browser=not args.no_browser)
    else:
        _run_qt()


if __name__ == "__main__":
    main()
