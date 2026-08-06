"""Dance Analyser desktop entry: python -m desktop"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import app` from backend/
_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from desktop.window import run_app  # noqa: E402


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
