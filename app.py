from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import main


def _translate_args(argv: list[str]) -> list[str]:
    """Translates flags like --run-id <id> --report into standard subcommands."""
    if "--report" in argv and "--run-id" in argv:
        idx = argv.index("--run-id")
        if idx + 1 < len(argv):
            run_id = argv[idx + 1]
            rest = [a for i, a in enumerate(argv) if i not in (idx, idx + 1) and a != "--report"]
            return rest + ["report", run_id]
    return argv


if __name__ == "__main__":
    translated = _translate_args(sys.argv[1:])
    sys.exit(main(translated))
