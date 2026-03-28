#!/usr/bin/env python3
"""Entry-point shim: delegates to the root demo.py module."""

import sys
import os
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the root demo main()
from demo import main  # noqa: E402

if __name__ == "__main__":
    main()
