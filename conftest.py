"""pytest configuration — ensures project root is on sys.path."""
import sys
from pathlib import Path

# Make sure the project root is importable without installation
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
