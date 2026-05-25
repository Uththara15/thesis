"""Pytest configuration for the similarity service test suite.

Adds the project root to sys.path so `import app.main` works without
requiring an editable install (`pip install -e .`).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
