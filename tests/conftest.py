"""Pytest bootstrap for the pysf2 suite.

Copyright 2025-2026 Stan Shumsky

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

The engine has no host/ComfyUI dependency, so the suite runs in any venv
with numpy + pytest: the only path surgery needed is making the repo root
(``import pysf2``) and the tests dir (``import _sf2gen`` / ``_notes``)
importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
