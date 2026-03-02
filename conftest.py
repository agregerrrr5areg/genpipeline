# conftest.py — project-level pytest configuration
#
# PSEUDOCODE:
#   AT pytest startup (before any test collects):
#     find the venv/bin directory next to this file
#     IF it exists AND it is not already on PATH:
#         prepend it to PATH
#   WHY: the ninja binary lives in venv/bin after `pip install ninja`.
#        torch.utils.cpp_extension.verify_ninja_availability() requires ninja
#        to be on the system PATH; prepending venv/bin satisfies that without
#        requiring a system-wide ninja install.
import os
from pathlib import Path

_VENV_BIN = Path(__file__).parent / "venv" / "bin"
if _VENV_BIN.exists():
    _path = os.environ.get("PATH", "")
    if str(_VENV_BIN) not in _path.split(os.pathsep):
        os.environ["PATH"] = str(_VENV_BIN) + os.pathsep + _path
