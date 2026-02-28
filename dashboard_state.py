"""dashboard_state.py — shared live state between BO thread and Streamlit UI."""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FEMResult:
    stress_max: float
    displacement_max: float
    mass: float
    h_mm: float
    r_mm: float


@dataclass
class IterResult:
    step: int
    objective: float
    z: list          # 16-dim float list
    voxel: Optional[np.ndarray]   # (32,32,32) binary, or None
    fem: Optional[FEMResult]      # only set when FreeCAD validates


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self.status: str = "idle"          # "idle" | "running" | "done"
        self.iterations: list[IterResult] = []
        self.best: Optional[IterResult] = None
        self.stop_requested: bool = False
        self.total_iters: int = 50
        self.selected: Optional[str] = None   # filename from design browser

    def add_iteration(self, r: IterResult) -> None:
        with self._lock:
            self.iterations.append(r)
            if self.best is None or r.objective < self.best.objective:
                self.best = r

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def reset(self, total_iters: int = 50) -> None:
        with self._lock:
            self.iterations.clear()
            self.best = None
            self.stop_requested = False
            self.status = "idle"
            self.total_iters = total_iters
