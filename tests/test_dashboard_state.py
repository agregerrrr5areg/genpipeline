import threading
from dashboard_state import AppState, IterResult

def test_initial_state():
    s = AppState()
    assert s.status == "idle"
    assert s.iterations == []
    assert s.best is None

def test_add_iteration():
    s = AppState()
    r = IterResult(step=1, objective=-0.05, z=[0.0]*16, voxel=None, fem=None)
    s.add_iteration(r)
    assert len(s.iterations) == 1
    assert s.best.objective == -0.05

def test_best_tracks_minimum():
    s = AppState()
    s.add_iteration(IterResult(step=1, objective=-0.03, z=[0.0]*16, voxel=None, fem=None))
    s.add_iteration(IterResult(step=2, objective=-0.08, z=[0.0]*16, voxel=None, fem=None))
    s.add_iteration(IterResult(step=3, objective=-0.05, z=[0.0]*16, voxel=None, fem=None))
    assert s.best.objective == -0.08
    assert s.best.step == 2

def test_stop_flag():
    s = AppState()
    assert not s.stop_requested
    s.request_stop()
    assert s.stop_requested

def test_thread_safety():
    s = AppState()
    errors = []
    def writer():
        for i in range(100):
            try:
                s.add_iteration(IterResult(step=i, objective=-i*0.01,
                                           z=[0.0]*16, voxel=None, fem=None))
            except Exception as e:
                errors.append(e)
    threads = [threading.Thread(target=writer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert len(s.iterations) == 400
