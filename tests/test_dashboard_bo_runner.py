import threading
from unittest.mock import patch, MagicMock
import numpy as np
from dashboard_state import AppState
from dashboard_bo_runner import BORunner


def _mock_vae():
    vae = MagicMock()
    vae.latent_dim = 16
    voxel_mock = MagicMock()
    voxel_mock.squeeze.return_value.cpu.return_value.numpy.return_value = np.zeros((32, 32, 32))
    vae.decode.return_value = voxel_mock
    return vae


def test_bo_runner_fills_state():
    state = AppState()
    state.reset(total_iters=3)
    vae = _mock_vae()

    with patch("dashboard_bo_runner.DesignOptimizer") as MockOpt:
        instance = MockOpt.return_value
        instance.x_history = [np.zeros(16)]
        instance.y_history = [-0.01]

        def fake_optimize_step():
            z = np.random.randn(16)
            instance.x_history.append(z)
            instance.y_history.append(-float(np.random.rand()))
            return z, {}
        instance.optimize_step.side_effect = fake_optimize_step
        instance.initialize_search.return_value = None

        runner = BORunner(state=state, vae=vae, device="cpu", n_iters=3, mode="bo-only")
        t = threading.Thread(target=runner.run)
        t.start()
        t.join(timeout=10)

    assert state.status == "done"
    assert len(state.iterations) >= 1


def test_bo_runner_stops_on_request():
    state = AppState()
    state.reset(total_iters=100)
    vae = _mock_vae()

    with patch("dashboard_bo_runner.DesignOptimizer") as MockOpt:
        instance = MockOpt.return_value
        call_count = [0]

        def fake_step():
            call_count[0] += 1
            if call_count[0] == 2:
                state.request_stop()
            z = np.zeros(16)
            instance.x_history.append(z)
            instance.y_history.append(-0.01 * call_count[0])
            return z, {}

        instance.optimize_step.side_effect = fake_step
        instance.initialize_search.return_value = None
        instance.y_history = [-0.01]
        instance.x_history = [np.zeros(16)]

        runner = BORunner(state=state, vae=vae, device="cpu", n_iters=100, mode="bo-only")
        t = threading.Thread(target=runner.run)
        t.start()
        t.join(timeout=10)

    assert state.status in ("done", "idle")
    assert call_count[0] <= 5
