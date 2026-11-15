import numpy as np

from csim.simulator import Simulator


def test_duration_and_repeated_runs():
    sim = Simulator(
        np.array([2.0]),
        t0=10.0,
        dt=0.5,
        max_steps=4,
        derivative=lambda t, x: np.ones_like(x),
    )

    # Truncated
    sim.simulate(t_final=0.75)
    np.testing.assert_allclose(sim.t, [10.0, 10.5, 10.75])
    np.testing.assert_allclose(sim.X[:, 0], [2.0, 2.5, 2.75])
    assert sim.t_curr == sim.t[-1]

    # Full duration
    sim.simulate()
    np.testing.assert_allclose(sim.X[:, 0], [2.0, 2.5, 3.0, 3.5, 4.0])


def test_stop_trims_history():
    sim = Simulator(
        np.array([0.0]),
        t0=0.0,
        dt=0.5,
        max_steps=100,
        derivative=lambda t, x: np.ones_like(x),
        stop_funcs=[lambda x: x[0] >= 1.0],
    )
    sim.simulate()
    np.testing.assert_allclose(sim.t, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(sim.X[:, 0], sim.t)  # derivative of 1 so x=t
