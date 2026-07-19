import numpy as np
import matplotlib.pyplot as plt

from csim.simulation import Simulator, rigid_body_step_fn
from csim.entities import Spacecraft

if __name__ == "__main__":
    dt = 0.01
    t0 = 0
    n_steps = 100000
    state0 = np.array([6800e3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    sc = Spacecraft(100, None)
    sim = Simulator(state0, t0, dt, n_steps, rigid_body_step_fn(dt, sc))

    sim.simulate()

    plt.plot(sim.X[:, 0])
    plt.show()
