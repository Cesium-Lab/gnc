# ruff: noqa: E741
import sys
import numpy as np
import pytest
import matplotlib.pyplot as plt

sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt

from csim.simulation import Simulator
from csim.entities import Spacecraft
from csim.world import MU_EARTH, R_EARTH
from csim.visualization import plot_orbit


# TODO: actual tests?
if __name__ == "__main__":
    r = R_EARTH + 14500e3

    v = np.sqrt(MU_EARTH/r)
    print(r)
    print(v)

    dt = 1
    t0 = 0
    n_steps = 100000
    i = 98 * np.pi/180
    state0 = np.array([
        r*np.cos(i),0,r*np.sin(i), 
        0,v,0, 
        1,0,0,0, 
        0.0001,0,0])
    
    m = 100
    I = np.diag([2,2,2])

    sc = Spacecraft(m, None)
    sim = Simulator(state0, t0, dt, n_steps, sc)

    # sim.simulate()

    # # n = 100
    # # plt.plot(sim.X[:n,0], sim.X[:n,1])
    # # plt.axis('equal')
    # # plt.xlim()
    # # plt.show()

    # plot_orbit(sim.X)
    
    # plt.plot(sim.X[:,:3])
    # plt.show()

    # plt.plot(sim.X[:,3:6])
    # plt.show()

    # plt.plot(sim.X[:,6:10])
    # plt.show()

    # plt.plot(sim.X[:,10:13])
    # plt.show()




# t0 = 0
# dt = 0.01
# n_steps = 10000

# state0 = np.array([0,0,0, 0,0,0, 0,0,0,0, 0,0,0])

# sc = Spacecraft(100, np.identity(3), state0)
# sim = Simulator(t0, dt, n_steps, sc)

# sim.simulate()

# from csim.simulation import Simulator
# from csim.entities import Spacecraft

# t0 = 0
# dt = 0.01
# n_steps = 10000

# state0 = np.array([0,0,0, 0,0,0, 0,0,0,0, 0,0,0])

# sc = Spacecraft(100, np.identity(3), state0)
# sim = Simulator(t0, dt, n_steps, sc)

# sim.simulate()

# print(sim.X.shape)
# plt.plot(sim.X[:,:3])
# plt.plot(sim.X[:,:6])
# plt.show()