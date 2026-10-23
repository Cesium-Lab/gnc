import numpy as np
from numpy.linalg import norm
from ...world.constants import G0, MU_EARTH
from ...constants import DEG_TO_RAD
from ...math.transformations import PCI_to_NTW
from ...math.coes import E_from_ta

# TODO: Example 6-4 p. 344
def guid_i_change(di: float, phi_fpa: float, v_i: float):
    """phi flight path angle"""
    dv = 2*v_i * np.cos(phi_fpa) * np.sin(di/2.0)
    return dv

# TODO: Example 6-5 p. 347
def guid_raan_change_circular(draan: float, i: float, v_i: float):
    ta = np.arccos(
        np.cos(i)**2 + np.sin(i)**2 * np.cos(draan)
    )
    dv = 2*v_i*np.sin(ta/2)
    return dv


