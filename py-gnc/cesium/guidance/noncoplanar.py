import numpy as np

# TODO: Example 6-4 p. 344
def guid_i_change(di: float, phi_fpa: float, v_i: float):
    """Delta-v for an inclination-only change.

    Parameters
    ----------
    di : float
        Change in inclination [rad].
    phi_fpa : float
        Flight path angle at burn point [rad].
    v_i : float
        Satellite speed at burn point [m/s].

    Returns
    -------
    dv : float
        Delta-v magnitude [m/s].

    References
    ----------
    Vallado 4e, Algorithm 39, p. 344.
    """
    dv = 2*v_i * np.cos(phi_fpa) * np.sin(di/2.0)
    return dv

# TODO: Example 6-5 p. 347
def guid_raan_change_circular(draan: float, i: float, v_i: float):
    """Delta-v for a RAAN-only change of a circular orbit.

    Parameters
    ----------
    draan : float
        Change in right ascension of the ascending node [rad].
    i : float
        Inclination of the orbit, unchanged by the maneuver [rad].
    v_i : float
        Circular orbital speed [m/s].

    Returns
    -------
    dv : float
        Delta-v magnitude [m/s].

    References
    ----------
    Vallado 4e, Algorithm 40, p. 347.
    """
    ta = np.arccos(
        np.cos(i)**2 + np.sin(i)**2 * np.cos(draan)
    )
    dv = 2*v_i*np.sin(ta/2)
    return dv


