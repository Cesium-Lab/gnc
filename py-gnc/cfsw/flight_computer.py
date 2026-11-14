"""Flight computer interface to everything else

Everything it knows arrives through its constructor (configuration and initial
estimate, loaded "before launch") or through `step(t, z)` (measurements)
"""

from dataclasses import dataclass, field

import numpy as np

from cesium.navigation.kf import Estimate, KfUpdate, kf_propagate_cov, kf_update
from cesium.navigation.measurements import meas_rel_pos, meas_rel_pos_jac
from cesium.orbital import hcw


@dataclass(frozen=True)
class PositionFcConfig:
    """Onboard configuration: the flight computer's models and assumed noise.

    `Q_rate * dt` is a first-order approximation of the discrete process noise

    References
    ----------
    NASA Navigation Filter Best Practices, Ch. 3 (process noise covariance).
    Maybeck, Stochastic Models, Estimation, and Control Vol. 1, performance analysis
    (truth model vs filter model).
    Van Loan, "Computing Integrals Involving the Matrix Exponential", IEEE TAC, 1978.
    Brown & Hwang, Introduction to Random Signals and Applied Kalman Filtering 4e
    (discrete process noise).
    """

    n: float
    """Target mean motion [rad/s]"""
    R: np.ndarray
    """M noise covariance (3, 3) [m^2]"""
    Q_rate: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    """Process noise added per second of propagation (6, 6); `Q = Q_rate * dt`.
    Zero by default since the truth is exactly HCW for now."""


class FlightComputer:
    """Relative navigation: HCW propagation plus Kalman updates from relative position."""

    def __init__(self, t0: float, est0: Estimate, config: PositionFcConfig) -> None:
        self.config = config
        self._t = t0
        self._est = Estimate(est0.x.astype(float).copy(), est0.P.astype(float).copy())
        self.last_update: KfUpdate | None = None

    ############################################################################
    #               Interface
    ############################################################################

    def step(self, t: float, z: np.ndarray | None) -> Estimate:
        """Propagate the estimate to `t`, then update with `z` if there is one.

        Parameters
        ----------
        t : float
            Time tag of the measurement [s]
        z : np.ndarray (3,) or None
            Relative position measurement [m], or None if no measurement
            this step (predict only).

        Returns
        -------
        self.estimate : Estimate
            Copy of the estimate after this step (telemetry).

        Notes
        -----
        Measurements are assumed processed in time order with no latency;

        References
        ----------
        NASA Navigation Filter Best Practices
        """
        dt = t - self._t
        if dt < 0:
            raise ValueError(f"time went backwards: t={t} < last t={self._t}")

        if dt > 0:
            x_next, Phi = self._propagate(self._est.x, dt)  # Propagate x
            P_next = kf_propagate_cov(self._est.P, Phi, self.config.Q_rate * dt)
            self._est = Estimate(x_next, P_next)
            self._t = t

        if z is not None:
            self.last_update = kf_update(
                self._est, z, meas_rel_pos, meas_rel_pos_jac, self.config.R
            )
            self._est = self.last_update.est

        return self.estimate  # COPY of estimate

    ############################################################################
    #               Telemetry (read-only)
    ############################################################################

    @property
    def estimate(self) -> Estimate:
        """Copy of the current estimate"""
        return Estimate(self._est.x.copy(), self._est.P.copy())

    @property
    def t(self) -> float:
        """Time of the current estimate [s]."""
        return self._t

    ############################################################################
    #               Internals
    ############################################################################

    # TODO: CHANGE TO MAKE GENERIC
    def _propagate(self, x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Mean propagation and the STM for the covariance.

        The seam for nonlinear dynamics later: integrate the full dynamics for `x` and
        the variational equations `Phi_dot = A(x) Phi` for `Phi` alongside it. For HCW
        both are exact and given by the same closed-form STM. For attitude (MEKF) the
        STM comes from the error-state dynamics instead.

        References
        ----------
        NASA Navigation Filter Best Practices, Ch. 3 (integrate the STM with the state),
        Ch. 1 (MEKF).
        """
        Phi = hcw.calc_stm(dt, self.config.n)
        x_next = Phi @ x
        return x_next, Phi
