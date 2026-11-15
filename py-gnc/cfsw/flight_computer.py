"""Flight computer interface to everything else"""

from dataclasses import dataclass, field

import numpy as np

from cesium.navigation.kf import Estimate, KfUpdate, kf_propagate_cov, kf_update
from cesium.orbital import hcw

from .measurements import MeasModel, Measurement


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
    Q_rate: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    """Process noise added per second of propagation (6, 6); `Q = Q_rate * dt`.
    Zero by default since the truth is exactly HCW for now."""


class FlightComputer:
    """Relative navigation: HCW propagation plus Kalman updates from tagged measurements."""

    def __init__(
        self,
        t0: float,
        est0: Estimate,
        config: PositionFcConfig,
        meas_models: dict[str, MeasModel] | None = None,
    ) -> None:
        self.config = config
        self._t = t0

        self._n = est0.x.shape[0]
        assert est0.x.shape == (self._n,)
        assert est0.P.shape == (self._n, self._n)

        self._est = Estimate(est0.x.astype(float).copy(), est0.P.astype(float).copy())

        self.last_update: KfUpdate | None = None
        self.meas_models = {} if meas_models is None else dict(meas_models)

    ############################################################################
    #               Interface
    ############################################################################

    def step(self, t: float, meas: list[Measurement] | None) -> Estimate:
        """Propagate the estimate to `t`, then update once per measurement.

        Parameters
        ----------
        t : float
            Time tag of the measurements [s]
        meas : list[Measurement] or None
            Measurements taken at `t`, each tagged with a sensor id in `meas_models`.
            None (or empty) if no sensor fired this step (predict only).

        Returns
        -------
        self.estimate : Estimate
            Copy of the estimate after this step (telemetry).

        Notes
        -----
        Measurements are assumed processed in time order with no latency. Several
        measurements at the same `t` are applied one after another, which equals a
        single stacked update when their noises are uncorrelated (block-diagonal R).

        References
        ----------
        NASA Navigation Filter Best Practices, Ch. 4 (measurement processing order).
        """
        dt = t - self._t
        if dt < 0:
            raise ValueError(f"time went backwards: t={t} < last t={self._t}")

        if dt > 0:
            x_next, Phi = self._propagate(self._est.x, dt)  # Propagate x
            P_next = kf_propagate_cov(self._est.P, Phi, self.config.Q_rate * dt)
            self._est = Estimate(x_next, P_next)
            self._t = t

        for m in meas or []:
            model = self.meas_models[m.sensor_id]  # KeyError: unknown sensor
            self.last_update = kf_update(
                self._est, m.z, model.h, model.H, model.R, model.residual
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
