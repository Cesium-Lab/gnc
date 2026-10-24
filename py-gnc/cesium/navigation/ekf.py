from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

##################################################
#               EKF (AA 228)
##################################################


@dataclass
class EkfParams:
    """Extended Kalman filter models for an N-state, M-measurement, U-input system.

    Attributes
    ----------
    f : Callable
        State transition model, `f(x (N,), u (U,)) -> x_next (N,)`.
    F : Callable
        Jacobian of `f` with respect to the state, `F(x (N,), u (U,)) -> (N, N)`.
    h : Callable
        Observation model, `h(x (N,)) -> z_expected (M,)`.
    H : Callable
        Jacobian of `h` with respect to the state, `H(x (N,)) -> (M, N)`.
    Q : np.ndarray (N, N)
        Process noise covariance.
    R : np.ndarray (M, M)
        Measurement noise covariance.
    """

    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    F: Callable[[np.ndarray, np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    H: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray


@dataclass
class EkfState:
    """Extended Kalman filter estimate for an N-state system.

    Attributes
    ----------
    x : np.ndarray (N,)
        State estimate (mean).
    P : np.ndarray (N, N)
        State estimate covariance.
    """

    x: np.ndarray  # Mean vector
    P: np.ndarray  # Covariance matrix


class EKF:
    """Extended Kalman filter.

    Important Attributes
    ----------
    state : EkfState
        Current state estimate and covariance, updated in place by `predict` and
        `update`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Extended_Kalman_filter
    - https://algorithmsbook.com/files/dm.pdf
    """

    def __init__(self, state0: EkfState, params: EkfParams):
        """
        Parameters
        ----------
        state0 : EkfState
            Initial state estimate and covariance.
        params : EkfParams
            Dynamics and measurement models and their noise covariances.
        """
        self.state = state0
        self.f = params.f
        self.F = params.F
        self.h = params.h
        self.H = params.H
        self.Q = params.Q
        self.R = params.R
        self.eye = np.eye(len(params.Q))

    def predict(self, u: np.ndarray) -> None:
        """Prediction step: propagate `self.state` to the next time step in place.

        Parameters
        ----------
        u : np.ndarray (U,)
            Control input applied over the step.
        """
        # I like getting variables out at the beginning
        x, P = self.state.x, self.state.P
        Q = self.Q

        x_next = self.f(x, u)

        F = self.F(x, u)
        P_next = F @ P @ F.T + Q

        self.state = EkfState(x_next, P_next)

    def update(self, z: np.ndarray) -> None:
        """Measurement update step: correct `self.state` in place with a measurement.

        Uses the Joseph-form covariance update for numerical stability.

        Parameters
        ----------
        z : np.ndarray (M,)
            Measurement vector.
        """
        # I like getting variables out at the beginning
        x_est, P_est = self.state.x, self.state.P
        R, eye = self.R, self.eye

        H = self.H(x_est)

        y_err = z - self.h(x_est)  # Measurement residual
        S = H @ P_est @ H.T + R  # Innovation covariance
        K = P_est @ H.T @ np.linalg.pinv(S)  # "near optimal" Kalman gain
        x_next = x_est + K @ y_err  # Update state
        P_next = (eye - K @ H) @ P_est @ (
            eye - K @ H
        ).T + K @ R @ K.T  # Update covariance

        # P_next = (I - K@H) @ P_est

        self.state = EkfState(x_next, P_next)
