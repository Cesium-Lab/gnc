import numpy as np
from dataclasses import dataclass
from collections.abc import Callable

##################################################
#               EKF
##################################################


@dataclass
class EkfParams:
    """Extended Kalman Filter Parameters for Nx1 state, Mx1 measurements, Ux1 inputs
    - `f`: **state transition model getter**
        f(state: np.ndarray [Nx1], input: np.ndarray [Ux1]) --> next_state_estimation: np.ndarray [Nx1]
    - `F`: **jacobian of state transition model getter**
        F(state: np.ndarray [Nx1], input: np.ndarray [Ux1]) --> jacobian_F: np.ndarray [NxN]
    - `h`: **observation model getter**
        h(state: np.ndarray [Nx1]) --> expected_measurement: np.ndarray [Mx1]
    - `H`: **jacobian of observation model getter**
        H(state: np.ndarray [Nx1]) --> jacobian_H: np.ndarray [MxN]
    - `Q`: **process noise covariance** (np.ndarray) [NxN]
    - `R`: **measurement noise covariance** (np.ndarray) [MxM]
    """

    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    F: Callable[[np.ndarray, np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    H: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray


@dataclass
class State:
    """Extended Kalman Filter State (Nx1 state)
    - `x`: **State vector** (np.ndarray) [Nx1]
    - `P`: **State covariance** (np.ndarray) [NxN]"""

    x: np.ndarray  # Mean vector
    P: np.ndarray  # Covariance matrix


class EKF:
    """https://en.wikipedia.org/wiki/Extended_Kalman_filter

    Important attributes:
    - `state`: np.ndarray (State [Nx1])
    """

    def __init__(self, state0: State, params: EkfParams):
        """
        Args:
            state0 (State): Starting state
            params (EkfParams): Starting EKF Parameters
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
        """Predicts state. Modifies member state and also returns it

        Args:
            u (np.ndarray): Input vector [Ux1]

        Returns:
            State: predicted next state k+1
        """
        # I like getting variables out at the beginning
        x, P = self.state.x, self.state.P
        Q = self.Q

        x_next = self.f(x, u)

        F = self.F(x, u)
        P_next = F @ P @ F.T + Q

        self.state = State(x_next, P_next)

    def update(self, z: np.ndarray) -> None:
        """Updates state. Modifies member state and also returns it

        Args:
            z (np.ndarray): Observation vector [Mx1]
            params (dict): (Optional) Dict with extra parameters for f() and F()
                (like params["dt"])

        Returns:
            State: final updated next state k+1
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

        self.state = State(x_next, P_next)
