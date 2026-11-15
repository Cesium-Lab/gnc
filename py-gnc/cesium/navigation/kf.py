"""Kalman filter steps as pure functions on a Gaussian estimate.

The caller (e.g. the flight computer) propagates the mean with whatever dynamics it
has and supplies the state transition matrix `Phi`; these functions handle the
covariance propagation and the measurement update. Nothing here holds state.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """Gaussian state estimate."""

    x: np.ndarray
    """State estimate (N,)"""
    P: np.ndarray
    """State covariance (N, N)"""


@dataclass(frozen=True)
class KfUpdate:
    est: Estimate
    """Estimate (N,)"""
    innovation: np.ndarray
    """Measurement residual (M,)"""
    S: np.ndarray
    """Innovation covariance (M, M)"""
    K: np.ndarray
    """Kalman gain (N, M)"""


def kf_propagate_cov(P: np.ndarray, Phi: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Covariance time update `P = Phi P Phi^T + Q`, symmetrized against round-off errors

    Parameters
    ----------
    P : np.ndarray (N, N)
        Covariance at the start of the step.
    Phi : np.ndarray (N, N)
        State transition matrix over the step (linearized about the estimate).
    Q : np.ndarray (N, N)
        Process noise covariance accumulated over the step.

    Returns
    -------
    P_next : np.ndarray (N, N)
        Propagated covariance.

    References
    ----------
    NASA Navigation Filter Best Practices, Algorithm 1.1 (The Linear Kalman Filter) p. 4

    """
    P_next = Phi @ P @ Phi.T + Q
    return 0.5 * (P_next + P_next.T)  # Positve definite


def plain_residual(z: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
    """Default measurement residual `z - z_hat`. Angles need wrapping instead."""
    return z - z_hat


def kf_update(
    est: Estimate,
    z: np.ndarray,
    h: Callable[[np.ndarray], np.ndarray],
    H: Callable[[np.ndarray], np.ndarray],
    R: np.ndarray,
    residual: Callable[[np.ndarray, np.ndarray], np.ndarray] = plain_residual,
) -> KfUpdate:
    """Measurement update (linear, or extended when `h` is nonlinear).

    Parameters
    ----------
    est : Estimate
        Prior estimate.
    z : np.ndarray (M,)
        Measurement.
    h : Callable
        Measurement model `h(x) -> z_expected (M,)`.
    H : Callable
        Jacobian of `h`, `H(x) -> (M, N)`, evaluated at the prior mean.
    R : np.ndarray (M, M)
        Measurement noise covariance assumed by the filter.
    residual : Callable
        `residual(z, z_hat) -> (M,)`, the innovation. Plain subtraction by default;
        pass an angle-wrapping version for angle measurements (e.g. azimuth) so a
        179 deg vs -179 deg pair gives -2 deg, not 358 deg.

    Returns
    -------
    KfUpdate
        Posterior estimate, innovation, innovation covariance, and gain.

    References
    ----------
    NASA Navigation Filter Best Practices, Algorithm 1.1 (The Linear Kalman Filter) p. 4
    """
    x, P = est.x, est.P
    H_k = H(x)

    innovation = residual(z, h(x))
    S = H_k @ P @ H_k.T + R
    K = np.linalg.solve(S, H_k @ P).T  # P H^T S^-1, using S and P symmetric

    I_KH = np.eye(len(x)) - K @ H_k
    P_next = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_next = 0.5 * (P_next + P_next.T)

    x_next = x + K @ innovation

    return KfUpdate(Estimate(x_next, P_next), innovation, S, K)
