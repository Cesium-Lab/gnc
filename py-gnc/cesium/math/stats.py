"""
Basic statistical math

Gaussian/uniform/multivariate sampling
Covariance/correlation calculations
White noise generation
"""

import numpy as np
from scipy.linalg import sqrtm

# TODO: test

def get_ellipse_points(mu: np.ndarray, Sigma: np.ndarray, P=0.95, N=200):
    t = np.linspace(0, 2*np.pi, N)
    circle = np.vstack((np.cos(t), np.sin(t)))
    rhs = -2*np.log(1-P)
    ellipse = np.sqrt(rhs) * circle
    scaled_ellipse = sqrtm(Sigma) @ ellipse # circle -> ellipse
    return scaled_ellipse + mu.reshape(2,1)

def noise(cov: np.ndarray, seed=None):
    if seed is not None:
        np.random.seed(seed)

    return np.random.multivariate_normal(
        mean=np.zeros_like(cov[0]),
        cov=cov
    )
    
def noise_vec(std_dev: float, n: int):
    """(n,) length vector of noise"""
    return np.random.normal(0.0, std_dev, size=n)

def rms(y: np.ndarray):
  """Calculates the Root Mean Square (RMS) of a numpy array."""
  return np.sqrt(np.mean(np.square(y)))

# def epsilon_and_nu(P: float, Sigma: np.ndarray):
#     nu = 1/ (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
#     epsilon = (1-P) * nu

#     return epsilon, nu

# def in_error_ellipse(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, P=0.95):
#     epsilon, nu = epsilon_and_nu(P, Sigma)
#     rhs = -2*np.log(epsilon / nu)   # = -2 log(1-P)
#     lhs = (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)
#     return lhs <= rhs
    
# def plot_gaussian(mu, Sigma, N):
#     x_arr = randn(N,2) @ sqrtm(Sigma) + mu

#     x_in = np.array([x for x in x_arr if in_error_ellipse(x, mu, Sigma)])
#     x_out = np.array([x for x in x_arr if not in_error_ellipse(x, mu, Sigma)])

#     e, _ = epsilon_and_nu(0.95, Sigma)
#     print(f"Percentage of points in ellipse {len(x_in) / N * 100}")
#     plt.scatter(x_in[:,0], x_in[:,1], marker='.')
#     plt.scatter(x_out[:,0], x_out[:,1], marker='.')
#     plt.title("95% Confidence Error Ellipse")
#     plt.show()