"""Circular Restricted Three-Body Problem"""

import numpy as np
import pytest

from ..math.base import jacobian


def lagrange_points(mu1: float, mu2: float, r12: float, rtol=1e-9, max_iter=100):
    """
    Lagrange points in the barycentric synodic frame.

    Parameters
    ----------
    mu1, mu2 : float
        Gravitational parameters of the bodies [m3/s2].
    r12 : float
        Distance between them [m].
    rtol : float, optional
        Relative tolerance on iterative convergence. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    L : ndarray (5, 3)
        Lagrange-point positions [m].
        
    References
    -------
    Curtis Equation 2.193, p. 94
        For L4, L5
    Curtis Equation 2.194, p. 94
        For L1, L2, L3
    """
    
    mu_small = min(mu1, mu2)
    
    # pi is just how it is notated in the book
    pi2 = mu_small / (mu1 + mu2)
    
    r1 = -pi2*r12
    r2 = (1-pi2)*r12
    assert abs(r12 - r2 + r1) < 1e-9
    
    def eq(x):
        return (
            (1 - pi2)*(x + pi2*r12)/abs(x + pi2*r12)**3
            + pi2*(x + pi2*r12 - r12)/abs(x + pi2*r12 - r12)**3
            - x / r12**3)
        
    L = np.zeros((5,3))
    L[0:3,0] = [r2 * 0.9, r2 * 1.1, -r12]
    
    for i,x in enumerate(L[0:3,0]):
        for _ in range(max_iter):
                
            # or update it
            df = jacobian(eq, x, 1, delta=0.1)
            x_guess = x - eq(x) / df

            if abs(x_guess - x) < rtol * r12:
                L[i,0] = x
                break
            x = x_guess
        else:
            raise RuntimeError(f"L{i+1} did not converge. Only got to {x_guess} with err of {x_guess - x}")
        
    # L4, L5
    L[3,0] = L[4,0] = r1 + r12/2
    L[3,1] = r12 * np.sqrt(3)/2
    L[4,1] = -L[3,1]
    
    
            
    return L
        
        
    
    
    
    
    
    