import numpy as np

##################################################
#               UKF (AA 228)
##################################################

# TODO: test with 273 examples
def UT(mu, Sigma, l_param):
    """returns x and w (point arrays)"""

    n = len(mu)
    x = np.zeros((2*n + 1, n))
    x[0] = mu

    factor = np.sqrt((l_param+n)) * np.linalg.cholesky(Sigma)
    for i in range(n): # Gives 0 to n-1 though
        
        # Getting the column of the factor
        x[i+1] = mu + factor[:,i]
        x[n+i+1] = mu - factor[:,i]

    w = np.ones((2*n + 1)) * 1 / 2 / (l_param + n) # Since every one is the same
    w[0] = l_param / (l_param + n)

    return x, w


def UT_inv(X, W, Q):
    """Returns mu, sigma"""
    assert isinstance(X, np.ndarray)
    assert isinstance(W, np.ndarray)

    n = int((len(W)-1) / 2) # Crude lowkey

    w_column = W.reshape(len(W),1)
    

    mu = np.sum(X * w_column, axis=0) # Element-wise

    Sigma = np.zeros((n,n))
    for w_, x in zip(W, X):
        Sigma += (
            w_ * np.outer((x - mu), (x - mu))
        )

    return mu, Sigma + Q