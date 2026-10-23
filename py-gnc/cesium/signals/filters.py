import numpy as np

# TODO: test like with attenuation or something and for dimensions
class LPF:
    """First-order discrete low-pass filter (exponential moving average).

    Attributes
    ----------
    alpha : float
        Smoothing factor in `[0, 1]`; weight given to the newest sample.
    y : float or np.ndarray
        Current filter output.
    """

    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Smoothing factor in `[0, 1]`. Larger values track input faster;
            smaller values smooth more.
        """
        assert isinstance(alpha, float)
        self.alpha = alpha
        self.y = 0.0
    
    def run(self, x: float | np.ndarray):
        """Filter new sample.

        Parameters
        ----------
        x : float or np.ndarray
            New input sample.

        Returns
        -------
        float or np.ndarray
            Updated filter output.
        """
        self.y = self.alpha*x + (1-self.alpha)*self.y
        # self.y = self.alpha*self.y + (1-self.alpha)*x
        return self.y