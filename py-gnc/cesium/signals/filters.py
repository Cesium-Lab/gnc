import numpy as np

# TODO: test like with attenuation or something 
class LPF:
    def __init__(self, alpha: float):
        assert isinstance(alpha, float)
        self.alpha = alpha
        self.y = 0.0
    
    def run(self, x: float | np.ndarray):
        self.y = self.alpha*x + (1-self.alpha)*self.y
        # self.y = self.alpha*self.y + (1-self.alpha)*x
        return self.y