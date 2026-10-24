import numpy as np

from ..math.stats import noise

##################################################
#               Particle Filter (AA 228)
##################################################


# TODO: test with 273 stuff and other examples
def pf_predict(state_model, particles, u, dt, Q):
    """Predicts with noise"""
    return np.array([state_model(particle, u, dt) + noise(Q) for particle in particles])


def pf_weights(meas_model, particles_predicted: np.ndarray, measurement: np.ndarray, R):
    weights = np.zeros((measurement.shape[0]))

    expected_measurements = np.array(
        [meas_model(particle) for particle in particles_predicted]
    )
    R_inv = np.linalg.inv(R)
    # Gaussian (scaled) https://www.investopedia.com/terms/n/normaldistribution.asp
    weights = np.array(
        [
            np.exp(-1 / 2 * ((measurement - meas).T @ R_inv @ (measurement - meas)))
            for meas in expected_measurements
        ]
    )

    return weights
