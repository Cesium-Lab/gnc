import numpy as np


# TODO: verify where also this could be wrong
def freq_spectrum(y: np.ndarray, dt: float):
    """y is (N,M) or (N,)"""
    N = len(y)

    # y_f = fft(y)
    # y_f = 1.0/N * np.abs(y_f[0:N//2])
    # x_f = fftfreq(N, dt)[:N//2]

    y_f = np.abs(np.fft.rfft(y, axis=0)) / N
    x_f = np.fft.rfftfreq(N, dt)

    # Double amplitudes except DC and Nyquist (when N is even)
    if N % 2 == 0:
        y_f[1:-1, ...] *= 2
    else:
        y_f[1:, ...] *= 2

    return x_f, y_f
