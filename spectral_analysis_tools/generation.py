import numpy as np
from numpy.random import randn
from .core import Df


def generate_autocorr(Lambda_f, mean_A=0., dt=1):
    """Generate a single batch for the variable 'A'.

    Parameters
    ----------
    Lambda_f : array-like
        Autocorrelation function.
    mean_A : float, optional
        Mean value of the variable `A`.
    dt : float, optional
        Sampling interval.

    Returns
    -------
    numpy.ndarray
        Generated time series.
    """
    Nf = len(Lambda_f)
    N = (Nf - 1) * 2
    alpha = np.zeros(Nf, dtype=complex)
    for i_f in range(Nf):
        sigma_f = np.sqrt(Lambda_f[i_f] / (2.0 * Df(i_f, Nf)))
        alpha[i_f] = (randn() + 1j * randn()) * sigma_f
    alpha[0] = np.sqrt(N) * mean_A
    return np.fft.irfft(alpha, norm="ortho") / np.sqrt(dt)


def generate_crosscorr(Lambda_A, Lambda_B, sf, phif, mean_A=0.0, mean_B=0.0, dt=1):
    """Generate a single batch for the variables 'A' and 'B'.

    Parameters
    ----------
    Lambda_A, Lambda_B : array-like
        Autocorrelation functions.
    sf, phif : array-like
        Correlation strength and phase.
    mean_A, mean_B : float, optional
        Mean values of the variables `A` and `B`.
    dt : float, optional
        Sampling interval.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Generated time series for `A` and `B`.
    """
    Nf = len(Lambda_A)
    N = (Nf - 1) * 2
    alpha = np.zeros(Nf, dtype=complex)
    beta = np.zeros(Nf, dtype=complex)
    for i_f in range(Nf):
        fac = 1.0 / (1.0 - sf[i_f] ** 2)
        h0 = fac / 2.0 * (1.0 / Lambda_A[i_f] + 1.0 / Lambda_B[i_f])
        hx = -fac * sf[i_f] * np.cos(phif[i_f]) / np.sqrt(Lambda_A[i_f] * Lambda_B[i_f])
        hy = -fac * sf[i_f] * np.sin(phif[i_f]) / np.sqrt(Lambda_A[i_f] * Lambda_B[i_f])
        hz = fac / 2.0 * (1.0 / Lambda_A[i_f] - 1.0 / Lambda_B[i_f])
        h = np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
        if h == hz:
            Q = np.eye(2)
        else:
            v_plus = 1.0 / np.sqrt(2.0 * h * (h + hz)) * np.array([h + hz, hx + 1j * hy])
            v_mins = 1.0 / np.sqrt(2.0 * h * (h - hz)) * np.array([h - hz, -hx - 1j * hy])
            Q = np.transpose(np.array([v_plus, v_mins]))
        Lambda_plus = 1.0 / (np.sqrt(N) * (h0 + h))
        Lambda_mins = 1.0 / (np.sqrt(N) * (h0 - h))
        sigma_plus = np.sqrt(Lambda_plus * np.sqrt(N) / (2.0 * Df(i_f, Nf)))
        sigma_mins = np.sqrt(Lambda_mins * np.sqrt(N) / (2.0 * Df(i_f, Nf)))
        gamma_f = (randn() + 1j * randn()) * sigma_plus
        delta_f = (randn() + 1j * randn()) * sigma_mins
        [[alpha[i_f]], [beta[i_f]]] = np.matmul(Q, np.array([[gamma_f], [delta_f]]))
        alpha[0] = np.sqrt(N) * mean_A
        beta[0] = np.sqrt(N) * mean_B
    A = np.fft.irfft(alpha, norm="ortho") / np.sqrt(dt)
    B = np.fft.irfft(beta, norm="ortho") / np.sqrt(dt)
    return A, B
