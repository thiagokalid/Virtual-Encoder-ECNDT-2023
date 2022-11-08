import numpy as np
from visual_encoder.dsp_utils import crosspower_spectrum
from scipy.sparse.linalg import svds


def linear_regression(x, y):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c


def phase_unwrapping(phase_vec):
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * 0.9)) + 2. * np.pi * (
                phase_diff < -(2 * np.pi * 0.9))
    return np.cumsum(corrected_difference)


def svd_estimate_shift(phase_vec, N):
    # Phase unwrapping:
    phase_unwrapped = phase_unwrapping(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2
    # Choosing a smaller window:
    x = r[M - 50:M + 50]
    y = phase_unwrapped[M - 50:M + 50]
    mu, c = linear_regression(x, y)
    delta = -mu * N / (2 * np.pi)
    return delta


def svd_method(f, g, frequency_window="Stone_et_al_2001"):
    M, N = f.shape
    q, Q = crosspower_spectrum(f, g, frequency_window)
    qu, s, qv = svds(Q, k=1)
    pu = np.angle(qu[:, 0])
    pv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(pu, M)
    deltax = svd_estimate_shift(pv, N)
    return deltax, deltay
