import numpy as np
from visual_encoder.dsp_utils import crosspower_spectrum, normalize_product, filter_array_by_maxmin
from scipy.sparse.linalg import svds


def linear_regression(x, y):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c


def phase_unwrapping(phase_vec, factor=0.7):
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
                phase_diff < -(2 * np.pi * factor))
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
    delta = mu * N / (2 * np.pi)
    return delta


def svd_method(img_beg, img_end, frequency_window="Stone_et_al_2001"):
    M, N = img_beg.shape
    q, Q = crosspower_spectrum(img_end, img_beg, frequency_window)
    qu, s, qv = svds(Q, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M)
    deltax = svd_estimate_shift(ang_qv, N)
    return deltax, deltay


def optimized_svd_method(processed_img_beg, processed_img_end, M, N, filter_values=False):
    Q = normalize_product(processed_img_end, processed_img_beg)
    qu, s, qv = svds(Q, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    if filter_values: # desativado, resolução inicial do bug nomeado de f153
        ang_qu = filter_array_by_maxmin(ang_qu)
        ang_qv = filter_array_by_maxmin(ang_qv)

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M)
    deltax = svd_estimate_shift(ang_qv, N)
    return deltax, deltay