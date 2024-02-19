import numpy as np
from visual_encoder.dsp_utils import crosspower_spectrum, normalize_product, filter_array_by_maxmin
from scipy.sparse.linalg import svds


def linear_regression(x, y):
    # Construir a matriz de design
    R = np.column_stack((x, np.ones_like(x)))

    # Calcular os coeficientes usando mínimos quadrados
    mu, c = np.linalg.lstsq(R, y, rcond=None)[0]

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
    #print(x.size)
    y = phase_unwrapped[M - 50:M + 50]
    mu, c = linear_regression(x, y)
    delta = mu * N / (2 * np.pi)
    return delta


def svd_estimate_shift_morereturns(phase_vec, N):
    # Phase unwrapping:
    phase_unwrapped = phase_unwrapping(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2
    x = r[M-70:M-20]
    y = phase_unwrapped[M-70:M-20]
    x2 = r[M+20:M+70]
    y2 = phase_unwrapped[M+20:M+70]
    mu2, c2 = linear_regression(x2, y2)
    mu1, c1 = linear_regression(x, y)
    mu = (mu1/2)+(mu2/2)
    delta = mu * N / (2 * np.pi)
    return -delta, phase_unwrapped, mu, c1

def svd_method(img_beg, img_end, frequency_window="Stone_et_al_2001"):
    M, N = img_beg.shape
    q, Q = crosspower_spectrum(img_end, img_beg)
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

    if filter_values:
        ang_qu = filter_array_by_maxmin(ang_qu)
        ang_qv = filter_array_by_maxmin(ang_qv)

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M)
    deltax = svd_estimate_shift(ang_qv, N)
    return deltax, deltay
