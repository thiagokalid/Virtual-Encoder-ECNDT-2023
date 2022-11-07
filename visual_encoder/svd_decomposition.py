import numpy as np
from visual_encoder.dsp_utils import crosspower_spectrum
from scipy.sparse.linalg import svds


def linear_regression(x, y):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    if (np.linalg.det(R.transpose() @ R) == 0):
        pass
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c



def svd_estimate_shift(angq, N, std_factor=10):
    # Sinal com as diferenças presentes pela inversão de fase:
    diff_q = np.diff(angq)
    corrected_difference = diff_q - 2. * np.pi * (diff_q > (2*np.pi * 0.9)) + 2. * np.pi * (diff_q < -(2*np.pi * 0.9))
    angq_unwrapped = np.cumsum(corrected_difference)

    #
    x_span = np.arange(0, diff_q.size)

    M = x_span.size // 2
    # Choosing a smaller window:
    x = x_span[M - 50:M + 50]
    y = angq_unwrapped[M - 50:M + 50]



    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y

    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o')
    plt.stem(x_span, angq_unwrapped)
    plt.plot(x, c + mu * x, color='r')
    delta = -mu * N/(2 * np.pi)
    return delta



def svd_method(f, g, std_factor=10):
    q, Q = crosspower_spectrum(f, g, 'Stone_et_al_2001')
    # num_Q
    qa, s, qb = svds(Q, k=1)
    ang_qa = np.angle(qa[:, 0])
    ang_qb = np.angle(qb[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qa, f.shape[0], std_factor)
    deltax = svd_estimate_shift(ang_qb, f.shape[1], std_factor)
    return deltax, deltay
