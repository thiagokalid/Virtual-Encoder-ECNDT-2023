import numpy as np
from dsp_utils import crosspower_spectrum


def pc_method(img1, img2):
    m, n = img1.shape
    q, Q = crosspower_spectrum(img1, img2)
    raw_deltay, raw_deltax = np.where(q == np.max(q))  # Máximo do normalized crosspower spectrum
    deltay = _fftpos2coordshift(raw_deltay, m)
    deltax = _fftpos2coordshift(raw_deltax, n)
    return deltax, deltay


def _fftpos2coordshift(shift, n):
    # Deslocamentos positivos pertencerão ao intervalo [0, N/2[ onde N é o número de amostras.
    # Deslocamentos negativos pertencerão ao intervalo [N/2, N[ onde N é o número de amostras.
    if shift > n / 2:
        return -(n - shift)
    else:
        return shift


