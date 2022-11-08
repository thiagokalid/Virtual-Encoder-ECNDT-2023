import numpy as np
from numpy.fft import fft2, fftshift

# Utilities associated with digital signal processing (DSP).


def ideal_lowpass(I, method='Stone_et_al_2001'):
    if method == 'Stone_et_al_2001':
        m = 0.6 * I.shape[0]/2
        n = 0.6 * I.shape[1]/2
        N = np.min([m,n])
        I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
            int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
        return I
    else:
        raise ValueError('Método não suportado.')


def crosspower_spectrum(f, g, method=None):
    # Reference: https://en.wikipedia.org/wiki/Phase_correlation
    F = fftshift(fft2(f))
    G = fftshift(fft2(g))
    if method is not None:
        F = ideal_lowpass(F, method=method)
        G = ideal_lowpass(G, method=method)
    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    q = np.real(np.fft.ifft2(Q))  # in theory, q must be fully real, but due to numerical approximations it is not.
    return q, Q


def ideal_lowpass2(I, method='Stone_et_al_2001'):
    if method == 'Stone_et_al_2001':
        m = 0.7 * I.shape[0]/2
        n = 0.7 * I.shape[1]/2
        N = np.min([m,n])
        I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
            int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
        return I, N
    else:
        raise ValueError('Método não suportado.')

