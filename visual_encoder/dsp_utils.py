import numpy as np
from numpy.fft import fft2, fftshift


# Utilities associated with digital signal processing (DSP).


def ideal_lowpass(I, factor=0.6, method='Stone_et_al_2001'):
    if method == 'Stone_et_al_2001':
        m = factor * I.shape[0] / 2
        n = factor * I.shape[1] / 2
        N = np.min([m, n])
        I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
            int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
        return I
    else:
        raise ValueError('Método não suportado.')

def ideal_lowpass2(I, method='Stone_et_al_2001'):
    if method == 'Stone_et_al_2001':
        m = 0.7 * I.shape[0] / 2
        n = 0.7 * I.shape[1] / 2
        N = np.min([m, n])
        I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
            int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
        return I, N
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


def image_preprocessing(image, method='Stone_et_al_2001', blackWindowing=True):
    if blackWindowing is True:
        image = apply_blackman_harris_window(image)
    fft_from_image = fftshift(fft2(image))
    if method is not None:
        fft_from_image = ideal_lowpass(fft_from_image, method=method)
    return fft_from_image


def apply_blackman_harris_window(image):
    # Obtenção das dimensões da imagem
    height, width = image.shape
    # Aplicação do janelamento de Blackman-Harris nas linhas e colunas da imagem
    window_row = blackman_harris_window(width)
    window_col = blackman_harris_window(height)
    # Cálculo da imagem janelada
    image_windowed = np.outer(window_col, window_row) * image
    return image_windowed


def blackman_harris_window(size, a0=0.35875, a1=0.48829, a2=0.14128, a3=0.01168):
    # a0, a1, a2 e a3 são os coeficientes de janelamento
    # Criação do vetor de amostras
    n = np.arange(size)
    # Cálculo da janela de Blackman-Harris
    window = a0 - a1 * np.cos(2 * np.pi * n / (size - 1)) + a2 * np.cos(4 * np.pi * n / (size - 1)) - a3 * np.cos(
        6 * np.pi * n / (size - 1))
    return window


def normalize_product(F: object, G: object) -> object:
    # Versão modificada de crosspower_spectrum() para melhorias de eficiência
    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    return Q


def filter_array_by_maxmin(arr):
    # Filtro para remover valores absolutos muito diferentes dentro de um array
    filtered_arr = []
    threshold = (arr.max() - arr.min()) / 4
    n = len(arr)

    # Se o array tiver menos que 3 elementos, não há o que filtrar
    if n < 3:
        return arr

    # Adiciona o primeiro elemento
    filtered_arr.append(arr[0])

    # Itera sobre os elementos do array, ignorando o primeiro e o último elemento
    for i in range(1, n - 1):
        diff_prev = abs(abs(arr[i]) - abs(arr[i - 1]))
        diff_next = abs(abs(arr[i]) - abs(arr[i + 1]))

        # Verifica se a diferença entre o elemento atual e o anterior, e entre o atual e o próximo
        # são menores que o threshol
        if (diff_prev < threshold and diff_next < threshold):
            filtered_arr.append(arr[i])
        else:
            median = arr[i - 1] / 2 + arr[i + 1] / 2
            if abs(median - arr[i - 1]) < threshold:
                filtered_arr.append(median)

    # Adiciona o último elemento
    filtered_arr.append(arr[-1])

    return np.array(filtered_arr)