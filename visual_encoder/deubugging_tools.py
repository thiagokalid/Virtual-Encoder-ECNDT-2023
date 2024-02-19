import numpy as np
from numpy.random import RandomState
from scipy.sparse.linalg import svds
from scipy.ndimage import rotate, uniform_filter
import matplotlib.pyplot as plt

from visual_encoder.dsp_utils import image_preprocessing, simplified_crosspower_spectrun
from visual_encoder.svd_decomposition import svd_estimate_shift_morereturns

import numpy as np
from numpy.random import RandomState



def img_gen(dx=0, dy=0, width=512, height=600, zoom=1, angle=0):
    img = np.zeros((width, height), dtype=float)
    prng = RandomState(1234)
    circ = prng.rand(100, 4)

    # Definir a grade de coordenadas X e Y
    X = np.arange(width)
    Y = np.arange(height)

    # Aplicar o deslocamento
    X = X - (width) + dx
    Y = Y - (height) + dy

    # Aplicar o zoom
    X = (X / zoom) + (width / 2)
    Y = (Y / zoom) + (height / 2)

    X, Y = np.meshgrid(X, Y)

    for i in range(circ.shape[0]):
        # Calcular o ângulo de rotação para a máscara
        angle_rad = np.radians(angle)

        # Calcular as coordenadas X e Y ajustadas com rotação para cada ponto da malha
        X_rotated = X * np.cos(angle_rad) - Y * np.sin(angle_rad)
        Y_rotated = X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # Ajustar as coordenadas dos centros dos círculos para centralizar
        circle_center_x = circ[i, 0] * width - (width / 2)
        circle_center_y = circ[i, 1] * height - (height / 2)

        # Calcular a máscara com base nas coordenadas rotacionadas e ajustadas
        mask = ((X_rotated - circle_center_x) ** 2 + (Y_rotated - circle_center_y) ** 2) < (circ[i, 2] * 100) ** 2

        # Atribuir à imagem
        img[mask.T] = circ[i, 3]  # Transpor a máscara antes de atribuir à imagem

    return img




def estimate_shift(img_0, img_1, method='Stone_et_al_2001', random_state = None):
    M, N = img_0.shape
    img_0_processed = image_preprocessing(img_0, method)
    img_1_processed = image_preprocessing(img_1, method)

    q, Q = simplified_crosspower_spectrun(img_0_processed, img_1_processed)
    qu, s, qv = svds(Q, k=1, random_state=random_state)

    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay, phasey_unwrapped, muy, cy = svd_estimate_shift_morereturns(ang_qu, M)
    deltax, phasex_unwrapped, mux, cx = svd_estimate_shift_morereturns(ang_qv, N)

    variables = {
        "img_0_processed": img_0_processed,
        "img_1_processed": img_1_processed,
        "q": q,
        "Q": Q,
        "qu": qu,
        "s": s,
        "qv": qv,
        "ang_qu": ang_qu,
        "ang_qv": ang_qv,
        "phasex_unwrapped": phasex_unwrapped,
        "phasey_unwrapped": phasey_unwrapped,
        "muy": muy,
        "mux": mux,
        "cy": cy,
        "cx": cx
    }

    return deltax, deltay, variables


def error_finder(img_0, img_1, estimated_value_y=None, estimated_value_x=None, max_variance_x=0.03, max_variance_y=0.03,
                 counter_limit=100, counter_only=False, random_mode=True, method='Stone_et_al_2001'):
    counter = 0
    deltax = 0
    deltay = 0
    variables = {}
    error_counter = np.zeros(2)
    absolute_varianxe_x = abs(max_variance_x*estimated_value_x)
    absolute_variance_y = abs(max_variance_y*estimated_value_y)
    while counter_limit > counter:
        counter = counter + 1

        if random_mode is True:
            deltax, deltay, variables = estimate_shift(img_0, img_1, method)
        else:
            deltax, deltay, variables = estimate_shift(img_0, img_1, method, counter)

        if (deltay > estimated_value_y + absolute_variance_y) or (deltay < estimated_value_y - absolute_variance_y):
            error_counter[0] = error_counter[0] + 1
            if counter_only is False:
                return deltax, deltay, variables, True

        if (deltax > estimated_value_x + absolute_varianxe_x) or (deltax < estimated_value_x - absolute_varianxe_x):
            error_counter[1] = error_counter[1] + 1
            if counter_only is False:
                return deltax, deltay, variables, True

    return deltax, deltay, variables, False



def showDebuggingImages(img_0, img_1, variables, centerText = ""):
    xx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
    xy_values = variables["mux"] * xx_values + variables["phasex_unwrapped"][0]

    yx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
    yy_values = variables["muy"] * yx_values + variables["phasey_unwrapped"][0]

    fig, ax = plt.subplots(2, 5, figsize=(12, 8))  # 3 linhas, 3 colunas

    ax[0, 0].imshow(img_0, cmap='gray')
    ax[0, 0].set_title(f'Original 0')

    ax[1, 0].imshow(img_1, cmap='gray')
    ax[1, 0].set_title(f'Original 1')

    ax[0, 1].imshow(np.log10(abs(variables["img_0_processed"])), cmap='gray')
    ax[0, 1].set_title(f'FFT 0')

    ax[1, 1].imshow(np.log10(abs(variables["img_1_processed"])), cmap='gray')
    ax[1, 1].set_title(f'FFT 1')

    x = range(0, variables["qu"].size)
    ax[0, 2].plot(variables["qu"])
    ax[0, 2].scatter(x, variables["qu"], s=5, color="red")
    ax[0, 2].set_title(f'Angulo qu')

    x = range(0, variables["qv"].size)
    ax[1, 2].scatter(x, variables["qv"], s=5, color="red")
    ax[1, 2].plot(variables["qv"])
    ax[1, 2].set_title(f'Angulo qv')

    x = range(0, variables["ang_qu"].size)
    ax[0, 3].scatter(x, variables["ang_qu"], s=5, color="red")
    ax[0, 3].plot(variables["ang_qu"])
    ax[0, 3].set_title(f'Angulo ang_qu')

    x = range(0, variables["ang_qv"].size)
    ax[1, 3].scatter(x, variables["ang_qv"], s=5, color="red")
    ax[1, 3].plot(variables["ang_qv"])
    ax[1, 3].set_title(f'Angulo ang_qv')

    ax[0, 4].plot(variables["phasey_unwrapped"])
    ax[0, 4].plot(yx_values, yy_values)
    ax[0, 4].set_title(f'Angulo Y')

    ax[1, 4].plot(variables["phasex_unwrapped"])
    ax[1, 4].plot(xx_values, xy_values)
    ax[1, 4].set_title(f'Angulo X')

    text_obj = fig.text(0.5, 0.94, centerText, ha='center', va='center', fontsize=12, transform=fig.transFigure)

    return plt

import numpy as np

def add_noise(img, noise_type='gaussian', noise_level=0.1):
    """
    Adiciona ruído à imagem.

    Args:
        img (ndarray): Imagem de entrada.
        noise_type (str): Tipo de ruído a ser adicionado. Opções disponíveis: 'gaussian', 'salt_and_pepper', 'speckle'.
        noise_level (float): Nível de ruído, padrão é 0.1.

    Returns:
        ndarray: Imagem com ruído adicionado.
    """
    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(img, noise_level)
    elif noise_type == 'salt_and_pepper':
        noisy_img = add_salt_and_pepper_noise(img, noise_level)
    elif noise_type == 'speckle':
        noisy_img = add_speckle_noise(img, noise_level)
    elif noise_type == "blur":
        noisy_img = add_blur(img, noise_level)
    else:
        raise ValueError("Tipo de ruído não suportado. Escolha entre 'gaussian', 'salt_and_pepper' ou 'speckle'.")

    return noisy_img

def add_gaussian_noise(img, noise_level=0.1):
    """
    Adiciona ruído gaussiano à imagem.

    Args:
        img (ndarray): Imagem de entrada.
        noise_level (float): Nível de ruído, padrão é 0.1.

    Returns:
        ndarray: Imagem com ruído gaussiano adicionado.
    """
    # Calcula a média e o desvio padrão da imagem
    mean = np.mean(img)
    std_dev = np.std(img)

    # Calcula o ruído gaussiano
    noise = np.random.normal(mean, std_dev * noise_level, img.shape)

    # Adiciona o ruído à imagem
    noisy_img = img + noise

    # Garante que os valores da imagem resultante estejam no intervalo [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)

    return noisy_img.astype(np.uint8)  # Converte para o tipo de dados uint8

def add_salt_and_pepper_noise(img, noise_level=0.05):
    """
    Adiciona ruído sal e pimenta à imagem.

    Args:
        img (ndarray): Imagem de entrada.
        noise_level (float): Nível de ruído, padrão é 0.05.

    Returns:
        ndarray: Imagem com ruído sal e pimenta adicionado.
    """
    noisy_img = np.copy(img)
    salt_and_pepper_pixels = np.random.rand(*img.shape)

    noisy_img[salt_and_pepper_pixels < noise_level / 2] = 0
    noisy_img[salt_and_pepper_pixels > 1 - noise_level / 2] = 255

    return noisy_img

def add_speckle_noise(img, noise_level=0.1):
    """
    Adiciona ruído speckle à imagem.

    Args:
        img (ndarray): Imagem de entrada.
        noise_level (float): Nível de ruído, padrão é 0.1.

    Returns:
        ndarray: Imagem com ruído speckle adicionado.
    """
    noise = np.random.normal(0, 1, img.shape)
    noisy_img = img + img * noise_level * noise

    # Garante que os valores da imagem resultante estejam no intervalo [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)

    return noisy_img.astype(np.uint8)  # Converte para o tipo de dados uint8

# Exemplo de uso:
# img = img_gen(dx=10, dy=20, width=512, height=600, zoom=1, angle=30)
# noisy_img = add_noise(img, noise_type='gaussian', noise_level=0.1)

def add_blur(img, kernel_size=3):
    """
    Adiciona um efeito de desfoque (blur) à imagem usando um filtro de média.

    Args:
        img (ndarray): Imagem de entrada.
        kernel_size (int): Tamanho do kernel do filtro de média, padrão é 3.

    Returns:
        ndarray: Imagem com o efeito de desfoque aplicado.
    """
    # Aplica o filtro de média à imagem
    blurred_img = uniform_filter(img.astype(float), size=kernel_size)

    return blurred_img  # Converte para o tipo de dados uint8
