from matplotlib.animation import FFMpegWriter
from numpy import meshgrid
from scipy.sparse.linalg import svds

from visual_encoder.deubugging_tools import apply_blackman_harris_window
from visual_encoder.dsp_utils import image_preprocessing, simplified_crosspower_spectrun
from visual_encoder.svd_decomposition import svd_estimate_shift_morereturns
from visual_encoder.utils import get_img

import os
import numpy as np
import matplotlib.pyplot as plt

n_beg = 153
n_end = 154
n_images = 233

data_root = "C:/Users/dsant/OneDrive/Ambiente de Trabalho/experimento/fotos/"
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

total_deltax = 0
total_deltay = 0

text_obj = None

j = 153

estimated_value = 7
max_variance = 0.5
counter = 0

deltay = estimated_value

img_0 = get_img(153, data_root)
img_1 = get_img(154, data_root)

img_0 = apply_blackman_harris_window(img_0)
img_1 = apply_blackman_harris_window(img_1)

M, N = img_0.shape

while True:
    counter = counter + 1

    img_0_processed = image_preprocessing(img_0)
    img_1_processed = image_preprocessing(img_1)

    q, Q = simplified_crosspower_spectrun(img_0_processed, img_1_processed)
    qu, s, qv = svds(Q, k=1)


    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])


    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay, phasey, muy, cy = svd_estimate_shift_morereturns(ang_qu, M)
    deltax, phasex, mux, cx = svd_estimate_shift_morereturns(ang_qv, N)

    total_deltax = deltax + total_deltax
    total_deltay = deltay + total_deltay

    if (deltay > estimated_value + max_variance):
        break
    elif (deltay < estimated_value - max_variance):
        break
    break

fig, ax = plt.subplots(2, 5, figsize=(12, 8))  # 3 linhas, 3 colunas

xx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
xy_values = mux * xx_values + phasex[0]

yx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
yy_values = muy * yx_values + phasey[0]

for a in ax.flatten():
    a.clear()

# Exibir as imagens processadas em uma grade de subplots
ax[0, 0].imshow(img_0, cmap='gray')
ax[0, 0].set_title(f'Original {j}')

ax[1, 0].imshow(img_1, cmap='gray')
ax[1, 0].set_title(f'Original {j + 1}')

ax[0, 1].imshow(np.log10(abs(img_0_processed)), cmap='gray')
ax[0, 1].set_title(f'FFT {j}')

ax[1, 1].imshow(np.log10(abs(img_1_processed)), cmap='gray')
ax[1, 1].set_title(f'FFT {j + 1}')

# Exibir os ângulos ang_qu e ang_qv em subplots adicionais à direita

x = range(0, qu.size)
ax[0, 2].plot(qu)
ax[0, 2].scatter(x,qu, s = 5,color="red")
ax[0, 2].set_title(f'Angulo qu')

x = range(0, qv.size)
ax[1, 2].scatter(x,qv, s = 5,color="red")
ax[1, 2].plot(qv)
ax[1, 2].set_title(f'Angulo qv')


#x = range(0, ang_qu.size, 1)
#ax[0, 3].scatter(x,ang_qu)
x = range(0, ang_qu.size)
ax[0, 3].scatter(x,ang_qu, s = 5,color="red")
ax[0, 3].plot(ang_qu)
ax[0, 3].set_title(f'Angulo qu')

x = range(0, ang_qv.size)
ax[1, 3].scatter(x,ang_qv, s = 5,color="red")
ax[1, 3].plot(ang_qv)
ax[1, 3].set_title(f'Angulo qv')

ax[0, 4].plot(phasey)
ax[0, 4].plot(yx_values, yy_values)
ax[0, 4].set_title(f'Angulo Y {j}')

ax[1, 4].plot(phasex)
ax[1, 4].plot(xx_values, xy_values)
ax[1, 4].set_title(f'Angulo X {j}')

# Adicionar texto acima de tudo usando fig.text()
text_str = f'deltax: {deltax:.2f}, deltay: {deltay:.2f}\n'
text_str += f'total_deltax: {total_deltax:.2f}, total_deltay: {total_deltay:.2f}'
# text_str += f'cx: {cx:.2f}, cy: {cy:.2f}'

# Remover texto antigo se existir
if text_obj:
    text_obj.remove()


text_obj = fig.text(0.5, 0.95, text_str, ha='center', va='center', fontsize=12, transform=fig.transFigure)

plt.show()
