from matplotlib.animation import FFMpegWriter
from scipy.sparse.linalg import svds

from visual_encoder.deubugging_tools import apply_blackman_harris_window
from visual_encoder.dsp_utils import image_preprocessing, simplified_crosspower_spectrun, filter_array_by_maxmin
from visual_encoder.svd_decomposition import svd_estimate_shift_morereturns
from visual_encoder.utils import get_img

import os
import numpy as np
import matplotlib.pyplot as plt

n_beg = 153
n_end = 154
n_images = 233

data_root = "C:/Users/dsant/OneDrive/Ambiente de Trabalho/experimento/fotos/"
plt.rcParams[
    'animation.ffmpeg_path'] = r'C:\Users\dsant\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1.1-full_build\bin\ffmpeg.exe'

fig, ax = plt.subplots(1, 2, figsize=(12, 8))  # 3 linhas, 3 colunas

total_deltax = 0
total_deltay = 0

text_obj = None

j = 153

estimated_value = 7
max_variance = 0.5
counter = 0

deltay = estimated_value

img_0_0 = get_img(153, data_root)
img_1_0 = get_img(154, data_root)
#img_0_0 = apply_blackman_harris_window(img_0_0)
#img_1_0 = apply_blackman_harris_window(img_1_0)
M, N = img_0_0.shape




problems_without_median = 0
problems_with_median = 0

while True:
    counter = counter + 1

    img_0_processed = image_preprocessing(img_0_0)
    img_1_processed = image_preprocessing(img_1_0)

    q, Q = simplified_crosspower_spectrun(img_0_processed, img_1_processed)
    Q2 = Q - np.average(Q)

    qu, s, qv = svds(Q, k=1, random_state=counter)
    qu2, s2, qv2 = svds(Q2, k=1, random_state=counter)

    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    ang_qu2 = np.angle(qu2[:, 0])
    ang_qv2 = np.angle(qv2[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:

    #ang_qu = filter_array(ang_qu, 0.5)

    #ang_qu = filter_array_by_maxmin(ang_qu)

    deltay, phasey, muy, cy = svd_estimate_shift_morereturns(ang_qu, M)
    deltax, phasex, mux, cx = svd_estimate_shift_morereturns(ang_qv, N)

    deltay2, phasey2, muy2, cy2 = svd_estimate_shift_morereturns(ang_qu2, M)
    deltax2, phasex2, mux2, cx2 = svd_estimate_shift_morereturns(ang_qv2, N)

    if (deltay > estimated_value + max_variance):
        problems_without_median = problems_without_median + 1
        print(counter)

    elif (deltay < estimated_value - max_variance):
        problems_without_median = problems_without_median + 1
        print(counter)




    # print(deltay)
    # print(deltay2)
    # print("Problemas encontrados sem a média: {}".format(problems_without_median))
    # print("Problemas encontrados com a média: {}".format(problems_with_median))

    #break

xx_values = np.linspace(0, 100, 1000)  # Valores arbitrários para x
xy_values = mux * xx_values + phasex[0]

yx_values = np.linspace(0, 100, 1000)  # Valores arbitrários para x
yy_values = muy * yx_values + phasey[0]

for a in ax.flatten():
    a.clear()

# Exibir os ângulos ang_qu e ang_qv em subplots adicionais à direita
x = range(0, ang_qu.size, 1)
ax[0].scatter(x, ang_qu)
ax[0].set_title(f'Angulo ang_qu {j}')

ax[1].plot(phasey)
ax[1].plot(yx_values, yy_values)
ax[1].set_title(f'Angulo Y {j}')

plt.show()
