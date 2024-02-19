from matplotlib.animation import FFMpegWriter
from scipy.sparse.linalg import svds

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
plt.rcParams[
    'animation.ffmpeg_path'] = r'C:\Users\dsant\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1.1-full_build\bin\ffmpeg.exe'

writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Número do arquivo de vídeo formatado
video_number = 1

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
print(img_0)
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

    print(deltay)
    print(counter)

    if (deltay > estimated_value + max_variance):
        break
    elif (deltay < estimated_value - max_variance):
        break
    # break


