from matplotlib.animation import FFMpegWriter
from scipy.sparse.linalg import svds

from visual_encoder.dsp_utils import image_preprocessing, simplified_crosspower_spectrun
from visual_encoder.svd_decomposition import svd_estimate_shift_morereturns
from visual_encoder.utils import get_imgs
import os
import numpy as np
import matplotlib.pyplot as plt

n_beg = 1
n_images = 233
data_root = "C:/Users/dsant/OneDrive/Ambiente de Trabalho/experimento/fotos/"
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\dsant\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.1.1-full_build\bin\ffmpeg.exe'
fig, ax = plt.subplots(2, 4, figsize=(12, 8))  # 3 linhas, 3 colunas
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
video_number = 1
output_directory = "animacoes"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_path = os.path.join(output_directory, f'animacao_{video_number:02d}.mp4')
total_deltax = 0
total_deltay = 0
text_obj = None

with writer.saving(fig, output_path, 100):
    imgs = get_imgs(n_images, data_root)
    for i in range(n_beg, n_images - n_beg + 1):
        img_0 = imgs[i - 1]
        img_1 = imgs[i]

        M, N = img_0.shape

        j = i - n_beg + 1

        img_0_processed = image_preprocessing(img_0)
        img_1_processed = image_preprocessing(img_1)

        # Limpar os eixos antes de adicionar novas imagens
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

        # Calcular o crosspower spectrum
        q, Q = simplified_crosspower_spectrun(img_0_processed, img_1_processed)
        qu, s, qv = svds(Q, k=1)
        ang_qu = np.angle(qu[:, 0])
        ang_qv = np.angle(qv[0, :])

        # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
        deltay, phasey, muy, cy = svd_estimate_shift_morereturns(ang_qu, M)
        deltax, phasex, mux, cx = svd_estimate_shift_morereturns(ang_qv, N)

        total_deltax = deltax + total_deltax
        total_deltay = deltay + total_deltay

        # Exibir os ângulos ang_qu e ang_qv em subplots adicionais à direita
        ax[0, 2].plot(ang_qu)
        ax[0, 2].set_title(f'Angulo ang_qu {j}')

        ax[1, 2].plot(ang_qv)
        ax[1, 2].set_title(f'Angulo ang_qv {j}')

        xx_values = np.linspace(0, 100, 1000)  # Valores arbitrários para x
        xy_values = mux * xx_values + phasex[0]

        yx_values = np.linspace(0, 100, 1000)  # Valores arbitrários para x
        yy_values = muy * yx_values + phasey[0]

        ax[0, 3].plot(phasey)
        ax[0, 3].plot(yx_values, yy_values)
        ax[0, 3].set_title(f'Angulo Y {j}')

        ax[1, 3].plot(phasex)
        ax[1, 3].plot(xx_values, xy_values)
        ax[1, 3].set_title(f'Angulo X {j}')

        # Adicionar texto acima de tudo usando fig.text()
        text_str = f'deltax: {deltax:.2f}, deltay: {deltay:.2f}\n'
        text_str += f'total_deltax: {total_deltax:.2f}, total_deltay: {total_deltay:.2f}'
        #text_str += f'cx: {cx:.2f}, cy: {cy:.2f}'

        # Remover texto antigo se existir
        if text_obj:
            text_obj.remove()

        text_obj = fig.text(0.5, 0.95, text_str, ha='center', va='center', fontsize=12, transform=fig.transFigure)

        # Adicionar uma pausa entre os frames (ajuste conforme necessário)
        plt.pause(0.1)

        # Gravar o frame atual
        writer.grab_frame()

print(f"Vídeo gerado em: {output_path}")
