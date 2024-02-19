from matplotlib.animation import FFMpegWriter
from scipy.sparse.linalg import svds

from visual_encoder.dsp_utils import image_preprocessing, normalize_product
from visual_encoder.svd_decomposition import svd_estimate_shift_morereturns
from visual_encoder.utils import get_imgs, get_imgs_cutted
import os
import numpy as np
import matplotlib.pyplot as plt
import csv


image_path = "C:/Users/dsant/OneDrive/Ambiente de Trabalho/experimento/textura/teste.jpg"



with open('data.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for pixel_interval in range(20, 1200):
        if (360//(pixel_interval+1) != 360//(pixel_interval)) or pixel_interval == 360:
            imgs = get_imgs_cutted(image_path, pixel_interval)
            total_deltax = 0
            print(pixel_interval)
            for i in range(1, imgs.shape[0]):
                img_0 = imgs[i - 1]
                img_1 = imgs[i]

                M, N = img_0.shape

                img_0_processed = image_preprocessing(img_0)
                img_1_processed = image_preprocessing(img_1)

                Q = normalize_product((img_0_processed, img_1_processed)
                qu, s, qv = svds(Q, k=1)
                ang_qu = np.angle(qu[:, 0])
                ang_qv = np.angle(qv[0, :])

                # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
                deltay, phasey, muy, cy = svd_estimate_shift_morereturns(ang_qu, M)
                deltax, phasex, mux, cx = svd_estimate_shift_morereturns(ang_qv, N)

                total_deltax = deltax + total_deltax
                total_deltay = deltay + total_deltay
                print("   {}".format(i))

            writer.writerow([total_deltax,total_deltay,pixel_interval])


