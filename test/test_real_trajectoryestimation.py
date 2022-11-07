import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from visual_encoder.svd_decomposition import *
from visual_encoder.phase_correlation import *
from visual_encoder.trajectory_estimators import *
from scipy import signal
from PIL import Image
import time

# matplotlib.use('TkAgg')

data_root = "/home/tekalid/repos/AUSPEX/test/visual_encoder/data/images/images_senoide/"
# myImage = Image.open(data_root + "rgb_example.jpeg");
# img_rgb = np.array(myImage)
img_list = list()
rgb2gray = lambda img_rgb: img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114

nmax = 150
for i in range(1, nmax):
    myImage = Image.open(data_root + f"image{i:02d}.jpg")
    img_rgb = np.array(myImage)
    img_gray = rgb2gray(img_rgb)
    img_list.append(img_gray)

img_list = img_list[:nmax]

print("Começando a estimação por SVD...")
t0 = time.time()
computed_coords_svd = compute_total_trajectory(img_list, x0=0, y0=0, method='svd', window_type='Blackman-Harris')
print(f"Tempo SVD: {time.time() - t0}")
print("Fim a estimação por SVD.")

print("Começando a estimação por PC...")
t0 = time.time()
computed_coords_pc = compute_total_trajectory(img_list, x0=0, y0=0, method='pc')
print(f"Tempo PC: {time.time() - t0}")
print("Começando a estimação por PC.")

# print("Salvando os vetores...")
# np.save(computed_coords_svd, 'computed_coords_svd.npy')
# np.save(computed_coords_pc, 'computed_coords_pc.npy')


# Parâmetros de geraçã ode vídeo:
video_title = "video/SVD_versus_PC_senoide_completa_v03"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure(figsize=(14, 9))
generate_video = False

print("Início do loop de geração de vídeo...")
if generate_video == True:
    # with writer.saving(fig, result_foldername + "/" + video_title + ".mp4", dpi=300):
    with writer.saving(fig, video_title + ".mp4", dpi=300):
        for shot in range(1, len(img_list)):
            plt.suptitle(f"Shot = {shot}")
            plt.subplot(2, 1, 1)
            plt.imshow(img_list[shot][::-1, :], cmap='gray',
                       extent=[computed_coords_svd[shot, 0],  # Esquerda X
                               computed_coords_svd[shot, 0] + img_list[0].shape[1],  # Direita X
                               computed_coords_svd[shot, 1],  # Topo Y
                               computed_coords_svd[shot, 1] + img_list[0].shape[0]]  # Base Y
                       )
            plt.xlim([computed_coords_svd[shot, 0], computed_coords_svd[shot, 0] + img_list[0].shape[1]])
            plt.ylim([computed_coords_svd[shot, 1], computed_coords_svd[shot, 1] + img_list[0].shape[0]])
            offset_y = img_list[0].shape[1] / 2
            offset_x = img_list[1].shape[1] / 2

            plt.plot(computed_coords_svd[:shot + 1][:, 0] + offset_x, computed_coords_svd[:shot + 1][:, 1] + offset_y,
                     ':o',
                     color='r', label='SVD')
            plt.plot(computed_coords_pc[:shot + 1][:, 0] + offset_x, computed_coords_svd[:shot + 1][:, 1] + offset_y,
                     ':o',
                     color='g', label='PC')
            plt.legend()

            print(f"frame = {shot + 1}")

            plt.subplot(2, 1, 2)
            plt.plot(computed_coords_svd[:shot + 1][:, 0] + offset_x, computed_coords_svd[:shot + 1][:, 1] + offset_y,
                     ':o',
                     color='r', label='SVD')
            plt.plot(computed_coords_pc[:shot + 1][:, 0] + offset_x, computed_coords_pc[:shot + 1][:, 1] + offset_y,
                     ':o',
                     color='g', label='PC')
            # plt.xlim([computed_coords[:, 0].min(), computed_coords[:, 0].max()])
            # plt.ylim([computed_coords[:, 1].min(), computed_coords[:, 1].max()])

            writer.grab_frame()
            plt.clf()
    print("Fim do loop de geração de vídeo.")
else:
    # CASO QUE ESTIMOU INCORRETAMENTE:
    shot = 69
    f = img_list[shot]
    g = img_list[shot + 1]

    plt.figure()
    plt.suptitle("Estimativa Correta")
    plt.plot(computed_coords_svd[:shot, 0], computed_coords_svd[:shot, 1], ':o',
             color='r', label='SVD')
    plt.plot(computed_coords_pc[:shot, 0], computed_coords_pc[:shot, 1],
             ':o',
             color='g', label='PC')
    plt.title("Trajetória total.")
    plt.xlabel("Distância em pixels")
    plt.ylabel("Distância em pixels")

    ### ----------------------------------------------------------------------------------------------------------- ###

    img_beg, img_end = apply_window(f, g, 'Blackman-Harris')
    window1dy_bm = signal.windows.blackman(f.shape[0])
    window1dy_bh = signal.windows.blackmanharris(f.shape[0])
    plt.figure()
    plt.plot(window1dy_bm, label='Blackman')
    plt.plot(window1dy_bh, label='Blackman-Harris')
    plt.legend()

    q_blackman, Q_blackman = crosspower_spectrum(img_beg, img_end)
    q_square, Q_square = crosspower_spectrum(f, g)
    f_sq = f[f.shape[0]//2 - 140: f.shape[0]//2 + 140, f.shape[1]//2 - 140: f.shape[1]//2 + 140]
    g_sq = g[g.shape[0]//2 - 140: g.shape[0]//2 + 140, g.shape[1]//2 - 140: g.shape[1]//2 + 140]
    q_square2, Q_square2 = crosspower_spectrum(f_sq, g_sq)
    qa, s, qb = svds(Q_blackman, k=1)
    ang_qa = np.angle(qa[:, 0])
    ang_qb = np.angle(qb[0, :])

    plt.figure()
    plt.plot(np.angle(qa), 'o')
    #

    plt.figure()
    plt.suptitle("Caso sem sucesso.")
    plt.subplot(2, 5, 1)
    plt.imshow(f, cmap='gray')
    plt.title(f"$f[x,y]$ ou Shot {shot}")

    plt.subplot(2, 5, 2)
    plt.imshow(g, cmap='gray')
    plt.title(f"$g[x,y]$ ou Shot {shot+1}")

    plt.subplot(2, 5, 3)
    plt.imshow(np.angle(Q_square), cmap='gray')
    plt.title("Ângulo de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Retângular")

    plt.subplot(2, 5, 4)
    plt.imshow(np.angle(Q_square2), cmap='gray')
    plt.title("Ângulo de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Retângular Menor")

    plt.subplot(2, 5, 5)
    plt.imshow(np.angle(Q_blackman), cmap='gray')
    plt.title("Ângulo de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Blackman-Harris")

    plt.subplot(2, 5, 6)
    plt.imshow(np.angle(qa @ np.diag(s) @ qb), cmap='gray')
    plt.title("Ângulo do resultado da decomposição SVD.")

    plt.subplot(2, 5, 7)
    plt.imshow(img_beg, cmap='gray')
    plt.title(f"Shot {shot} janelado")

    plt.subplot(2, 5, 8)
    plt.imshow(img_end, cmap='gray')
    plt.title(f"Shot {shot+1} janelado")

    plt.subplot(2, 5, 9)
    plt.imshow(f_sq, cmap='gray')
    plt.title(f"Shot {shot} janelado")

    plt.subplot(2, 5, 10)
    plt.imshow(g_sq, cmap='gray')
    plt.title(f"Shot {shot+1} janelado")

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(np.angle(qa @ np.diag(s) @ qb), cmap='gray')
    # plt.title("Ângulo do resultado da decomposição SVD com janelamento Blackman-Harris")


    # Com janelamento tipo Blackman-Harris
    deltax_pc, deltay_pc = pc_method(img_beg, img_end)
    deltax_svd, deltay_svd = svd_method(img_beg, img_end)
    # Com janela retangular:
    deltax_pc, deltay_pc = pc_method(f, g)
    deltax_svd, deltay_svd = svd_method(f, g)

    plt.subplot(2, 5, 8)
    plt.imshow(q_blackman, cmap='gray')
    plt.title(f"IFFT de "+ r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Blackman-Harris")

    plt.subplot(2, 5, 9)
    plt.imshow(q_square, cmap='gray')
    plt.title(f"Shot {shot + 1} janelado")
    plt.title(f"IFFT de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Retangular")



    # CASO QUE ESTIMOU CORRETAMENTE:
    shot = 75
    f = img_list[shot]
    g = img_list[shot + 1]

    plt.figure()
    plt.suptitle("Estimativa Correta")
    plt.plot(computed_coords_svd[:shot, 0], computed_coords_svd[:shot, 1], ':o',
             color='r', label='SVD')
    plt.plot(computed_coords_pc[:shot, 0], computed_coords_pc[:shot, 1],
             ':o',
             color='g', label='PC')
    plt.title("Trajetória total.")
    plt.xlabel("Distância em pixels")
    plt.ylabel("Distância em pixels")

    ### ----------------------------------------------------------------------------------------------------------- ###

    img_beg, img_end = apply_window(f, g, 'Blackman-Harris')

    q_blackman, Q_blackman = crosspower_spectrum(img_beg, img_end)
    q_square2, Q_square2 = crosspower_spectrum(img_beg, img_end)
    q_square, Q_square = crosspower_spectrum(f, g)
    qa, s, qb = svds(Q_blackman, k=1)
    ang_qa = np.angle(qa[:, 0])
    ang_qb = np.angle(qb[0, :])

    plt.figure()
    plt.suptitle("Caso com sucesso.")
    plt.subplot(2, 5, 1)
    plt.imshow(f, cmap='gray')
    plt.title(f"$f[x,y]$ ou Shot {shot}")

    plt.subplot(2, 5, 2)
    plt.imshow(g, cmap='gray')
    plt.title(f"$g[x,y]$ ou Shot {shot + 1}")

    plt.subplot(2, 5, 3)
    plt.imshow(np.angle(Q_square), cmap='gray')
    plt.title("Ângulo de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Retângular")

    plt.subplot(2, 5, 4)
    plt.imshow(np.angle(Q_blackman), cmap='gray')
    plt.title("Ângulo de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Blackman-Harris")

    plt.subplot(2, 5, 5)
    plt.imshow(np.angle(qa @ np.diag(s) @ qb), cmap='gray')
    plt.title("Ângulo do resultado da decomposição SVD.")

    plt.subplot(2, 5, 6)
    plt.imshow(img_beg, cmap='gray')
    plt.title(f"Shot {shot} janelado")

    plt.subplot(2, 5, 7)
    plt.imshow(img_end, cmap='gray')
    plt.title(f"Shot {shot + 1} janelado")

    plt.subplot(2, 5, 8)
    plt.imshow(q_blackman, cmap='gray')
    plt.title(f"IFFT de "+ r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Blackman-Harris")

    plt.subplot(2, 5, 9)
    plt.imshow(q_square, cmap='gray')
    plt.title(f"Shot {shot + 1} janelado")
    plt.title(f"IFFT de " + r'$\frac{F[x,y] \circ G[x,y]}{|F[x,y] * G[x,y]^*|}$' + "\n Janelamento Retangular")


    # Com janelamento tipo Blackman-Harris
    deltax_pc, deltay_pc = pc_method(img_beg, img_end)
    deltax_svd, deltay_svd = svd_method(img_beg, img_end)
    # Com janela retangular:
    deltax_pc, deltay_pc = pc_method(f, g)
    deltax_svd, deltay_svd = svd_method(f, g)

