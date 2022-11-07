import matplotlib.pyplot as plt
from visual_encoder.trajectory_estimators import *
from visual_encoder.svd_decomposition import *
from PIL import Image


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


data_root = "/home/tekalid/repos/AUSPEX/test/visual_encoder/data/images/"
myImage = Image.open(data_root + "rgb_example.jpeg");
img_rgb = np.array(myImage)

# Conversão RGB -> Grey
base_img = img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114

# # Geração de imagens com ruído:
gauss_noise = 5
print("Gerando os pontos artificialmente...")
img_list, xshifts, yshifts, coord = generate_artifical_shifts(base_img, x0=200, y0=200, gaussian_noise_db=gauss_noise)
print("Gerado os pontos artificiais.")

# Cálculo da trajetória:
computed_coords_pc = compute_total_trajectory(img_list, x0=200, y0=200, method='pc')
computed_coords_svd = compute_total_trajectory(img_list, x0=200, y0=200, method='svd', window_type='Blackman-Harris')

plt.imshow(base_img, cmap='gray')
plt.plot(coord[:, 0], coord[:, 1], '-o', color='r', label='Expected')
plt.plot(computed_coords_svd[:, 0], computed_coords_svd[:, 1], '-o', color='b', label='SVD')
plt.plot(computed_coords_pc[:, 0], computed_coords_pc[:, 1], '-o', color='g', label='PC')
plt.legend()

# # Debugando o SVD:
# plt.figure()
# shot = 25
# f = img_list[shot]
# n_jumps = 1
# g = img_list[shot + n_jumps]
# ideal_deltax = 3 * n_jumps
# ideal_deltay = 2 * n_jumps
# plt.subplot(2, 3, 1)
# plt.imshow(f, cmap='gray')
# plt.subplot(2, 3, 2)
# plt.imshow(g, cmap='gray')
# #
# # q, Q, N = crosspower_spectrum2(f, g, method='Stone_et_al_2001')
# qa, s, qb = svds(Q, k=1)
# # Ângulo de qa e qb:
# ang_qb = np.angle(qb[0, :])
# ang_qa = np.angle(qa[:, 0])
#
# diff_q = np.diff(ang_qb)
# diff_mean = np.mean(diff_q)
# diff_std = np.std(diff_q)
#
# std_factor = 3
# # diff_q2 = moving_average(diff_q, 15)
# idx = [(np.mean(diff_q) - np.std(diff_q) * std_factor) <= d <= (np.mean(diff_q) + np.std(diff_q) * std_factor) for d
#        in diff_q]  # ìndices para serem mantidos
#
# ###
# freq_y = np.fft.fftfreq(f.shape[0])
# freq_x = np.fft.fftfreq(f.shape[1])
# filt_freq_y = freq_y[int(f.shape[0] // 2 - N): int(f.shape[0] // 2 + N)]
# filt_freq_x = freq_x[int(f.shape[1] // 2 - N): int(f.shape[1] // 2 + N):]
# filt_freq_x2 = filt_freq_x[:-1][idx]
# filt_freq_y2 = filt_freq_y[:-1][idx]
#
# filt_diff = diff_q[idx]
# # filt_ang_qb = ang_qb[:-1][idx]
#
# plt.subplot(2, 3, 1)
# plt.stem(ang_qb)
# plt.subplot(2, 3, 2)
# plt.stem(ang_qa)
#
#
# plt.title("$q_b$")
# plt.subplot(2, 3, 4)
# plt.stem(diff_q)
# plt.title("Diferença de $q_b$")
#
# plt.subplot(2, 3, 5)
# plt.stem(filt_diff)
# plt.title(f"Diferença da Filtragem de $q_b$ para pontos até cte$\cdot\sigma$ onde cte={std_factor}")
#
# ############
# delta = 2 * np.pi / freq_x.size
# tanx = -np.mean(filt_diff) / delta
# deltax = tanx
#
# plt.figure()
# plt.subplot(2, 3, 1)
# plt.imshow(f, cmap='gray')
# plt.subplot(2, 3, 2)
# plt.imshow(g, cmap='gray')
# plt.subplot(2, 3, 3)
# plt.stem(ang_qa)
# plt.title("$ângulo de q_a$")
# plt.subplot(2, 3, 6)
# plt.stem(ang_qb)
# plt.title("$ângulo de q_b$")
#
# plt.subplot(2, 3, 4)
# plt.imshow(np.angle(Q), cmap='gray')
# plt.title("Matriz Q")
#
# plt.subplot(2, 3, 5)
# plt.imshow(np.angle(qa @ np.diag(s) @ qb), cmap='gray')
# plt.title("Aproximação SVD de Q")
#
# plt.suptitle(f"Ruído Gaussiano Artificial de {gauss_noise} dB")
