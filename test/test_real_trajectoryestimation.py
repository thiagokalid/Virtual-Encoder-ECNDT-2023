import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from visual_encoder.svd_decomposition import *
from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
from scipy import signal
from PIL import Image
import time

# matplotlib.use('TkAgg')

data_root = "/home/tekalid/repos/AUSPEX/test/visual_encoder/data/images/images_senoide/"
# myImage = Image.open(data_root + "rgb_example.jpeg");
# img_rgb = np.array(myImage)
img_list = list()
rgb2gray = lambda img_rgb: img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114

nmax = 100
for i in range(1, nmax):
    myImage = Image.open(data_root + f"image{i:02d}.jpg")
    img_rgb = np.array(myImage)
    img_gray = rgb2gray(img_rgb)
    img_list.append(img_gray)

img_list = img_list[:nmax]

print("Começando a estimação por SVD...")
t0 = time.time()
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001")
svd_traj = TrajectoryParams(svd_param)
svd_traj.compute_total_trajectory(img_list)
print(f"Tempo SVD: {time.time() - t0}")
print("Fim a estimação por SVD.")

print("Começando a estimação por PC...")
pc_param = DisplacementParams(method="pc", spatial_window=None)
pc_traj = TrajectoryParams(pc_param)
t0 = time.time()
pc_traj.compute_total_trajectory(img_list)
print(f"Tempo PC: {time.time() - t0}")
print("Começando a estimação por PC.")

# Parâmetros de geração de vídeo:
results_path = "/home/tekalid/repos/Virtual-Encoder-ECNDT-2023/results/video/"
videotitle = "SVD_vs_PC_v02"
metadata = dict(title=results_path + videotitle, artist='Matplotlib',
                comment='Movie support!')

writer = FFMpegWriter(fps=5, metadata=metadata)

fig = plt.figure(figsize=(14, 9))
generate_video = False
# Video-type 1 refeers to the original image and trace video
# Video-type 2 refeers to the debug video, where it contains not only the image and trace, but also additional info.
video_type = 2

print("Início do loop de geração de vídeo...")
if generate_video == True:
    # with writer.saving(fig, result_foldername + "/" + video_title + ".mp4", dpi=300):
    with writer.saving(fig, results_path + videotitle + ".mp4", dpi=300):
        for shot in range(1, len(img_list)):
            if video_type == 1:
                plt.suptitle(f"Shot = {shot}")
                plt.subplot(2, 1, 1)
                plt.imshow(img_list[shot][::-1, :], cmap='gray',
                           extent=[svd_traj.get_coords(shot)[0],  # Esquerda X
                                   svd_traj.get_coords(shot)[0] + img_list[0].shape[1],  # Direita X
                                   svd_traj.get_coords(shot)[1],  # Topo Y
                                   svd_traj.get_coords(shot)[1] + img_list[0].shape[0]]  # Base Y
                           )
                plt.xlim([svd_traj.get_coords(shot)[0], svd_traj.get_coords(shot)[0] + img_list[0].shape[1]])
                plt.ylim([svd_traj.get_coords(shot)[1], svd_traj.get_coords(shot)[1] + img_list[0].shape[0]])
                offset_y = img_list[0].shape[1] / 2
                offset_x = img_list[1].shape[1] / 2

                plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                         svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                         ':o',
                         color='r', label='SVD')
                plt.plot(pc_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                         pc_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                         ':o',
                         color='g', label='PC')
                plt.legend()

                print(f"frame = {shot + 1}")

                plt.subplot(2, 1, 2)
                plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                         svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                         ':o',
                         color='r', label='SVD')
                plt.plot(pc_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                         pc_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                         ':o',
                         color='g', label='PC')

                writer.grab_frame()
                plt.clf()
            elif video_type == 2:
                print(f"frame = {shot + 1}")
                M, N = img_list[0].shape
                q, Q = svd_traj.compute_cps(img_list, shot=shot)
                qu, s, qv = svd_traj.compute_svd(img_list, shot=shot)
                pu = np.angle(qu[:, 0])
                pv = np.angle(qv[0, :])
                pu_unwrapped = phase_unwrapping(pu)
                pv_unwrapped = phase_unwrapping(pv)
                deltay, muy, cy, xy, yy = svd_traj.estimate_shift(pu_unwrapped, M)
                deltax, mux, cx, xx, yx = svd_traj.estimate_shift(pv_unwrapped, N)

                plt.suptitle(f"Shot = {shot}")
                plt.subplot(2, 3, 1)
                plt.imshow(img_list[shot][::-1, :], cmap='gray',
                           extent=[svd_traj.get_coords(shot)[0],  # Esquerda X
                                   svd_traj.get_coords(shot)[0] + img_list[0].shape[1],  # Direita X
                                   svd_traj.get_coords(shot)[1],  # Topo Y
                                   svd_traj.get_coords(shot)[1] + img_list[0].shape[0]]  # Base Y
                           )
                plt.xlim([svd_traj.get_coords(shot)[0], svd_traj.get_coords(shot)[0] + img_list[0].shape[1]])
                plt.ylim([svd_traj.get_coords(shot)[1], svd_traj.get_coords(shot)[1] + img_list[0].shape[0]])
                offset_y = img_list[0].shape[1] / 2
                offset_x = img_list[1].shape[1] / 2

                plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                         svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                         ':o',
                         color='r', label='SVD')
                plt.plot(pc_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                         pc_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                         ':o',
                         color='g', label='PC')

                plt.subplot(2, 3, 4)
                plt.imshow(np.angle(qu @ np.diag(s) @ qv), cmap='gray')
                plt.title("Crosspower spectrum")

                plt.subplot(2, 3, 2)
                x_pu = np.arange(0, pu_unwrapped.size)
                plt.stem(x_pu, pu[:-1])
                plt.title(r"$p_u$")

                plt.subplot(2, 3, 3)
                x_pv = np.arange(0, pv_unwrapped.size)
                plt.stem(x_pv, pv[:-1])
                plt.title(r"$p_v$")

                plt.subplot(2, 3, 5)
                x_pu = np.arange(0, pu_unwrapped.size)
                plt.stem(x_pu, pu_unwrapped)
                plt.plot(xy, xy * muy + cy, color='r')
                plt.title(r'unwrap$(p_u)$')

                plt.subplot(2, 3, 6)
                x_pv = np.arange(0, pv_unwrapped.size)
                plt.stem(x_pv, pv_unwrapped)
                plt.plot(xx, xx * mux + cx, color='r')
                plt.title(r'unwrap$(p_v)$')

                # Limpa
                writer.grab_frame()
                plt.clf()
    print("Fim do loop de geração de vídeo.")
else:
    shot = 70
    M, N = img_list[0].shape
    q, Q = svd_traj.compute_cps(img_list, shot=shot)
    qu, s, qv = svd_traj.compute_svd(img_list, shot=shot)
    pu = np.angle(qu[:, 0])
    pv = np.angle(qv[0, :])
    pu_unwrapped = phase_unwrapping(pu)
    pv_unwrapped = phase_unwrapping(pv)
    qu2, s2, qv2 = svd_traj.compute_svd(img_list, shot=shot)
    pu2 = np.angle(qu2[:, 0])
    pv2 = np.angle(qv2[0, :])

    plt.subplot(4,1,1)
    r = np.arange(0, pu.size)
    plt.stem(r, pu)
    plt.plot(r, pu2, 's', color='g')

    plt.subplot(4,1,2)
    r = np.arange(0, pu.size)
    plt.stem(r[:-1], np.diff(pu))
    plt.plot(r[:-1], np.diff(pu2), 's', color='g')
    plt.plot(r, 2*np.pi*0.9*np.ones_like(r), 'r:')
    plt.plot(r, -2*np.pi*0.9*np.ones_like(r), 'r:')

    pu2_unwrapped = phase_unwrapping(pu2)
    pv2_unwrapped = phase_unwrapping(pv2)

    plt.subplot(4,1,3)
    r = np.arange(0, pu_unwrapped.size)
    plt.stem(r, pu_unwrapped)
    plt.plot(r, pu2_unwrapped, 's', color='g')

    # phase_diff = np.diff(pu)
    # corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * 0.9)) + 2. * np.pi * (
    #             phase_diff < -(2 * np.pi * 0.9))
    # plt.stem(np.cumsum(corrected_difference))
    # plt.plot(pu2_unwrapped, 's', color='g')




    deltay, muy, cy, xy, yy = svd_traj.estimate_shift(pu_unwrapped, M)
    deltay2, muy2, cy2, xy2, yy2 = svd_traj.estimate_shift(pu2_unwrapped, M)
    deltax, mux, cx, xx, yx = svd_traj.estimate_shift(pv_unwrapped, N)

    plt.subplot(4,1,4)
    plt.stem(xy, yy)
    plt.plot(xy2, yy2, 's', color='g')
    plt.xlim(r[0], r[-1])

    plt.suptitle(f"Shot = {shot}\n " + r"$(\Delta x, \Delta y)$ = " + f"({deltax:.2f}, {deltay:.2f})")
    plt.subplot(2, 3, 1)
    plt.imshow(img_list[shot][::-1, :], cmap='gray',
               extent=[svd_traj.get_coords(shot)[0],  # Esquerda X
                       svd_traj.get_coords(shot)[0] + img_list[0].shape[1],  # Direita X
                       svd_traj.get_coords(shot)[1],  # Topo Y
                       svd_traj.get_coords(shot)[1] + img_list[0].shape[0]]  # Base Y
               )
    plt.xlim([svd_traj.get_coords(shot)[0], svd_traj.get_coords(shot)[0] + img_list[0].shape[1]])
    plt.ylim([svd_traj.get_coords(shot)[1], svd_traj.get_coords(shot)[1] + img_list[0].shape[0]])
    offset_y = img_list[0].shape[1] / 2
    offset_x = img_list[1].shape[1] / 2

    plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
             svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
             ':o',
             color='r', label='SVD')
    plt.plot(pc_traj.get_coords()[:shot + 1][:, 0] + offset_x,
             pc_traj.get_coords()[:shot + 1][:, 1] + offset_y,
             ':o',
             color='g', label='PC')

    plt.subplot(2, 3, 4)
    plt.imshow(np.angle(qu @ np.diag(s) @ qv), cmap='gray')
    plt.title("Crosspower spectrum")

    plt.subplot(2, 3, 2)
    x_pu = np.arange(0, pu_unwrapped.size)
    plt.stem(x_pu, pu[:-1])
    plt.title(r"$p_u$")

    plt.subplot(2, 3, 3)
    x_pv = np.arange(0, pv_unwrapped.size)
    plt.stem(x_pv, pv[:-1])
    plt.title(r"$p_v$")

    plt.subplot(2, 3, 5)
    x_pu = np.arange(0, pu_unwrapped.size)
    plt.stem(x_pu, pu_unwrapped)
    plt.plot(xy, xy * muy + cy, color='r')
    plt.title(r'unwrap$(p_u)$')

    plt.subplot(2, 3, 6)
    x_pv = np.arange(0, pv_unwrapped.size)
    plt.stem(x_pv, pv_unwrapped)
    plt.plot(xx, xx * mux + cx, color='r')
    plt.title(r'unwrap$(p_v)$')
