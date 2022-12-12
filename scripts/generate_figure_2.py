import matplotlib.pyplot as plt
from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import get_img

# Arbitrary shot and
shot = 500
image_size = (480, 640)
filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001")
svd_traj = TrajectoryParams(svd_param)

# Current image
f = get_img(shot, data_root + filename[0])[:, int(image_size[1]/2 - image_size[0]/2):int(image_size[1]/2 + image_size[0]/2)]
# Next image
g = get_img(shot+1, data_root + filename[0])[:, int(image_size[1]/2 - image_size[0]/2):int(image_size[1]/2 + image_size[0]/2)]

# Apply 2D FFT:
F = np.fft.fftshift(np.fft.fft2(f))
G = np.fft.fftshift(np.fft.fft2(g))

# Cross power-spectrum:
q, Q = crosspower_spectrum(f, g)

fig, axis = plt.subplots(nrows=2, ncols=7)

axis[0, 0].imshow(f, cmap="gray")
axis[0, 0].set_title("$f(x,y)$")
axis[1, 0].imshow(g, cmap="gray")
axis[1, 0].set_title("$g(x,y)$")
axis[0, 1].imshow(np.log10(np.abs(F) + 1), cmap="gray")
axis[0, 1].set_title("$log_{10}(FFT(|f(x,y)| + 1)$")
axis[1, 1].imshow(np.log10(np.abs(G) + 1), cmap="gray")
axis[1, 1].set_title("$log_{10}(FFT(|g(x,y)| + 1)$")
axis[0, 2].imshow(np.log10(np.abs(Q)), cmap='gray')
axis[0, 2].set_title("$Q_n$")
axis[1, 2].imshow(np.log10(np.abs(Q)), cmap='gray')
axis[1, 2].set_title("$Q_n \cdot w(x,y)$")
#
# M, N = curr_img.shape

# qu, s, qv = svd_traj.compute_svd(shot, data_root)
# pu = np.angle(qu[:, 0])
# pv = np.angle(qv[0, :])
# pu_unwrapped = phase_unwrapping(pu)
# pv_unwrapped = phase_unwrapping(pv)
# deltay, muy, cy, xy, yy = svd_traj.estimate_shift(pu_unwrapped, M)
# deltax, mux, cx, xx, yx = svd_traj.estimate_shift(pv_unwrapped, N)
#
# plt.suptitle(f"Shot = {shot}")
# plt.subplot(2, 4, 1)
# plt.imshow(curr_img[::-1, :], cmap='gray',
#        extent=[svd_traj.get_coords(shot)[0],  # Esquerda X
#                svd_traj.get_coords(shot)[0] + curr_img.shape[1],  # Direita X
#                svd_traj.get_coords(shot)[1],  # Topo Y
#                svd_traj.get_coords(shot)[1] + curr_img.shape[0]]  # Base Y
#        )
# plt.xlim([svd_traj.get_coords(shot)[0], svd_traj.get_coords(shot)[0] + curr_img.shape[1]])
# plt.ylim([svd_traj.get_coords(shot)[1], svd_traj.get_coords(shot)[1] + curr_img.shape[0]])
# offset_y = curr_img.shape[1] / 2
# offset_x = curr_img.shape[1] / 2
#
# plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
#      svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
#      ':o',
#      color='r', label='SVD')
# plt.plot(pc_traj.get_coords()[:shot + 1][:, 0] + offset_x,
#      pc_traj.get_coords()[:shot + 1][:, 1] + offset_y,
#      ':o',
#      color='g', label='PC')
#
# plt.subplot(2, 4, 5)
# plt.imshow(np.angle(qu @ np.diag(s) @ qv), cmap='gray')
# plt.title("Crosspower spectrum")
#
# plt.subplot(2, 4, 2)
# x_pu = np.arange(0, pu_unwrapped.size)
# plt.stem(x_pu, pu[:-1])
# plt.title(r"$p_u$")
#
# plt.subplot(2, 4, 3)
# x_pu = np.arange(0, pu_unwrapped.size)
# plt.stem(x_pu[:-1], np.diff(pu[:-1]))
# factor = 0.8
# plt.plot(x_pu, 2*np.pi*factor*np.ones_like(pu[:-1]), ":r")
# plt.plot(x_pu, -2*np.pi*factor*np.ones_like(pu[:-1]), ":r")
# plt.title(r"diff$(p_u)$")
#
# plt.subplot(2, 4, 4)
# x_pu = np.arange(0, pu_unwrapped.size)
# plt.stem(x_pu, pu_unwrapped)
# plt.plot(xy, xy * muy + cy, color='r')
# plt.title(r'unwrap$(p_u)$')
#
# plt.subplot(2, 4, 6)
# x_pv = np.arange(0, pv_unwrapped.size)
# plt.stem(x_pv, pv[:-1])
# plt.title(r"$p_v$")
#
# plt.subplot(2, 4, 7)
# x_pu = np.arange(0, pu_unwrapped.size)
# plt.stem(x_pu[:-1], np.diff(pu[:-1]))
# factor = 0.9
# plt.plot(x_pu, 2 * np.pi * factor * np.ones_like(pu[:-1]), ":r")
# plt.plot(x_pu, -2 * np.pi * factor * np.ones_like(pu[:-1]), ":r")
# plt.title(r"diff$(p_u)$")
#
# plt.subplot(2, 4, 8)
# x_pu = np.arange(0, pv_unwrapped.size)
# plt.stem(x_pu, pv_unwrapped)
# plt.plot(xx, xx * mux + cx, color='r')
# plt.title(r'unwrap$(p_v)$')
