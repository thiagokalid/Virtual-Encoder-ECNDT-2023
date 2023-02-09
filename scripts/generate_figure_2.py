import matplotlib.pyplot as plt
from visual_encoder.dsp_utils import ideal_lowpass
from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import get_img
from scipy.sparse.linalg import svds
from visual_encoder.svd_decomposition import phase_unwrapping, linear_regression

# Arbitrary shot and
shot = 250
image_size = (480, 640)
filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001")
svd_traj = TrajectoryParams(svd_param)

# Current imageq
f = get_img(shot, data_root + filename[0])[:,
    int(image_size[1] / 2 - image_size[0] / 2):int(image_size[1] / 2 + image_size[0] / 2)]
# Next image
g = get_img(shot + 1, data_root + filename[0])[:,
    int(image_size[1] / 2 - image_size[0] / 2):int(image_size[1] / 2 + image_size[0] / 2)]

# Apply 2D FFT:
F = np.fft.fftshift(np.fft.fft2(f))
G = np.fft.fftshift(np.fft.fft2(g))

# Cross power-spectrum:
q, Q = crosspower_spectrum(f, g)
m = 0.8 * Q.shape[0] / 2
n = 0.8 * Q.shape[1] / 2


# Window size:
Q_wind = ideal_lowpass(Q, factor=0.3)

# SVD Truncation:
qu, s, qv = svds(Q_wind, k=1)
qu = qu[:, 0]
qv = qv[0, :]
pu = np.angle(qu)
pv = np.angle(qv)
pu_unwrapped = phase_unwrapping(pu[:])
pv_unwrapped = phase_unwrapping(pv[:])
N = pu_unwrapped.size // 2

r = np.arange(0, pu_unwrapped.size)
M = r.size // 2
# Choosing a smaller window:
x1 = r
y = pu_unwrapped
mu1, c1 = linear_regression(x1, y)

r = np.arange(0, pv_unwrapped.size)
M = r.size // 2
# Choosing a smaller window:
x2 = r
y = pv_unwrapped
mu2, c2 = linear_regression(x2, y)

# Values for custom xlim and ylim:
amplitude = np.abs(np.max(pv_unwrapped)) + np.abs(np.min(pv_unwrapped))
middle_pu = np.mean(pu_unwrapped)
middle_pv = np.mean(pv_unwrapped)

fig, axis = plt.subplots(nrows=2, ncols=7)

axis[0, 0].imshow(f, cmap="gray")
axis[0, 0].set_title("$f_n(x,y)$")
axis[0, 0].tick_params(bottom=False, left=False)
axis[0, 0].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[1, 0].imshow(g, cmap="gray")
axis[1, 0].set_title("$f_{n-1}(x,y)$")
axis[1, 0].tick_params(bottom=False, left=False)
axis[1, 0].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[0, 1].imshow(np.log10(np.abs(F) + 1), cmap="gray")
axis[0, 1].set_title("$log_{10}(|FFT(f_{n}(x,y)| + 1)$")
axis[0, 1].tick_params(bottom=False, left=False)
axis[0, 1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[1, 1].imshow(np.log10(np.abs(G) + 1), cmap="gray")
axis[1, 1].set_title("$log_{10}(|FFT(f_{n-1}(x,y)| + 1)$")
axis[1, 1].tick_params(bottom=False, left=False)
axis[1, 1].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[0, 2].imshow(np.log10(np.abs(Q)), cmap='gray')
axis[0, 2].set_title("$Q_n$")
axis[0, 2].tick_params(bottom=False, left=False)
axis[0, 2].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[1, 2].imshow(np.log10(np.abs(Q_wind)), cmap='gray')
axis[1, 2].set_title("$Q_n \cdot w(x,y)$")
axis[1, 2].tick_params(bottom=False, left=False)
axis[1, 2].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[0, 3].plot(np.real(qu), '-o', markersize=3, label="$Re(q_u)$", color="#FF1F5B")
axis[0, 3].plot(np.imag(qu), '-o', markersize=3, label="$Im(q_u)$", color="#FFC61E")
axis[0, 3].set_title("$q_u(u,v)$")
axis[0, 3].legend()
axis[0, 3].tick_params(bottom=False, left=False)
axis[0, 3].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label


axis[1, 3].plot(np.real(qv), '-o', markersize=3, label="$Re(q_v)$", color="#FF1F5B")
axis[1, 3].plot(np.imag(qv), '-o', markersize=3, label="$Im(q_v)$", color="#FFC61E")
axis[1, 3].set_title("$q_v(u,v)$")
axis[1, 3].legend()
axis[1, 3].tick_params(bottom=False, left=False)
axis[1, 3].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[0, 4].plot(pu, '-o', markersize=4, color="#FF1F5B")
axis[0, 4].set_title("$p_u(u,v)$")
axis[0, 4].tick_params(bottom=False, left=False)
axis[0, 4].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[1, 4].plot(pv, '-o', markersize=4, color="#FF1F5B")
axis[1, 4].set_title("$p_v(u,v)$")
axis[1, 4].tick_params(bottom=False, left=False)
axis[1, 4].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[0, 5].plot(pu_unwrapped, 'o', markersize=4, color="#FF1F5B")
axis[0, 5].set_title("unwrap($p_u(u,v)$)")
axis[0, 5].set_ylim([middle_pu - amplitude/1.8, middle_pu + amplitude/1.8])
axis[0, 5].tick_params(bottom=False, left=False)
axis[0, 5].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[1, 5].plot(pv_unwrapped, 'o', markersize=4, color="#FF1F5B")
axis[1, 5].set_title("unwrap($p_v(u,v)$)")
axis[1, 5].set_ylim([middle_pv - amplitude/1.8, middle_pv + amplitude/1.8])
axis[1, 5].tick_params(bottom=False, left=False)
axis[1, 5].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

x_span = np.linspace(0, pu_unwrapped.size, pu_unwrapped.size)
axis[0, 6].plot(x1, pu_unwrapped, 'o', markersize=4, color="#FF1F5B")
axis[0, 6].plot(x1, x1 * mu1 + c1, color='#009ADE', linewidth=2)
axis[0, 6].set_title("Linear fit of $p_u$")
axis[0, 6].set_title("unwrap($p_u(u,v)$)")
axis[0, 6].set_ylim([middle_pu - amplitude/1.8, middle_pu + amplitude/1.8])
axis[0, 6].tick_params(bottom=False, left=False)
axis[0, 6].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

axis[1, 6].plot(x1, pv_unwrapped, 'o', markersize=4, color="#FF1F5B")
axis[1, 6].plot(x2, x2 * mu2 + c2, color='#009ADE', linewidth=2)
axis[1, 6].set_title("Linear fit of $p_v$")
axis[1, 6].set_ylim([middle_pv - amplitude/1.8, middle_pv + amplitude/1.8])
axis[1, 6].tick_params(bottom=False, left=False)
axis[1, 6].set(xlabel=None, xticklabels=[], ylabel=None, yticklabels=[])  # remove the axis label

# fig = plt.figure(figsize=(3, 3))
# ax = plt.gca()
# plt.plot(x2, pv_unwrapped, 'o', markersize=12, color="#FF1F5B")
# plt.plot(x2, x2 * mu2 + c2, color='#009ADE', linewidth=2)
# plt.axis(False)
# plt.xlim([-10, 150])
# plt.ylim([5, -45])
# plt.tick_params(axis='both', which='both', bottom=False,
#                 top=False, left=False, right=False,
#                 labelbottom=False, labeltop=False,
#                 labelright=False, labelleft=False)
# plt.tight_layout()
# plt.savefig("../figs/Fig2.png", dpi=300, format=None, metadata=None,
#         bbox_inches="tight"
#        )

# fig = plt.figure(figsize=(3, 3))
# ax = plt.gca()
# plt.plot(x1, pv_unwrapped, 'o', markersize=4, color="#FF1F5B")
# plt.plot(x2, x2 * mu2 + c2, color='#009ADE', linewidth=2)
# plt.axis(False)
# plt.tick_params(axis='both', which='both', bottom=False,
#                 top=False, left=False, right=False,
#                 labelbottom=False, labeltop=False,
#                 labelright=False, labelleft=False)
# plt.tight_layout()
# plt.savefig("../figs/Fig2.png", dpi=300, format=None, metadata=None,
#         bbox_inches="tight"
#        )
