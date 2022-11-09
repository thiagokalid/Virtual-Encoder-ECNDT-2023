import matplotlib.pyplot as plt
from visual_encoder.trajectory_estimators import *
from visual_encoder.svd_decomposition import *
from PIL import Image
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams

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
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001")
svd_traj = TrajectoryParams(svd_param, x0=200, y0=200)
svd_traj.compute_total_trajectory(img_list)
pc_param = DisplacementParams(method="pc", spatial_window=None)
pc_traj = TrajectoryParams(pc_param, x0=200, y0=200)
pc_traj.compute_total_trajectory(img_list)

plt.imshow(base_img, cmap='gray')
plt.plot(coord[:, 0], coord[:, 1], '-o', color='r', label='Expected')
plt.plot(svd_traj.get_coords()[:, 0], svd_traj.get_coords()[:, 1], '-o', color='b', label='SVD')
plt.plot(pc_traj.get_coords()[:, 0], pc_traj.get_coords()[:, 1], '-o', color='g', label='PC')
plt.legend()
