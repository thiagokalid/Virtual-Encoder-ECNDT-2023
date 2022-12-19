import numpy as np
import os
from visual_encoder.dip_utils import gaussian_noise, salt_and_pepper, apply_window
from visual_encoder.phase_correlation import pc_method
from visual_encoder.svd_decomposition import svd_method
from scipy.spatial.transform import Rotation as R


def compute_new_position(deltax, deltay, x0, y0, z0, rot_calibration=0, quaternion=None):
    deltax_corrected = deltax * np.cos(rot_calibration) - deltay * np.sin(rot_calibration)
    deltay_corrected = deltax * np.sin(rot_calibration) + deltay * np.cos(rot_calibration)
    new_pos = np.array([
        x0 + deltax_corrected,
        y0 + deltay_corrected,
        z0
    ])

    if quaternion is None:
        return new_pos
    else:
        r = R.from_quat(quaternion)
        rotated_pos = r.apply(new_pos)
        return rotated_pos


def compute_trajectory(img_beg, img_end, x0, y0, z0, traj_params, quaternion=None):
    # Windowing in applied in order to mitigate edging effects:
    img_beg, img_end = apply_window(img_beg, img_end, traj_params.spatial_window)

    if traj_params.method == 'pc':
        deltax, deltay = pc_method(img_beg, img_end, traj_params.frequency_window)
    elif traj_params.method == 'svd':
        deltax, deltay = svd_method(img_beg, img_end, traj_params.frequency_window)
    else:
        raise ValueError('Selected method not supported.')
    deltax = deltax * traj_params.xy_res[0]
    deltay = deltay * traj_params.xy_res[1]
    xf, yf, zf = compute_new_position(deltax, deltay, x0, y0, z0,
                                      quaternion=quaternion, rot_calibration=traj_params.rot_calibration)
    return xf, yf, zf


def compute_total_trajectory_path(data_root, n_images, traj_params, n_beg=1, quat_data=None):
    positions = np.zeros((n_images, 3))
    x0, y0, z0 = positions[0, :] = traj_params.get_init_coord()
    for i in range(n_beg + 1, n_images + n_beg):
        img_0 = get_img(i - 1, data_root)
        img_f = get_img(i, data_root)
        j = i - n_beg
        if quat_data is not None:
            quaternion = quat_data[:, j]
            positions[j, :] = compute_trajectory(img_0, img_f, x0, y0, z0, traj_params,
                                                 quaternion=quaternion)
        else:
            positions[j, :] = compute_trajectory(img_0, img_f, x0, y0, z0, traj_params)

        x0, y0, z0 = positions[j, :]
    return positions


def get_img(i, data_root):
    image_list = os.listdir(data_root)
    image_name = list(filter(lambda x: f"image{i:02d}_" in x, image_list))[0]
    # image_name = f"image{i:02d}.jpg"
    from PIL import Image
    rgb2gray = lambda img_rgb: img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114
    myImage = Image.open(data_root + image_name)
    img_rgb = np.array(myImage)
    img_gray = rgb2gray(img_rgb)
    return img_gray


def generate_artifical_shifts(base_image, width=None, height=None, x0=0, y0=0, xshifts=None, yshifts=None, steps=100,
                              gaussian_noise_db=None, salt_pepper_noise_prob=None):
    if xshifts is None or yshifts is None:
        # Trajetória em diagonal
        xshifts = np.zeros(steps)
        yshifts = np.zeros(steps)
        xshifts[:steps // 4] = 4
        yshifts[:steps // 4] = 0
        xshifts[steps // 4:2 * steps // 4] = 0
        yshifts[steps // 4:2 * steps // 4] = 4
        xshifts[2 * steps // 4:3 * steps // 4] = -4
        yshifts[2 * steps // 4:3 * steps // 4] = 0
        xshifts[3 * steps // 4:4 * steps // 4] = 0
        yshifts[3 * steps // 4:4 * steps // 4] = -4

    xshifts[0] = 0
    yshifts[0] = 0
    if width is None or height is None:
        width = int(base_image.shape[0] / 4)
        height = int(base_image.shape[1] / 4)
    img_list = list()
    coordinates = np.zeros((steps, 2))
    for i in range(steps):
        coordinates[i, :] = (x0, y0)
        y0 = int(y0 + yshifts[i])
        x0 = int(x0 + xshifts[i])
        yf = int(y0 + height)
        xf = int(x0 + width)
        shifted_img = base_image[y0:yf, x0:xf]
        if gaussian_noise_db is not None:
            shifted_img = gaussian_noise(shifted_img, gaussian_noise_db)
        if salt_pepper_noise_prob is not None:
            shifted_img = salt_and_pepper(shifted_img, prob=salt_pepper_noise_prob)
            # plt.imshow(shifted_img)

        img_list.append(shifted_img)
    salt_and_pepper(shifted_img)
    return img_list, xshifts, yshifts, coordinates


def convert_to_3d(coords_2d, quaternion_vector):
    coords_3d = np.zeros(shape=(coords_2d.shape[0], 3))
    coords_3d[0, :2] = coords_2d[0, :]
    for i in range(1, coords_2d.shape[0]):
        delta_2d = coords_2d[i] - coords_2d[i - 1]
        delta_3d = np.array([delta_2d[0], delta_2d[1], 0])
        quat = quaternion_vector[i]
        r = R.from_quat(quat)
        delta_3d = r.apply(delta_3d)
        coords_3d[i, :] = coords_3d[i - 1, :] + delta_3d
    return coords_3d


def get_quat_data(data_root, filename="quat_data", n=995):
    quat_data = np.zeros(shape=(n, 4))
    with open(data_root + "/" + filename + '.txt', 'r') as f:
        for i, line in enumerate(f):
            corrected_line = line.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(',')
            if "None" in corrected_line:
                corrected_line = previous_line
            quat_data[i, :] = np.array([float(x) for x in corrected_line])
            previous_line = corrected_line
    return quat_data
