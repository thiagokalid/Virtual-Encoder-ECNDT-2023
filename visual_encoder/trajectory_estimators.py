from visual_encoder.dip_utils import gaussian_noise, salt_and_pepper, apply_window
from visual_encoder.phase_correlation import pc_method
from visual_encoder.svd_decomposition import svd_method
from scipy.spatial.transform import Rotation as R
from visual_encoder.utils import *


def compute_new_position(deltax, deltay, x0, y0, z0, rot_calibration=0, quaternion=None, euler=None):
    initial_position = np.array([x0, y0, z0])
    deltax_corrected = deltax * np.cos(rot_calibration) - deltay * np.sin(rot_calibration)
    deltay_corrected = deltax * np.sin(rot_calibration) + deltay * np.cos(rot_calibration)
    delta_3d = np.array([deltax_corrected, deltay_corrected, 0])

    if quaternion is None and euler is None:
        return initial_position + delta_3d
    elif quaternion is None and euler is not None:
        r = R.from_euler("yxz", euler, degrees=True)
        delta_3d_rotated = r.apply(delta_3d)
        return initial_position + delta_3d_rotated
    elif quaternion is not None and euler is None:
        r = R.from_quat(quaternion)
        delta_3d_rotated = r.apply(delta_3d)
        return initial_position + delta_3d_rotated
    else:
        raise ValueError("Incorrect orientation data.")


def compute_trajectory(img_beg, img_end, x0, y0, z0, traj_params, quaternion=None, euler=None):
    # Windowing in applied in order to mitigate edging effects:
    img_beg, img_end = apply_window(img_beg, img_end, traj_params.spatial_window)

    if traj_params.method == 'pc':
        deltax, deltay = pc_method(img_beg, img_end, traj_params.frequency_window)
    elif traj_params.method == 'svd':
        deltax, deltay = svd_method(img_beg, img_end, traj_params.frequency_window)
    else:
        raise ValueError('Selected method not supported.')
    deltax = deltax * traj_params.xy_res[0]
    deltay = deltay * traj_params.xy_res[1] * -1  # The minus is consequence of the coordinate system difference.
    # The used system consider ascending row order equals to descending y-axis values.
    xf, yf, zf = compute_new_position(deltax, deltay, x0, y0, z0,
                                      quaternion=quaternion, euler=euler, rot_calibration=traj_params.rot_calibration)
    return xf, yf, zf


def compute_total_trajectory_path(data_root, n_images, traj_params, n_beg=1, quat_data=None, euler_data=None):
    positions = np.zeros((n_images, 3))
    x0, y0, z0 = positions[0, :] = traj_params.get_init_coord()
    for i in range(n_beg + 1, n_images + n_beg):
        img_0 = get_img(i - 1, data_root)
        img_f = get_img(i, data_root)
        j = i - n_beg
        if quat_data is not None:
            quaternion = quat_data[j, :]
            positions[j, :] = compute_trajectory(img_0, img_f, x0, y0, z0, traj_params,
                                                 quaternion=quaternion)
        elif euler_data is not None:
            euler = euler_data[j, :]
            positions[j, :] = compute_trajectory(img_0, img_f, x0, y0, z0, traj_params,
                                                 euler=euler)
        else:
            positions[j, :] = compute_trajectory(img_0, img_f, x0, y0, z0, traj_params)

        x0, y0, z0 = positions[j, :]
    return positions


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


# def convert_to_3d(coords_2d, quaternion_vector):
#     coords_3d = np.zeros(shape=(coords_2d.shape[0], 3))
#     coords_3d[0, :2] = coords_2d[0, :]
#     for i in range(1, coords_2d.shape[0]):
#         delta_2d = coords_2d[i] - coords_2d[i - 1]
#         delta_3d = np.array([delta_2d[0], delta_2d[1], 0])
#         quat = quaternion_vector[i]
#         r = R.from_quat(quat)
#         delta_3d = r.apply(delta_3d)
#         coords_3d[i, :] = coords_3d[i - 1, :] + delta_3d
#     return coords_3d


def convert_to_3d(coords_2d, euler_data):
    coords_3d = np.zeros(shape=(coords_2d.shape[0], 3))
    for i, euler in enumerate(euler_data):
        r = R.from_euler('yxz', euler, degrees=True)
        coords_3d[i, :] = r.apply(coords_2d[i, :])
    return coords_3d

def gen_artificial_traj(width, height, num=400, type="rectangular", dim=3):
    # Ideal coordinates:
    quarter = num//4
    positions = np.zeros(shape=(num, dim))
    positions[:quarter, 0] = np.linspace(0, width, num=quarter)
    positions[:quarter, 1] = 0
    positions[quarter:quarter*2, 0] = width
    positions[quarter:quarter*2, 1] = np.linspace(0, height, num=quarter)
    positions[quarter*2:quarter*3, 0] = np.linspace(0, width, num=quarter)[::-1]
    positions[quarter*2:quarter*3, 1] = height
    positions[quarter*3:, 0] = 0
    positions[quarter*3:, 1] = np.linspace(0, height, num=quarter)[::-1]
    return positions


def gen_artificial_euler(init_ang=-48.8, end_ang=48.8, num=400, dim=3):
    quarter = num//4
    euler_data = np.zeros(shape=(num, dim))
    euler_data[:quarter*1, 1] = init_ang
    euler_data[quarter*1:quarter*2, 1] = np.linspace(init_ang, end_ang, num=quarter)
    euler_data[quarter*2:quarter*3, 1] = end_ang
    euler_data[quarter*3:quarter*4, 1] = np.linspace(init_ang, end_ang, num=quarter)[::-1]
    return euler_data
