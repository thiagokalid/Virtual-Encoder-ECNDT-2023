import matplotlib.pyplot as plt
from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import get_img, get_quat_data

def get_euler_data(data_root, filename="eul_data", n=995):
    euler_data = np.zeros(shape=(n, 3))
    with open(data_root + "/" + filename + '.txt', 'r') as f:
        for i, line in enumerate(f):
            corrected_line = line.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(',')
            if "None" in corrected_line:
                corrected_line = previous_line
            euler_data[i, :] = np.array([float(x) for x in corrected_line])
            previous_line = corrected_line
    return euler_data


# Figure 3 (d) info:
# Geometry : Plane
# Test type: Contact (air)


# Measured planar specimen dimensions:
measured_shortest_dist_y = np.mean([200.1, 200, 200.9])  # 3 measurements were made, where distance is express in mm
measured_longest_dist_x = np.mean([359, 358, 357])  # 3 measurements were made, where distance is expressed in mm

filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/cylindrical/"
print("Starting the SVD based estimation method...")
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               resolution_unit="mm/pixels", xy_resolution=(0.022625319220898433, 0.024491131391332216),
                               rotation_correction=-0.03581000622178206)
# Result: x_res, y_res = (0.022625319220898433, 0.024491131391332216)
# rotation = -0.03581000622178206

euler_data = get_euler_data(data_root + filename[0], n=1496)
svd_traj = TrajectoryParams(svd_param)
svd_traj.compute_total_trajectory_path(data_root + filename[0], n_images=1496, euler_data=euler_data)
print("End of the SVD based estimation method.")

plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(svd_traj.coords[:, 0], svd_traj.coords[:, 1], svd_traj.coords[:, 2], color="#FF1F5B", label="Estimated")
ax.set_xlabel("x-axis / mm")
ax.set_ylabel("y-axis / mm")
ax.set_zlabel("z-axis / mm")
ax.axis("Equal")

