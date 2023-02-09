import matplotlib.pyplot as plt
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import *
from visual_encoder.utils import *
from scipy.spatial.transform import Rotation as R

# Measured planar specimen dimensions:
measured_longest_dist_x = np.mean([298, 299, 298])  # distance in millimeters
measured_shortest_dist_y = np.mean([215, 220, 215])  # distance in millimeters

# Estimating trajectory:
filename = [f"water_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/cylindrical/"
print("Starting the SVD based estimation method...")
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               resolution_unit="mm/pixels", xy_resolution=(0.022625319220898433, 0.024491131391332216),
                               rotation_correction=0.035810006221782056)

svd_traj = TrajectoryParams(svd_param)
svd_traj.compute_total_trajectory_path(data_root + filename[0], n_images=1496)
print("End of the SVD based estimation method.")

# Read IMU data:
euler_data = get_euler_data(data_root + filename[0], n=1496)
euler_data[:, 0] = np.median(euler_data[:, 0])
euler_data[1256, 1] = euler_data[1255, 1]
euler_data[1388, 1] = euler_data[1387, 1]
euler_data[:, 2] = np.median(euler_data[:, 2])


# Ideal trajectory:
coords_2d_ideal = gen_artificial_traj(measured_longest_dist_x, measured_shortest_dist_y)
euler_data_ideal = gen_artificial_euler(init_ang=euler_data[:, 1].min(), end_ang=euler_data[:, 1].max())
    coords_3d_ideal = convert_to_3d(coords_2d_ideal, euler_data_ideal)

# Apply IMU data in estimated coordiantes:
coords_3d = convert_to_3d(svd_traj.get_coords(), euler_data)

# Apply rotation and correction to all points in order to correct cylinder rotation along x-axis:
r = R.from_euler('yxz', [0, -euler_data[:, 1].max() + 180, 0], degrees=True)
coords_3d_ideal = r.apply(coords_3d_ideal)
coords_3d_ideal[:, 1] = -coords_3d_ideal[:, 1]
coords_3d = r.apply(coords_3d)
coords_3d[:, 1] = -coords_3d[:, 1]

# Plot data:
fig = plt.figure(figsize=(7.5, 5.25))
ax = plt.axes(projection="3d")
ax.view_init(11, -32)
ax.plot3D(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], 'o', color="#FF1F5B", label='Reconstructed')
ax.plot3D(coords_3d_ideal[:, 0], coords_3d_ideal[:, 1], coords_3d_ideal[:, 2], linewidth=3, color="#009ADE",
          label="True")
ax.set_xlim([-10, measured_longest_dist_x + 10])
ax.set_ylim([-51, measured_shortest_dist_y + 40])
ax.set_zlim([-130, 180])
ax.set_xlabel("x-axis / mm")
ax.set_ylabel("y-axis / mm")
ax.set_zlabel("z-axis / mm")
ax.set_xticks(np.linspace(0, measured_longest_dist_x, 5))
ax.set_yticks(np.linspace(0, measured_shortest_dist_y, 5))
ax.set_zticks(np.linspace(0, coords_3d_ideal[:, 2].max(), 4))
plt.tight_layout()
plt.savefig("../figures/Figure3f.eps", format=None, metadata=None,
            bbox_inches="tight"
            )
plt.legend()
plt.title("(f) Cylindrical path (3-D)")

################################################
fig = plt.figure(figsize=(5, 3.5))
ax = plt.axes()
ax.plot(coords_3d[:, 0], coords_3d[:, 1], 'o', color="#FF1F5B", label='Reconstructed')
ax.plot(coords_3d_ideal[:, 0], coords_3d_ideal[:, 1], linewidth=3, color="#009ADE", label="True")
ax.set_xlim([-20, measured_longest_dist_x + 20])
ax.set_ylim([-61, measured_shortest_dist_y + 50])
ax.set_xlabel("x-axis / mm")
ax.set_ylabel("y-axis / mm")
ax.set_xticks(np.linspace(0, measured_longest_dist_x, 5))
ax.set_yticks(np.linspace(0, measured_shortest_dist_y, 5))
plt.grid()
plt.tight_layout()
plt.savefig("../figures/Figure3e.eps", format=None, metadata=None,
            bbox_inches="tight"
            )
plt.title("(e) Cylindrical path (2-D)")
plt.legend()
