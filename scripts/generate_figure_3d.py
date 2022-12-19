import matplotlib.pyplot as plt
from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams

# Figure 3 (d) info:
# Geometry : Plane
# Test type: Contact (air)


# Measured planar specimen dimensions:
measured_shortest_dist_y = np.mean([200.1, 200, 200.9])  # 3 measurements were made, where distance is express in mm
measured_longest_dist_x = np.mean([359, 358, 357])  # 3 measurements were made, where distance is expressed in mm

filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               xy_resolution=(0.02151467169232321, 0.027715058926976663), resolution_unit="mm/pixels",
                               rotation_correction=-0.034107828176162376)
svd_traj = TrajectoryParams(svd_param)
svd_traj.compute_total_trajectory_path(data_root + filename[0], n_images=1496)
print("End of the SVD based estimation method.")


# Plotting the results:
fig = plt.figure(figsize=(5, 3.5))
# Estimated trajectory:
plt.plot(svd_traj.get_coords()[:, 0],
         svd_traj.get_coords()[:, 1] * -1,
         ':o',
         color="#FF1F5B", label='Reconstructed')
# Ideal/Expected trajectory:
x_range = np.arange(0, measured_longest_dist_x, 1e-2)
y_range = np.arange(0, measured_shortest_dist_y, 1e-2)
plt.plot(x_range, 0 * x_range, linewidth=3, color="#009ADE", label="True")
plt.plot(x_range, measured_shortest_dist_y * np.ones_like(x_range), linewidth=3, color="#009ADE", label="_")
plt.plot(0 * y_range, y_range, linewidth=3, color="#009ADE", label="_")
plt.plot(measured_longest_dist_x * np.ones_like(y_range), y_range, linewidth=3, color="#009ADE", label="_")
plt.xlabel("x-axis / [mm]")
plt.ylabel("y-axis / [mm]")
slack = .1
plt.xlim([- measured_longest_dist_x * slack, measured_longest_dist_x * (1 + slack)])
plt.ylim([- measured_shortest_dist_y * slack, measured_shortest_dist_y * (1 + slack)])
plt.xticks(np.linspace(0, measured_longest_dist_x, 5))
plt.yticks(np.linspace(0, measured_shortest_dist_y, 5))
plt.legend()
plt.tight_layout()
plt.grid()


plt.tick_params(axis='both', which='both')
plt.tight_layout()
plt.savefig("../figures/Figure3d.eps", format=None, metadata=None,
        bbox_inches="tight"
       )
plt.axis('equal')
plt.title("(d) Planar path (2-D)")
