from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
import matplotlib.pyplot as plt
import pandas as pd

# Geometry : Plane
# Test type: Contact (air)

# Measured planar specimen dimensions:
measured_longest_dist_x = np.mean([359, 358, 357])  # distance in millimeters
measured_shortest_dist_y = np.mean([200.1, 200, 200.9])  # distance in millimeters

filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")
# General SVD related parameters:
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               xy_resolution=(0.02151467169232321, 0.027715058926976663), resolution_unit="mm/pixels",
                               rotation_correction=-0.034107828176162376)

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Closed-loop case:
print("Medium: Air. Beginning of closed-loop...")
closed_loop_air = TrajectoryParams(svd_param)
closed_loop_air.compute_total_trajectory_path(data_root + filename[0], n_images=1496)
print("End of closed-loop...")

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Single-X case:
print("Beginning of single-x...")
single_x_air = TrajectoryParams(svd_param)
single_x_air.compute_total_trajectory_path(data_root + filename[1], n_images=594)
print("End of single-x...")

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Single-Y case:
print("Beginning of single-y...")
single_y_air = TrajectoryParams(svd_param)
single_y_air.compute_total_trajectory_path(data_root + filename[2], n_images=593)
print("End of single-y...")

fig = plt.figure(figsize=(9, 5))
xrange = np.arange(0, measured_longest_dist_x)
yrange = np.arange(0, measured_shortest_dist_y)
plt.suptitle("Ensaios fora d'gua")
factor = .3
plt.subplot(1, 3, 1)
plt.plot(single_x_air.coords[:, 0], single_x_air.coords[:, 1] * (-1), 'o', color="#FF1F5B")
plt.title("Single-X")
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.xlim([measured_longest_dist_x * (-factor), measured_longest_dist_x * (1+factor)])
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(single_y_air.coords[:, 0], single_y_air.coords[:, 1] * (-1), 'o', color="#FF1F5B")
plt.title("Single-Y")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="True")
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(closed_loop_air.coords[:, 0], closed_loop_air.coords[:, 1] * (-1), 'o', color="#FF1F5B")
plt.title("Closed-loop")
plt.ylim([measured_shortest_dist_y * (-factor), measured_shortest_dist_y * (1+factor)])
factor = .15
plt.xlim([measured_longest_dist_x * (-factor), measured_longest_dist_x * (1+factor)])
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.plot(xrange, xrange * 0 + measured_shortest_dist_y, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0 + measured_longest_dist_x, yrange, linewidth=2, color="#009ADE", label="_")
plt.legend()
plt.axis('equal')
plt.grid()

# Geometry : Plane
# Test type: Immersion (water)
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               xy_resolution=(0.02035460911563854, 0.025723094098896573), resolution_unit="mm/pixels",
                               rotation_correction=-0.02428690717359043)
filename = [f"water_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")

# Medium: Water (Immersion)
# Geometry: Planar
# Test type: Closed-loop case:
print("Medium: Water. Beginning of closed-loop...")
closed_loop_water = TrajectoryParams(svd_param)
closed_loop_water.compute_total_trajectory_path(data_root + filename[0], n_images=1496)
print("End of closed-loop...")

# Medium: Water (Immersion)
# Geometry: Planar
# Test type: Single-X case:
print("Beginning of single-x...")
single_x_water = TrajectoryParams(svd_param)
single_x_water.compute_total_trajectory_path(data_root + filename[1], n_images=594)
print("End of single-x...")

# Medium: Water (Immersion)
# Geometry: Planar
# Test type: Single-Y case:
print("Beginning of single-y...")
single_y_water = TrajectoryParams(svd_param)
single_y_water.compute_total_trajectory_path(data_root + filename[2], n_images=594)
print("End of single-y...")


fig = plt.figure(figsize=(9, 5))
xrange = np.arange(0, measured_longest_dist_x)
yrange = np.arange(0, measured_shortest_dist_y)
plt.suptitle("Ensaios dentro d'gua")
factor = .3
plt.subplot(1, 3, 1)
plt.plot(single_x_water.coords[:, 0], single_x_water.coords[:, 1] * (-1), 'o', color="#FF1F5B")
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.title("Single-X")
plt.xlim([measured_longest_dist_x * (-factor), measured_longest_dist_x * (1+factor)])
plt.axis('equal')
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(single_y_water.coords[:, 0], single_y_water.coords[:, 1] * (-1), 'o', color="#FF1F5B")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="True")
plt.title("Single-Y")
plt.axis('equal')
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(closed_loop_water.coords[:, 0], closed_loop_water.coords[:, 1] * (-1), 'o', color="#FF1F5B")
plt.title("Closed-loop")
plt.ylim([measured_shortest_dist_y * (-factor), measured_shortest_dist_y * (1+factor)])
factor = .15
plt.xlim([measured_longest_dist_x * (-factor), measured_longest_dist_x * (1+factor)])
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.plot(xrange, xrange * 0 + measured_shortest_dist_y, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0 + measured_longest_dist_x, yrange, linewidth=2, color="#009ADE", label="_")
plt.axis('equal')
plt.grid()
plt.legend()

df_plane_single_x = pd.DataFrame({'x': [single_x_air.coords[-1, 0], single_x_water.coords[-1, 0], measured_longest_dist_x],
                                  'y': [-1 * single_x_air.coords[-1, 1], -1 * single_x_water.coords[-1, 1], 0]})
df_plane_single_y = pd.DataFrame({'x': [single_y_air.coords[-1, 0], single_y_water.coords[-1, 0], 0],
                                  'y': [-1 * single_y_air.coords[-1, 1], -1 * single_y_water.coords[-1, 1], measured_shortest_dist_y]})
df_plane_closed_loop = pd.DataFrame({'x': [closed_loop_air.coords[-1, 0], closed_loop_water.coords[-1, 0], 0],
                                  'y': [-1 * closed_loop_air.coords[-1, 1], -1 * closed_loop_water.coords[-1, 1], 0]})
