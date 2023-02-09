from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.trajectory_estimators import *
import matplotlib.pyplot as plt
import pandas as pd
from visual_encoder.utils import *
from scipy.spatial.transform import Rotation as R

def rms(vec):
    return 1

# Geometry : Plane
# Test type: Contact (air)

# Measured planar specimen dimensions:
plan_measured_longest_dist_x = np.mean([359, 358, 357])  # distance in millimeters
plan_measured_shortest_dist_y = np.mean([200.1, 200, 200.9])  # distance in millimeters

filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")
# General SVD related parameters:
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               xy_resolution=(0.02151467169232321, 0.027715058926976663), resolution_unit="mm/pixels",
                               rotation_correction=0.034107828176162376)

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Closed-loop case:
print("Medium: Air. Beginning of closed-loop...")
plan_closed_loop_air = TrajectoryParams(svd_param)
plan_closed_loop_air.compute_total_trajectory_path(data_root + filename[0], n_images=1496)
print("End of closed-loop...")

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Single-X case:
print("Beginning of single-x...")
plan_single_x_air = TrajectoryParams(svd_param)
plan_single_x_air.compute_total_trajectory_path(data_root + filename[1], n_images=594)
print("End of single-x...")

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Single-Y case:
print("Beginning of single-y...")
plan_single_y_air = TrajectoryParams(svd_param)
plan_single_y_air.compute_total_trajectory_path(data_root + filename[2], n_images=593)
print("End of single-y...")

fig = plt.figure(figsize=(9, 5))
xrange = np.arange(0, plan_measured_longest_dist_x)
yrange = np.arange(0, plan_measured_shortest_dist_y)
plt.suptitle("Geometry: Plane - Contact")
factor = .3
plt.subplot(1, 3, 1)
plt.plot(plan_single_x_air.coords[:, 0], plan_single_x_air.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Single-X")
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.xlim([plan_measured_longest_dist_x * (-factor), plan_measured_longest_dist_x * (1+factor)])
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(plan_single_y_air.coords[:, 0], plan_single_y_air.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Single-Y")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="True")
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(plan_closed_loop_air.coords[:, 0], plan_closed_loop_air.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Closed-loop")
plt.ylim([plan_measured_shortest_dist_y * (-factor), plan_measured_shortest_dist_y * (1+factor)])
factor = .15
plt.xlim([plan_measured_longest_dist_x * (-factor), plan_measured_longest_dist_x * (1+factor)])
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.plot(xrange, xrange * 0 + plan_measured_shortest_dist_y, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0 + plan_measured_longest_dist_x, yrange, linewidth=2, color="#009ADE", label="_")
plt.legend()
plt.axis('equal')
plt.grid()

# Geometry : Plane
# Test type: Immersion (water)
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               xy_resolution=(0.02035460911563854, 0.025723094098896573), resolution_unit="mm/pixels",
                               rotation_correction=0.02428690717359043)
filename = [f"water_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/planar/"
print("Starting the SVD based estimation method...")

# Medium: Water (Immersion)
# Geometry: Planar
# Test type: Closed-loop case:
print("Medium: Water. Beginning of closed-loop...")
plan_closed_loop_water = TrajectoryParams(svd_param)
plan_closed_loop_water.compute_total_trajectory_path(data_root + filename[0], n_images=1496)
print("End of closed-loop...")

# Medium: Water (Immersion)
# Geometry: Planar
# Test type: Single-X case:
print("Beginning of single-x...")
plan_single_x_water = TrajectoryParams(svd_param)
plan_single_x_water.compute_total_trajectory_path(data_root + filename[1], n_images=594)
print("End of single-x...")

# Medium: Water (Immersion)
# Geometry: Planar
# Test type: Single-Y case:
print("Beginning of single-y...")
plan_single_y_water = TrajectoryParams(svd_param)
plan_single_y_water.compute_total_trajectory_path(data_root + filename[2], n_images=594)
print("End of single-y...")


fig = plt.figure(figsize=(9, 5))
xrange = np.arange(0, plan_measured_longest_dist_x)
yrange = np.arange(0, plan_measured_shortest_dist_y)
plt.suptitle("Geometry: Plane - Immersion")
factor = .3
plt.subplot(1, 3, 1)
plt.plot(plan_single_x_water.coords[:, 0], plan_single_x_water.coords[:, 1], 'o', color="#FF1F5B")
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.title("Single-X")
plt.xlim([plan_measured_longest_dist_x * (-factor), plan_measured_longest_dist_x * (1+factor)])
plt.axis('equal')
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(plan_single_y_water.coords[:, 0], plan_single_y_water.coords[:, 1], 'o', color="#FF1F5B")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="True")
plt.title("Single-Y")
plt.axis('equal')
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(plan_closed_loop_water.coords[:, 0], plan_closed_loop_water.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Closed-loop")
plt.ylim([plan_measured_shortest_dist_y * (-factor), plan_measured_shortest_dist_y * (1+factor)])
factor = .15
plt.xlim([plan_measured_longest_dist_x * (-factor), plan_measured_longest_dist_x * (1+factor)])
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.plot(xrange, xrange * 0 + plan_measured_shortest_dist_y, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0 + plan_measured_longest_dist_x, yrange, linewidth=2, color="#009ADE", label="_")
plt.axis('equal')
plt.grid()
plt.legend()

# Geometry : Cylindrical
# Measured planar specimen dimensions:
cyl_measured_longest_dist_x = np.mean([298, 299, 298])  # distance in millimeters
cyl_measured_shortest_dist_y = np.mean([215, 220, 215])  # distance in millimeters

# Test type: Contact (air)
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               resolution_unit="mm/pixels", xy_resolution=(0.022625319220898422, 0.024491131391332216),
                               rotation_correction=0.035810006221782056)
filename = [f"air_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/cylindrical/"
print("Starting the SVD based estimation method...")

# Medium: Air (Contact)
# Geometry: Cylindrical
# Test type: Closed-loop case:
print("Medium: Air. Geometry: Cylinder. Beginning of closed-loop...")
cyl_closed_loop_air = TrajectoryParams(svd_param)
cyl_closed_loop_air.compute_total_trajectory_path(data_root + filename[0], n_images=1496)

# Read IMU data:
cyl_closed_loop_air_euler_data = get_euler_data(data_root + filename[0], n=1496)
cyl_closed_loop_air_euler_data[:, 0] = np.median(cyl_closed_loop_air_euler_data[:, 0])
cyl_closed_loop_air_euler_data[:, 2] = np.median(cyl_closed_loop_air_euler_data[:, 2])

# Apply IMU data in estimated coordiantes:
cyl_closed_loop_air_coords_3d = convert_to_3d(cyl_closed_loop_air.get_coords(), cyl_closed_loop_air_euler_data)

# Apply rotation and correction to all points in order to correct cylinder rotation along x-axis:
r = R.from_euler('yxz', [0, -cyl_closed_loop_air_euler_data[:, 1].max() + 180, 0], degrees=True)
cyl_closed_loop_air_coords_3d = r.apply(cyl_closed_loop_air_coords_3d)
cyl_closed_loop_air_coords_3d[:, 1] = -cyl_closed_loop_air_coords_3d[:, 1]
print("End of closed-loop...")

# Medium: Air (Contact)
# Geometry: Cylindrical
# Test type: Single-X case:
print("Beginning of single-x...")
cyl_single_x_air = TrajectoryParams(svd_param)
cyl_single_x_air.compute_total_trajectory_path(data_root + filename[1], n_images=594)
print("End of single-x...")

# Medium: Air (Contact)
# Geometry: Cylindrical
# Test type: Single-Y case:
print("Beginning of single-y...")
cyl_single_y_air = TrajectoryParams(svd_param)
cyl_single_y_air.compute_total_trajectory_path(data_root + filename[2], n_images=594)
print("End of single-y...")

fig = plt.figure(figsize=(9, 5))
xrange = np.arange(0, cyl_measured_longest_dist_x)
yrange = np.arange(0, cyl_measured_shortest_dist_y)
plt.suptitle("Geometry: Cylinder - Contact")
factor = .3
plt.subplot(1, 3, 1)
plt.plot(cyl_single_x_air.coords[:, 0], cyl_single_x_air.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Single-X")
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.xlim([cyl_measured_longest_dist_x * (-factor), cyl_measured_longest_dist_x * (1+factor)])
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(cyl_single_y_air.coords[:, 0], cyl_single_y_air.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Single-Y")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="True")
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(cyl_closed_loop_air_coords_3d[:, 0], cyl_closed_loop_air_coords_3d[:, 1], 'o', color="#FF1F5B")
plt.title("Closed-loop")
plt.ylim([cyl_measured_shortest_dist_y * (-factor), cyl_measured_shortest_dist_y * (1+factor)])
factor = .15
plt.xlim([cyl_measured_longest_dist_x * (-factor), cyl_measured_longest_dist_x * (1+factor)])
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.plot(xrange, xrange * 0 + cyl_measured_shortest_dist_y, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0 + cyl_measured_longest_dist_x, yrange, linewidth=2, color="#009ADE", label="_")
plt.legend()
plt.axis('equal')
plt.grid()

# Test type: Immersion (water)
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               resolution_unit="mm/pixels", xy_resolution=(0.022625319220898433, 0.024491131391332216),
                               rotation_correction=0.035810006221782056)
filename = [f"water_{name}" + "/" for name in ['closed_loop', 'single_x', 'single_y']]
data_root = "../data/cylindrical/"
print("Starting the SVD based estimation method...")

# Medium: Water (Immersion)
# Geometry: Cylindrical
# Test type: Closed-loop case:
print("Medium: Air. Geometry: Cylinder. Beginning of closed-loop...")
cyl_closed_loop_water = TrajectoryParams(svd_param)
cyl_closed_loop_water.compute_total_trajectory_path(data_root + filename[0], n_images=1496)

# Read IMU data:
cyl_closed_loop_water_euler_data = get_euler_data(data_root + filename[0], n=1496)
cyl_closed_loop_water_euler_data[:, 0] = np.median(cyl_closed_loop_water_euler_data[:, 0])
cyl_closed_loop_water_euler_data[1256, 1] = cyl_closed_loop_water_euler_data[1255, 1]
cyl_closed_loop_water_euler_data[1388, 1] = cyl_closed_loop_water_euler_data[1387, 1]
cyl_closed_loop_water_euler_data[:, 2] = np.median(cyl_closed_loop_water_euler_data[:, 2])

# Apply IMU data in estimated coordiantes:
cyl_closed_loop_water_coords_3d = convert_to_3d(cyl_closed_loop_water.get_coords(), cyl_closed_loop_water_euler_data)

# Apply rotation and correction to all points in order to correct cylinder rotation along x-axis:
r = R.from_euler('yxz', [0, -cyl_closed_loop_water_euler_data[:, 1].max() + 180, 0], degrees=True)
cyl_closed_loop_water_coords_3d = r.apply(cyl_closed_loop_water_coords_3d)
cyl_closed_loop_water_coords_3d[:, 1] = -cyl_closed_loop_water_coords_3d[:, 1]
print("End of closed-loop...")


# Plot data:
fig = plt.figure(figsize=(7.5, 5.25))
ax = plt.axes(projection="3d")
ax.view_init(13, 30)
ax.plot3D(cyl_closed_loop_water_coords_3d[:, 0], cyl_closed_loop_water_coords_3d[:, 1], cyl_closed_loop_water_coords_3d[:, 2], 'o', color="#FF1F5B", label='Reconstructed')



# Medium: Water (Immersion)
# Geometry: Cylindrical
# Test type: Single-X case:
print("Beginning of single-x...")
cyl_single_x_water = TrajectoryParams(svd_param)
cyl_single_x_water.compute_total_trajectory_path(data_root + filename[1], n_images=593)
print("End of single-x...")

# Medium: Water (Immersion)
# Geometry: Cylindrical
# Test type: Single-Y case:
print("Beginning of single-y...")
cyl_single_y_water = TrajectoryParams(svd_param)
cyl_single_y_water.compute_total_trajectory_path(data_root + filename[2], n_images=594)
print("End of single-y...")


fig = plt.figure(figsize=(9, 5))
xrange = np.arange(0, cyl_measured_longest_dist_x)
yrange = np.arange(0, cyl_measured_shortest_dist_y)
plt.suptitle("Geometry: Cylinder - Immersion")
factor = .3
plt.subplot(1, 3, 1)
plt.plot(cyl_single_x_water.coords[:, 0], cyl_single_x_water.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Single-X")
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.xlim([cyl_measured_longest_dist_x * (-factor), cyl_measured_longest_dist_x * (1+factor)])
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(cyl_single_y_water.coords[:, 0], cyl_single_y_water.coords[:, 1], 'o', color="#FF1F5B")
plt.title("Single-Y")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="True")
plt.axis('equal')
plt.grid()
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(cyl_closed_loop_water_coords_3d[:, 0], cyl_closed_loop_water_coords_3d[:, 1], 'o', color="#FF1F5B")
plt.title("Closed-loop")
plt.ylim([cyl_measured_shortest_dist_y * (-factor), cyl_measured_shortest_dist_y * (1+factor)])
factor = .15
plt.xlim([cyl_measured_longest_dist_x * (-factor), cyl_measured_longest_dist_x * (1+factor)])
plt.plot(xrange, xrange * 0, linewidth=2, color="#009ADE", label="True")
plt.plot(xrange, xrange * 0 + cyl_measured_shortest_dist_y, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0, yrange, linewidth=2, color="#009ADE", label="_")
plt.plot(yrange * 0 + cyl_measured_longest_dist_x, yrange, linewidth=2, color="#009ADE", label="_")
plt.legend()
plt.axis('equal')
plt.grid()


# latex_table_code = r"""
# \begin{table}[t]
# \centering
# \caption{Accumulated displacements measured for the 2-D trajectories reconstructed from the data provided by the virtual encoder.}
# \begin{tabular}{lllllll|llllll}
# \cline{2-13}
# Geometry& \multicolumn{6}{c|}{Plane} \multicolumn{6}{c}{Cylinder}
# \\ \cline{2-13}
#  Test type & \multicolumn{2}{c}{Single-x} & \multicolumn{2}{c}{Single-y} & \multicolumn{2}{c|}{Closed loop} & \multicolumn{2}{c}{Single-x} & \multicolumn{2}{c}{Single-y} & \multicolumn{2}{c}{Closed loop} \\ \cline{2-13}
# Results & x & y & x & y & x & y & x & y & x & y & x & y \\ \hline
# \multicolumn{1}{l|}{Contact}""" + \
# f"""& {plan_single_x_air.coords[-1, 0]:.2f} & {plan_single_x_air.coords[-1, 1]:.2f} &
#       {plan_single_y_air.coords[-1, 0]:.2f} & {plan_single_y_air.coords[-1, 1]:.2f} &
#       {plan_closed_loop_air.coords[-1, 0]:.2f} & {plan_closed_loop_air.coords[-1, 1]:.2f} &
#       {cyl_single_x_air.coords[-1, 0]:.2f} & {cyl_single_x_air.coords[-1, 1]:.2f} &
#       {cyl_single_y_air.coords[-1, 0]:.2f} & {cyl_single_y_air.coords[-1, 1]:.2f} &
#       {cyl_closed_loop_air.coords[-1, 0]:.2f} & {cyl_closed_loop_air.coords[-1, 1]:.2f}
#       """ + \
# r"""\\ \hline
# \multicolumn{1}{l|}{Immersion}""" + \
# f"""& {plan_single_x_water.coords[-1, 0]:.2f} & {plan_single_x_water.coords[-1, 1]:.2f} &
#       {plan_single_y_water.coords[-1, 0]:.2f} & {plan_single_y_water.coords[-1, 1]:.2f} &
#       {plan_closed_loop_water.coords[-1, 0]:.2f} & {plan_closed_loop_water.coords[-1, 1]:.2f} &
#       {cyl_single_x_water.coords[-1, 0]:.2f} & {cyl_single_x_water.coords[-1, 1]:.2f} &
#       {cyl_single_y_water.coords[-1, 0]:.2f} & {cyl_single_y_water.coords[-1, 1]:.2f} &
#       {cyl_closed_loop_water.coords[-1, 0]:.2f} & {cyl_closed_loop_water.coords[-1, 1]:.2f}
#       """ + \
# r""" \\ \hline
# \textbf{True}""" + \
# f"""&
#     {plan_single_x_air.coords[-1, 0]:.2f} & {plan_single_x_air.coords[-1, 1]:.2f} &
#     {plan_single_y_air.coords[-1, 0]:.2f} & {plan_single_y_air.coords[-1, 1]:.2f} &
#     {plan_closed_loop_air.coords[-1, 0]:.2f} & {plan_closed_loop_air.coords[-1, 1]:.2f}""" + \
# r"""\end{tabular}
# \label{table:results}
# \end{table}
# """

table_1_plane = r"""
\begin{table}[t]
\centering
\caption{Accumulated displacements measured for the 2-D trajectories reconstructed from the data provided by the virtual encoder.}
\begin{tabular}{lllllll}
\cline{2-7}
 Geometry& \multicolumn{6}{c}{Plane}  \\ \cline{2-7}
 Test type & \multicolumn{2}{c}{Single-x} & \multicolumn{2}{c}{Single-y} & \multicolumn{2}{c}{Closed loop} \\ \cline{2-7}
Results & x & y & x & y & x & y \\ \hline
\multicolumn{1}{l|}{Contact}&""" + \
f"""
        {plan_single_x_air.coords[-1, 0]:.2f} & {plan_single_x_air.coords[-1, 1]:.2f} &
        {plan_single_y_air.coords[-1, 0]:.2f} & {plan_single_y_air.coords[-1, 1]:.2f} &
        {plan_closed_loop_air.coords[-1, 0]:.2f} & {plan_closed_loop_air.coords[-1, 1]:.2f}
""" + \
r"""
      \\ \hline
\multicolumn{1}{l|}{Immersion}& """ + \
f"""
        {plan_single_x_water.coords[-1, 0]:.2f} & {plan_single_x_water.coords[-1, 1]:.2f} &
        {plan_single_y_water.coords[-1, 0]:.2f} & {plan_single_y_water.coords[-1, 1]:.2f} &
        {plan_closed_loop_water.coords[-1, 0]:.2f} & {plan_closed_loop_water.coords[-1, 1]:.2f}
""" + \
r"""
      \\ \hline
\textbf{True} & """ + \
f"""
        {plan_measured_longest_dist_x:.2f} & {0:.2f} &
        {0:.2f} & {plan_measured_shortest_dist_y:.2f} &
        {0:.2f} & {0:.2f}
""" + \
r"""
\end{tabular}
\label{table:results}
\end{table}
"""

table_1_cyl = r"""
\begin{table}[t]
\centering
\caption{Accumulated displacements measured for the 2-D trajectories reconstructed from the data provided by the virtual encoder.}
\begin{tabular}{lllllll}
\cline{2-7}
 Geometry& \multicolumn{6}{c}{Cylinder}  \\ \cline{2-7}
 Test type & \multicolumn{2}{c}{Single-x} & \multicolumn{2}{c}{Single-y} & \multicolumn{2}{c}{Closed loop} \\ \cline{2-7}
Results & x & y & x & y & x & y \\ \hline
\multicolumn{1}{l|}{Contact}&""" + \
f"""
        {cyl_single_x_air.coords[-1, 0]:.2f} & {cyl_single_x_air.coords[-1, 1]:.2f} &
        {cyl_single_y_air.coords[-1, 0]:.2f} & {cyl_single_y_air.coords[-1, 1]:.2f} &
        {cyl_closed_loop_air.coords[-1, 0]:.2f} & {cyl_closed_loop_air.coords[-1, 1]:.2f}
""" + \
r"""
      \\ \hline
\multicolumn{1}{l|}{Immersion}& """ + \
f"""
        {cyl_single_x_water.coords[-1, 0]:.2f} & {cyl_single_x_water.coords[-1, 1]:.2f} &
        {cyl_single_y_water.coords[-1, 0]:.2f} & {cyl_single_y_water.coords[-1, 1]:.2f} &
        {cyl_closed_loop_water.coords[-1, 0]:.2f} & {cyl_closed_loop_water.coords[-1, 1]:.2f}
""" + \
r"""
      \\ \hline
\textbf{True} & """ + \
f"""
        {cyl_measured_longest_dist_x:.2f} & {0:.2f} &
        {0:.2f} & {cyl_measured_shortest_dist_y:.2f} &
        {0:.2f} & {0:.2f}
""" + \
r"""
\end{tabular}
\label{table:results}
\end{table}
"""

