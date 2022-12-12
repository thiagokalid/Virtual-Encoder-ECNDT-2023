from visual_encoder.phase_correlation import *
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams

medium = "air"
filename = [f"{medium}_planar_{name}_side" for name in ['shortest', 'longest']]
data_root = "../data/calibration/"

print("Starting the SVD based estimation method...")

measured_longest_dist_x = np.mean([359, 358, 357])  # distance in millimeters
measured_shortest_dist_y = np.mean([200.1, 200, 200.9])  # distance in millimeters
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001")
svd_traj = TrajectoryParams(svd_param)
y_res, x_res = svd_traj.calibrate(data_root,
                                  filename_list=filename,
                                  measured_coords=[measured_shortest_dist_y, measured_longest_dist_x],
                                  n_images=[1494, 1994]
                                  )
# Result: x_res, y_res = (0.021592538251603233, 0.02780955452804545)