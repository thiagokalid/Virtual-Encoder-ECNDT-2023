import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from visual_encoder.trajectory_estimators import get_img
from visual_encoder.tajectory_params import TrajectoryParams
from visual_encoder.displacement_params import DisplacementParams

# Where to get the data:
geometries = ["planar", "cylindrical"]
media = ["air", "water"]
excurtion_types = ["closed_loop", "single_x", "single_y"]
chosen_type = 0
data_root = f"../../data/{geometries[0]}/{media[0]}_{excurtion_types[chosen_type]}/"
media_root = f"../../media/{geometries[0]}/{media[0]}/{excurtion_types[chosen_type]}_video/"

# General SVD related parameters:
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001",
                               xy_resolution=(0.02151467169232321, 0.027715058926976663), resolution_unit="mm/pixels",
                               rotation_correction=-0.034107828176162376)

# Medium: Air (Contact)
# Geometry: Planar
# Test type: Closed-loop case:
print("Medium: Air. Beginning of closed-loop...")
svd_traj = TrajectoryParams(svd_param)
svd_traj.compute_total_trajectory_path(data_root, n_images=1496)
print("End of closed-loop...")

# Parâmetros de geração de vídeo:
results_path = "../../Virtual-Encoder-ECNDT-2023/results/video/"
videotitle = "Video_v01"
metadata = dict(title=results_path + videotitle, artist='Matplotlib',
                comment='Movie support!')

writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure(figsize=(9, 9))
generate_video = False
shot = 1

print("Início do loop de geração de vídeo...")
if generate_video == True:
    # with writer.saving(fig, result_foldername + "/" + video_title + ".mp4", dpi=300):
    with writer.saving(fig, results_path + videotitle + ".mp4", dpi=300):
        for shot in range(2, svd_traj.get_coords().shape[0]):
            plt.suptitle(f"Shot = {shot}")
            plt.subplot(2, 2, 1)
            curr_img = get_img(shot, data_root)
            plt.imshow(curr_img[::-1, :], cmap='gray',
                       extent=[svd_traj.get_coords(shot)[0],  # Esquerda X
                               svd_traj.get_coords(shot)[0] + curr_img.shape[1],  # Direita X
                               svd_traj.get_coords(shot)[1],  # Topo Y
                               svd_traj.get_coords(shot)[1] + curr_img.shape[0]]  # Base Y
                       )
            plt.xlim([svd_traj.get_coords(shot)[0], svd_traj.get_coords(shot)[0] + curr_img.shape[1]])
            plt.ylim([svd_traj.get_coords(shot)[1], svd_traj.get_coords(shot)[1] + curr_img.shape[0]])
            offset_y = curr_img.shape[1] / 2
            offset_x = curr_img.shape[1] / 2

            plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                     svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                     ':o',
                     color='r', label='SVD')
            plt.legend()

            print(f"frame = {shot + 1}")

            plt.subplot(2, 2, 2)
            plt.plot(svd_traj.get_coords()[:shot + 1][:, 0] + offset_x,
                     svd_traj.get_coords()[:shot + 1][:, 1] + offset_y,
                     ':o',
                     color='r', label='SVD')

            plt.subplot(2, 2, 4)
            plt.title("Recording")
            photo = get_img(shot, media_root, rgb=True)
            plt.imshow(photo)
            plt.axis(False)

            writer.grab_frame()
            plt.clf()
    print("Fim do loop de geração de vídeo.")