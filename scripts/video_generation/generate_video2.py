# Import visual_encoder packages:
from visual_encoder.trajectory_estimators import get_img

# Import external packages:
import matplotlib.pyplot as plt

# Where to get the data:
geometries = ["planar", "cylindrical"]
media = ["air", "water"]
excurtion_types = ["closed_loop", "single_x", "single_y"]
data_root = f"../../data/{geometries[0]}/{media[0]}_{excurtion_types[0]}/"
media_root = f"../../media/{geometries[0]}/{media[0]}/{excurtion_types[0]}_video/"

# Creating ploting window:
fig = plt.figure(figsize=(9, 9))
generate_video = True

shot = 1
plt.subplot(2, 2, 1)
plt.title("Raspberry pi camera")
img = get_img(shot, data_root)
plt.imshow(img, cmap='gray')

plt.subplot(2, 2, 2)
ax = plt.axes(projection="3d")
ax.scatter(1, 1, 1)

plt.subplot(2, 2, 3)
plt.title("Recording")
photo = get_img(shot, media_root, rgb=True)
plt.imshow(photo)