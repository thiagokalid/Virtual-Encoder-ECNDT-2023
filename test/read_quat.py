from visual_encoder.trajectory_estimators import get_quat_data, convert_to_3d
import numpy as np
import matplotlib.pyplot as plt

data_root = "/home/tekalid/Downloads/images"
quat_data = get_quat_data(data_root, n=2744)
span = np.linspace(0, 10, 2744)

coords_2d = np.array([span, np.zeros_like(span), np.zeros_like(span)]).transpose()
coords_3d = convert_to_3d(coords_2d, quat_data)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], 'o')