import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
plt.xlim(-10, 10)
plt.ylim(-10,10)

for i in range(10):
    y = np.random.random()
    if i > 0:
        point.remove()
    point = plt.scatter(i, y,color = 'blue')
    plt.pause(0.05)

plt.show()