import cv2

from visual_encoder.svd_decomposition import optimized_svd_method
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from visual_encoder.dsp_utils import image_preprocessing, cv2_to_nparray_grayscale

#img_width = 640
#img_height = 480
camera_id = 1 #Altere o id da câmera aqui
total_rec_time = 60 #seconds
escalaPyCamX = -0.02151467169232321
escalaPyCamY = 0.027715058926976663

# define a video capture object
print('Requesting access to camera. This may take a while...')
vid = cv2.VideoCapture(camera_id)
print('Got access to camera!')

# # Set resolution
# print('Setting camera resolution. This may take a while...')
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
#
# print('Resolution is set! Setting exposure...')
# vid.set(cv2.CAP_PROP_EXPOSURE, 10)
# print("Exposure is set!")

frame_num = 0 #para guardar o número de frames.

#variáveis para guardar o deslocamento
deltax = 0
deltay = 0

#variáveis para guardar o deslocamento total
total_deltax = 0
total_deltay = 0

plt.xlim(-10, 10)
plt.ylim(-10,10)
point = plt.scatter(0, 0, color="blue")

start_time = time.time()

while time.time() <= start_time + total_rec_time:
    ret, frame = vid.read()
    img_array = cv2_to_nparray_grayscale(frame)
    img_processed = image_preprocessing(img_array)
    if(frame_num == 0):
        M, N = img_array.shape
    elif frame_num == 10:
        start_time = time.time()
    elif frame_num > 10:
        deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N)
        total_deltax = total_deltax + (deltax*escalaPyCamX)
        total_deltay = total_deltay + (deltay*escalaPyCamY)
        point.remove()
        point = plt.scatter(total_deltax, total_deltay, color = "blue")
        plt.title('x:%.2f   y:%.2f' % (total_deltax,total_deltay))
        cv2.imshow('frame',frame)
        plt.pause(0.05)

    frame_num = frame_num + 1
    img_processed_old = img_processed

passed_time = (time.time() - start_time)
frame_num = frame_num - 10

print("--- %s seconds ---" % passed_time)
print("--- %s  frames ---" % frame_num)

fps = frame_num/passed_time
print("--- %s     fps ---" % fps)

vid.release()