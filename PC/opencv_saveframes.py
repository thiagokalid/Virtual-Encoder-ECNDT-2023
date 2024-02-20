import cv2
import time
import shutil

camera_id = 1
total_rec_time = 60

# Define a video capture object
print('Requesting access to camera. This may take a while...')
vid = cv2.VideoCapture(camera_id)
print('Got access to camera!')


# Criar uma pasta para salvar os frames
import os

if os.path.exists('frames'):
    shutil.rmtree('frames')
else:
    os.makedirs('frames')

frame_num = 0
start_time = time.time()

while time.time() <= start_time + total_rec_time:
    ret, frame = vid.read()
    frame_num += 1

    # Salvando o frame como uma imagem
    cv2.imwrite(f'frames/frame_{frame_num}.jpg', frame)

    passed_time = (time.time() - start_time)

print("--- %s seconds ---" % passed_time)
print("--- %s  frames ---" % frame_num)

fps = frame_num / passed_time
print("--- %s     fps ---" % fps)

vid.release()
