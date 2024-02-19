import cv2
video_type = ["animacao_01"]
for i in range(len(video_type)):
  data_root = f"C:/Users/dsant/PycharmProjects/Virtual-Encoder-ECNDT-2023/raspberrypi/newCode/animacoes/"
  vidcap = cv2.VideoCapture(data_root + video_type[i]+".mp4")
  success, image = vidcap.read()
  count = 1


  while success:
    cv2.imwrite(data_root + f"/fotos/image{count:02d}_.jpg", image)
    success, image = vidcap.read()
    print('Saved image ', count)
    count += 1
