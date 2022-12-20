import cv2
video_type = ["calibration_x", "calibration_y", "single_x", "single_y", "closed_loop"]
for i in range(len(video_type)):
  data_root = f"../../media/planar/water/{video_type[i]}"
  vidcap = cv2.VideoCapture(data_root + ".mp4")
  success, image = vidcap.read()
  count = 1


  while success:
    cv2.imwrite(data_root + "_video/image%d.jpg" % count, image)
    success, image = vidcap.read()
    print('Saved image ', count)
    count += 1
