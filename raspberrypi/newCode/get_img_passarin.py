# import the opencv library
import cv2

# define a video capture object
print('Requesting access to camera. This may take a while...')
vid = cv2.VideoCapture(2)
print('Got access to camera!')

# Set resolution
print('Setting camera resolution. This may take a while...')
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Seems like this is not necessary
print('Resolution is set!')

i = 0
for exposure in range(0, 10, 1):

    # Set exposure
    vid.set(cv2.CAP_PROP_EXPOSURE, exposure)

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(exposure), (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    #cv2.imshow('frame', frame)
    cv2.imwrite(str(i).zfill(5) + '.jpg', frame)

    print('Recorded image ' + str(i))

    i += 1

print('End.')
# After the loop release the cap object
vid.release()