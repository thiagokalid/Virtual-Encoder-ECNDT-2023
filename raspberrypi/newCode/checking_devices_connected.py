import cv2

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        cap, img = cap.read()
        if cap:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

print(returnCameraIndexes())