import queue
import threading
import time
import cv2
from visual_encoder.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing
from visual_encoder.svd_decomposition import optimized_svd_method

global frame_queue, processed_frame_queue, displacement_queue, pause_threads
global last_two_frames, accumulated_displacement, start_time, end_time, total_frames

frame_queue = queue.Queue()
processed_frame_queue = queue.Queue()
displacement_queue = queue.Queue()
accumulated_displacement = [0, 0]
last_two_frames = [None, None]
total_frames = 0
pause_threads = False
start_time = 0

def get_frame(run_event):
    global start_time
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if start_time == 0:
        start_time = time.time()
    while ret and run_event.is_set():
        ret, frame = cap.read()
        largest_queue = max(frame_queue.qsize(), processed_frame_queue.qsize(), displacement_queue.qsize())
        if largest_queue == frame_queue.qsize() and frame_queue.qsize() < 3:
            frame_queue.put(frame)
        elif largest_queue == processed_frame_queue.qsize():
            frame_queue.put(frame)
        elif largest_queue == displacement_queue.qsize():
            frame_queue.put(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def preprocess_frame(run_event):
    while run_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            grayscale_frame = cv2_to_nparray_grayscale(frame)
            processed_frame = image_preprocessing(grayscale_frame)
            processed_frame_queue.put(processed_frame)
            global total_frames
            total_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def calculate_displacement(run_event):
    while run_event.is_set():
        if not processed_frame_queue.empty():
            processed_frame = processed_frame_queue.get()
            last_two_frames[0] = last_two_frames[1]
            last_two_frames[1] = processed_frame
            if last_two_frames[0] is not None and last_two_frames[1] is not None:
                M, N = last_two_frames[0].shape
                dx, dy = optimized_svd_method(last_two_frames[0], last_two_frames[1], M, N)
                displacement_array = [dx, dy]
                displacement_queue.put(displacement_array)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def send_displacement_array(run_event):
    while run_event.is_set():
        if not displacement_queue.empty():
            displacement_array = displacement_queue.get()
            accumulated_displacement[0] += displacement_array[0]
            accumulated_displacement[1] += displacement_array[1]
            print(accumulated_displacement)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def timed_output(name, delay, run_event):
    while run_event.is_set():
        time.sleep(delay)
        print(name,": New Message!")


def main():
    run_event = threading.Event()
    run_event.set()

    p1 = threading.Thread(target=get_frame, args=(run_event,))
    p2 = threading.Thread(target=preprocess_frame, args=(run_event,))
    p3 = threading.Thread(target=calculate_displacement, args=(run_event,))
    p4 = threading.Thread(target=send_displacement_array, args=(run_event,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    try:
        while 1:
            time.sleep(.1)
    except KeyboardInterrupt:
        print("attempting to close threads")
        run_event.clear()
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        print ("threads successfully closed")

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = total_frames / elapsed_time
        print("FPS:", fps)

if __name__ == '__main__':
    main()
