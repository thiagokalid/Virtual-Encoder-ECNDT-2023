import io
import time
import threading
from picamera2 import Picamera2
from picamera2.outputs import FileOutput
from picamera2.encoders import JpegEncoder

import logging
import sys
import board
import adafruit_bno055

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import fft2, fftshift
from visual_encoder.svd_decomposition import svd_method
from visual_encoder.displacement_params import DisplacementParams
from visual_encoder.tajectory_params import TrajectoryParams, get_img
from io import BytesIO
import serial
import serial.threaded
import queue
# O que cada parâmetro representa?
svd_param = DisplacementParams(method="svd", spatial_window='Blackman-Harris', frequency_window="Stone_et_al_2001")
svd_traj = TrajectoryParams(svd_param)
x0, y0, z0 = (0, 0, 0)
curr_img = 0
prev_img = 0
fps = 15
period = 1/fps
tot_time = 30
frame_limit = 1/fps * 1e6 # in us
img_width = 640
img_height = 480

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        print(self)
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)     #inicia steam de video?
                    frame_num = self.owner.frame_num    #atribui um número para o frame
                    self.owner.frame_num += 1
                    curr_img = np.asarray(ImageOps.grayscale(Image.open(self.stream))) # Lê e guarda foto atual
                    self.owner.img_buffer[frame_num, :, :] = curr_img       #salva imagens
                    #self.owner.fft_buffer[frame_num, :, :] = fft2(curr_img)
                    if self.owner.frame_num > 10:       #descarta primeiros frames
                        #F = self.owner.fft_buffer[self.owner.frame_num, :, :]          #previa de buffer de fft
                        #G = self.owner.fft_buffer[self.owner.frame_num - 1, :, :]      #a idéia é ter treads para só
                        f = self.owner.img_buffer[frame_num, :, :]
                        g = self.owner.img_buffer[frame_num-1, :, :]
                        deltax, deltay = svd_method(f, g, downsampling_factor=2)
                        deltax = deltax * 0.02151467169232321       #parametros de calibração
                        deltay = deltay * 0.027715058926976663      #parametros de calibração
                        self.owner.x_coord += deltax
                        self.owner.y_coord += deltay
                        coord = [deltax * 1000, deltay * 1000, 0.]          #use priority queue?
                        #self.owner.q#.put(coord)
                        message = (str(coord)+"\n").encode('utf-8')
                        with self.owner.serial_lock:
                            self.owner.ser.write(message)
                        print(f"coord inside thread= ({coord[0]:>3.2f}, {coord[1]:>3.2f}, {coord[2]:>3.2f})")
                        #print(f"delta = ({deltax:>20.2f}, {deltay:>20.2f}), acumulado = ({self.owner.x_coord:>10.2f}, {self.owner.y_coord:>10.2f})")
                    elif self.owner.frame_num >= fps * tot_time:
                        self.terminated = True
                    else:
                        print(frame_num) # Ignore the first few frames
                 

                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)                 

class ProcessOutput(io.BufferedIOBase):
    def __init__(self):
        self.done = False            
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(3)]
        self.processor = None
        self.frame_num = 0
        self.img_buffer = np.zeros(shape=(fps * tot_time, img_height, img_width))
        self.fft_buffer = np.zeros(shape=(fps * tot_time, img_height, img_width), dtype=complex)
        self.x_coord = 0.
        self.y_coord = 0.
        self.acumulador = np.zeros(3)


        #self.protocol = serial.threaded.ReaderThread(self.ser, PrintLines)

        #self.q = queue.Queue()
        time.sleep(0.1)
        
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            #print(self.processor)
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
                
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    print("--------------------- All threads are busy. ----------------------------")
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def terminate(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
                print("Inside Lock")
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    break # pool is empty
            proc.terminated = True
            proc.join()
        time.sleep(1)
        #print(self.ser.in_waiting)
        # print(self.ser.out_waiting)
        time.sleep(1)
            

camera = Picamera2()
camera.set_controls({"ExposureTime": 500})
camera.start_preview()
#camera.preview_configuration.controls.FrameRate = 100.0
#camera.preview_configuration.size = (640, 480)



config = camera.create_preview_configuration(main = {"size" : (img_width, img_height)})
camera.configure(config)
camera.set_controls({'FrameDurationLimits' : (int(frame_limit), 100000)})

#worker_thread.start()

# Give the camera some warm-up time
time.sleep(3)
output = ProcessOutput()

encoder = JpegEncoder(q=70)
print(" Starting to record...")

file_like_object = FileOutput(output)
camera.start_recording(encoder, file_like_object)
start = time.time()
time.sleep(tot_time)
print("recording...")
camera.stop_recording()
finish = time.time()
print('Captured %d frames at %.2ffps' % (
output.frame_num,
output.frame_num / (finish - start)))
time.sleep(1)
output.terminate()
# Ending serial commmunication
print("End")
print("Queue fechada")
