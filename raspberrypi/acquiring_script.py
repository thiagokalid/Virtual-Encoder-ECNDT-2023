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
i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c, 0x29)
dataFile = open("images/quat_data.txt", "w")
dataFileEuler = open("images/eul_data.txt", "w")

class SplitFrames(io.BufferedIOBase):
    def __init__(self):
        self.frame_num = 0
        self.output = None
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; close the old one (if any) and
            # open a new output
            
            if self.output:
                self.output.close()
            self.frame_num += 1
            #Pega os valores dos sensores e o timestamp atual e salva como uma nova linha no txt
            timestamp = (time.time() - start) * 1000 # in ms
            dataFile.write(
            str(sensor.quaternion)+
            "\n")
            dataFileEuler.write(str(sensor.euler) + "\n")
            self.output = io.open(f'images/image{self.frame_num:02d}_t{timestamp:.1f}.jpg', 'wb')    
        self.output.write(buf)


camera = Picamera2()
camera.set_controls({"ExposureTime": 500})
camera.start_preview()
#camera.preview_configuration.controls.FrameRate = 100.0
#camera.preview_configuration.size = (640, 480)

fps = 60
period = 1/fps
frame_limit = 1/fps * 1e6 # in us

config = camera.create_preview_configuration(main = {"size" : (640, 480)})
camera.configure(config)
camera.set_controls({'FrameDurationLimits' : (int(frame_limit), 100000)})

# Give the camera some warm-up time
time.sleep(3)
output = SplitFrames()

encoder = JpegEncoder(q=70)
print(" Starting to record...")

camera.start_recording(encoder, FileOutput(output))
start = time.time()
time.sleep(15)
camera.stop_recording()
finish = time.time()
dataFile.close()
dataFileEuler.close()
print('Captured %d frames at %.2ffps' % (
output.frame_num,
output.frame_num / (finish - start)))
