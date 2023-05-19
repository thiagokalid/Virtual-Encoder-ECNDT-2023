import RPi.GPIO as gpio
import time

Encoder1PinPhaseA = 5
Encoder1PinPhaseB = 6

Encoder2PinPhaseA = 13
Encoder2PinPhaseB = 19

Encoder3PinPhaseA = 26
Encoder3PinPhaseB = 21

encoderPrimary = [Encoder1PinPhaseA,Encoder2PinPhaseA,Encoder3PinPhaseA]
encoderSecondary = [Encoder1PinPhaseB,Encoder2PinPhaseB,Encoder3PinPhaseB]

arrayUtilitario = [[gpio.HIGH,gpio.HIGH],[gpio.HIGH,gpio.LOW],[gpio.LOW,gpio.LOW],[gpio.LOW,gpio.HIGH]]

gpio.setmode(gpio.BCM)
for i in range(3):
    gpio.setup(encoderPrimary[i], gpio.OUT)
    gpio.setup(encoderSecondary[i], gpio.OUT)

arrayOfCoordinates = [10000, 10000, 10000]
multiplicador = [0,0,0]

for i in range(3):
    if arrayOfCoordinates[i] > 0:
	    multiplicador[i] = 1
    elif arrayOfCoordinates[i] < 0:
	    multiplicador[i] = -1

while multiplicador != 0:
    arrayOfCoordinates = arrayOfCoordinates - multiplicador
    contador = (contador - multiplicador)%4
    gpio.output(Encoder3PinPhaseA, arrayUtilitario[contador][0])
    gpio.output(Encoder3PinPhaseB, arrayUtilitario[contador][1])
    gpio.output(Encoder2PinPhaseA, arrayUtilitario[contador][0])
    gpio.output(Encoder2PinPhaseB, arrayUtilitario[contador][1])
    gpio.output(Encoder1PinPhaseA, arrayUtilitario[contador][0])
    gpio.output(Encoder1PinPhaseB, arrayUtilitario[contador][1])

    time.sleep(0.001)
    if(arrayOfCoordinates == 0):
        multiplicador = 0
        exit
