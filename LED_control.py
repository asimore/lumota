"""This application is to control APA 102 LEDs on the device"""

import socket
import time
import sys
import threading

sys.path.append('/home/pi/projects/lumin/Lumin_FW_Src/audio_application/python/src')
from pixels import pixels

timer_alive = 0

#Socket  initialization for messages from other device applications

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)
sock.bind(server_address)
sock.listen(1)
connection, client_address = sock.accept()

#LED initial pattern

pixels.on()
#pixels.think()

#LED control function to go back to initial state, when the timer ends

def timer_end():
        pixels.off()
        time.sleep(1)
        pixels.on()
        #pixels.think()
        print("Timer end")
        global timer_alive
        timer_alive = 0

#LED control function on BLE connection message from BLE Gatt Server

def on_BLE():

	global timer_alive
	if(timer_alive == 1):
		return 
	pixels.off()
	time.sleep(1)
	pixels.on()
	#pixels.speak()
	print("Timer start")
	timer = threading.Timer(30.0,timer_end)
	timer.start()
	timer_alive = 1


while True:

    data = connection.recv(64)
    if(data == b'connect'):
    	on_BLE()

sock.close()
