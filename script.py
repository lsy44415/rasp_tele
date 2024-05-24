import time
import threading
import os
import re
import RPi.GPIO as GPIO
import logging 
wi_fi = False
start_flag = False
GPIO.setmode(GPIO.BCM)
GPIO.setup(15,GPIO.OUT)
GPIO.setup(14,GPIO.OUT)

def test_wifi():
    global wi_fi
    a =os.popen('ifconfig wlan0 | egrep inet')
    logging.info("using command to check wifi")
    for data in a.readlines():
        if len(re.findall(r"inet \d+",data))>0:
            print("wifi works")
            wi_fi = True
            GPIO.output(15, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(15, GPIO.LOW)
            time.sleep(0.5)
            GPIO.output(15, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(15, GPIO.LOW)

def startprgm():
    global start_flag 
    time.sleep(3)
    logging.info("start run")
    
    while start_flag == False:
        print("start check wifi")
        test_wifi()
        if wi_fi == True:
            time.sleep(1)            
            GPIO.output(14, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(14, GPIO.LOW)
            time.sleep(1)
            GPIO.output(14, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(14, GPIO.LOW)
            
            logging.critical('Running: fastapi.py')
            os.system("sudo -H -u pi /usr/bin/python3 /home/pi/fastapi-mongo/app/main.py &")
            logging.critical('Running: telepathy.py')
            time.sleep(3)
            os.system("python3 /home/pi/telepathy/simple1.py &")
            start_flag = True
        else:
            continue


startprgm()
