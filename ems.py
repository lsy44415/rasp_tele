import requests
import json
from threading import Timer
import RPi.GPIO as GPIO
import time 


GPIO.setmode(GPIO.BCM)
GPIO.setup(13,GPIO.OUT)




start_sti = False

def get_data(pid):
    url = "http://0.0.0.0:8000/bci/"
    full_url = url +str(pid)
    x = requests.get(full_url)

    result =x.json()

    #print(result['data'][0]['emo'])
    search_data = result['data'][0]
    return search_data

def stimulate(pid):
    global start_sti
    start_sti = True
    count = 0
    
    while start_sti:
        data = get_data(pid)
        emo = data['emo']
        print('data is',data)
        print("current status: ",emo)
        if emo =='Happy':
            GPIO.output(13, GPIO.HIGH)
            print("current pwm: work")
        elif emo == 'Sad':
            GPIO.output(13, GPIO.LOW)
            print("current pwm: not work")
        time.sleep(10)
        count +=1
     
        if count ==10:
            start_sti = False
            GPIO.cleanup()
            print("stimulation stop")
    

stimulate(1)
#print(result['data'][0]['time'])
