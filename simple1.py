import time
import os
import re
import logging
import csv
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations, WaveletTypes, NoiseEstimationLevelTypes,WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes, NoiseTypes
from statistics import mean
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

from scipy.stats import zscore
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import requests
import pickle
from threading import Thread

import json
from threading import Timer
import RPi.GPIO as GPIO


datalogging = False
board_id = None
params = None
board = None
master_board_id = None
sampling_rate = None
nfft = None
restfulness_params = None
restfulness = None
eeg_channels = None
data = []
GPIO.setmode(GPIO.BCM)
GPIO.setup(15,GPIO.OUT)
GPIO.setup(14,GPIO.OUT)
wi_fi = False
file_dir = '/home/pi/telepathy/log/'

def get_data(pid):
    url = "http://0.0.0.0:8000/bci/"
    full_url = url +str(pid)
    x = requests.get(full_url)

    result =x.json()

    #print(result['data'][0]['emo'])
    search_data = result['data'][0]
    return search_data


def stimulate_one(pid):

    data = get_data(pid)
    emo = data['emo']
    print('partner data is',data)
    print("their emo status: ",emo)
    if emo =='1':
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(14, GPIO.LOW)
        print("current pwm: 15 work - happy")
    elif emo == '-1':
        GPIO.output(15, GPIO.LOW)
        GPIO.output(14, GPIO.HIGH)
        print("current pwm: 14 work -sad")
    else:
        print("netural emotion")
        GPIO.output(15, GPIO.LOW)
        GPIO.output(14, GPIO.LOW)


def test_wifi():
    a =os.popen('ifconfig wlan0 | egrep inet')
    for data in a.readlines():
        if len(re.findall(r"inet \d+",data))>0:
            print("wifi works")
            wi_fi = True


def init_bci():
    global board_id
    global params
    global board
    global sampling_rate
    global nfft
    global restfulness_params
    global restfulness
    global eeg_channels


    BoardShim.enable_dev_board_logger()
    # turns on the loggers for additional debug output during dev
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()


    params = BrainFlowInputParams()
    params.serial_port = "/dev/ttyUSB0"
    #params.serial_port = "/dev/cu.usbserial-DM01MPXO"
    #params.serial_port = "/dev/cu.usbserial-DO015QAW"
    board_id = BoardIds.CYTON_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    board = BoardShim(board_id, params)

    print("board_descr", board_descr)
    # sampling_rate = BoardShim.get_sampling_rate(board_id)
    #sampling_rate = int(board_descr['sampling_rate'])
    sampling_rate = 250


    # Connect device to BCI and start streaming data
    logging.critical("Connecting to BCI")
    board.prepare_session()

    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start in the main thread')

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    eeg_channels = board_descr['eeg_channels']


def filter_signal(_data, _eeg_channels): # this is for cleaning the data

    for channel in _eeg_channels:
        # print("before",_data[channel])
        #0.1hz - 75hz bandpass
        DataFilter.perform_bandpass(_data[channel], 200, 0.3, 75, 3, FilterTypes.BESSEL.value, 0)
        # 50hz filter
        DataFilter.perform_bandstop(_data[channel], 200, 49, 51, 4, FilterTypes.BUTTERWORTH.value, 0)
        # try more Denoise methods !!!!
        try:
        # print("denoise",_data[channel])

        # DataFilter.perform_rolling_filter(_data[channel], 3, AggOperations.MEAN.value)
        # DataFilter.perform_rolling_filter(_data[channel], 3, AggOperations.MEDIAN.value)
            DataFilter.perform_wavelet_denoising(_data[channel], WaveletTypes.BIOR3_9, 3,
                                             WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                             WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)
        except:
            print("denoise error")
            update_data(_eeg_channels)
        # print("after",_data[channel])
        # print(_data)
    return _data



def update_data(_eeg_channels):
    
    global data
    global today
    data = []
    data = board.get_board_data()

    while len(data[_eeg_channels]) ==0:
        print("data is empty")
        print("get data again now........")
        time.sleep(1)
        data = board.get_board_data()
        time.sleep(1)
        # grabs the eeg data currently stored in the boardShim buffer and makes an array called "data
      # uses the filter signal function above to clean data

    if datalogging == True:
        _timestamps = []
        data_to_log = data[_eeg_channels]
        # print(data_to_log)
        for count in range(data_to_log.shape[1]):
            dt = datetime.now()
            ts = datetime.timestamp(dt)
            _timestamps.append([ts])

        timearray = np.array(_timestamps)
        np.reshape(timearray, [len(timearray), 1])
        timearray = np.transpose(timearray)

        # print(timearray)
        stamped_data = np.vstack((data_to_log, timearray))
        today = str(date.today())
        file_name = file_dir+today+'EEG_pid'+str(pid)+'.csv'
        #print("stamped_data ",stamped_data)
        with open(file_name,'a+') as f:
            write = csv.writer(f)
            write.writerows(data_to_log)
        #DataFilter.write_file(stamped_data, today + '-EEG_log.csv', 'w') # use 'a' for append mode
    return data

def get_band_power(_data, _eeg_channels,_emoSVM,current_dateTime):
    brain_wave = []
    error_flag=False

    channel_band_power = []
    for channel in _eeg_channels:
        try:
            # DataFilter.detrend(newsample_data, DetrendOperations.LINEAR.value)
            # # z-score normalise raw data
            # data_zscored = zscore(newsample_data)
            DataFilter.detrend(_data[channel], DetrendOperations.LINEAR.value)
            # z-score normalise raw data
            data_zscored = zscore(_data[channel])
            # clip between -3 and 3
            data_zscored = np.clip(data_zscored, -3, 3)
            # print('zscore', data_zscored)
            #psd = DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate, WindowOperations.BLACKMAN_HARRIS.value)
            #print(psd)
            # _psd = DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate,WindowOperations.HAMMING.value)

            # _psd = DataFilter.get_psd_welch(data_zscored, nfft, nfft // 2, sampling_rate,WindowOperations.HAMMING.value)

            #_psd = DataFilter.get_psd_welch(_data[channel], 512, 0, sampling_rate,WindowOperations.BLACKMAN_HARRIS.value)
            _psd = DataFilter.get_psd_welch(_data[channel], 256, 256 // 2, 250, WindowOperations.HAMMING.value)
            delta = DataFilter.get_band_power(_psd, 1, 4)
            theta = DataFilter.get_band_power(_psd, 5, 8)
            alpha = DataFilter.get_band_power(_psd, 9, 14)
            beta = DataFilter.get_band_power(_psd, 14.0, 30.0)
            gamma = DataFilter.get_band_power(_psd, 30.0, 50.0)

            channel_band_power = [delta,theta,alpha,beta,gamma]
            # print("channel_band_power: ",channel_band_power)
            # print("channel_band_power_zscore: ", zscore(channel_band_power))
            brain_wave.append(channel_band_power)
        except:
            print("get PSD error")
            return -9

            
    brain_wave = np.concatenate([brain_wave], axis=None)
    bands_array = np.array([brain_wave])
    arr = np.array(bands_array[0])
    arr = arr.reshape(1,40)
    file_time = file_dir+current_dateTime + 'pid_'+str(pid)+'.csv'
    with open(file_time,'a+') as f:
            write = csv.writer(f)
            write.writerows(arr)
    # print(bands_array)
    _valance = _emoSVM.predict(bands_array)
    return _valance


def get_mindfulness(_bands):
    #feature_vector = np.concatenate((_bands[0], _bands[1]))
    bands = DataFilter.get_avg_band_powers(_bands, eeg_channels, sampling_rate, True)

    feature_vector = np.concatenate((bands[0], bands[1]))
    # print("feature_vector",feature_vector)
    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    mind = mindfulness.predict(feature_vector)

    mindfulness.release()


    return mind

def get_bands(_data, _eeg_channels):

    return DataFilter.get_avg_band_powers(_data, _eeg_channels, sampling_rate, True)

def get_valance_bands(_data, _eeg_channels,_emoSVM):

    row_of_bands = []
    # headset channel order: t7,p3,f3,fp1,fp2,f4,t8,p4

    t7_bands = DataFilter.get_avg_band_powers(_data, [1], sampling_rate, True)
    p3_bands = DataFilter.get_avg_band_powers(_data, [2], sampling_rate, True)
    f3_bands = DataFilter.get_avg_band_powers(_data, [3], sampling_rate, True)
    fp1_bands = DataFilter.get_avg_band_powers(_data, [4], sampling_rate, True)
    fp2_bands = DataFilter.get_avg_band_powers(_data, [5], sampling_rate, True)
    f4_bands = DataFilter.get_avg_band_powers(_data, [6], sampling_rate, True)
    t8_bands = DataFilter.get_avg_band_powers(_data, [7], sampling_rate, True)
    p4_bands = DataFilter.get_avg_band_powers(_data, [8], sampling_rate, True)


    row_of_bands = [t7_bands[0], p3_bands[0], f3_bands[0], fp1_bands[0], fp2_bands[0], f4_bands[0], t8_bands[0], p4_bands[0]]
    #print("row_of_bands",row_of_bands)
    # print(row_of_bands)
    row_of_bands = np.concatenate([row_of_bands], axis=None)
    bands_array = np.array([row_of_bands])
    ave_bands = []
    for i in range(1,9):
        ave_bands.append(mean(_data[i]))
    # _valance= _emoSVM.predict(bands_array)
    ave_array = np.array([ave_bands])
    _valance = _emoSVM.predict(ave_array)
    return _valance

def get_emotion(_valance, _mindfulness):
    _emotion = 'null'


    # if _mindfulness >= 0.5 and _valance > 0:
        # _emotion = "Happy"
    # if _mindfulness >= 0.5 and _valance <0:
        # _emotion = "Angry"
    # if _mindfulness < 0.5 and _valance >0:
        # _emotion = "Calm"
    # if _mindfulness < 0.5 and _valance <0:
        # _emotion = "Sad"
    if _valance == 0:
        _emotion = "Neutral"
    if _valance == 1:
        _emotion = "Happy"
    if _valance == -1:
        _emotion = "Sad"
    return _emotion

def main(p1,p2):
    global pid
    global p2id
    pid = p1
    p2id = p2
    svm_class = pickle.load(open('/home/pi/telepathy/svm_only_seed_linear_2user.sav', 'rb'))
    #svm_class = pickle.load(open('svm_new_model.sav', 'rb'))
    #rf_class=pickle.load(open('rf_class.sav', 'rb'))
    # initialise BCI parameters
    # bci.init_bci("Synthetic") # Use this for synthetically generated test data
    init_bci()  # Use this when using the actual bci
    global datalogging
    datalogging= True
    count = 0
    current_dateTime = str(datetime.now())
    # MAIN LOOP #
    while True:
        board.start_stream()
        time.sleep(8)
        _data = update_data(eeg_channels)  # prompts the BCI to pull any new data from the data buffer
        board.stop_stream()
        _data = filter_signal(_data, eeg_channels)

        #bands = get_bands(data, eeg_channels)
        #mindfulness = get_mindfulness(_data)
        emotion = get_band_power(_data,eeg_channels,svm_class,current_dateTime)[0]
        # new_valance = get_valance_bands(_data,eeg_channels,valance_class)
        # print("new_valance",new_valance)
        # print("all band", bands)

        # valance = get_valance_bands(data, valance_class)
        #valance = get_valance_bands(data, new_class) # svm predict
        #valance = get_valance_bands(brain_wave, svm_class) #random forest predict

        #emotion = get_emotion(valance, mindfulness)
#        f = open(current_dateTime+"_emo.txt", "a")
#         f.writelines('mindfulness: ' + str(mindfulness) + "\n")
#         f.writelines('valance: '+str(valance)+"\n")
 #       f.writelines('emotion: ' + str(emotion) + "\n")
   #     f.writelines("\n")
        emo_name= file_dir+today+'_emo_pid'+str(pid)+'.csv'
        with open(emo_name,'a+') as f:
            write = csv.writer(f)
            write.writerow('emo:' + str(emotion))

        url = 'http://0.0.0.0:8000/bci/'
        bci_data = {
            "pid": str(pid),
            "emo":str(emotion),
            "time":str(datetime.now())
        }
        x = requests.post(url, json=bci_data)
       
        print("emotion: ", emotion)
        
        time.sleep(1)
        logging.critical("rest 1 second and then start 4 second stimulation")
        if count <220:
            stimulate_one(p2id)
            time.sleep(4)
        else:
            GPIO.cleanup()
            break
        count +=1
        GPIO.output(15, GPIO.LOW)
        GPIO.output(14, GPIO.LOW)
        logging.critical("stimulation stop, break for 2 second")
        time.sleep(2)
        print("current count is ",count)




        # print('Cooldown: %s' % str(cooldown))

        # # update cooldown
        # if cooldown > 0:
        #     cooldown -= 5
        #
        # print("mean alpha theta:")
        # print(mean_alpha_theta)
        #
        # if mean_alpha_theta > 1.29:
        #     threshold_summation += 1
        # else:
        #     threshold_summation = 0
        #
        # print("threshold_summation:")
        # print(threshold_summation)
        #
        # # if sleepiness == 0:
        # #     print("not sleepy")
        #
        # # if sleepiness == 1:
        # #     print("sleepy")
        #
        # # if user is sleepy and cooldown is not active, begin stimulation and activate cooldown
        # if threshold_summation == 3 and cooldown == 0:
        #     cooldown = 28800  # 28800 seconds = 480 minutes = 8 hours. Lets the user sleep
        #     # thread_2 = Thread(target=stimulation_thread)
        #     # thread_2.start()
        #     # thread_2.join



if __name__ == "__main__":
    # first number is own id, second one is partner's id
    main(11,12)
