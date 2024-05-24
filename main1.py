import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations, WaveletTypes, NoiseEstimationLevelTypes,WaveletExtensionTypes, ThresholdTypes, WaveletDenoisingTypes, NoiseTypes

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
GPIO.setup(13,GPIO.OUT)
GPIO.setup(14,GPIO.OUT)




start_sti = False

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
    print('data is',data)
    print("current status: ",emo)
    if emo =='Happy':
            GPIO.output(13, GPIO.HIGH)
            GPIO.output(14, GPIO.LOW)
            print("current pwm: work")
        elif emo == 'Sad':
            GPIO.output(13, GPIO.LOW)
            GPIO.output(14, GPIO.HIGH)
            print("current pwm: not work")



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
            GPIO.output(14, GPIO.LOW)
            print("current pwm: work")
        elif emo == 'Sad':
            GPIO.output(13, GPIO.LOW)
            GPIO.output(14, GPIO.HIGH)
            print("current pwm: not work")
        time.sleep(8)
        count +=1
     
        if count ==30:
            start_sti = False
            GPIO.cleanup()
            print("stimulation stop")
    

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
    #params.serial_port = "/dev/cu.usbserial-DM01MPXO"
    params.serial_port = "/dev/ttyUSB0"
    board_id = BoardIds.CYTON_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    board = BoardShim(board_id, params)

    print("board_descr", board_descr)
    # sampling_rate = BoardShim.get_sampling_rate(board_id)
    #sampling_rate = int(board_descr['sampling_rate'])
    sampling_rate = 200
    print("sampling rate", sampling_rate)

    # Connect device to BCI and start streaming data
    print("Connecting to BCI")
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start in the main thread')

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    print(nfft)
    eeg_channels = board_descr['eeg_channels']


def filter_signal(_data, _eeg_channels): # this is for cleaning the data

    for channel in _eeg_channels:
        # print("before",_data[channel])
        #0.1hz - 75hz bandpass
        DataFilter.perform_bandpass(_data[channel], 200, 0.3, 125, 3, FilterTypes.BESSEL.value, 0)
        # 50hz filter
        DataFilter.perform_bandstop(_data[channel], 200, 49, 51, 4, FilterTypes.BUTTERWORTH.value, 0)
        # try more Denoise methods !!!!

        print("denoise",_data[channel])

        # DataFilter.perform_rolling_filter(_data[channel], 3, AggOperations.MEAN.value)
        # DataFilter.perform_rolling_filter(_data[channel], 3, AggOperations.MEDIAN.value)
        DataFilter.perform_wavelet_denoising(_data[channel], WaveletTypes.BIOR3_9, 3,
                                             WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                             WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)
        DataFilter.remove_environmental_noise(_data[channel], sampling_rate, NoiseTypes.FIFTY.value)

        print("after",_data[channel])
    return _data



def update_data(_eeg_channels):
    global data
    data = []
    data = board.get_board_data()
    if len(data) == 0:
        print("data is empty")
        init_bci()
        data = board.get_board_data()
        # grabs the eeg data currently stored in the boardShim buffer and makes an array called "data
    data = filter_signal(data, _eeg_channels)  # uses the filter signal function above to clean data

    if datalogging == True:
        _timestamps = []
        data_to_log = data[_eeg_channels]
        for count in range(data_to_log.shape[1]):
            dt = datetime.now()
            ts = datetime.timestamp(dt)
            _timestamps.append([ts])

        timearray = np.array(_timestamps)
        np.reshape(timearray, [len(timearray), 1])
        timearray = np.transpose(timearray)
        stamped_data = np.vstack((data_to_log, timearray))
        today = str(date.today())

        DataFilter.write_file(stamped_data, today + '-EEG_log.csv', 'a') # use 'a' for append mode
    return data

def get_band_power(_data, _eeg_channels,_emoSVM):
    brain_wave = []
    channel_band_power = []
    for channel in _eeg_channels:
        DataFilter.detrend(_data[channel], DetrendOperations.LINEAR.value)
        # z-score normalise raw data
        data_zscored = zscore(_data[channel])
        # clip between -3 and 3
        data_zscored = np.clip(data_zscored, -3, 3)
        print("zscored",data_zscored)
        # print(_data[channel])
        #psd = DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate, WindowOperations.BLACKMAN_HARRIS.value)
        #print(psd)
        # _psd = DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate,WindowOperations.HAMMING.value)

        _psd = DataFilter.get_psd_welch(data_zscored, nfft, nfft // 2, sampling_rate,
                                        WindowOperations.HAMMING.value)

        delta = DataFilter.get_band_power(_psd, 1, 4)
        theta = DataFilter.get_band_power(_psd, 4, 8)
        alpha = DataFilter.get_band_power(_psd, 8, 14)
        beta = DataFilter.get_band_power(_psd, 14.0, 30.0)
        gamma = DataFilter.get_band_power(_psd, 30.0, 50.0)

        channel_band_power = [delta,theta,alpha,beta,gamma]
        # print("channel_band_power: ",channel_band_power)
        brain_wave.append(channel_band_power)


    brain_wave = np.concatenate([brain_wave], axis=None)
    bands_array = np.array([brain_wave])

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

    # restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
    #                                           BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    # restfulness = MLModel(restfulness_params)
    # restfulness.prepare()
    # print('Restfulness: %s' % str(restfulness.predict(feature_vector)))
    # restfulness.release()

    return mind

def get_bands(_data, _eeg_channels):

    return DataFilter.get_avg_band_powers(_data, _eeg_channels, sampling_rate, True)

def get_valance_bands(_data, _emoSVM):

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

    # print(row_of_bands)
    row_of_bands = np.concatenate([row_of_bands], axis=None)
    bands_array = np.array([row_of_bands])

    _valance= _emoSVM.predict(bands_array)

    return _valance

def get_emotion(_valance, _mindfulness):
    _emotion = 'null'


    if _mindfulness >= 0.5 and _valance > 0:
        _emotion = "Happy"
    if _mindfulness >= 0.5 and _valance <0:
        _emotion = "Angry"
    if _mindfulness < 0.5 and _valance >0:
        _emotion = "Calm"
    if _mindfulness < 0.5 and _valance <0:
        _emotion = "Sad"
    if _valance == 0:
        _emotion = "Neutral"
    return _emotion

def main():
    pid = 4
    p2id = 4
    threshold_summation = 0
    cooldown = 0
    svm_class = pickle.load(open('/home/pi/telepathy/svm_class.sav', 'rb'))
    #svm_class = pickle.load(open('svm_class.sav', 'rb'))
    #rf_class=pickle.load(open('rf_class.sav', 'rb'))
    # initialise BCI parameters
    # bci.init_bci("Synthetic") # Use this for synthetically generated test data
    init_bci()  # Use this when using the actual bci
    global datalogging
    datalogging= True
    #current_dateTime = str(datetime.now())
    # MAIN LOOP #
    count = 0
    while True:

        time.sleep(8)
        _data = update_data(eeg_channels)  # prompts the BCI to pull any new data from the data buffer


        #bands = get_bands(data, eeg_channels)
        mindfulness = get_mindfulness(_data)
        valance = get_band_power(_data,eeg_channels,svm_class)

        # print("all band", bands)

        # valance = get_valance_bands(data, valance_class)
        #valance = get_valance_bands(data, new_class) # svm predict
        #valance = get_valance_bands(brain_wave, svm_class) #random forest predict

        emotion = get_emotion(valance, mindfulness)
        # f = open(current_dateTime+".txt", "w")
        # f.writelines(emotion)
        url = 'http://0.0.0.0:8000/bci/'
        bci_data = {
            "pid": str(pid),
            "emo":str(emotion),
            "time":str(datetime.now())
        }
        x = requests.post(url, json=bci_data)
        print("mindfulness: ", mindfulness)
        print("valance: ", valance)
        print("emotion: ", emotion)
        if count <20:
            stimulate_one(pid)
        else:
            GPIO.cleanup()
            break
        count +=1
        print(count)


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
    main()
