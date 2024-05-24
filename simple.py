import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations,FilterTypes
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

from scipy.stats import zscore
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import requests

import pickle
from threading import Thread




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
pid = 1

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
    #params.serial_port = "/dev/cu.usbserial-DO015QAW"
    
    params.serial_port = "/dev/ttyUSB0"
    board_id = BoardIds.CYTON_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    board = BoardShim(board_id, params)

    print("board_descr", board_descr)
    # sampling_rate = BoardShim.get_sampling_rate(board_id)
    #sampling_rate = int(board_descr['sampling_rate'])
    sampling_rate = 200
    print("sampling id", sampling_rate)

    # Connect device to BCI and start streaming data
    print("Connecting to BCI")
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start in the main thread')

    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    eeg_channels = board_descr['eeg_channels']

def filter_signal(_data, _eeg_channels): # this is for cleaning the data
    for channel in _eeg_channels:
        #0.1hz - 75hz bandpass
        DataFilter.perform_bandpass(_data[channel], 200, 0.1, 75, 3, FilterTypes.BESSEL.value, 0)
        # 50hz filter
        DataFilter.perform_bandstop(_data[channel], 200, 49, 51, 4, FilterTypes.BUTTERWORTH.value, 0)
        # Denoise
        # DataFilter.perform_wavelet_denoising(_data[channel], 'coif3', 3)

    return _data



def update_data(_eeg_channels):
    global data
    data = []
    data = board.get_board_data()  # grabs the eeg data currently stored in the boardShim buffer and makes an array called "data
    if len(data) ==0:
        print("data is empty")
        init_bci()
        data = board.get_board_data() 
    
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
        DataFilter.write_file(stamped_data, today + '-EEG_log.csv', 'a')


def get_band_power(_data, _eeg_channels):
    _alpha_theta_array = []

    for channel in _eeg_channels:
        # z-score normalise raw data
        data_zscored = zscore(_data[channel])
        # clip between -3 and 3
        data_zscored = np.clip(data_zscored, -3, 3)

        _psd = DataFilter.get_psd_welch(data_zscored, nfft, nfft // 2, sampling_rate,
                                        WindowOperations.BLACKMAN_HARRIS.value)

        alpha = DataFilter.get_band_power(_psd, 8, 12)
        theta = DataFilter.get_band_power(_psd, 4, 8)
        delta = DataFilter.get_band_power(_psd, 0.5, 4.0)

        beta = DataFilter.get_band_power(_psd, 12.0, 35.0)
        gamma = DataFilter.get_band_power(_psd, 35.0, 45.0)

        _ratio = alpha / theta

        _alpha_theta_array.append(_ratio)

    _relative_alpha_theta_df = pd.DataFrame([_alpha_theta_array])
    return _relative_alpha_theta_df



def get_mindfulness(_bands):
    #feature_vector = np.concatenate((_bands[0], _bands[1]))
    feature_vector = _bands[0]
    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    mind = mindfulness.predict(feature_vector)
    # print('Mindfulness: %s' % str(mind))
    mindfulness.release()

    restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    # print('Restfulness: %s' % str(restfulness.predict(feature_vector)))
    restfulness.release()

    return mind

def get_bands(_data, _eeg_channels):
    return DataFilter.get_avg_band_powers(_data, _eeg_channels, sampling_rate, True)

def get_valance_bands(_data, _emoSVM):

    row_of_bands = []

    c1_bands = DataFilter.get_avg_band_powers(_data, [1], sampling_rate, True)
    c2_bands = DataFilter.get_avg_band_powers(_data, [2], sampling_rate, True)
    c3_bands = DataFilter.get_avg_band_powers(_data, [3], sampling_rate, True)
    c4_bands = DataFilter.get_avg_band_powers(_data, [4], sampling_rate, True)
    c5_bands = DataFilter.get_avg_band_powers(_data, [5], sampling_rate, True)
    c6_bands = DataFilter.get_avg_band_powers(_data, [6], sampling_rate, True)
    c7_bands = DataFilter.get_avg_band_powers(_data, [7], sampling_rate, True)
    c8_bands = DataFilter.get_avg_band_powers(_data, [8], sampling_rate, True)

    row_of_bands = [c1_bands[0], c2_bands[0], c3_bands[0], c4_bands[0], c5_bands[0], c6_bands[0], c7_bands[0], c8_bands[0]]

    row_of_bands = np.concatenate([row_of_bands], axis=None)
    bands_array = np.array([row_of_bands])

    _valance= _emoSVM.predict(bands_array)

    return _valance

def get_emotion(_valance, _mindfulness):
    _emotion = 'null'
    if _mindfulness >= 0.5:
        _arousal = "High"
    if _mindfulness < 0.5:
        _arousal = "Low"

    if _arousal == "High" and _valance == 1:
        _emotion = "Happy"
    if _arousal == "High" and _valance == -1:
        _emotion = "Angry"
    if _arousal == "Low" and _valance == 1:
        _emotion = "Calm"
    if _arousal == "Low" and _valance == -1:
        _emotion = "Sad"
    return _emotion

def main():
    threshold_summation = 0
    cooldown = 0
    

    new_class = pickle.load(open('/home/pi/telepathy/svm_class.sav', 'rb'))
    time.sleep(10)
    # initialise BCI parameters
    init_bci()  # Use this when using the actual bci
    datalogging = True

    # MAIN LOOP #
    while True:

        time.sleep(10)
        update_data(eeg_channels)  # prompts the BCI to pull any new data from the data buffer



        bands = get_bands(data, eeg_channels)
        # print("all band", bands)
        mindfulness = get_mindfulness(bands)
        # valance = get_valance_bands(data, valance_class)
        valance = get_valance_bands(data, new_class)
        emotion = get_emotion(valance, mindfulness)
        url = 'http://0.0.0.0:8000/bci/'
        current_time = str(datetime.now())
        bci_data = {
            "pid": str(pid),
            "emo":str(emotion),
            "time":current_time
        }
        x = requests.post(url, json=bci_data)
        print("mindfulness: ", mindfulness)
        print("valance: ", valance)
        print("emotion: ", emotion)
        print("time: ",current_time)




if __name__ == "__main__":
    main()
