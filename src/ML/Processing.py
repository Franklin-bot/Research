import random
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import numpy as np


def augment_temporal_minus_one(data):
    r = data.shape[0]
    c = data.shape[1]

    res = np.zeros((r-1, c*2))
    for j in range(c):
        res[:, j*2] = data[1:, j]
        res[:, j*2+1] = data[0:r-1, j]
    
    return res

def augment_temporal_plus_one(data):
    r = data.shape[0]
    c = data.shape[1]

    res = np.zeros((r-1, c*2))
    for j in range(c):
        res[:, j*2] = data[0:r-1, j]
        res[:, j*2+1] = data[1:, j]
    return res

def mask(data):
    r = data.shape[0]
    c = data.shape[1]
    res = np.copy(data)

    for j in range(0, 38, 2):
        res[:, j] = -2
    return np.vstack((data, res))

def removeZeros(data):
    return np.delete(data, [0, 16], axis=1)

def scaleDataInstance(data):
    for i in range(data.shape[1]):
        if max(abs(data[:, i])) != 0:
            data[:, i] = data[:, i] / max(abs(data[:, i]))
    return data

# get random index for IMU sensor
def sampleSensor(sensorName):
    index = random.randint(0, 49)
    return sensorName if index == 0 else sensorName + str(index)

def lowpass_filter(data, cutoff=10, fs=200, order=4):

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs  # Nyquist frequency
        normal_cutoff = cutoff / nyq  # Normalize the cutoff frequency
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Apply the filter to each column of the data
    for i in range(data.shape[1]):
        data[:, i] = butter_lowpass_filter(data[:, i].copy(), cutoff, fs, order)
    return data
    

