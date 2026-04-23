import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import islice
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from tqdm import tqdm

# parse the OpenSim .mot file
def parse_motion_file(input_file_path):
    # get header
    input_file = open(input_file_path)
    header = "".join(list(islice(input_file, 10)))
    input_file.close()

    # get data in dataframe
    dataframe = pd.read_csv(filepath_or_buffer=input_file_path, skipinitialspace=True, sep='\t', header=8, engine="python", dtype=np.float64)
    # display(dataframe)

    time = dataframe['time']
    dataframe = dataframe.drop(labels='time', axis=1)

    return header, dataframe, time

# read IMU data file and parse it into a header and dataframe
def parse_IMU_file(input_file_path):
    # get header
    input_file = open(input_file_path)
    header = "".join(list(islice(input_file, 4)))
    input_file.close()

    # get data in dataframe
    dataframe = pd.read_csv(filepath_or_buffer=input_file_path, skipinitialspace=True, sep='\t', header=4, engine="python", dtype="str")
    # display(dataframe)
    display(dataframe)

# create a .mot file using a header and modified dataframe
def write_motion_file(header, dataframe, time, new_file_name):

    # insert header
    new_file_path = "/Users/FranklinZhao/OpenSimProject/Dataset/Motion/Augmented/" + new_file_name + ".mot"
    new_file = open(new_file_path, "a")
    new_file.write(header)
    new_file.close()

    # insert dataframe
    dataframe.insert(loc=0, column = 'time', value = time)
    dataframe.to_csv(path_or_buf=new_file_path, sep='\t', header=True, mode="a", index=False)


# lowpass filter
def lowpass_filter(data, fs, fc, order, axis):
    w = fc / (fs / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, data, axis=axis)
    return output

# plot perturbed curves and original curve for each feature
# exports as pdf to filename path
def plotMulvariateCurves(filename, dataset, original_data, time, feature_headers):
    num_features = dataset.shape[2]
    num_generated = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in tqdm(range(num_features)):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(num_generated):
                ax.plot(time, (dataset[j][:, i]))
                ax.plot(time, (original_data[:, i]), color="black")
            ax.set_xlabel("time")
            ax.set_ylabel("Kinematics values")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()

# plot the splines of each perturbed curve for each feature
# will be exported as pdf to filename path
def plotMulvariateSplines(filename, dataset, time, time_points, distortion, feature_headers):
    num_features = dataset.shape[0]

    with PdfPages(filename) as pdf:
        for i in tqdm(range(num_features)):
            fig, ax = plt.subplots(figsize=(8, 6))
            for j in range(dataset.shape[1]):
                ax.plot(time, (dataset[i][j, :]))
                ax.scatter(time_points[i][j, :], distortion[i][j, :], color='red')
            ax.set_xlabel("time")
            ax.set_title(feature_headers[i])
            pdf.savefig(fig)
            plt.close(fig)
            
        plt.tight_layout()

# convert a signal to the 360 degree convention
def convertTo360Degrees(signal):
    signal = [math.radians(deg) for deg in signal]
    signal = [math.cos(rad) for rad in signal]
    signal = [math.acos(rad) for rad in signal]
    signal = [math.degrees(rad) for rad in signal]
    return signal