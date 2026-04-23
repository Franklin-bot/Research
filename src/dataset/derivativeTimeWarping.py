import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import pywt
from PyEMD import EMD
from scipy import signal
from Processing import parse_motion_file, lowpass_filter
from vmdpy import VMD
import emd as emd2
from tqdm import tqdm, trange
from scipy import signal
import scipy.interpolate as interpolate
import scipy.io

def vmd_decomp(data):
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0            # noise-tolerance (no strict fidelity enforcement)  
    K = 8              # modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly
    tol = 1e-7

    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    return u, omega[-1:]      


def noTimeWarp(imfs, warping, data, time, n_perturb):

    # create warpings for each range
    warping = np.zeros([n_perturb, 8, 7510])
    for j in range(n_perturb):
        for i in range(8):
            warping[j, i, :] = DerivMagnitudeWarp(time, 50, data).flatten()

    plt.figure(1)
    # # warp VMD IMFS
    new_vmd_imfs = np.zeros(imfs.shape)
    new_VMDs = np.zeros([n_perturb, len(data)])
    for j in trange(n_perturb):
        for i in range(8):
            new_vmd_imfs[i, :] = imfs[i, :] * warping[j, i, :]
        new_VMD = np.sum(new_vmd_imfs, axis=0)
        new_VMDs[j] = new_VMD
        plt.plot(new_VMD, alpha=0.7)
    plt.plot(data, color='k', label='original')
    plt.title('No Time Warp')

    m_VMD, u_VMD, l_VMD = calculateMeanStdDev(new_VMDs)
    plt.plot(m_VMD, color=(1, 0, 1), label='mean')
    plt.legend()

    plt.figure(2)
    plt.plot(m_VMD, color="red", label='mean')
    plt.fill_between(range(len(m_VMD)), u_VMD, l_VMD, color='red', alpha=0.5)
    plt.title('No Time Warp Distribution')

def endTimeWarp(imfs, warping, data, time, n_perturb):

    # create warpings for each range
    warping = np.zeros([n_perturb, 8, 7510])
    for j in range(n_perturb):
        for i in range(8):
            warping[j, i, :] = DerivMagnitudeWarp(time, 50, data).flatten()

    plt.figure(3)
    # # warp VMD IMFS
    new_vmd_imfs = np.zeros(imfs.shape)
    new_VMDs = np.zeros([n_perturb, len(data)])
    for j in trange(n_perturb):
        for i in range(8):
            new_vmd_imfs[i, :] = imfs[i, :] * warping[j, i, :]
        new_VMD = np.sum(new_vmd_imfs, axis=0)
        new_VMD = TimeWarp(new_VMD, time, 20)
        new_VMDs[j] = new_VMD
        plt.plot(new_VMD, alpha=0.7)
    plt.plot(data, color='k', label='original')
    plt.title('End Time Warp')

    m_VMD, u_VMD, l_VMD = calculateMeanStdDev(new_VMDs)
    plt.plot(m_VMD, color=(1, 0, 1), label='mean')
    plt.legend()

    plt.figure(4)
    plt.plot(m_VMD, color="red", label='mean')
    plt.fill_between(range(len(m_VMD)), u_VMD, l_VMD, color='red', alpha=0.5)
    plt.title('End Time Warp Distribution')

def imfTimeWarpPre(imfs, warping, data, time, n_perturb):

    # create warpings for each range
    warping = np.zeros([n_perturb, 8, 7510])
    for j in range(n_perturb):
        for i in range(8):
            warping[j, i, :] = DerivMagnitudeWarp(time, 50, data).flatten()

    plt.figure(5)
    # # warp VMD IMFS
    new_vmd_imfs = np.zeros(imfs.shape)
    new_VMDs = np.zeros([n_perturb, len(data)])
    for j in trange(n_perturb):
        for i in range(8):
            new_vmd_imfs[i, :] = TimeWarp(imfs[i, :], time, 20)
            new_vmd_imfs[i, :] = new_vmd_imfs[i, :] * warping[j, i, :]
        new_VMD = np.sum(new_vmd_imfs, axis=0)
        new_VMDs[j] = new_VMD
        plt.plot(new_VMD, alpha=0.7)
    plt.plot(data, color='k', label='original')
    plt.title('IMF Time Warp (before magwarp)')

    m_VMD, u_VMD, l_VMD = calculateMeanStdDev(new_VMDs)
    plt.plot(m_VMD, color=(1, 0, 1), label='mean')
    plt.legend()

    plt.figure(6)
    plt.plot(m_VMD, color="red", label='mean')
    plt.fill_between(range(len(m_VMD)), u_VMD, l_VMD, color='red', alpha=0.5)
    plt.title('IMF Time Warp Distribution (before magwarp)')

def imfTimeWarpPost(imfs, warping, data, time, n_perturb):

    # create warpings for each range

    plt.figure(7)
    # # warp VMD IMFS
    new_vmd_imfs = np.zeros(imfs.shape)
    new_VMDs = np.zeros([n_perturb, len(data)])
    for j in trange(n_perturb):
        for i in range(8):
            new_vmd_imfs[i, :] = imfs[i, :] * warping[j, i, :]
            new_vmd_imfs[i, :] = TimeWarp(new_vmd_imfs[i, :], time, 20)
        new_VMD = np.sum(new_vmd_imfs, axis=0)
        new_VMDs[j] = new_VMD
        plt.plot(new_VMD, alpha=0.7)
    plt.plot(data, color='k', label='original')
    plt.title('IMF Time Warp (after magwarp)')

    m_VMD, u_VMD, l_VMD = calculateMeanStdDev(new_VMDs)
    plt.plot(m_VMD, color=(1, 0, 1), label='mean')
    plt.legend()

    plt.figure(8)
    plt.plot(m_VMD, color="red", label='mean')
    plt.fill_between(range(len(m_VMD)), u_VMD, l_VMD, color='red', alpha=0.5)
    plt.title('IMF Time Warp Distribution (after magwarp)')

    return new_VMDs

def compareTimeWarp(data, time, n_perturb):
    imfs, target_freqs = vmd_decomp(data)

    magWarping = np.zeros([n_perturb, 8, 7510])
    for j in range(n_perturb):
        for i in range(8):
            magWarping[j, i, :] = DerivMagnitudeWarp(time, 50, data).flatten()



    noTimeWarp(imfs, magWarping, data, t, n_perturb)
    endTimeWarp(imfs, magWarping, data, t, n_perturb)
    imfTimeWarpPre(imfs, magWarping, data, t, n_perturb)
    imfTimeWarpPost(imfs, magWarping, data, t, n_perturb)
    plt.show()


def calculateMeanStdDev(dataset):
    num_columns = dataset.shape[1]
    mean = np.zeros(num_columns)
    upper = np.zeros(num_columns)
    lower = np.zeros(num_columns)

    for i in range(num_columns):
        column_data = dataset[:, i]
        mean[i] = np.mean(column_data)
        std = np.std(column_data)
        upper[i] = mean[i] + std
        lower[i] = mean[i] - std
        
    return mean, upper, lower

def DerivMagnitudeWarp(time, n_points, signal):

    scale = 0.8

    derivatives = np.diff(signal)
    derivatives = np.abs(np.append(derivatives, derivatives[-1]))
    derivatives /= np.max(derivatives)

    step = int(len(time) / n_points)
    time_indices = np.arange(0, len(time), step, dtype=int)
    time_points = time[time_indices]

    distortion = []

    for i in range(len(time_indices)):
        distortion.append(np.random.normal(loc=1, scale=scale*derivatives[time_indices[i]], size=1))

    cs = interpolate.CubicSpline(time_points, distortion)
    return cs(time)

def TimeWarp(data, time, n_points):

    step = int(len(time)/(n_points))
    time_indices = (np.arange(0, len(time), step, dtype=int))
    time_points = time[time_indices]
    distortion = np.random.normal(loc=1, scale=0.05, size=len(time_indices))

    new_data = np.zeros(len(data))
    cs = interpolate.CubicSpline(time_points, distortion)
    distorted_time = ((np.cumsum(cs(time)))/len(time))*np.max(time)
    new_data = np.interp(time, distorted_time, data)
    return new_data


if __name__ == "__main__":

    h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
    original_data = df.to_numpy()
    feature = original_data[:, 3]
    t = np.array(t)

    compareTimeWarp(feature, t, 20)

