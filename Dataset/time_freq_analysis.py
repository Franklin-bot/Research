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

def stft_decomp(data):
    sampling_freq = 200
    window_size = 2048
    overlap = 2047

    # s, f, t, im = plt.specgram(data, Fs=sampling_freq, NFFT=window_size, noverlap=overlap, cmap='rainbow')
    # print(f[:20])
    # plt.ylim(0, 20)

    f, t, Zxx = signal.stft(data, fs=sampling_freq, nfft=window_size, nperseg=window_size, noverlap=overlap)
    return f, t, Zxx

def wavelet_decomp(data):
    fs = 200
    wavelet = 'cmor1.5-1'
    frequencies = np.array(np.arange(start=0.01, stop=20.01, step=0.05)) / fs
    scales = pywt.frequency2scale(wavelet, frequencies)
    print(len(scales))

    coef, freqs = pywt.cwt(data, scales, wavelet)

    return coef, freqs
    # print(coef.shape)
    # print(len(freqs))
    
    # graph
    # plt.imshow(abs(coef), extent=[0, 7510, 20, 0] , aspect='auto', cmap='rainbow')
    # plt.gca().invert_yaxis()
    # plt.xticks(ticks=np.arange(0, coef.shape[1], step=1000), labels=np.arange(0, coef.shape[1], step=1000)/fs)

def emd_decomp(data):
    imf = emd2.sift.sift(data)
    
    # instantaneous frequency, phase and amplitude
    IP, IF, IA = emd2.spectra.frequency_transform(imf, 200, 'hilbert')
    return imf

def graphEMDModeSpectrums(data, filename):
    imf = emd_decomp(data)
    with PdfPages(filename) as pdf:
        for i in trange(imf.shape[1]):
            fig, ax = plt.subplots(figsize=(8, 6))
            fft_result = np.fft.fft(imf[:, i])
            frequencies = np.fft.fftfreq(len(fft_result), 1/200)
            ax.set_title(f'IMF {i+1} Spectrum')
            ax.semilogy(frequencies, np.abs(fft_result))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
            ax.set_xlim(0, frequencies.max())
            pdf.savefig(fig)
            plt.close(fig)

def vmd_decomp(data):
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0            # noise-tolerance (no strict fidelity enforcement)  
    K = 8              # modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly
    tol = 1e-7

    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    return u, omega[-1:]
    
def graphVMDModeSpectrums(data, filename):
    u, u_hat, omega, K = vmd_decomp(data)


    with PdfPages(filename) as pdf:
        for i in trange(K):
            fig, ax = plt.subplots(figsize=(8, 6))
            magnitude_spectrum = np.abs(u_hat[:, i])
            freqs = np.fft.fftshift(np.fft.fftfreq(len(magnitude_spectrum)))
            
            ax.semilogy(freqs, magnitude_spectrum)
            
            # Set labels and title
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude Spectrum')
            ax.set_title(f'Mode {i + 1} Spectrum')
            ax.set_xlim(0, freqs.max())
            pdf.savefig(fig)
            plt.close(fig)
    
    print(omega[-1:])        


def comparePerturbation(data, time, n_perturb):
    imfs, target_freqs = vmd_decomp(data)
    target_freqs = target_freqs[0]
    freq_segments = []

    plt.figure()
    for i in range(imfs.shape[0]):
        plt.plot(imfs[i])

    # create ranges for each VMD IMF frequency
    for i in range(1, len(target_freqs)-1):
        freq_segments.append(((target_freqs[i] + target_freqs[i-1])/2, (target_freqs[i] + target_freqs[i+1])/2))

    freq_segments.insert(0, (0, freq_segments[0][0]))
    freq_segments.append((freq_segments[-1][1], target_freqs[-1] + (target_freqs[-1] - freq_segments[-1][1])))

    # create warpings for each range
    warping = np.zeros([n_perturb, 8, 7510])
    for j in range(n_perturb):
        for i in range(8):
            warping[j, i, :] = MagnitudeWarp(time, 8)


    plotWarping(warping[0])
    show_buckets(freq_segments, data)


    plt.figure()
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
    plt.title('VMD Perturbed Trajectories')

    m_VMD, u_VMD, l_VMD = calculateMeanStdDev(new_VMDs)
    plt.plot(m_VMD, color=(1, 0, 1), label='mean')
    plt.legend()
    


    # # # warp STFT
    f, t, Zxx = stft_decomp(data)
    new_Zxx = np.zeros(Zxx.shape)
    new_STFTS = np.zeros([n_perturb, len(data)])

    plt.figure()
    show_stft_buckets(freq_segments, f, t, Zxx)

    plt.figure()
    for k in trange(n_perturb):
        for i in range(len(f)):
            for j in range(len(freq_segments)):
                if (freq_segments[j])[0] <= f[i] < (freq_segments[j])[1]:
                    new_Zxx[i][:7510] = Zxx[i][:7510] * warping[k, j, :]
        new_STFT = signal.istft(new_Zxx, fs=200, nperseg=2048, nfft=2048, noverlap=2047)
        new_STFTS[k, :] = new_STFT[1]
        plt.plot(new_STFT[1], alpha=0.7)
    plt.plot(data, color='k', label='original')
    plt.title('STFT Perturbed Trajectories')

    m_STFT, u_STFT, l_STFT = calculateMeanStdDev(new_STFTS)
    plt.plot(m_STFT, color=(1, 0, 1), label='mean')
    plt.legend()
    

    scipy.io.savemat('data.mat', mdict={'data': data})
    scipy.io.savemat('warpings.mat', mdict={'warpings': warping})


    # # # Warp EMD
    imf = emd_decomp(data)

    # plt.figure()
    new_emd_imfs = np.zeros(imf.shape)
    new_EMDs = np.zeros([n_perturb, len(data)])
    max_frequencies = np.zeros(8)

    for i in range(imf.shape[1]):
        fft_result = np.fft.fft(imf[:, i])
        frequencies = np.fft.fftfreq(len(fft_result), 1/200)
        magnitude = np.abs(fft_result)
        max_amplitude_index = np.argmax(magnitude)
        frequency_with_max_amplitude = frequencies[max_amplitude_index]
        max_frequencies[i] = frequency_with_max_amplitude

    show_emd_buckets(freq_segments, imf, max_frequencies)

    plt.figure()
    for k in trange(n_perturb):
        for i in range(imf.shape[1]):
            for j in range(len(freq_segments)):
                if (freq_segments[j])[0] <= np.abs(max_frequencies[i]) < (freq_segments[j])[1]:
                    new_emd_imfs[:, i] = imf[:, i] * warping[k, j, :]
        new_IMF = np.sum(new_emd_imfs.T, axis=0)
        new_EMDs[k, :] = new_IMF
        plt.plot(new_IMF, alpha=0.7)
    
    
    plt.plot(data, color='k', label='original')
    plt.title('EMD Perturbed Trajectories')
    m_EMD, u_EMD, l_EMD = calculateMeanStdDev(new_EMDs)
    plt.plot(m_EMD, color=(1, 0, 1), label='mean')
    plt.legend()




    plt.figure()
    plt.title('Different Method Trajectories Overview')
    plt.plot(m_VMD, color='r', label='VMD')
    plt.fill_between(range(len(m_VMD)), u_VMD, l_VMD, color='r', alpha = 0.2)
    plt.plot(m_STFT, color='b', label='STFT')
    plt.fill_between(range(len(m_STFT)), u_STFT, l_STFT, color='b', alpha = 0.2)
    plt.plot(m_EMD, color='g', label='EMD')
    plt.fill_between(range(len(m_EMD)), u_EMD, l_EMD, color='g', alpha = 0.2)
    plt.plot(data, color='k')
    plt.legend()
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

def MagnitudeWarp(time, n_points):
    step = int(len(time) / n_points)
    time_indices = np.arange(0, len(time), step, dtype=int)
    time_points = time[time_indices]

    distortion = np.random.normal(loc=1, scale=0.2, size=len(time_indices))

    cs = interpolate.CubicSpline(time_points, distortion)
    return cs(time)





h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
feature = original_data[:, 6]
t = np.array(t)

def plotWarping(warping):
    colors = ['#FF3333', '#FF9933', '#E6EC38', '#BBFF33', '#33FFD4', '#3377FF', '#7733FF', '#F333FF']
    n_warp = warping.shape[0]
    fig, axs = plt.subplots(n_warp, 1)

    for i in range(n_warp):
        axs[i].axhline(y=1, color='b', linestyle='--')
        axs[i].plot(warping[i][:], color = colors[i])
        axs[i].set_title(f"Warping {i+1}")
        axs[i].set_ylim(0.25, 1.75)
    fig.suptitle("Warpings")

def show_buckets(buckets, signal):
    plt.figure()
    colors = ['#FF3333', '#FF9933', '#E6EC38', '#BBFF33', '#33FFD4', '#3377FF', '#7733FF', '#F333FF']

    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/200)
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(frequencies)//2]))
    for i in range(len(buckets)):
        plt.axvspan(buckets[i][0], buckets[i][1], alpha=0.5, color=colors[i])
    plt.title("Frequency-Amplitude of Signal with Warping Segments")
    plt.xlim(0, 3)
    plt.title("Buckets vs Original Signal FFT")

def show_stft_buckets(buckets, f, t, Zxx):
    plt.figure()
    colors = ['#FF3333', '#FF9933', '#E6EC38', '#BBFF33', '#33FFD4', '#3377FF', '#7733FF', '#F333FF']
    spectrogram = np.abs(Zxx)**2
    plt.pcolormesh(t, f, 10 * np.log10(spectrogram), shading='gouraud', cmap='gray')  # Using logarithmic scale for better visualization
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, 3)
    plt.title("Spectrogram of SIgnal with Warping Segments")
    for i in range(len(buckets)):
        plt.axhspan(buckets[i][0], buckets[i][1], alpha=0.5, color=colors[i])
    plt.title("Buckets vs STFT Spectrogram")


def show_emd_buckets(buckets, imf, max_frequency):
    imf = imf.T
    fig, axs = plt.subplots(imf.shape[0], 1)
    colors = ['#FF3333', '#FF9933', '#E6EC38', '#BBFF33', '#33FFD4', '#3377FF', '#7733FF', '#F333FF']

    for i in range(imf.shape[0]):
        for j in range(len(buckets)):
            if (buckets[j])[0] <= np.abs(max_frequency[i]) < (buckets[j])[1]:
                axs[i].plot(imf[i], color=colors[j])
            else:
                axs[i].plot(imf[i], color='black')
            axs[i].set_title(f"imf {i}")
    fig.suptitle("Buckets vs EMD IMFs")

# comparePerturbation(feature, t, 1)

imfs, target_freqs = vmd_decomp(feature)
target_freqs = target_freqs[0]
freq_segments = []

plt.figure()
for i in range(imfs.shape[0]):
    plt.plot(imfs[i])
plt.show()