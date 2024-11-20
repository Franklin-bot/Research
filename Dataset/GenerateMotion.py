import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Processing import parse_motion_file, write_motion_file
from derivativeTimeWarping import DerivMagnitudeWarp, vmd_decomp, TimeWarp
from tqdm import tqdm

h, df, t = parse_motion_file("/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/inverse_kinematics_data/SN001_0024_tug_01.mot")
original_data = df.to_numpy()
feature = original_data[:, 3]
t = np.array(t)



def imfTimeWarpPost(imfs, warping, data, time):

    new_vmd_imfs = np.zeros(imfs.shape)
    for i in range(8):
        new_vmd_imfs[i, :] = imfs[i, :] * warping[i, :]
        new_vmd_imfs[i, :] = TimeWarp(new_vmd_imfs[i, :], time, 20)
    new_VMD = np.sum(new_vmd_imfs, axis=0)
    return new_VMD

def generateFeature(data, time):

    # get IMFs
    imfs, target_freqs = vmd_decomp(data)

    # create magnitude warpings
    magWarping = np.zeros([8, 7510])
    for i in range(8):
        magWarping[i, :] = DerivMagnitudeWarp(time, 50, data).flatten()
    
    # create new feature from magnitude and time warping
    new_feature = imfTimeWarpPost(imfs, magWarping, data, t)
    return new_feature
    

# def write_motion_file(header, dataframe, time, new_file_name):
def generateMotion(data, time):

    n_features = data.shape[1]
    new_motion = np.zeros(data.shape)

    for f in tqdm(range(n_features)):
        curr_feature = data[:, f]
        new_motion[:, f] = generateFeature(curr_feature, time)
    
    write_motion_file(h, pd.DataFrame(new_motion), time, "test_motion")
    

generateMotion(original_data, t)





