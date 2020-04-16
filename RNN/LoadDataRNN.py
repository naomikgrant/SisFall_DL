'''
This Python file loads data for RNN to be carried out on the SisFall database and then carries out
the pre-processing stage of labelling falls from ADL's into respective pickle files.
Note to self: Include creation & copy of ADL folders in this .py file at the end.
'''

import csv
import glob, os          # Iterates through directories and joins paths
import cv2               # Carries out image & cv operations
import shutil
import numpy as np       # Carried out array operations
import math
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import random
import pickle

# Paths for results from dataset
main_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new"
dataset_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new/SisFall_dataset"
ADL_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new/ADL"
FALL_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new/FALL"

# Variable and array declarations for applying RNN
sample_num = 4500                  # Setting number of desired samples
time_steps = 200                   # Setting number of desired time steps
feat_num = 1                       # Setting number of desired features
AAcc_RNNx, AAcc_RNNz = [], []      # Array for ADL RNN samples
FAcc_RNNx, FAcc_RNNz = [], []      # Array for FALL RNN samples
AvectorsumBF_RNN = []              # Array for ADL RNN vectorsums before filtering
FvectorsumBF_RNN = []              # Array for FALL RNN vectorsums before filtering
train_samplesBF_RNN = []           # Array for all final RNN samples before filtering
train_labelsBF_RNN = []            # Array for all labels of 10000 samples before filtering
partA = []
partB = []
#train_samplesAF = []              # Array for all random 10000# Creating folder to place all ADL text files
#train_labelsAF = []               # Array for all labels of 10000 samples after filtering

# Dividing data into time slices
for ADLfile in os.scandir(ADL_path):
    with open(ADLfile, 'r') as ADL_RNNcsv:
        lines = ADL_RNNcsv.readlines()
        # Clearing white spaces
    with open(ADLfile, 'w') as ADL_RNNcsv:
        lines = filter(lambda x: x.strip(), lines)
        print("Clearing white spaces from ADL RNN files...")
        ADL_RNNcsv.writelines(lines)
        # Appending ADL x-axis & y-axis lists
    with open(ADLfile, 'rt') as ADL_RNNcsv:
      ADL_RNNdata = (ADL_RNNcsv.readlines()[0:time_steps])
      ADL_RNNsamples = csv.reader(ADL_RNNdata, delimiter=',')
      for i in ADL_RNNsamples:
        print("Importing ADL RNN accelerometer readings...")
        AAcc_RNNx.append(int(i[0]))
        AAcc_RNNz.append(int(i[2]))

for FALLfile in os.scandir(FALL_path):
    with open(FALLfile, 'r') as FALL_RNNcsv:
        lines = FALL_RNNcsv.readlines()
        # Clearing white spaces
    with open(FALLfile, 'w') as FALL_RNNcsv:
        lines = filter(lambda x: x.strip(), lines)
        print("Clearing white spaces from FALL RNN files...")
        FALL_RNNcsv.writelines(lines)
        # Appending ADL x-axis & y-axis lists
    with open(FALLfile, 'rt') as FALL_RNNcsv:
      FALL_RNNdata = (FALL_RNNcsv.readlines()[0:time_steps])
      FALL_RNNsamples = csv.reader(FALL_RNNdata, delimiter=',')
      for i in FALL_RNNsamples:
        print("Importing FALL RNN accelerometer readings...")
        FAcc_RNNx.append(int(i[0]))
        FAcc_RNNz.append(int(i[2]))

# Function to calculate horizontal plane vectorsum feature
def vectorsum (Acc_x, Acc_z, vectorsum):
    x = []
    z = []
    print("Preparing vectorsum feature...")
    x = [i*i for i in Acc_x]
    z = [i*i for i in Acc_z]
    sum = [a+b for a,b in zip(x,z)]
    vectorsum = [math.sqrt(i) for i in sum]
    return vectorsum

# Function to calculate dividing list with imported readings
def div_timesteps (list, n):
    print("Dividing list into timesteps...")
    for i in range (0, len(list), n):
        yield list[i:i + n]

# Calculating horizontal plane vectorsum feature
AvectorsumBF_RNN = vectorsum(AAcc_RNNx, AAcc_RNNz, AvectorsumBF_RNN)
FvectorsumBF_RNN = vectorsum(FAcc_RNNx, FAcc_RNNz, FvectorsumBF_RNN)

# Dividing list with imported readings into specified timesteps
# in order to create final input datasets for RNN model
partA = list(div_timesteps(AvectorsumBF_RNN, time_steps))
for i in partA:
    print("Appending ADL RNN label in array...")
    train_labelsBF_RNN.append(0)

partB = list(div_timesteps(FvectorsumBF_RNN, time_steps))
for i in partB:
    print("Appending FALL RNN label in array...")
    train_labelsBF_RNN.append(1)

train_samplesBF_RNN = partA.copy()
train_samplesBF_RNN.extend(partB)

# Placing training samples and labels into numpy arays
train_samplesBF_RNN = np.array(train_samplesBF_RNN)
train_labelsBF_RNN = np.array(train_labelsBF_RNN)

# Scaling & reshaping data from 2D -> 3D
print("Scaling data samples...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samplesBF_RNN = scaler.fit_transform((train_samplesBF_RNN).reshape(-1, 1))
scaled_train_samplesBF_RNN = scaled_train_samplesBF_RNN.reshape(sample_num, time_steps, feat_num)

# Saving data
print("Saving data samples before filtering...")
pickle_out = open("Samples_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesBF_RNN, pickle_out)
pickle_out.close()

pickle_out = open("Labels_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsBF_RNN, pickle_out)
pickle_out.close()
'''
print("Saving data samples after filtering...")
pickle_out = open("SamplesAF.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesAF, pickle_out)
pickle_out.close()

pickle_out = open("LabelsAF.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsAF, pickle_out)
pickle_out.close()
'''