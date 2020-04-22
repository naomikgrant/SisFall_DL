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
from scipy.signal import butter,filtfilt
from sklearn.preprocessing import MinMaxScaler
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
AvectorsumAF_RNN = []              # Array for ADL RNN vectorsums after filtering
FvectorsumAF_RNN = []              # Array for FALL RNN vectorsums after filtering
train_samplesBF_RNN = []           # Array for all final RNN samples before filtering
train_labelsBF_RNN = []            # Array for all labels of samples before filtering
train_samplesAF_RNN = []           # Array for all final RNN samples after filtering
train_labelsAF_RNN = []            # Array for all labels of samples after filtering
partA1, partA2 = [], []
partB1, partB2 = [], []


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

# Function to convert the data from bits -> g
def conversion (Acc_x, Acc_z):
    x = []
    z = []
    print("Converting data in bits to g...")
    x = [(32.0 / 8192.0)*i for i in Acc_x]
    z = [(32.0 / 8192.0)*i for i in Acc_z]
    return x, z

AAcc_RNNx, AAcc_RNNz = conversion(AAcc_RNNx, AAcc_RNNz)
FAcc_RNNx, FAcc_RNNz = conversion(FAcc_RNNx, FAcc_RNNz)

# Function to calculate horizontal plane vectorsum feature BEFORE filtering
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

##################### START FILTERING #####################
# Filter creation and application
print("Filtering w/ 4th order Butterworth filter w/ fc = 5Hz...")
fs = 200            # Sampling frequency in Hz
order = 4           # Order signal
fc = 5              # Cut-off frequency of the filter in Hz
nyquist = 0.5*fs    # Nyquist frequency
t = 370800          # Elderly = 1.5h*15*3600, Youth = 3.5h*23*3600 so total time = 370,800

def filter(data, cutoff, fs, order):
    normal_cutoff = fc / nyquist
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Array -> Signal conversion
print("Converting array w/ data into sine wave...")
sigAAcc_RNNx = [i*np.sin(2*np.pi*fs*t) for i in AAcc_RNNx]
sigAAcc_RNNz = [i*np.sin(2*np.pi*fs*t) for i in AAcc_RNNz]
sigFAcc_RNNx = [i*np.sin(2*np.pi*fs*t) for i in FAcc_RNNx]
sigFAcc_RNNz = [i*np.sin(2*np.pi*fs*t) for i in FAcc_RNNz]

# Filter application
print("Applying filter...")
AAcc_RNNx = filter(sigAAcc_RNNx, fc, fs, order)
AAcc_RNNz = filter(sigAAcc_RNNz, fc, fs, order)
FAcc_RNNx = filter(sigFAcc_RNNx, fc, fs, order)
FAcc_RNNz = filter(sigFAcc_RNNz, fc, fs, order)
##################### END FILTERING #####################

# Calculating horizontal plane vectorsum feature AFTER filtering
AvectorsumAF_RNN = vectorsum(AAcc_RNNx, AAcc_RNNz, AvectorsumAF_RNN)
FvectorsumAF_RNN = vectorsum(FAcc_RNNx, FAcc_RNNz, FvectorsumAF_RNN)

# Dividing list with imported readings into specified timesteps
# in order to create final input datasets for RNN model
partA1 = list(div_timesteps(AvectorsumBF_RNN, time_steps))
partA2 = list(div_timesteps(AvectorsumAF_RNN, time_steps))
for i in partA1:
    print("Appending ADL RNN label in array...")
    train_labelsBF_RNN.append(0)
for i in partA2:
    print("Appending ADL RNN label in array...")
    train_labelsAF_RNN.append(0)

partB1 = list(div_timesteps(FvectorsumBF_RNN, time_steps))
partB2 = list(div_timesteps(FvectorsumAF_RNN, time_steps))
for i in partB1:
    print("Appending FALL RNN label in array...")
    train_labelsBF_RNN.append(1)
for i in partB2:
    print("Appending FALL RNN label in array...")
    train_labelsAF_RNN.append(1)

train_samplesBF_RNN = partA1.copy()
train_samplesAF_RNN = partA2.copy()
train_samplesBF_RNN.extend(partB1)
train_samplesAF_RNN.extend(partB2)

# Placing training samples and labels into numpy arays
train_samplesBF_RNN = np.array(train_samplesBF_RNN)
train_labelsBF_RNN = np.array(train_labelsBF_RNN)
train_samplesAF_RNN = np.array(train_samplesAF_RNN)
train_labelsAF_RNN = np.array(train_labelsAF_RNN)

# Scaling & reshaping data from 2D -> 3D
print("Scaling data samples...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samplesBF_RNN = scaler.fit_transform((train_samplesBF_RNN).reshape(-1, 1))
#scaled_train_samplesBF_RNN = scaled_train_samplesBF_RNN.reshape(sample_num, time_steps, feat_num)
scaled_train_samplesBF_RNN = scaled_train_samplesBF_RNN.reshape(sample_num, time_steps)
scaled_train_samplesAF_RNN = scaler.fit_transform((train_samplesAF_RNN).reshape(-1, 1))
scaled_train_samplesAF_RNN = scaled_train_samplesAF_RNN.reshape(sample_num, time_steps)

# Saving data
print("Saving data samples before filtering...")
pickle_out = open("SamplesBF_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesBF_RNN, pickle_out)
pickle_out.close()

pickle_out = open("LabelsBF_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsBF_RNN, pickle_out)
pickle_out.close()

print("Saving data samples after filtering...")
pickle_out = open("SamplesAF_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesAF_RNN, pickle_out)
pickle_out.close()

pickle_out = open("LabelsAF_RNN.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsAF_RNN, pickle_out)
pickle_out.close()
