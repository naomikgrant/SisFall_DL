'''
This Python file loads a specific number of randomly selected data
from the SisFall database and then carries out the pre-processing stage
of labelling falls from ADL's into respective pickle files.
'''
import tensorflow as tf
import csv
import os                # Iterates through directories and joins paths
import shutil
import numpy as np       # Carried out array operations
import math
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import random
import pickle

# Path for results from dataset
main_path = "./SisFall_new"
dataset_path = "./SisFall_dataset"
ADL_path = "./SisFall_new/ADL"
FALL_path = "./SisFall_new/FALL"

feat_num = 1                       # Setting number of desired features
'''
main_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new"
dataset_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new/SisFall_dataset"
ADL_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new/ADL"
FALL_path = "/PyCharm 2019.2.3/PycharmProjects/SisFall_new/FALL"
'''
'''
# Creating folder to place all ADL text files
os.scandir(main_path)
ADLfolder = os.path.join(main_path, 'ADL')
if not os.path.exists(ADLfolder):
    os.makedirs(ADLfolder)

# Creating folder to place FALL text files
os.scandir(main_path)
FALLfolder = os.path.join(main_path, 'FALL')
if not os.path.exists(FALLfolder):
    os.makedirs(FALLfolder)
'''

# Sorting ADL files and FALL files in respective folders
for root, dirs, files in os.walk(os.path.normpath((dataset_path)), topdown=False):
    for name in files:
        src = os.path.join(root, name)
        if name.endswith(".txt") and name.startswith("D"):
           #if not os.path.exists(ADL_path):                # Add this line AFTER files are copied to folder
                print("ADL files:" + name)
                shutil.copy(src, ADL_path)
        if name.endswith(".txt") and name.startswith("F"):
            #if not os.path.exists(FALL_path):               # Add this line AFTER files are copied to folder
                print("FALL files:" + name)
                shutil.copy(src, FALL_path)

# Array declarations
AAcc_x, AAcc_z = [], []            # Array for ADL samples
FAcc_x, FAcc_z = [], []            # Array for FALL samples
AvectorsumBF = []                  # Array for ADL vectorsums before filtering
FvectorsumBF = []                  # Array for FALL vectorsums before filtering
AvectorsumAF = []                  # Array for ADL vectorsums after filtering
FvectorsumAF = []                  # Array for FALL vectorsums after filtering
train_samplesBF = []               # Array for all random 10000 samples before filtering
train_labelsBF = []                # Array for all labels of 10000 samples before filtering
train_samplesAF = []               # Array for all random 10000 samples after filtering
train_labelsAF = []                # Array for all labels of 10000 samples after filtering

# Importing data from .txt files that were copied to new ADL and FALL folders
# Clearing white spaces from lines before appending to lists
for ADLfile in os.scandir(ADL_path):
    with open(ADLfile, 'r') as ADLcsv:
        lines = ADLcsv.readlines()
    with open(ADLfile, 'w') as ADLcsv:
        lines = filter(lambda x: x.strip(), lines)
        print("Clearing white spaces from ADL files...")
        ADLcsv.writelines(lines)
    with open(ADLfile, 'rt') as ADLcsv:
        ADLsamples = csv.reader(ADLcsv, delimiter = ',')     # ADLsamples = array holding ADL samples
        for i in ADLsamples:
            print("Importing ADL accelerometer readings...")
            AAcc_x.append(int(i[0]))
            AAcc_z.append(int(i[2]))

for FALLfile in os.scandir(FALL_path):
    with open(FALLfile, 'r') as FALLcsv:
        lines = FALLcsv.readlines()
    with open(FALLfile, 'w') as FALLcsv:
        lines = filter(lambda x: x.strip(), lines)
        print("Clearing white spaces from FALL files...")
        FALLcsv.writelines(lines)
    with open(FALLfile, 'rt') as FALLcsv:
        FALLsamples = csv.reader(FALLcsv, delimiter=',')  # ADLsamples = array holding ADL samples
        for i in FALLsamples:
            print("Importing FALL accelerometer readings...")
            FAcc_x.append(int(i[0]))
            FAcc_z.append(int(i[2]))

# Converting data in bits to g
# For accelerometer ADXL345, Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
# where AD = acceleration data, Range = 16g, Resolution = 2^13 = 8192
def conversion (Acc_x, Acc_z):
    x = []
    z = []
    print("Converting data in bits to g...")
    x = [(32.0 / 8192.0)*i for i in Acc_x]
    z = [(32.0 / 8192.0)*i for i in Acc_z]
    return x, z

AAcc_x, AAcc_z = conversion(AAcc_x, AAcc_z)
FAcc_x, FAcc_z = conversion(FAcc_x, FAcc_z)

# Preparing the horizontal plane vectorsum feature before filtering
def vectorsum (Acc_x, Acc_z, vectorsum):
    x = []
    z = []
    print("Preparing vectorsum feature...")
    x = [i*i for i in Acc_x]
    z = [i*i for i in Acc_z]
    sum = [a+b for a,b in zip(x,z)]
    vectorsum = [math.sqrt(i) for i in sum]
    return vectorsum

AvectorsumBF = vectorsum(AAcc_x, AAcc_z, AvectorsumBF)
for i in range(1000):
    AsampleBF = random.choice(AvectorsumBF)
    print("Appending ADL sample in array...")
    train_samplesBF.append(AsampleBF)
    print("Appending ADL label in array...")
    train_labelsBF.append(0)

FvectorsumBF = vectorsum(FAcc_x, FAcc_z, FvectorsumBF)
for i in range(1000):
    FsampleBF = random.choice(FvectorsumBF)
    print("Appending FALL sample in array...")
    train_samplesBF.append(FsampleBF)
    print("Appending FALL label in array...")
    train_labelsBF.append(1)

# Placing arrays into numpy arrays due to expection from Keras
print("Placing into numpy arrays prior to filtering...")
train_labelsBF = np.array(train_labelsBF)
train_samplesBF = np.array(train_samplesBF)

print('No. of randomly selected accelerometer readings BEFORE filtering:', train_samplesBF.shape[0])

##################### START FILTERING #####################
# Filter creation and application
print("Filtering w/ 4th order Butterworth filter w/ fc = 5Hz...")
fs = 200            # Sampling frequency in Hz
order = 4           # Order signal
cutOff = 5          # cut-off frequency of the filter in Hz
nyquist = 0.5*fs    # Nyquist frequency
fc = cutOff/nyquist # Normalized cut-off frequency of the filter
t = 370800          # Elderly = 1.5h*15*3600, Youth = 3.5h*23*3600 so total time = 370,800

def filter(data, fc, order):
    # Get the filter coefficients
    sos = signal.butter(order, fc, output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

# Array -> Signal conversion
print("Converting array w/ data into sine wave...")
sigAAcc_x = [i*np.sin(2*np.pi*fs*t) for i in AAcc_x]
sigAAcc_z = [i*np.sin(2*np.pi*fs*t) for i in AAcc_z]
sigFAcc_x = [i*np.sin(2*np.pi*fs*t) for i in FAcc_x]
sigFAcc_z = [i*np.sin(2*np.pi*fs*t) for i in FAcc_z]

# Filter application
print("Applying filter...")
AAcc_x = filter(sigAAcc_x, fc, order)
AAcc_z = filter(sigAAcc_z, fc, order)
FAcc_x = filter(sigFAcc_x, fc, order)
FAcc_z = filter(sigFAcc_z, fc, order)
##################### END FILTERING #####################

# Preparing the horizontal plane vectorsum feature after filtering
AvectorsumAF = vectorsum(AAcc_x, AAcc_z, AvectorsumAF)
for i in range(1000):
    AsampleAF = random.choice(AvectorsumAF)
    print("Appending ADL sample in array...")
    train_samplesAF.append(AsampleAF)
    print("Appending ADL label in array...")
    train_labelsAF.append(0)

FvectorsumAF = vectorsum(FAcc_x, FAcc_z, FvectorsumAF)
for i in range(1000):
    FsampleAF = random.choice(FvectorsumAF)
    print("Appending FALL sample in array...")
    train_samplesAF.append(FsampleAF)
    print("Appending FALL label in array...")
    train_labelsAF.append(1)

# Placing arrays into numpy arrays due to expection from Keras
print("Placing into numpy arrays after filtering...")
train_labelsAF = np.array(train_labelsAF)
train_samplesAF = np.array(train_samplesAF)

print('No. of randomly selected accelerometer readings AFTER filtering:', train_samplesAF.shape[0])

# Hot encoding
print("Hot encoding labels...")
train_labelsBF=tf.keras.utils.to_categorical(train_labelsBF)
train_labelsAF=tf.keras.utils.to_categorical(train_labelsAF)

# Scaling data
print("Scaling data samples...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samplesBF = scaler.fit_transform((train_samplesBF).reshape(-1, 1))
scaled_train_samplesAF = scaler.fit_transform((train_samplesAF).reshape(-1, 1))


# Saving data
print("Saving data samples before filtering...")
pickle_out = open("SamplesBF.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesBF, pickle_out)
pickle_out.close()

pickle_out = open("LabelsBF.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsBF, pickle_out)
pickle_out.close()

print("Saving data samples after filtering...")
pickle_out = open("SamplesAF.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(scaled_train_samplesAF, pickle_out)
pickle_out.close()

pickle_out = open("LabelsAF.pickle", "wb")
pickle_out.truncate(0)
pickle.dump(train_labelsAF, pickle_out)
pickle_out.close()

#pickle_in = open("SamplesBF.pickle", "rb")
#scaled_train_samplesBF = pickle.load(pickle_in)
