
'''
This Python file loads the data from the respective pickle files
and carries out different RNN models to predict falls from ADL's.
'''

import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

# Loading data to go through neural networks
X1 = pickle.load(open("SamplesBF_RNN.pickle", "rb"))
Y1 = pickle.load(open("LabelsBF_RNN.pickle", "rb"))
X2 = pickle.load(open("SamplesAF_RNN.pickle", "rb"))
Y2 = pickle.load(open("LabelsAF_RNN.pickle", "rb"))

print('Before Filtering: Shape of X1:', X1.shape, 'Shape of Y1:', Y1.shape)
print('After Filtering: Shape of X2:', X2.shape, 'Shape of Y2:', Y2.shape)

# Separating Train & Test Datasets
#split_size = int(X.shape[0]*0.7)
#X1_train, X1_test = X[:split_size], X[split_size:]
#Y1_train, Y1_test = Y[:split_size], Y[split_size:]

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 0)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.3, random_state = 0)

print('Before Filtering: Shape of X1_train:', X1_train.shape, 'Shape of Y1_train:', Y1_train.shape)
print('After Filtering: Shape of X2_train:', X2_train.shape, 'Shape of Y2_train:', Y2_train.shape)

# Building recurrent neural network model
modelBF_RNN = tf.keras.models.Sequential\
([
     tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X1_train.shape[1], X1_train.shape[2])),    # Takes number of timesteps & features as input
     tf.keras.layers.LSTM(32, return_sequences=True),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(16),
     tf.keras.layers.Dropout(0.1),
     tf.keras.layers.Dense(2, activation='sigmoid')
])
modelBF_RNN.summary()     # Displays parameters within model
modelBF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])

modelAF_RNN = tf.keras.models.Sequential\
([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X2_train.shape[1], X2_train.shape[2])),    # Takes number of timesteps & features as input
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, activation='sigmoid')
])
modelAF_RNN.summary()     # Displays parameters within model
modelAF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Training model
print("Training model w/ data after filtering...")
historyAF_RNN = modelAF_RNN.fit(X2_train, Y2_train, validation_data=(X2_test, Y2_test), batch_size=40, epochs=24, shuffle = False, verbose=1)

print("Training model w/ data before filtering...")
historyBF_RNN = modelBF_RNN.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test), batch_size=40, epochs=24, shuffle = False, verbose=1)

# Prediction Stage
print("Evaluating models...")
loss1, acc1 = modelBF_RNN.evaluate(X1_test, Y1_test, verbose=1)
loss2, acc2 = modelAF_RNN.evaluate(X2_test, Y2_test, verbose=1)
print("Evaluated Accuracy")
print("------------------")
print("Before Filter: {:4.4f}%" .format(100*acc1))
print("After Filter: {:4.4f}%" .format(100*acc2))

# Saving Stage
print("Saving history of model without filtering...")
pickle_out = open("HistoryBF_RNN.pickle", "wb")
pickle.dump(historyBF_RNN.history, pickle_out)
pickle_out.close()

print("Saving history of model with filtering...")
pickle_out = open("HistoryAF_RNN.pickle", "wb")
pickle.dump(historyAF_RNN.history, pickle_out)
pickle_out.close()
