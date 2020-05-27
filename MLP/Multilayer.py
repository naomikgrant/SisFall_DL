'''
This Python file loads the data from the pickle files
and carries out a MLP model to categorize falls from ADL's.
'''

import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

# Loading data to go through neural networks
X1 = pickle.load(open("SamplesBF.pickle", "rb"))
Y1 = pickle.load(open("LabelsBF.pickle", "rb"))
X2 = pickle.load(open("SamplesAF.pickle", "rb"))
Y2 = pickle.load(open("LabelsAF.pickle", "rb"))

print('Before Filtering: Shape of X1:', X1.shape, 'Shape of Y1:', Y1.shape)
print('After Filtering: Shape of X2:', X2.shape, 'Shape of Y2:', Y2.shape)
'''
# Separating Train & Test Datasets
split_size = int(X.shape[0]*0.7)
X_train, X_test = X[:split_size], X[split_size:]
Y_train, Y_test = Y[:split_size], Y[split_size:]
'''
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 0)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.3, random_state = 0)

print('Before Filtering: Shape of X1_train:', X1_train.shape, 'Shape of Y1_train:', Y1_train.shape)
print('After Filtering: Shape of X2_train:', X2_train.shape, 'Shape of Y2_train:', Y2_train.shape)

# Building basic neural network model
modelBF = tf.keras.models.Sequential\
([
    tf.keras.layers.Dense(32, input_shape=(X1_train.shape[1],)),  # 1st layer
    tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='sigmoid') # Last Layer
])
modelBF.summary()     # Displays parameters within model
modelBF.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

modelAF = tf.keras.models.Sequential\
([
    tf.keras.layers.Dense(32, input_shape=(X2_train.shape[1],)),  # 1st layer
    tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='sigmoid') # Last Layer
])
modelAF.summary()     # Displays parameters within model
modelAF.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Training model
print("Training model w/ data after filtering...")
historyAF = modelAF.fit(X2, Y2, validation_data=(X2_test, Y2_test), batch_size=20, epochs=50, shuffle = True, verbose=2)

print("Training model w/ data before filtering...")
historyBF = modelBF.fit(X1, Y1, validation_data=(X1_test, Y1_test), batch_size=20, epochs=50, shuffle = True, verbose=2)

# Prediction Stage
print("Evaluating models...")
loss1, acc1 = modelBF.evaluate(X1_test, Y1_test, verbose=1)
loss2, acc2 = modelAF.evaluate(X2_test, Y2_test, verbose=1)
print("Evaluated Accuracy")
print("------------------")
print("Before Filter: {:4.4f}%" .format(100*acc1))
print("After Filter: {:4.4f}%" .format(100*acc2))

# Saving Stage
print("Saving history of model without filtering...")
pickle_out = open("HistoryBF.pickle", "wb")
pickle.dump(historyBF.history, pickle_out)
pickle_out.close()

print("Saving history of model with filtering...")
pickle_out = open("HistoryAF.pickle", "wb")
pickle.dump(historyAF.history, pickle_out)
pickle_out.close()
