
'''
This Python file loads the data from the respective pickle files
and carries out different RNN models to predict falls from ADL's.
'''

import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Loading data to go through neural networks
X1 = pickle.load(open("Samples_RNN.pickle", "rb"))
Y1 = pickle.load(open("Labels_RNN.pickle", "rb"))
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
print('Before Filtering: Shape of X2_train:', X2_train.shape, 'Shape of Y2_train:', Y2_train.shape)

# Building recurrent neural network model
'''
modelBF_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.LSTM(32, return_sequences=True, activation='relu', input_shape=(X1_train.shape[1], X1_train.shape[2])),      # Takes number of timesteps & features as input
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(32, activation='relu',),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='softmax')
])
modelBF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), metrics=['accuracy'], loss='binary_crossentropy', verbose=0)

# Training model
print("Training model...")
historyBF_RNN = modelBF_RNN.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test), batch_size=64, epochs=10, shuffle = False)
modelBF_RNN.summary()     # Displays parameters within model
'''

embed_vec_length = 32
modelBF_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.Embedding(X1_train.shape[0], embed_vec_length, input_length=X1_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
modelBF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'], verbose=0)

modelAF_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.Embedding(X2_train.shape[0], embed_vec_length, input_length=X2_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
modelAF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'], verbose=0)

# Training model
print("Training model w/ data after filtering...")
historyAF_RNN = modelAF_RNN.fit(X2_train, Y2_train, validation_data=(X2_test, Y2_test), batch_size=64, epochs=15, shuffle = False)
modelAF_RNN.summary()     # Displays parameters within model

print("Training model w/ data before filtering...")
historyBF_RNN = modelBF_RNN.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test), batch_size=64, epochs=15, shuffle = False)
modelBF_RNN.summary()     # Displays parameters within model

# Prediction Stage
print("Evaluating models...")
loss1, acc1 = modelBF_RNN.evaluate(X1_test, Y1_test, verbose=1)
loss2, acc2 = modelBF_RNN.evaluate(X2_test, Y2_test, verbose=1)
print("Evaluated Accuracy")
print("------------------")
print("Before Filter: {:4.4f}%" .format(100*acc1))
print("After Filter: {:4.4f}%" .format(100*acc2))

print("Predictions Accuracy")
print("--------------------")
predictions1 = modelBF_RNN.predict_classes(X1_test, batch_size=64, verbose=1)    # N.B.-Predictions are rounded
predictions2 = modelAF_RNN.predict_classes(X2_test, batch_size=64, verbose=1)    # N.B.-Predictions are rounded]
pred_acc1 = accuracy_score(predictions1, Y1_test)
pred_acc2 = accuracy_score(predictions2, Y2_test)
print("Before Filter: {:4.4f}%" .format(100*pred_acc1))
print("After Filter: {:4.4f}%" .format(100*pred_acc2))

print("Confusion Matrix")
print("----------------")
cm1=confusion_matrix(Y1_test, predictions1)
cm2=confusion_matrix(Y2_test, predictions2)
print("Before Filter:")
print(cm1)
print("After Filter:")
print(cm2)

# Saving Stage
print("Saving history of model without filter...")
pickle_out = open("HistoryBF_RNN.pickle", "wb")
pickle.dump(historyBF_RNN.history, pickle_out)
pickle_out.close()

print("Saving history of model without filter...")
pickle_out = open("HistoryAF_RNN.pickle", "wb")
pickle.dump(historyAF_RNN.history, pickle_out)
pickle_out.close()
