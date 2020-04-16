
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
X = pickle.load(open("Samples_RNN.pickle", "rb"))
Y = pickle.load(open("Labels_RNN.pickle", "rb"))

print('Shape of X:', X.shape, 'Shape of Y:', Y.shape)

# Separating Train & Test Datasets
#split_size = int(X.shape[0]*0.7)
#X_train, X_test = X[:split_size], X[split_size:]
#Y_train, Y_test = Y[:split_size], Y[split_size:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

print('Shape of X_train:', X_train.shape, 'Shape of Y_train:', Y_train.shape)

# Building recurrent neural network model
model_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])),    # Takes number of timesteps & features as input
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_RNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'], loss='binary_crossentropy', verbose=0)

# Training model
print("Training model...")
historyBF_RNN = model_RNN.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=50, epochs=10, shuffle = False)
model_RNN.summary()     # Displays parameters within model

'''
embed_vec_length = 32
model_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.Embedding(X_train.shape[0], embed_vec_length, input_length=X_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),    # Takes number of timesteps & feature
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_RNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'], verbose=0)
'''

# Training model
print("Training model...")
historyBF_RNN200 = model_RNN.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=50, epochs=10, shuffle = False)
model_RNN.summary()     # Displays parameters within model

# Prediction Stage
print("Evaluating model...")
loss, acc = model_RNN.evaluate(X_test, Y_test, verbose=1)
print("Evaluated accuracy: {:4.4f}%" .format(100*acc))

print("Making predictions...")
predictions = model_RNN.predict_classes(X_test, batch_size=50, verbose=1)    # N.B.-Predictions are rounded
pred_acc = accuracy_score(predictions, Y_test)
print("Predicted accuracy: {:4.4f}%" .format(100*pred_acc))

print("Confusion Matrix:")
cm=confusion_matrix(Y_test, predictions)
print(cm)

# Saving Stage
print("Saving history of model without filter...")
pickle_out = open("HistoryBF_RNN.pickle", "wb")
pickle.dump(historyBF_RNN200.history, pickle_out)
pickle_out.close()
