'''
This Python file loads the data from the pickle files
and carries out a MLP model to categorize falls from ADL's.
'''

import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Loading data to go through neural networks
X = pickle.load(open("Samples.pickle", "rb"))
Y = pickle.load(open("Labels.pickle", "rb"))

print("Shape of X:")
print(X.shape)

# Separating Train & Test Datasets
split_size = int(X.shape[0]*0.7)
X_train, X_test = X[:split_size], X[split_size:]
Y_train, Y_test = Y[:split_size], Y[split_size:]

print("Shape of X_train:")
print(X_train.shape)

# Building basic neural network model
model = tf.keras.models.Sequential\
([
    tf.keras.layers.Dense(32, input_shape=(1,), activation = 'relu'),  # 1st layer
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='softmax') # Last Layer
])
model.summary()     # Displays parameters within model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
print("Training model...")
historyBF = model.fit(X, Y, validation_split=(0.3), batch_size=20, epochs=50, shuffle = True, verbose=2)

# Prediction Stage
print("Evaluating model...")
loss, acc = model.evaluate(X_test, Y_test, verbose=1)
print("Evaluated accuracy: {:4.4f}%" .format(100*acc))

print("Making predictions...")
predictions = model.predict_classes(X_test, batch_size=30, verbose=1)    # N.B.-Predictions are rounded
pred_acc = accuracy_score(predictions, Y_test)
print("Predicted accuracy: {:4.4f}%" .format(100*pred_acc))

print("Confusion Matrix:")
cm=confusion_matrix(Y_test, predictions)
print(cm)

# Saving Stage
print("Saving model without filter...")
model.save('SisFall_Model_BF.h5')

print("Saving history of model without filter...")
pickle_out = open("HistoryBF.pickle", "wb")
pickle.dump(historyBF.history, pickle_out)
pickle_out.close()
