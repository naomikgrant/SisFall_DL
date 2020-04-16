import tensorflow as tf
import pickle
import matplotlib
import matplotlib.pyplot as plt

# Loading saved model & model history (before adding filter)
print("Loading model (before filter)...")
model_bf = tf.keras.models.load_model('SisFall_Model_BF.h5')
print("Loading model's history (before filter)...")
historyBF = pickle.load(open("HistoryBF.pickle", "rb"))

print("Plotting graphs...")
# Plot training & validation accuracy values
fig1 = plt.figure(1)
plt.plot(historyBF['acc'])
plt.plot(historyBF['val_acc'])
plt.title('Model Accuracy Before Filter')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_Accuracy(BF).png')

# Plot training & validation loss values
plt.figure(2)
plt.plot(historyBF['loss'])
plt.plot(historyBF['val_loss'])
plt.title('Model Loss Before Filter')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_Loss(BF).png')

