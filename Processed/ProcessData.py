import tensorflow as tf
import pickle
import matplotlib
import matplotlib.pyplot as plt

# Loading saved model history (before adding filter)
print("Loading MLP's model's history before and after filtering...")
historyBF = pickle.load(open("HistoryBF.pickle", "rb"))
historyAF = pickle.load(open("HistoryAF.pickle", "rb"))

print("Loading RNN's model's history before and after filtering...")
historyBF_RNN = pickle.load(open("HistoryBF_RNN.pickle", "rb"))
historyAF_RNN = pickle.load(open("HistoryAF_RNN.pickle", "rb"))

print("Plotting graphs for MLP model...")
# Plot training & validation accuracy values
fig1 = plt.figure(1)
plt.plot(historyBF['accuracy'])
plt.plot(historyBF['val_accuracy'])
plt.title('Model Accuracy Before Filter via MLP Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_MLP_Accuracy(BF).png')

# Plot training & validation loss values
plt.figure(2)
plt.plot(historyAF['accuracy'])
plt.plot(historyAF['val_accuracy'])
plt.title('Model Accuracy After Filter via MLP Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_MLP_Accuracy(AF).png')

print("Plotting graphs for RNN model...")
# Plot training & validation accuracy values
fig1 = plt.figure(3)
plt.plot(historyBF_RNN['accuracy'])
plt.plot(historyBF_RNN['val_accuracy'])
plt.title('Model Accuracy Before Filter via RNN Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_RNN_Accuracy(BF).png')

# Plot training & validation loss values
plt.figure(4)
plt.plot(historyAF_RNN['accuracy'])
plt.plot(historyAF_RNN['val_accuracy'])
plt.title('Model Accuracy After Filter via RNN Model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_RNN_Accuracy(AF).png')
