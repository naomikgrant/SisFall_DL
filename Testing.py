
import pickle

Y1 = pickle.load(open("LabelsBF.pickle", "rb"))
Y2 = pickle.load(open("LabelsAF.pickle", "rb"))
Y3 = pickle.load(open("LabelsBF_RNN.pickle", "rb"))
Y4 = pickle.load(open("LabelsAF_RNN.pickle", "rb"))


print("LabelsBF:")
print(Y1)
print(Y1.shape)

print("LabelsAF:")
print(Y2)
print(Y2.shape)

print("LabelsBF_RNN:")
print(Y3)
print(Y3.shape)

print("LabelsAF_RNN:")
print(Y4)
print(Y4.shape)