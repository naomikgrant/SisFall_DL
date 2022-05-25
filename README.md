# SisFall_DL
Deep learning on SisFall Dataset for mobile phone fall detection.

To download the complete SisFall dataset, please visit this link where it is publicly available: http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/

UPDATE: The link is broken! To contact the authors for the files, you can find their names here: https://www.mdpi.com/1424-8220/17/1/198
I'll try to upload a download link here as I still have the files.

N.B.: Ensure to run the file LoadData.py in the MLP folder before running any of the other python files.
This specific file creates ADL and FALL folders to store the respective readings. 

Abbreviations and their meanings
MLP = Multilayer Perceptron
RNN = Recurrent Neural Network
BF = Before Filtering
AF = After Filtering

The Testing.py file serves to observe the shape, size and content of the labels and samples 
produced after running LoadData.py & LoadDataRNN.py
