import os
#numpy + pyplot together
import numpy as np
import matplotlib.pyplot as plt
#tensorflow with keras wrapper
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

ANN = 'false_pos2.keras'
# file = 'false_pos-20240930T151741.txt'
# file = 'false_pos-20240930T153614.txt'
# file = 'false_pos-20240930T153709.txt'
# file = 'false_pos-20240930T161616.txt'
# file = 'false_pos-20240930T164835.txt'
# file = 'false_pos-20241001T090710.txt'
# file = 'false_pos-20241001T152256.txt'
# file = 'false_pos-testing.txt'
file = 'spectra0.txt'

test_data = np.loadtxt(file)[:,:-1]

model = tf.keras.models.load_model(ANN)

# Using the model to make a prediction
# YPred is the predicted value from the network
YPred = model.predict(test_data, verbose =1)

#Format will be [Data, Predicted]
predicted_data = np.hstack((test_data, YPred))

np.savetxt("Predicted.dat", predicted_data)

#save the tested data
filepath = os.path.splitext(file)[0]
# longname = filepath.split("/")[1]
test_name = filepath.split("_")[0]


# Highlighting potential hits
potential = predicted_data[np.where(YPred > 0.8)[0],:]
np.savetxt('Potential_Detections_in_'+test_name+'_with_'+ANN+'.dat', potential)

# #creating a histogram
hbin = np.linspace(min(YPred), max(YPred), num = 100).reshape(-1)
plt.hist(YPred[np.where(YPred < 0.8)[0]], bins = hbin, density = False, label = 'Non-Detections')
plt.hist(YPred[np.where(YPred > 0.8)[0]], bins = hbin, density = False, label = 'Potential Detections')
plt.xlabel("Predicted likelihood of being a real detection by network")
plt.ylabel("Number of spectra")
plt.title(ANN+" trained network predicting "+test_name)
plt.yscale('log')
plt.show()