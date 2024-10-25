import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# The cutoff for what is considered a line
cutoff = 0.8

# The neural network to test
ANN = 'LINER/LINER_1_ANN_0.keras'

# The data to test on
file = 'LINER/LINER_all_0.txt'
data = np.loadtxt(file)

filepath = os.path.splitext(file)[0]
test_name = filepath.split('/')[-1]

ANN_path = os.path.splitext(ANN)[0]
ANN_name = ANN_path.split('/')[-1]

test_data = data[:,:-8] # The amplitudes
lines_present = data[:,-8:].T # Whether the lines are present or not

model = tf.keras.models.load_model(ANN)

# YPred is the predicted value from the network
YPred = model.predict(test_data, verbose=1)

# Find AoNs of the four categories
lines_present = lines_present.T

status = lines_present - YPred

real_det = np.dstack(np.where((status >= 0) & (status <= 1 - cutoff)))[0]
miss_det = np.dstack(np.where(status > 1 - cutoff))[0]
nothing = np.dstack(np.where((status > 0 - cutoff) & (status <= 0)))[0]
false_pos = np.dstack(np.where(status <= 0 - cutoff))[0]

real_AoNs = data[real_det[:,0], real_det[:,1]]
miss_AoNs = data[miss_det[:,0], miss_det[:,1]]
noth_AoNs = data[nothing[:,0], nothing[:,1]]
false_AoNs = data[false_pos[:,0], false_pos[:,1]]

# Histogram
density = True
cumulative = False
stacked = False

hbin = np.linspace(0, np.median(real_AoNs) + 3*np.std(real_AoNs), 50)
plt.hist(real_AoNs, bins=hbin, density=density, label='real detections', histtype='step', cumulative=cumulative, stacked=stacked)
plt.hist(miss_AoNs, bins=hbin, density=density, label='missed detections', histtype='step', cumulative=cumulative, stacked=stacked)
plt.hist(noth_AoNs, bins=hbin, density=density, label='true negatives', histtype='step', cumulative=cumulative, stacked=stacked)
plt.hist(false_AoNs, bins=hbin, density=density, label='false positives', histtype='step', cumulative=cumulative, stacked=stacked)

if cumulative == True:
    plt.title(f'A/N cumulative distribution of lines detected by\n{ANN_name} in {test_name}')
else:
    plt.title(f'A/N distribution of lines detected by\n{ANN_name} in {test_name}')
plt.xlabel('A/N out')
plt.legend()

plt.show()



# Separate for each line
real_det = [] # real detections
miss_det = [] # missed detections
false_pos = [] # false positives
nothing = [] # true negatives

real_AoNs = []
miss_AoNs = []
noth_AoNs = []
false_AoNs = []

for i in range(len(status[0])):
    real_det.append(np.where((status[:,i] >= 0) & (status[:,i] <= 1 - cutoff))[0])
    miss_det.append(np.where(status[:,i] > 1 - cutoff)[0])
    nothing.append(np.where((status[:,0] > 0 - cutoff) & (status[:,0] <= 0))[0])
    false_pos.append(np.where(status[:,i] <= 0 - cutoff)[0])

    real_AoNs.append(data[real_det[i], i])
    miss_AoNs.append(data[miss_det[i], i])
    noth_AoNs.append(data[nothing[i], i])
    false_AoNs.append(data[false_pos[i], i])


    hbin = np.linspace(0, np.median(real_AoNs[i]) + 3*np.std(real_AoNs[i]), 10)

    plt.hist(real_AoNs[i], bins=hbin, density=density, label='real detections', histtype='step', cumulative=cumulative, stacked=stacked)
    plt.hist(miss_AoNs[i], bins=hbin, density=density, label='missed detections', histtype='step', cumulative=cumulative, stacked=stacked)
    plt.hist(noth_AoNs[i], bins=hbin, density=density, label='true negatives', histtype='step', cumulative=cumulative, stacked=stacked)
    plt.hist(false_AoNs[i], bins=hbin, density=density, label='false positives', histtype='step', cumulative=cumulative, stacked=stacked)


    if cumulative == True:
        plt.title(f'A/N cumulative distribution of lines detected by {ANN_name}\nin {test_name} in line {i}')
    else:
        plt.title(f'A/N distribution of lines detected by {ANN_name}\nin {test_name} in line {i}')
    plt.xlabel('A/N out')
    plt.legend()

    plt.show()




AoN_medians = []
AoN_stds = []

for i in range(len(status[0])):
    AoN_medians.append(np.median(real_AoNs[i]))
    AoN_stds.append(np.std(real_AoNs[i]))
    
hbin = np.linspace(0, max(AoN_medians) + 3*max(AoN_stds), 10)

for i in range(len(status[0])):
    plt.hist(real_AoNs[i], bins=hbin, density=density, label=f'real detections, line {i}', histtype='step', cumulative=cumulative, stacked=stacked)

plt.legend()
plt.show()










