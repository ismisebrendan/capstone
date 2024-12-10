import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# The cutoff for what is considered a line
cutoff = 0.8

# How many outputs the ANN gives
outputs = 1

# The index of the brightest line
brightest = 4

# The neural network to test
ANN = 'planetary_nebula/planetary_nebula_0_to_9_mod_ANN_0.keras'

# The data to test on
file = 'planetary_nebula/planetary_nebula_10_to_37_mod.txt'
data = np.loadtxt(file)

# Filenames and paths
filepath = os.path.splitext(file)[0]
test_name = filepath.split('/')[-1]

ANN_path = os.path.splitext(ANN)[0]
ANN_name = ANN_path.split('/')[-1]
ANN_path = ANN_path.split('/')[0:-1]
ANN_path = '/'.join(ANN_path)


test_data = data[:,:-outputs] # The amplitudes
lines_present = data[:,-outputs:].T # Whether the lines are present or not

model = tf.keras.models.load_model(ANN)

# The predicted value from the network
YPred = model.predict(test_data, verbose=1)

# Separate the lines into the different categories and store AoNs
lines_present = lines_present.T

status = lines_present - YPred

cutoffs = np.arange(0.1, 1.0, 0.1)
for cutoff in cutoffs:
    real_det = np.dstack(np.where((status >= 0) & (status <= 1 - cutoff)))[0]
    miss_det = np.dstack(np.where(status > 1 - cutoff))[0]
    nothing = np.dstack(np.where((status > 0 - cutoff) & (status < 0)))[0]
    false_pos = np.dstack(np.where(status <= 0 - cutoff))[0]
    
    if outputs != 1:
        real_AoNs = data[real_det[:,0], real_det[:,1]]
        miss_AoNs = data[miss_det[:,0], miss_det[:,1]]
        noth_AoNs = data[nothing[:,0], nothing[:,1]]
        false_AoNs = data[false_pos[:,0], false_pos[:,1]]
    else:
        real_AoNs = data[real_det[:,0], 4]
        miss_AoNs = data[miss_det[:,0], 4]
        noth_AoNs = data[nothing[:,0], 4]
        false_AoNs = data[false_pos[:,0], 4]
    
    # Create histograms
    density = True
    cumulative = False
    stacked = False
    no_bins = 50
    
    
    hbin = np.linspace(0, np.median(real_AoNs) + 3*np.std(real_AoNs), no_bins)
    plt.hist(real_AoNs, bins=hbin, density=density, label='real detections', histtype='step', cumulative=cumulative, stacked=stacked)
    plt.hist(miss_AoNs, bins=hbin, density=density, label='missed detections', histtype='step', cumulative=cumulative, stacked=stacked)
    plt.hist(noth_AoNs, bins=hbin, density=density, label='true negatives', histtype='step', cumulative=cumulative, stacked=stacked)
    plt.hist(false_AoNs, bins=hbin, density=density, label='false positives', histtype='step', cumulative=cumulative, stacked=stacked)
    
    if cumulative == True:
        plt.title(f'A/N cumulative distribution of lines detected by\n{ANN_name} in {test_name}, cutoff={cutoff}')
    else:
        plt.title(f'A/N distribution of lines detected by\n{ANN_name} in {test_name}, cutoff={cutoff}')
    plt.xlabel('A/N out')
    plt.legend()
    
    plt.show()


# Plot different cutoffs of AoN
ind_over_1 = real_AoNs > 1
ind_over_2 = real_AoNs > 2
density = False

plt.hist(real_AoNs, bins=hbin, density=density, label='real detections', histtype='step', cumulative=cumulative, stacked=stacked)
plt.hist(real_AoNs[ind_over_1], bins=hbin, density=density, label='real detections A/N > 1', histtype='step', cumulative=cumulative, stacked=stacked)
plt.hist(real_AoNs[ind_over_2], bins=hbin, density=density, label='real detections A/N > 2', histtype='step', cumulative=cumulative, stacked=stacked)
# plt.hist(miss_AoNs, bins=hbin, density=density, label='missed detections', histtype='step', cumulative=cumulative, stacked=stacked)
# plt.hist(noth_AoNs, bins=hbin, density=density, label='true negatives', histtype='step', cumulative=cumulative, stacked=stacked)
# plt.hist(false_AoNs, bins=hbin, density=density, label='false positives', histtype='step', cumulative=cumulative, stacked=stacked)

if cumulative == True:
    plt.title(f'A/N cumulative distribution of lines detected by\n{ANN_name} in {test_name}')
else:
    plt.title(f'A/N distribution of lines detected by\n{ANN_name} in {test_name}')
plt.xlabel('A/N out')
plt.legend()

plt.show()

cutoffs = np.arange(0.1, 1.1, 0.1)

for cutoff in cutoffs:
    false_pos = np.dstack(np.where(status <= 0 - cutoff))[0]

    false_AoNs = data[false_pos[:,0], 4]
    
    plt.hist(false_AoNs, bins=hbin, density=density, label='real detections', histtype='step', cumulative=cumulative, stacked=stacked)





