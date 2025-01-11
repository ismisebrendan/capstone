import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# The cutoff for what is considered a spectrum
cutoff = 0.8

outputs = 1

# The neural network to test
ANN = 'spectra_0_to_4_mod_ANN_0.keras'

# The data to test on
file = 'spectra_5_to_37_mod.txt'
data = np.loadtxt(file)

filepath = os.path.splitext(file)[0]
test_name = filepath.split('/')[-1]

ANN_path = os.path.splitext(ANN)[0]
ANN_name = ANN_path.split('/')[-1]
ANN_path = ANN_path.split('/')[0:-1]
ANN_path = '/'.join(ANN_path)

test_data = data[:,:-outputs] # The amplitudes
lines_present = data[:,-outputs:].T # Whether the lines are present or not

model = tf.keras.models.load_model(ANN)

# YPred is the predicted value from the network
YPred = model.predict(test_data, verbose=1)

# Create histograms for each peak
hbin = np.linspace(min(YPred), max(YPred), num=100).reshape(-1)
plt.hist(YPred[np.where(YPred < cutoff)[0]], bins=hbin, density=False, label='Non-Detections')
plt.hist(YPred[np.where(YPred > cutoff)[0]], bins=hbin, density=False, label='Potential Detections')
plt.xlabel('Predicted likelihood of being a real detection by network')
plt.ylabel('Number of spectra')
plt.title(f'{ANN_name} trained network predicting {test_name}')
plt.yscale('log')
plt.show()

# Separate the lines into the different categories
lines_present = lines_present.T

status = lines_present - YPred
    
real_det = np.dstack(np.where((status >= 0) & (status <= 1 - cutoff)))[0]
miss_det = np.dstack(np.where(status > 1 - cutoff))[0]
nothing = np.dstack(np.where((status > 0 - cutoff) & (status < 0)))[0]
false_pos = np.dstack(np.where(status <= 0 - cutoff))[0]    

# Confusion matrix overall
confusion = np.array([[len(real_det), len(false_pos)], [len(miss_det), len(nothing)]], dtype='f')

# Raw numbers
ax = plt.gca()
ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
plt.imshow(confusion, cmap='Greys')
plt.colorbar()
plt.xticks([0, 1], ['Spectrum present', 'Spectrum not present'])
plt.yticks([0, 1], ['Spectrum detected', 'Spectrum not detected'], rotation='vertical', va='center')
plt.ylabel('Is the spectrum detected?')
plt.xlabel('Is the spectrum present?')
plt.title(f'Confusion matrix for all spectra in {test_name}\nusing {ANN_name}')
plt.text(0, 0, int(confusion[0][0]), ha='center', backgroundcolor='w')
plt.text(0, 1, int(confusion[1][0]), ha='center', backgroundcolor='w')
plt.text(1, 0, int(confusion[0][1]), ha='center', backgroundcolor='yellow')
plt.text(1, 1, int(confusion[1][1]), ha='center', backgroundcolor='w')
plt.show()

# Relative values
no_false, no_lines = np.unique(lines_present, return_counts=True)[1]

confusion[:,0] /= no_lines
confusion[:,1] /= no_false

ax = plt.gca()
ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 
plt.imshow(confusion, cmap='Greys', vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xticks([0, 1], ['Spectrum present', 'Spectrum not present'])
plt.yticks([0, 1], ['Spectrum detected', 'Spectrum not detected'], rotation='vertical', va='center')
plt.ylabel('Is the spectrum detected?')
plt.xlabel('Is the spectrum present?')
plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN_name}')
plt.text(0, 0, np.round(confusion[0][0], 5), ha='center', backgroundcolor='w')
plt.text(0, 1, np.round(confusion[1][0], 5), ha='center', backgroundcolor='w')
plt.text(1, 0, np.round(confusion[0][1], 5), ha='center', backgroundcolor='yellow')
plt.text(1, 1, np.round(confusion[1][1], 5), ha='center', backgroundcolor='w')
plt.show()
