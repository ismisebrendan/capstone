import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# The cutoff for what is considered a line
cutoff = 0.8

outputs = 1

# The neural network to test
ANN = 'planetary_nebula/planetary_nebula_0_to_9_mod_ANN_0.keras'

# The data to test on
file = 'planetary_nebula/planetary_nebula_10_to_37_mod.txt'
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

predicted_data = np.hstack((test_data, YPred))

# np.savetxt(f'{ANN_path}/Predicted_Detections_in_{test_name}_with_{ANN_name}.dat', predicted_data)

# Highlighting potential hits
potential = predicted_data[np.unique(np.where(YPred > cutoff)[0]),:]
# np.savetxt(f'{ANN_path}/Potential_Detections_in_{test_name}_with_{ANN_name}.dat', potential)

# cutoffs = np.arange(0.5, 1.1, 0.1)

# for cutoff in cutoffs:
# Create histograms for each peak
for i in range(len(YPred[0])):
    hbin = np.linspace(min(YPred[:,i]), max(YPred[:,i]), num=100).reshape(-1)
    plt.hist(YPred[np.where(YPred[:,i] < cutoff)[0]][:,i], bins=hbin, density=False, label='Non-Detections')
    plt.hist(YPred[np.where(YPred[:,i] > cutoff)[0]][:,i], bins=hbin, density=False, label='Potential Detections')
    plt.xlabel('Predicted likelihood of being a real detection by network')
    plt.ylabel('Number of spectra')
    plt.title(f'{ANN_name} trained network predicting {test_name} for peak {i}')
    plt.yscale('log')
    plt.show()
    # print(f'Found peak {i} in {len(np.where(YPred[:,i] > cutoff)[0])/len(YPred) *100}% of spectra')
    # print(f'Expected value: {np.unique(lines_present[i], return_counts=True)[1][1] / len(YPred) * 100}')
    # print(f'Accuracy: {len(np.where(YPred[:,i] > cutoff)[0]) / np.unique(lines_present[i], return_counts=True)[1][1]}\n' )

# Separate the lines into the different categories
lines_present = lines_present.T

status = lines_present - YPred
    
real_det = np.dstack(np.where((status >= 0) & (status <= 1 - cutoff)))[0]
miss_det = np.dstack(np.where(status > 1 - cutoff))[0]
nothing = np.dstack(np.where((status > 0 - cutoff) & (status <= 0)))[0]
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
plt.xticks([0, 1], ['Line present', 'Line not present'])
plt.yticks([0, 1], ['Line detected', 'Line not detected'], rotation='vertical', va='center')
plt.ylabel('Is the line detected?')
plt.xlabel('Is the line present?')
plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN_name}')
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
plt.xticks([0, 1], ['Line present', 'Line not present'])
plt.yticks([0, 1], ['Line detected', 'Line not detected'], rotation='vertical', va='center')
plt.ylabel('Is the line detected?')
plt.xlabel('Is the line present?')
plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN_name}')
plt.text(0, 0, np.round(confusion[0][0], 5), ha='center', backgroundcolor='w')
plt.text(0, 1, np.round(confusion[1][0], 5), ha='center', backgroundcolor='w')
plt.text(1, 0, np.round(confusion[0][1], 5), ha='center', backgroundcolor='yellow')
plt.text(1, 1, np.round(confusion[1][1], 5), ha='center', backgroundcolor='w')
plt.show()


# Separately for each line

output = []

for i in range(len(lines_present[0])):
    confusion = np.array([[np.unique(real_det[:,1], return_counts=True)[1][i], np.unique(false_pos[:,1], return_counts=True)[1][i]], [np.unique(miss_det[:,1], return_counts=True)[1][i], np.unique(nothing[:,1], return_counts=True)[1][i]]], dtype='f')
    
    ax = plt.gca()
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.imshow(confusion, cmap='Greys')
    plt.colorbar()
    plt.xticks([0, 1], ['Line present', 'Line not present'])
    plt.yticks([0, 1], ['Line detected', 'Line not detected'], rotation='vertical', va='center')
    plt.ylabel('Is the line detected?')
    plt.xlabel('Is the line present?')
    plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN_name} for line {i}')
    plt.text(0, 0, int(confusion[0][0]), ha='center', backgroundcolor='w')
    plt.text(0, 1, int(confusion[1][0]), ha='center', backgroundcolor='w')
    plt.text(1, 0, int(confusion[0][1]), ha='center', backgroundcolor='yellow')
    plt.text(1, 1, int(confusion[1][1]), ha='center', backgroundcolor='w')
    plt.show()
    
    # Relative values
    no_false, no_lines = np.unique(lines_present[:,0], return_counts=True)[1]

    confusion[:,0] /= no_lines
    confusion[:,1] /= no_false

    ax = plt.gca()
    ax.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.imshow(confusion, cmap='Greys', vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks([0, 1], ['Line present', 'Line not present'])
    plt.yticks([0, 1], ['Line detected', 'Line not detected'], rotation='vertical', va='center')
    plt.ylabel('Is the line detected?')
    plt.xlabel('Is the line present?')
    plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN_name} for line {i}')
    plt.text(0, 0, np.round(confusion[0][0], 5), ha='center', backgroundcolor='w')
    plt.text(0, 1, np.round(confusion[1][0], 5), ha='center', backgroundcolor='w')
    plt.text(1, 0, np.round(confusion[0][1], 5), ha='center', backgroundcolor='yellow')
    plt.text(1, 1, np.round(confusion[1][1], 5), ha='center', backgroundcolor='w')
    plt.show()
    
    
    output.append(f'{confusion[0][0]}, {confusion[0][1]}, {confusion[1][0]}, {confusion[1][1]}')

# # Save the confusion matrix as a file
# f = open(f'{ANN_path}/confusion_{ANN_name}_{test_name}.txt', 'w')

# f.write('# Rows are each individual peak.\t\t Columns are: True positives, False postives, False negatives, True negatives\n')

# for i in range(len(output)):
#     f.write(f'{output[i]}\n')
# f.close()







