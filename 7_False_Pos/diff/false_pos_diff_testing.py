import os
#numpy + pyplot together
import numpy as np
import matplotlib.pyplot as plt
#tensorflow with keras wrapper
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

ANN = 'false_data1_1.keras'
file = 'false_data1.txt'

data = np.loadtxt(file)

test_data = data[:,:-8]
lines_present = data[:,-8:].T

model = tf.keras.models.load_model(ANN)

# Using the model to make a prediction
# YPred is the predicted value from the network
YPred = model.predict(test_data, verbose =1)

#Format will be [Data, Predicted]
predicted_data = np.hstack((test_data, YPred))

# np.savetxt("Predicted.dat", predicted_data)

#save the tested data
filepath = os.path.splitext(file)[0]
# longname = filepath.split("/")[1]
test_name = filepath.split(".")[0]

# The cutoff for what is considered a hit
cutoff = 0.8

# Highlighting potential hits
potential = predicted_data[np.unique(np.where(YPred > cutoff)[0]),:]
# np.savetxt('Potential_Detections_in_'+test_name+'_with_'+ANN+'.dat', potential)

# Create histograms for each peak
for i in range(len(YPred[0])):
    hbin = np.linspace(min(YPred[:,i]), max(YPred[:,i]), num = 100).reshape(-1)
    plt.hist(YPred[np.where(YPred[:,i] < cutoff)[0]][:,i], bins = hbin, density = False, label = 'Non-Detections')
    plt.hist(YPred[np.where(YPred[:,i] > cutoff)[0]][:,i], bins = hbin, density = False, label = 'Potential Detections')
    plt.xlabel("Predicted likelihood of being a real detection by network")
    plt.ylabel("Number of spectra")
    plt.title(ANN+" trained network predicting "+test_name+f" for peak {i+1}")
    plt.yscale('log')
    plt.show()
    print(f'Found peak {i+1} in {len(np.where(YPred[:,i] > cutoff)[0])/len(YPred) *100}% of spectra')
    print(f'Expected value: {np.unique(lines_present[i], return_counts=True)[1][1] / len(YPred) * 100}')
    print(f'{len(np.where(YPred[:,i] > cutoff)[0]) / np.unique(lines_present[i], return_counts=True)[1][1]}\n' )

lines_present = lines_present.T

real_det = np.array([]) # real detections
miss_det = np.array([]) # missed detections
false_pos = np.array([]) # false positives
nothing = np.array([]) # actually nothing there and reported as nothing

for i in range(len(YPred)):
    for j in range(len(YPred[i])):
        status = lines_present[i][j] - YPred[i][j]
        
        if status >= 0 and status <= 1 - cutoff:
            real_det = np.append(real_det, [i,j])
        elif status > 1 - cutoff:
            miss_det = np.append(miss_det, [i,j])
        elif status > 0 - cutoff and status <= 0:
            nothing = np.append(nothing, [i,j])
        elif status <= 0 - cutoff:
            false_pos = np.append(false_pos, [i,j])

real_det = np.reshape(real_det, (int(len(real_det)/2), 2))
miss_det = np.reshape(miss_det, (int(len(miss_det)/2), 2))
false_pos = np.reshape(false_pos, (int(len(false_pos)/2), 2))
nothing = np.reshape(nothing, (int(len(nothing)/2), 2))

confusion = np.array([[len(real_det), len(false_pos)], [len(miss_det), len(nothing)]], dtype='f')


# Raw numbers
ax = plt.gca()
ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')
plt.imshow(confusion, cmap='Greys')
plt.colorbar()
plt.yticks([0, 1], ['Line present', 'Line not present'], rotation='vertical', va='center')
plt.xticks([0, 1], ['Line detected', 'Line not detected'])
plt.xlabel('Is the line detected?')
plt.ylabel('Is the line present?')
plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN}')
plt.text(0, 0, confusion[0][0], ha='center', backgroundcolor='w')
plt.text(0, 1, confusion[0][1], ha='center', backgroundcolor='w')
plt.text(1, 0, confusion[1][0], ha='center', backgroundcolor='w')
plt.text(1, 1, confusion[1][1], ha='center', backgroundcolor='w')
plt.show()

# Relative values
no_false, no_lines = np.unique(lines_present, return_counts=True)[1]

confusion[:,0] /= no_lines
confusion[:,1] /= no_false

ax = plt.gca()
ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none') 
plt.imshow(confusion, cmap='Greys')
plt.colorbar()
plt.yticks([0, 1], ['Line present', 'Line not present'], rotation='vertical', va='center')
plt.xticks([0, 1], ['Line detected', 'Line not detected'])
plt.xlabel('Is the line detected?')
plt.ylabel('Is the line present?')
plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN}')
plt.text(0, 0, np.round(confusion[0][0], 5), ha='center', backgroundcolor='w')
plt.text(0, 1, np.round(confusion[0][1], 5), ha='center', backgroundcolor='w')
plt.text(1, 0, np.round(confusion[1][0], 5), ha='center', backgroundcolor='w')
plt.text(1, 1, np.round(confusion[1][1], 5), ha='center', backgroundcolor='w')
plt.show()



# Separately for each line

output = []

for i in range(len(lines_present[0])):
    confusion = np.array([[np.unique(real_det[:,1], return_counts=True)[1][i], np.unique(false_pos[:,1], return_counts=True)[1][i]], [np.unique(miss_det[:,1], return_counts=True)[1][i], np.unique(nothing[:,1], return_counts=True)[1][i]]], dtype='f')

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.imshow(confusion, cmap='Greys')
    plt.colorbar()
    plt.yticks([0, 1], ['Line present', 'Line not present'], rotation='vertical', va='center')
    plt.xticks([0, 1], ['Line detected', 'Line not detected'])
    plt.xlabel('Is the line detected?')
    plt.ylabel('Is the line present?')
    plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN} for line {i+1}')
    plt.text(0, 0, confusion[0][0], ha='center', backgroundcolor='w')
    plt.text(0, 1, confusion[0][1], ha='center', backgroundcolor='w')
    plt.text(1, 0, confusion[1][0], ha='center', backgroundcolor='w')
    plt.text(1, 1, confusion[1][1], ha='center', backgroundcolor='w')
    plt.show()
    
    # Relative values
    no_false, no_lines = np.unique(lines_present[:,0], return_counts=True)[1]

    confusion[:,0] /= no_lines
    confusion[:,1] /= no_false

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.imshow(confusion, cmap='Greys')
    plt.colorbar()
    plt.yticks([0, 1], ['Line present', 'Line not present'], rotation='vertical', va='center')
    plt.xticks([0, 1], ['Line detected', 'Line not detected'])
    plt.xlabel('Is the line detected?')
    plt.ylabel('Is the line present?')
    plt.title(f'Confusion matrix for all lines in {test_name}\nusing {ANN} for line {i+1}')
    plt.text(0, 0, np.round(confusion[0][0], 5), ha='center', backgroundcolor='w')
    plt.text(0, 1, np.round(confusion[0][1], 5), ha='center', backgroundcolor='w')
    plt.text(1, 0, np.round(confusion[1][0], 5), ha='center', backgroundcolor='w')
    plt.text(1, 1, np.round(confusion[1][1], 5), ha='center', backgroundcolor='w')
    plt.show()
    
    
    output.append(f'{confusion[0][0]}, {confusion[0][1]}, {confusion[1][0]}, {confusion[1][1]}')


f = open(f'confusion_{ANN[:-6]}_{test_name}.txt', 'w')

for i in range(len(output)):
    f.write(f'{output[i]}\n')
f.close()







