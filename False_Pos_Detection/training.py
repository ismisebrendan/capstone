import os
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Import data
file = 'spectra_all_0.txt'

epochs = 500

training_data = np.loadtxt(file)

samples = len(training_data)
outputs = 8

layer1 = int(np.round(np.sqrt((outputs + 2) * samples) + 2 * np.sqrt(samples / (outputs + 2))))
layer2 = int(np.round(outputs * np.sqrt(samples / (outputs + 2))))

print(f'{layer1} nodes in hidden layer 1')
print(f'{layer2} nodes in hidden layer 2')

# Create the Network using the Keras 
# 30000 data points to train on, 8 outputs (1 for each line)
model = Sequential()
model.add(Dense(8, activation = 'tanh', kernel_initializer='random_normal'))
#Adding hidden layers
model.add(Dense(layer1, activation = 'tanh', kernel_initializer='random_normal'))
model.add(Dense(layer2, activation = 'tanh', kernel_initializer='random_normal'))
#Adding output layer
model.add(Dense(8, activation = 'sigmoid', kernel_initializer='random_normal'))
model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train the ANN using stochastic gradient descent
# with a validation split of 20%
model.fit(training_data[:,:-8], training_data[:,-8:], epochs=epochs, batch_size=64, verbose=1, validation_split=0.2)


# Evaluate the model accuracy
_, accuracy = model.evaluate(training_data[:,:-8], training_data[:,-8:])



#save the trained network
filepath = os.path.splitext(file)[0]
name = filepath.split('/')[-1]

folder = filepath.split('/')[0:-1]
folder = '/'.join(folder)
print('%0.3f' % accuracy)


files_present = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[:len(name)] == f'{name}' and f[-6:] == '.keras']


model.save(f'{folder}/{name}_ANN_{len(files_present)}.keras')

print(f'Model: {name}_ANN_{len(files_present)}.keras generated')