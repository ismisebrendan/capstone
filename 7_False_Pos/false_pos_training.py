import os
#numpy + pyplot together
import numpy as np
#tensorflow with keras wrapper
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



# Import data
file = "false_pos-20240930T161616.txt"

training_data = np.loadtxt(file, skiprows = 1)

# Using same settings as Andrew
# Create the Network using the Keras 
model = Sequential()
model.add(Dense(8, activation = "tanh", kernel_initializer='random_normal'))
#Adding hidden layers
model.add(Dense(129, activation = "tanh", kernel_initializer='random_normal'))
model.add(Dense(26, activation = "tanh", kernel_initializer='random_normal'))
#Adding output layer
model.add(Dense(1, activation = "sigmoid", kernel_initializer='random_normal'))
model.compile(loss = "mean_squared_error", optimizer='adam', metrics=['accuracy'])

# Train the ANN using stochastic gradient descent
# with a validation split of 20%
model.fit(training_data[:,:-1], training_data[:,-1], epochs=500, batch_size=64, verbose=1, validation_split=0.2)


# Evaluate the model accuracy
_, accuracy = model.evaluate(training_data[:,:-1], training_data[:,-1])



#save the trained network
filepath = os.path.splitext(file)[0]
name = filepath.split("-")[0]
print("%0.3f" % accuracy)
model.save(f'{name}.keras')