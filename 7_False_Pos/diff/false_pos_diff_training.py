import os
#numpy + pyplot together
import numpy as np
#tensorflow with keras wrapper
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



# Import data
file = "false_data1.txt"

training_data = np.loadtxt(file)

# # Shuffle the data before input
# np.random.shuffle(training_data)

# Using same settings as Andrew
# Create the Network using the Keras 
# 10000 data points to train on, 8 outputs (1 for each line)
model = Sequential()
model.add(Dense(8, activation = "tanh", kernel_initializer='random_normal'))
#Adding hidden layers
model.add(Dense(114, activation = "tanh", kernel_initializer='random_normal'))
model.add(Dense(76, activation = "tanh", kernel_initializer='random_normal'))
#Adding output layer
model.add(Dense(8, activation = "sigmoid", kernel_initializer='random_normal'))
model.compile(loss = "mean_squared_error", optimizer='adam', metrics=['accuracy'])

# Train the ANN using stochastic gradient descent
# with a validation split of 20%
model.fit(training_data[:,:-8], training_data[:,-8:], epochs=500, batch_size=64, verbose=1, validation_split=0.2)


# Evaluate the model accuracy
_, accuracy = model.evaluate(training_data[:,:-8], training_data[:,-8:])



#save the trained network
filepath = os.path.splitext(file)[0]
name = filepath.split("-")[0]
print("%0.3f" % accuracy)


files_present = [f for f in os.listdir() if os.path.isfile(os.path.join(".", f)) and f[:len(name)] == f"{name}" and f[-6:] == ".keras"]


model.save(f'{name}_{len(files_present)}.keras')