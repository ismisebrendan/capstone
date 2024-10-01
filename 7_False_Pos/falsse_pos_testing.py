import os
#numpy + pyplot together
import numpy as np
#tensorflow with keras wrapper
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

ANN = 'false_pos.keras'
file = 'false_pos-testing.txt'

test_data = np.loadtxt(file)

model = tf.keras.models.load_model(ANN)
