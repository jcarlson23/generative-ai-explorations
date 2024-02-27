import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test  = utils.to_categorical(y_test, NUM_CLASSES)

# get the green channel of pixel (12,13)
print("The green channel at (12,13) is %f" % x_train[54,12,13,1])

# builds the MLP (sequential layers of the NN)
"""
Sequential method of building the MLP

model = models.Sequential( [
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(200, activation = 'relu'),
    layers.Dense(150, activation = 'relu'),
    layers.Dense(10, activation = 'softmax'),
])

"""

""" 
Building the MLP via the functional API
"""
input_layers = layers.Input(shape=(32,32,3))
x = layers.Flatten()(input_layer)
x = layers.Dense(units=200, activation = 'relu')(x)
x = layers.Dense(units=150, activation = 'relu')(x)
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(input_layer, output_layer)

# The 'relu' and 'softmax' are activation functions, or
# more accurately, a description of them. ReLU is
# "Rectified Linear Unit"
# More information can be found here:
#
# https://medium.com/@cmukesh8688/activation-functions-sigmoid-tanh-relu-leaky-relu-softmax-50d3778dcea5
#


