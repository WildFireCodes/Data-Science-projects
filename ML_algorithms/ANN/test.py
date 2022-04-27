from keras.datasets import mnist
from keras.utils import np_utils
from activation_functions import Tanh
from neural_network import NeuralNetwork
from layers import Activation, Dense
from loss_functions import Mse

import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

#loss function - Mse object (MSE, MSE_derivative)
loss = Mse()
#activation function - Tanh object (Tanh, Tanh_derivative)
activation = Tanh()
#neural network with loss function MSE
network = NeuralNetwork(loss)

network.add(Dense(28*28, 100))
network.add(Activation(activation))

network.add(Dense(100, 50))
network.add(Activation(activation))

network.add(Dense(50, 10))
network.add(Activation(activation))

network.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1, verbose = 1)

out = network.predict(x_test[0:5])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])