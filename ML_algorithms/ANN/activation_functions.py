import numpy as np

#class?
def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def sigmoid_derivative(x):
    sigmoid = sigmoid(x)

    return sigmoid * (1 - sigmoid)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return x * (x > 0)

def relu_derivative(x):
    return 1 * (x > 0)
