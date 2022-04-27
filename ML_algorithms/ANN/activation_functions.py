import numpy as np

class ActivationFunction():
    def activation(self):
        raise NotImplementedError
    
    def activation_derivative(self):
        raise NotImplementedError

class Sigmoid:
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        sigmoid = sigmoid(x)

        return sigmoid * (1 - sigmoid)
        
class Tanh:
    def activation(self, x):
        return np.tanh(x)

    def activation_derivative(self, x):
        return 1 - np.tanh(x) ** 2

class ReLu:
    def activation(self, x):
        return x * (x > 0)

    def activation_derivative(x):
        return 1. * (x > 0)
