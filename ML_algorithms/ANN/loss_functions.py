import numpy as np

class Loss:
    def loss(self):
        raise NotImplementedError
    
    def loss_derivative(self):
        raise NotImplementedError

class Mse(Loss):
    def loss(self, y_predicted, y_true):
        return np.mean((y_predicted - y_true) ** 2)

    def loss_derivative(self, y_predicted, y_true):
        return 2 * (y_predicted - y_true) / len(y_true)

