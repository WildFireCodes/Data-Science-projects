import numpy as np

class Loss:
    def __init__(self, y_predicted, y_true):
        self.y_predicted = None
        self.y_true = None

    def loss(self):
        raise NotImplementedError
    
    def loss_derivative(self):
        raise NotImplementedError

class Mse(Loss):
    def __init__(self, y_predicted, y_true):
        self.y_predicted = y_predicted
        self.y_true = y_true

    def loss(self):
        return np.mean(np.power(self.y_true - self.y_predicted, 2))

    def loss_derivative(self):
        return 2 * (self.y_predicted - self.y_true) / len(self.y_true)

