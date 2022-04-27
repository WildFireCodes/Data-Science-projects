import numpy as np

class NeuralNetwork:
    def __init__(self, loss_function):
        self.loss_function = loss_function
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x_train, y_train, epochs = 100, learning_rate = 0.001, verbose = 0):
        for i in range(epochs):
            error = 0
            for j in range(len(x_train)):
                output = x_train[j]
                for l in self.layers:
                    output = l.forward_propagation(output)

                error += self.loss_function.loss(output, y_train[j])
                gradient = self.loss_function.loss_derivative(output, y_train[j])

                for l in reversed(self.layers):
                    gradient = l.backward_propagation(gradient, learning_rate)
            
            if verbose:
                error = error/len(x_train)
                print(f"Epoch {i}, loss: {error}")

    def predict(self, x_test):
        predicted = []

        for i in range(len(x_test)):
            output = x_test[i]
            for l in self.layers:
                output = l.forward_propagation(output)
        
        predicted.append(output)
        return predicted
