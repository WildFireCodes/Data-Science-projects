import numpy as np

class Layer:
    '''base class from which we will inherit'''

    def __init__(self):
        self.x = None
        self.y = None

    def forward_propagation(self, x):
        '''computes the output Y of a layer for a given X (input)'''
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        '''computes dE/dX for a given dE/dY (and update parameters if any)'''
        raise NotImplementedError
        
class Activation(Layer):
    def __init__(self, activation_function):
        self.activation_function = activation_function
    
    def forward_propagation(self, x):
        self.x = x
        self.output = self.activation_function.activation(x)

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error * self.activation_function.activation_derivative(self.x)

class Dense(Layer):
    def __init__(self, n_input, n_output):
        '''n_input = number of neurons in the first layer
           n_output = number of neurons in the last layer'''
        # if n_input == 0 or n_output == 0:
        #     raise ValueError

        self.weight = np.random.randn(n_input, n_output) * np.sqrt(2 / n_input)
        self.bias = np.random.randn(1, n_output)
    
    def forward_propagation(self, x):
        self.x = x
        self.output = np.dot(self.x, self.weight) + self.bias
        
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        calculated_weights_error = np.dot(self.x.T, output_error)
        gradient = np.dot(output_error, self.weight.T)

        self.weight -= learning_rate * calculated_weights_error
        self.bias -= learning_rate * output_error

        return gradient
    
