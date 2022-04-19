import numpy as np

class Layer:
    '''base class from which we will inherit'''

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        '''computes the output Y of a layer for a given X (input)'''
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        '''computes dE/dX for a given dE/dY (and update parameters if any)'''
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    pass
