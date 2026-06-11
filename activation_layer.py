import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError


class ReLU(Layer):
    def __init__(self):
        self.mask = None  # mask is variable which is assigned nothing for now

    def forward(self, x):
        self.mask = (x > 0)          # this will return true or false i.e. 1/0
        return x * self.mask         # all the negative numbers will become zero

    def backward(self, grad):
        return grad * self.mask      # pass gradient only when input was > 0
