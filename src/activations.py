import numpy as np
import math


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class GELU:
    def __init__(self, approximate='none'):
        self.approximate = approximate
        self.x = None

    def forward(self, x):
        self.x = x
        if self.approximate == 'tanh':
            inner = np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
            return 0.5 * x * (1 + np.tanh(inner))
        else:
            erf_vec = np.vectorize(math.erf)
            return 0.5 * x * (1 + erf_vec(x / np.sqrt(2)))

    def backward(self, grad):
        x = self.x
        if self.approximate == 'tanh':
            c = np.sqrt(2 / np.pi)
            inner = c * (x + 0.044715 * np.power(x, 3))
            tanh_inner = np.tanh(inner)
            d_inner = c * (1.0 + 3.0 * 0.044715 * np.square(x))
            dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - np.square(tanh_inner)) * d_inner
            return grad * dx
        else:
            erf_vec = np.vectorize(math.erf)
            cdf = 0.5 * (1.0 + erf_vec(x / np.sqrt(2)))
            pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.square(x))
            dx = cdf + x * pdf
            return grad * dx

