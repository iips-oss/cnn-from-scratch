import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        # small weights no gradient explosion
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.dweights = None
        self.dbiases = None

    def forward(self, input_data):
        self.input = input_data  # falttened 1D vector
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate=None):
        # calculating gradient
        self.dweights = self.input.T @ output_gradient  # @ is matrix product
        self.dbiases = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = output_gradient @ self.weights.T

        # updating parameters if learning_rate is provided
        if learning_rate is not None:
            self.weights -= learning_rate * self.dweights
            self.biases -= learning_rate * self.dbiases

        return input_gradient


class Conv:
    def __init__(self, input_shape, kernel_size, num_kernels):
        input_depth, input_height, input_width = input_shape

        self.input_shape = input_shape
        self.input_depth = input_depth
        self.num_kernels = num_kernels

        # output shape depends on num kernels
        # ie. num kernels = num of output feature maps
        # no input_depth in output shape because each kernel process all input channels
        self.output_shape = (
            num_kernels,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )

        self.kernel_shape = (
            num_kernels,
            input_depth,  # color channels
            kernel_size,  # k_height
            kernel_size,  # k_width
        )
        self.kernels = np.random.randn(*self.kernel_shape) * 0.1
        self.biases = np.zeros((num_kernels, 1, 1))
        self.dkernels = None
        self.dbiases = None

    def forward(self, input_data):
        self.input = input_data
        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(input_data, (self.kernel_shape[2], self.kernel_shape[3]), axis=(1, 2))
        self.output = np.einsum('jyxkl,ijkl->iyx', patches, self.kernels) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(self.input, (self.kernel_shape[2], self.kernel_shape[3]), axis=(1, 2))
        
        self.dkernels = np.einsum('iyx,jyxkl->ijkl', output_gradient, patches)
        self.dbiases = np.sum(output_gradient, axis=(1, 2), keepdims=True)

        input_gradient = np.zeros(self.input_shape)
        for y in range(self.output_shape[1]):
            for x in range(self.output_shape[2]):
                input_gradient[:, y : y + self.kernel_shape[2], x : x + self.kernel_shape[3]] += np.tensordot(
                    output_gradient[:, y, x], self.kernels, axes=(0, 0)
                )

        if learning_rate is not None:
            self.kernels -= learning_rate * self.dkernels
            self.biases -= learning_rate * self.dbiases

        return input_gradient


class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input = input_data

        depth, height, width = input_data.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        self.output_shape = (depth, out_height, out_width)

        from numpy.lib.stride_tricks import sliding_window_view
        patches = sliding_window_view(input_data, (self.pool_size, self.pool_size), axis=(1, 2))
        patches = patches[:, ::self.stride, ::self.stride]
        
        return np.max(patches, axis=(3, 4))

    def backward(self, output_gradient):
        input_gradient = np.zeros_like(self.input)

        depth, out_height, out_width = self.output_shape

        for row in range(out_height):
            for col in range(out_width):
                start_y = row * self.stride
                start_x = col * self.stride
                
                patch = self.input[:, start_y : start_y + self.pool_size, start_x : start_x + self.pool_size]
                max_val = np.max(patch, axis=(1, 2), keepdims=True)
                mask = (patch == max_val)
                
                input_gradient[:, start_y : start_y + self.pool_size, start_x : start_x + self.pool_size] += (
                    output_gradient[:, row, col][:, np.newaxis, np.newaxis] * mask
                )

        return input_gradient


class Flatten:
    def __init__(self):
        self.input_shape = None
        self.batch_size = 0
        
    def forward(self, x):
        self.input_shape = x.shape
        self.batch_size = x.shape[0]
        return x.reshape(self.batch_size, -1)
        
    def backward(self, grad_input):
        return grad_input.reshape(self.input_shape)
