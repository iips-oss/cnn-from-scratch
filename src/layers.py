import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        # small weights no gradient explosion
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data  # falttened 1D vector
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        # calculating gradient
        weights_gradient = self.input.T @ output_gradient  # @ is matrix product
        input_gradient = output_gradient @ self.weights.T

        # updating parameters
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

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

    def forward(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape) + self.biases

        for i in range(self.num_kernels):  # go through every kernel
            for j in range(self.input_depth):  # go through every channel of image
                for y in range(self.output_shape[1]):  # go through every row
                    for x in range(self.output_shape[2]):  # go through every column
                        # extract region
                        region_of_intrest = self.input[
                            j,
                            # a : b => slicing a to b
                            y : y + self.kernel_shape[2],
                            x : x + self.kernel_shape[3],
                        ]

                        # apply kernel filter to region
                        self.output[i, y, x] += np.sum(
                            region_of_intrest * self.kernels[i, j]
                        )

        return self.output

    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                for y in range(self.output_shape[1]):
                    for x in range(self.output_shape[2]):
                        region_of_intrest = self.input[
                            j,
                            y : y + self.kernel_shape[2],
                            x : x + self.kernel_shape[3],
                        ]
                        # weight sharing accumilation
                        kernel_gradient[i, j] += (
                            output_gradient[i, y, x] * region_of_intrest
                        )

                        # input gradient accumilation
                        input_gradient[
                            j,
                            y : y + self.kernel_shape[2],
                            x : x + self.kernel_shape[3],
                        ] += output_gradient[i, y, x] * self.kernels[i, j]

        self.kernels -= learning_rate * kernel_gradient
        bias_gradient = np.sum(output_gradient, axis=(1, 2), keepdims=True)

        self.biases -= learning_rate * bias_gradient
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

        output = np.zeros(self.output_shape)

        for d in range(depth):
            for row in range(out_height):
                for col in range(out_width):
                    start_y = row * self.stride
                    start_x = col * self.stride

                    # taking current window
                    patch = input_data[
                        d,
                        start_y : start_y + self.pool_size,
                        start_x : start_x + self.pool_size,
                    ]

                    output[d, row, col] = np.max(patch)

        return output

    def backward(self, output_gradient):
        input_gradient = np.zeros_like(self.input)

        depth, out_height, out_width = self.output_shape

        for d in range(depth):
            for row in range(out_height):
                for col in range(out_width):
                    start_y = row * self.stride
                    start_x = col * self.stride

                    patch = self.input[
                        d,
                        start_y : start_y + self.pool_size,
                        start_x : start_x + self.pool_size,
                    ]

                    max_val = np.max(patch)

                    # gradient only goes to max value position
                    for i in range(self.pool_size):
                        for j in range(self.pool_size):
                            if patch[i, j] == max_val:
                                input_gradient[
                                    d,
                                    start_y + i,
                                    start_x + j,
                                ] += output_gradient[d, row, col]

        return input_gradient
