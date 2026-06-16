import numpy as np
from layer import Layer

class Conv(Layer):
    def __init__(self,input_shape,kernel_size,num_kernels):
        input_depth,input_height,input_width = input_shape
        self.num_kernels = num_kernels
        self.input_shape = input_shape
        self.input_depth = input_depth

        self.output_shape = (num_kernels,
                          input_height - kernel_size + 1,
                          input_width - kernel_size + 1) 
        #output shape will depend upon num kernels

        self.kernel_shape = (num_kernels,
                                input_depth,
                                kernel_size,
                                kernel_size
                            )
        self.kernels = np.random.randn(*self.kernel_shape) * 0.1
        self.biases = np.zeros((num_kernels, 1, 1))
    def forward(self,input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape) + self.biases

        #go through every kernel
        for i in range(self.num_kernels):
            #go through every channel of image
            for j in range(self.input_depth):
                # go through every row
                for y in range(self.output_shape[1]):
                    #go through every column
                    for x in range(self.output_shape[2]):
                        region_of_intrest = self.input[j,y : y + self.kernel_shape[2],x : x + self.kernel_shape[3]] # extract region
                        self.output[i,y,x] += np.sum(region_of_intrest * self.kernels[i,j]) # do the operation 
        
        
        return self.output
    def backward(self,output_gradient,learning_rate):
        kernel_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                for y in range(self.output_shape[1]):
                    for x in range(self.output_shape[2]):
                        region_of_intrest = self.input[j,y : y + self.kernel_shape[2],x : x + self.kernel_shape[3]]
                        kernel_gradient[i,j] += output_gradient[i,y,x] * region_of_intrest

                        input_gradient[j,y : y + self.kernel_shape[2],x : x+ self.kernel_shape[3]] += (output_gradient[i,y,x] * self.kernels[i,j])
        
        self.kernels -= learning_rate * kernel_gradient
        bias_gradient = np.sum(
                            output_gradient,
                            axis=(1, 2),
                            keepdims=True
                        )

        self.biases -= learning_rate * bias_gradient
        return input_gradient