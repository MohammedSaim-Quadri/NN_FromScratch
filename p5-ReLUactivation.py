'''
Every neuron has two parts to it, first part performs the summation(linear transformation) i.e. inputs*weights + bias - 
and the second part applies an activation function to the output of this step, this is to introduce non-linearity to NN.
This whole process of summation and then applying activation function to give a output is called the Forward Propagation,
and is done for all neurons in the network.
'''

import numpy as np
from create_data import spiral_data # using function to create data using np, instead of typing manually
np.random.seed(0)

# input data and labels/classes
X,y = spiral_data(100, 3)


# defining the hidden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # 0.10 is multiplied to keep the weights in range of -1 to 1
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # weights that will be size of input multiplied by number of neurons
        self.biases = np.zeros((1, n_neurons)) # by default biases are initialized to 0 unless necessary
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU : {x if x>=0; 0 if otherwise}
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

# using objects to create the layers
layer1 = Layer_Dense(2,5) #(R1,c1)
# layer2 = Layer_Dense(5,2) # (R2=c1, c2), because the output of layer1 is input for layer2
activation1 = Activation_ReLU()

layer1.forward(X)
print("layer 1 output:",layer1.output)
print("___________________________")
activation1.forward(layer1.output)
print("activation function output:",activation1.output)