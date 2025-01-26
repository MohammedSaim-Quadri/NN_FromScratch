'''
The last remaining parts to the forward propagation is to apply another activation function specific-
to the output layer, before calculation of loss(error), which is prerequisite to backpropagation.
Here we are implementing the activation function for the output layer/neurons in case of a multiclass scenario, which is called softmax.
'''
'''
The reason behind applying a separate specific activation just to the output layer/neurons, is that after the the last layer(layer just befor o/p layer) -
gives an output, it often has negative values or zeros, so anyother acitvation fucntion like ReLU/sigmoid etc would by default give zero as output, which would be a problem - 
especially as this wont allow the model to learn, since these 0 values during backpropagation could cause other values to diminish, reducing the performance of the model drastically.
'''

import numpy as np
from create_data import spiral_data # using function to create data using np, instead of typing manually
np.random.seed(0)


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

# formula for softmax : e^x / summation of e^x from 1 to n
# the numerator is the exponentiation part and the denominator part is called the normalization part
# the exponentiation is done to get rid of the negative while the normalization helps retain the meaning of negative terms.

class Activation_Softmax:
    def forward(self,inputs):
        # exponentiate
        exp_vals = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) # subtracting largest value in the layer from each value in that layer inorder to prevent the problem of overflow due to exponentiation.
        # normalize
        probabilities = exp_vals / np.sum(exp_vals,axis=1,keepdims=True) # axis=1 to sum the rows of exp_vals matrix, and keepdims retains the shape
        self.output = probabilities


# input data and labels/classes
X,y = spiral_data(points = 100, classes= 3)

# using objects to create the layers
dense1 = Layer_Dense(2,3) #first HL
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3) # output layer
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
'''
[[0.33333333 0.33333333 0.33333333]
 [0.33331734 0.33331832 0.33336434]
 [0.3332888  0.33329153 0.33341967]
 [0.33325941 0.33326395 0.33347665]
 [0.33323311 0.33323926 0.33352763]]
'''