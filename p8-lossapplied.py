'''
applying the loss function to the neural network code by creating its own class and objects
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

"""
This code defines a base `Loss` class for computing loss in a neural network and a 
specific implementation `Loss_CategoricalCrossentropy` for categorical cross-entropy loss.

- The `Loss` class includes a `calculate` method to compute the average loss for a batch 
  of data by calling the `forward` method implemented in derived classes.

- The `Loss_CategoricalCrossentropy` class calculates the negative log likelihood for 
  categorical data:
  1. Clipping predictions to prevent numerical instability, such as taking the negative 
     log of zero, which would result in infinity.
  2. Handling two cases for labels:
     - Integer labels (sparse): Select the predicted probabilities using indices.
     - One-hot encoded labels: Compute the dot product between predictions and labels.
  3. Applying the negative logarithm to the correct class confidences to compute the loss.
"""

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


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

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)