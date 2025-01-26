import numpy as np
from create_data import vertical_data # using function to create data using np, instead of typing manually
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
X,y = vertical_data(samples = 100, classes= 3)

# using objects to create the layers
dense1 = Layer_Dense(2,3) #first HL
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3) # output layer
activation2 = Activation_Softmax()

# dense1.forward(X)
# activation1.forward(dense1.output)

# dense2.forward(activation1.output)
# activation2.forward(dense2.output)

# print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
# loss = loss_function.calculate(activation2.output, y)

# print("Loss:", loss)
'''
create some variables to track the best loss and the associated weights and biases-
We initialized the loss to a large value and will decrease it when a new, lower, loss is found. We
are also copying weights and biases (copy() ensures a full copy instead of a reference to the
object)
'''
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):# consider this to be 1 epoch having 10000 iterations

    #update the weights and biases with some small random values
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    #performing forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    #calculate accuracy from output of activation2 and target
    predictions = np.argmax(activation2.output, axis = 1)
    accuracy = np.mean(predictions == y)

    #if loss is smaller -- print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration', iteration,
              "loss:",loss, "acc:", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else: 
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

'''
Though this approach work well for our vertical(linear) data, it doesnt really work as well with the spiral data that we had been using-
earlier. later, we'll learn that the most probable reason for this is called a local minimum of loss. the data complexity is also not irrelevant here.
'''
# Random weight/bias changes fail due to infinite combinations and varied loss impacts, depending on parameters, inputs, and non-linearities.
# Loss is influenced indirectly by weights/biases through model output.
# Gradient Descent adjusts weights/biases by calculating their effect on loss using partial derivatives and backpropagation.
# Derivatives measure slope or impact of input on output; for nonlinear functions, the slope varies and is measured using tangents.
# Numerical differentiation approximates the derivative by calculating the slope of a tangent using two closely spaced points.

"""
Random search for weights/biases is inefficient due to infinite combinations and non-linear impacts on loss. Loss function indirectly depends on weights/biases via model output.
Gradients and partial derivatives measure how parameters influence loss, guiding optimization. 
Linear slopes (e.g., y=2x) are calculated as Δy/Δx, while nonlinear slopes (e.g., y=2x^2) use tangents to approximate derivatives with a small Δx for numerical differentiation.
A tangent's equation y=mx+b requires slope (m) and intercept (b). Approximating derivatives via small deltas balances accuracy and numerical stability. 
"""


