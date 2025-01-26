'''
The final step before backpropagation is to calculate how right or how wrong the network is with its prediction -
this is calculated using something called a loss function, which calculate the error in the prediction wrt the actual values - 
simply put this function calculates the difference between the predicted and the actual value.
'''
"""
for classfication problems we make use of cross entropy as a loss function rather than the usua MAE, MSE, RMSE-
which are used in regression problems.
Categorical cross entropy : L = −Σ yi,j * log(ŷi,j)
Li - sample loss value ;  y - target values
i - i-th sample in a set ; ŷ - predicted values
j - label/output index
"""
import math

softmax_output = [0.7, 0.1, 0.2] # outputs from the output layer after activation layer applied
target_output = [1, 0, 0] # target values based on one-hot encoding

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

print(-math.log(0.7))
print(-math.log(0.5))