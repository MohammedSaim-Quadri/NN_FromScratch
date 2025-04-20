'''
First we will implement backpropagation through a single neuron to understand the flow of the process and 
see how the loss gets calculated & minimimized, using chaing rule of derivative.
'''

# import numpy as np

# # Initial parameters
# weights = np.array([-3.0, -1.0, 2.0])
# bias = 1.0
# inputs = np.array([1.0, -2.0, 3.0])
# target_output = 0.0
# learning_rate = 0.001

# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1.0, 0.0)

# for iteration in range(200):
#     # Forward pass
#     linear_output = np.dot(weights, inputs) + bias
#     output = relu(linear_output)
#     loss = (output - target_output) ** 2

#     # Backward pass
#     dloss_doutput = 2 * (output - target_output)
#     doutput_dlinear = relu_derivative(linear_output)
#     dlinear_dweights = inputs
#     dlinear_dbias = 1.0

#     dloss_dlinear = dloss_doutput * doutput_dlinear
#     dloss_dweights = dloss_dlinear * dlinear_dweights
#     dloss_dbias = dloss_dlinear * dlinear_dbias

#     # Update weights and bias
#     weights -= learning_rate * dloss_dweights
#     bias -= learning_rate * dloss_dbias

#     # Print the loss for this iteration
#     print(f"Iteration {iteration + 1}, Loss: {loss}")

# print("Final weights:", weights)
# print("Final bias:", bias)


'''
Now we will implement the same backpropagation but through an entire layer of neurons using the same process,
and see how the process works in an actual neural network.
'''

# # Initial inputs
# inputs = np.array([1, 2, 3, 4])

# # Initial weights and biases
# weights = np.array([
#     [0.1, 0.2, 0.3, 0.4],
#     [0.5, 0.6, 0.7, 0.8],
#     [0.9, 1.0, 1.1, 1.2]
# ])

# biases = np.array([0.1, 0.2, 0.3])

# # Learning rate
# learning_rate = 0.001

# # ReLU activation function and its derivative
# def relu(x):
#     return np.maximum(0, x)

# def relu_derivative(x):
#     return np.where(x > 0, 1, 0)

# # Training loop
# for iteration in range(200):
#     # Forward pass
#     z = np.dot(weights, inputs) + biases
#     a = relu(z)
#     y = np.sum(a)

#     # Calculate loss
#     loss = y ** 2

#     # Backward pass
#     # Gradient of loss with respect to output y
#     dL_dy = 2 * y

#     # Gradient of y with respect to a
#     dy_da = np.ones_like(a)

#     # Gradient of loss with respect to a
#     dL_da = dL_dy * dy_da

#     # Gradient of a with respect to z (ReLU derivative)
#     da_dz = relu_derivative(z)

#     # Gradient of loss with respect to z
#     dL_dz = dL_da * da_dz

#     # Gradient of z with respect to weights and biases
#     dL_dW = np.outer(dL_dz, inputs)
#     dL_db = dL_dz

#     # Update weights and biases
#     weights -= learning_rate * dL_dW
#     biases -= learning_rate * dL_db

#     # Print the loss every 20 iterations
#     if iteration % 20 == 0:
#         print(f"Iteration {iteration}, Loss: {loss}")

# # Final weights and biases
# print("Final weights:\n", weights)
# print("Final biases:\n", biases)


'''
as seen with above code, using gradient descent to adjust loss result in much better performance and much lower losses in models compared to our
previous approach.
Now lets intregrate this concept into our actual code that we have been building up until now.
'''

'''
Combining everything into one complete pipeline :
Forward Pass:
Input (X) → Dense1 → ReLU → Dense2 → Softmax → Loss

Backward Pass:
Loss gradient → Softmax gradient → Dense2 gradient → ReLU gradient → Dense1 gradient
'''

import numpy as np 
from create_data import spiral_data
np.random.seed(0)


# Dense layer 
class Layer_Dense:
    # Layer initialization 
    def __init__(self, n_inputs, n_neurons): 
        # Initialize weights and biases 
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1, n_neurons)) 

    # Forward pass 
    def forward(self, inputs): 
        # Remember input values 
        self.inputs = inputs 
        # Calculate output values from inputs, weights and biases 
        self.output = np.dot(inputs, self.weights) + self.biases 

    # Backward pass 
    def backward(self, dvalues): 
        # Gradient of loss with respect to weights: inputs^T @ dvalues
        # This follows from the chain rule of differentiation
        self.dweights = np.dot(self.inputs.T, dvalues) 

        # Gradient of loss with respect to biases: sum of dvalues across samples
        # We sum across axis 0 to get one gradient per neuron
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 

        # Gradient of loss with respect to inputs: dvalues @ weights^T
        # This is used to propagate gradients to earlier layers 
        self.dinputs = np.dot(dvalues, self.weights.T) 

# ReLU activation 
class Activation_ReLU: 
    # Forward pass 
    def forward(self, inputs): 
        # Remember input values 
        self.inputs = inputs 
        # Calculate output values from inputs 
        self.output = np.maximum(0, inputs) 

    # Backward pass 
    def backward(self, dvalues): 
        # Since we need to modify original variable, 
        # let's make a copy of values first 
        self.dinputs = dvalues.copy() 
        # ReLU derivative is 1 for inputs > 0 and 0 otherwise
        # Zero out gradients where inputs were negative
        self.dinputs[self.inputs <= 0] = 0 

# Softmax activation 
class Activation_Softmax: 
    # Forward pass 
    def forward(self, inputs): 
        # store inputs for backpropagation 
        self.inputs = inputs 
        # Subtract max value for numerical stability before exponentiation
        # This prevents overflow by keeping exponent values in a manageable range
        exp_values = np.exp(inputs - np.max(inputs, axis=1, 
                                            keepdims=True)) 
        # Normalize them for each sample 
        probabilities = exp_values / np.sum(exp_values, axis=1, 
                                            keepdims=True) 
        self.output = probabilities 

    # Backward pass 
    def backward(self, dvalues): 
        # Create uninitialized array 
        self.dinputs = np.empty_like(dvalues) 
        # Enumerate outputs and gradients 
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): 
            # Flatten output array 
            single_output = single_output.reshape(-1, 1) 
            # Calculate Jacobian matrix of softmax
            # For softmax: J(i,j) = S_i * (kronecker_delta(i,j) - S_j)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) 
            # Calculate sample-wise gradient 
            # and add it to the array of sample gradients 
            # Apply chain rule: dinputs = jacobian_matrix @ dvalues
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues) 


# Common loss class 
class Loss: 
    # Calculates the data and regularization losses 
    # given model output and ground truth values 
    def calculate(self, output, y): 
        # Calculate sample losses 
        sample_losses = self.forward(output, y) 
        # Calculate mean loss 
        data_loss = np.mean(sample_losses) 
        # Return loss 
        return data_loss 


# Cross-entropy loss -
# measures difference between predicted and true probability distributions 
class Loss_CategoricalCrossentropy(Loss): 
    # Forward pass 
    def forward(self, y_pred, y_true): 
        # Number of samples in a batch 
        samples = len(y_pred) 
        # Clip data to prevent division by 0 
        # Clip both sides to not drag mean towards any value 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) 
        # Handle different label formats:
        # If categorical labels (sparse, single integer per sample)
        if len(y_true.shape) == 1: 
            # Extract the predicted probability for the correct class
            correct_confidences = y_pred_clipped[ 
                range(samples), 
                y_true 
            ] 
        # Mask values - If one-hot encoded labels (vector of 0s with a 1 for correct class) 
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum( 
                y_pred_clipped * y_true, 
                axis=1 
            ) 

        '''
        #Cross-entropy loss: -log(correct_probability)
        # The negative log converts probabilities (0-1) to positive loss values
        # Higher confidence = lower loss, lower confidence = higher loss 
        '''
        negative_log_likelihoods = -np.log(correct_confidences) 
        return negative_log_likelihoods 

    # Backward pass 
    def backward(self, dvalues, y_true): 
        # Number of samples 
        samples = len(dvalues) 
        # Number of labels in every sample 
        # We'll use the first sample to count them 
        labels = len(dvalues[0]) 

        # If labels are sparse, turn them into one-hot vector 
        if len(y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true] 

        # Calculate gradient: -true_value / predicted_value
        # This derives from taking derivative of -log(x) with respect to x
        self.dinputs = -y_true / dvalues 
        # Normalize gradient 
        self.dinputs = self.dinputs / samples 


# Softmax classifier - combined Softmax activation 
# and cross-entropy loss for faster backward step 
'''
This provides numerical stability and computational efficiency
'''
class Activation_Softmax_Loss_CategoricalCrossentropy(): 
    # Creates activation and loss function objects 
    def __init__(self): 
        self.activation = Activation_Softmax() 
        self.loss = Loss_CategoricalCrossentropy() 

    # Forward pass 
    def forward(self, inputs, y_true): 
        # Output layer's activation function - apply softmax
        self.activation.forward(inputs) 
        # Store output for backward pass
        self.output = self.activation.output 
        # Calculate and return loss value 
        return self.loss.calculate(self.output, y_true) 

    # Backward pass 
    def backward(self, dvalues, y_true): 
        # Number of samples 
        samples = len(dvalues) 
        # If labels are one-hot encoded, 
        # turn them into discrete class indices if needed 
        if len(y_true.shape) == 2: 
            y_true = np.argmax(y_true, axis=1) 
        # Copy dvalues to avoid modifying the original data 
        self.dinputs = dvalues.copy() 
        
        '''
        # Calculate gradient: softmax derivative combined with cross-entropy derivative
        # This simplified formula works because of how softmax and cross-entropy 
        # derivatives interact mathematically (they partially cancel out)
        '''
        self.dinputs[range(samples), y_true] -= 1 
        # Normalize gradient 
        self.dinputs = self.dinputs / samples 


# Create dataset 
X, y = spiral_data(points=100, classes=3)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Create Dense layer with 2 input features and 3 output values 
dense1 = Layer_Dense(2, 3)
print("\n[Layer Dense1] Initialized")

# Create ReLU activation (to be used with Dense layer): 
activation1 = Activation_ReLU()
print("[Activation ReLU1] Initialized")

# Create second Dense layer with 3 input features (as we take output 
# of previous layer here) and 3 output values (output values) 
dense2 = Layer_Dense(3, 3) 
print("[Layer Dense2] Initialized")

# Create Softmax classifier's combined loss and activation 
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
print("[Softmax + CrossEntropy] Initialized") 

print("\n--- FORWARD PASS ---")
# Perform a forward pass of our training data through this layer 
dense1.forward(X)
print("[Dense1] Output shape:", dense1.output.shape)

# Perform a forward pass through activation function 
# takes the output of first dense layer here 
activation1.forward(dense1.output)
print("[ReLU1] Output shape:", activation1.output.shape)

# Perform a forward pass through second Dense layer 
# takes outputs of activation function of first layer as inputs 
dense2.forward(activation1.output)
print("[Dense2] Output shape:", dense2.output.shape)

# Perform a forward pass through the activation/loss function 
# takes the output of second dense layer here and returns loss 
loss = loss_activation.forward(dense2.output, y)
print("[Softmax + Loss] Output shape:", loss_activation.output.shape)

# Let's see output of the first few samples: 
print("\nFirst 5 Softmax outputs:")
print(loss_activation.output[:5]) 
# Print loss value 
print('loss:', loss) 
# Calculate accuracy from output of activation2 and targets 
# calculate values along first axis 
predictions = np.argmax(loss_activation.output, axis=1) 
if len(y.shape) == 2:
    y = np.argmax(y, axis=1) 
accuracy = np.mean(predictions==y) 

# Print accuracy 
print('acc:', accuracy) 

# Backward pass 
print("\n--- BACKWARD PASS ---")
loss_activation.backward(loss_activation.output, y)
print("[Loss + Softmax Backward] dinputs shape:", loss_activation.dinputs.shape)

dense2.backward(loss_activation.dinputs)
print("[Dense2 Backward] dweights shape:", dense2.dweights.shape)
print("[Dense2 Backward] dbiases shape:", dense2.dbiases.shape)

activation1.backward(dense2.dinputs)
print("[ReLU1 Backward] dinputs shape:", activation1.dinputs.shape)

dense1.backward(activation1.dinputs)
print("[Dense1 Backward] dweights shape:", dense1.dweights.shape)
print("[Dense1 Backward] dbiases shape:", dense1.dbiases.shape)

# Print gradients 
print("\n--- FINAL GRADIENTS ---")
print("Dense1 weights gradient:\n", dense1.dweights)
print("Dense1 biases gradient:\n", dense1.dbiases)
print("Dense2 weights gradient:\n", dense2.dweights)
print("Dense2 biases gradient:\n", dense2.dbiases)

'''
But this is not how 100% the way its actually done, as we still not -
updating the weights using the caluculated gradients
'''