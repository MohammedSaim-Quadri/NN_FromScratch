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

# GD optimizer
class Optimizer_GD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# Create dataset
X, y = spiral_data(points=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation (to be used with Dense layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_GD()

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}')
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


'''
Even though this code is utilizing the gradient descent optimizer to update the weights based on the calculated gradient, the base gradient descent 
optimizer is not ideal for the best performance as it suffers from the following problems:
    - For small learning rates, the problem of getting stuck in a local minima occurs, where instead of optimizing for the global minima,
    the optimization occurs wrt to the local minima leading to loss stagnation i.e., no further reduction in loss after some epochs, ultimately
    leading to low performance of the NN.
    - The second issue is, using large learning rates we end up with destabilized or unstable updates/optimization, due to the 
    potential for overshooting the minimum. This can cause the loss to oscillate wildly or even diverge completely, preventing convergence.
    - Gradient descent uses a fixed learning rate for all parameters, which isn't optimal since different parameters might require 
    different update magnitudes based on their importance and gradient history.
    - It doesn't adapt to the geometry of the error surface, treating all directions equally regardless of curvature.

To address these limitations, more advanced optimizers have been developed such as:
    - learning rate decay: Reduces the learning rate over time to allow for more precise convergence as the model approaches a minimum.
    - Momentum: Helps accelerate convergence and reduce oscillation by adding a fraction of the previous update vector.
    - RMSprop: Adapts learning rates per-parameter by using the running average of squared gradients.
    - Adam: Combines ideas from momentum and RMSprop, maintaining both a running average of gradients and squared gradients.
    - AdaGrad: Adapts learning rates for each parameter based on historical gradient information.
'''