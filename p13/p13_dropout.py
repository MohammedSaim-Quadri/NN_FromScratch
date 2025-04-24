import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_data import spiral_data
from p11.p11_gradientdescent import Optimizer_Adam
np.random.seed(0)

'''
Lets add Dropout layer logic to our already existing code.
UPDATED:
- Added class Layer_Dropout after Layer_Dense class.
'''
class Layer_Dense:
#Layer initialization
    def __init__(self, n_inputs, n_neurons,
                weight_regularizer_l1=0, weight_regularizer_l2=0,
                bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass 
    def forward(self, inputs): 
        # Remember input values 
        self.inputs = inputs 
        # Calculate output values from inputs, weights and biases 
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass 
    def backward(self, dvalues): 
        # Gradients on parameters 
        self.dweights = np.dot(self.inputs.T, dvalues) 
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) 
        # Gradients on regularization 
        # L1 on weights 
        if self.weight_regularizer_l1 > 0: 
            dL1 = np.ones_like(self.weights) 
            dL1[self.weights < 0] = -1 
            self.dweights += self.weight_regularizer_l1 * dL1 
        # L2 on weights 
        if self.weight_regularizer_l2 > 0: 
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights 
        # L1 on biases 
        if self.bias_regularizer_l1 > 0: 
            dL1 = np.ones_like(self.biases) 
            dL1[self.biases < 0] = -1 
            self.dbiases += self.bias_regularizer_l1 * dL1 
        # L2 on biases 
        if self.bias_regularizer_l2 > 0: 
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases 
        # Gradient on values 
        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout 
class Layer_Dropout: 
    # Init 
    def __init__(self, rate): 
        # Store rate, we invert it as for example for dropout 
        # of 0.1 we need success rate of 0.9 
        self.rate = 1 - rate 

    # Forward pass 
    def forward(self, inputs): 
        # Save input values 
        self.inputs = inputs 
        # Generate and save scaled mask 
        self.binary_mask = np.random.binomial(1, self.rate,size=inputs.shape) / self.rate 
        # Apply mask to output values 
        self.output = inputs * self.binary_mask 

    # Backward pass 
    def backward(self, dvalues): 
        # Gradient on values 
        self.dinputs = dvalues * self.binary_mask


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
    # Regularization loss calculation 
    def regularization_loss(self, layer): 
        # 0 by default 
        regularization_loss = 0 

        # L1 regularization - weights 
        # calculate only when factor greater than 0 
        if layer.weight_regularizer_l1 > 0: 
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights)) 
        # L2 regularization - weights 
        if layer.weight_regularizer_l2 > 0: 
            regularization_loss += layer.weight_regularizer_l2 *    \
                                np.sum(layer.weights * layer.weights) 
        # L1 regularization - biases 
        # calculate only when factor greater than 0 
        if layer.bias_regularizer_l1 > 0: 
            regularization_loss += layer.bias_regularizer_l1 * \
                                    np.sum(np.abs(layer.biases)) 

        # L2 regularization - biases 
        if layer.bias_regularizer_l2 > 0: 
            regularization_loss += layer.bias_regularizer_l2 * \
                                    np.sum(layer.biases * layer.biases) 
        return regularization_loss 

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
print('Training started....')
X, y = spiral_data(points=10000, classes=3) 

# Create Dense layer with 2 input features and 64 output values 
dense1 = Layer_Dense(2, 256, weight_regularizer_l2=5e-4, 
                            bias_regularizer_l2=5e-4) 

# Create ReLU activation (to be used with Dense layer): 
activation1 = Activation_ReLU() 

# Create dropout layer 
dropout1 = Layer_Dropout(0.1) 

# Create second Dense layer with 64 input features (as we take output 
# of previous layer here) and 3 output values (output values) 
dense2 = Layer_Dense(256, 3) 

# Create Softmax classifier's combined loss and activation 
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy() 

# Create optimizer 
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5) 

# Train in loop 
for epoch in range(10001): 

    # Perform a forward pass of our training data through this layer 
    dense1.forward(X) 

    # Perform a forward pass through activation function 
    # takes the output of first dense layer here 
    activation1.forward(dense1.output) 

    # Perform a forward pass through Dropout layer  
    dropout1.forward(activation1.output) 

    # Perform a forward pass through second Dense layer 
    # takes outputs of activation function of first layer as inputs 
    dense2.forward(dropout1.output) 

    # Perform a forward pass through the activation/loss function 
    # takes the output of second dense layer here and returns loss 
    data_loss = loss_activation.forward(dense2.output, y) 

    # Calculate regularization penalty 
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2) 

    # Calculate overall loss 
    loss = data_loss + regularization_loss 

    # Calculate accuracy from output of activation2 and targets 
    # calculate values along first axis 
    predictions = np.argmax(loss_activation.output, axis=1) 
    if len(y.shape) == 2: 
        y = np.argmax(y, axis=1) 
    accuracy = np.mean(predictions==y) 

    if not epoch % 100: 
        print(f'epoch: {epoch}, ' + 
                f'acc: {accuracy:.3f}, ' + 
                f'loss: {loss:.3f} (' + 
                f'data_loss: {data_loss:.3f}, ' + 
                f'reg_loss: {regularization_loss:.3f}), ' + 
                f'lr: {optimizer.current_learning_rate}') 

    # Backward pass 
    loss_activation.backward(loss_activation.output, y) 
    dense2.backward(loss_activation.dinputs) 
    dropout1.backward(dense2.dinputs) 
    activation1.backward(dropout1.dinputs) 
    dense1.backward(activation1.dinputs) 

    # Update weights and biases 
    optimizer.pre_update_params() 
    optimizer.update_params(dense1) 
    optimizer.update_params(dense2) 
    optimizer.post_update_params() 

print('Training completed....')


# Validate the model 
print('Validation started....')
np.random.seed(1)
# Create test dataset 
X_test, y_test = spiral_data(samples=100, classes=3) 
# Perform a forward pass of our testing data through this layer 
dense1.forward(X_test) 
# Perform a forward pass through activation function 
# takes the output of first dense layer here 
activation1.forward(dense1.output) 
# Perform a forward pass through second Dense layer 
# takes outputs of activation function of first layer as inputs 
dense2.forward(activation1.output) 
# Perform a forward pass through the activation/loss function 
# takes the output of second dense layer here and returns loss 
loss = loss_activation.forward(dense2.output, y_test) 
# Calculate accuracy from output of activation2 and targets 
# calculate values along first axis 
predictions = np.argmax(loss_activation.output, axis=1) 
if len(y_test.shape) == 2: 
    y_test = np.argmax(y_test, axis=1) 
accuracy = np.mean(predictions==y_test) 
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

print('Validation completed....')