# Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.
# It works by iteratively adjusting the model parameters in the direction of the steepest descent of the loss function.
# The goal is to find the optimal parameters that minimize the loss function, leading to better model performance.
# The basic idea is to compute the gradient (or derivative) of the loss function with respect to the model parameters,
# and then update the parameters in the opposite direction of the gradient.
# This process is repeated until convergence, which is when the loss function reaches a minimum or stops changing significantly.
# Gradient descent can be applied to various types of models, including linear regression, logistic regression, and neural networks.

# Jump to the comprehensive optimizer summary: ctrl+f "OPTIMIZER_SUMMARY"


import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_data import spiral_data
from p10.p10_backpropagation import(
    Layer_Dense,
    Activation_ReLU,
    Activation_Softmax,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Loss_CategoricalCrossentropy,
    Loss
)

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


'''
Learning rate decay helps address one of the major issues with basic gradient descent: 
the fixed learning rate problem. By gradually reducing the learning rate over time:

1. We can start with a larger learning rate to make faster progress early in training
2. As we approach the minimum, we decrease the learning rate to make more precise adjustments
3. This helps overcome the oscillation problem that occurs with large fixed learning rates
4. It also helps escape shallow local minima that might trap an optimizer with a small fixed learning rate

Common decay strategies include:
- Time-based decay (used here): lr = initial_lr / (1 + decay * iteration)
- Step decay: lr = initial_lr * drop^(floor(epoch/epochs_drop))
- Exponential decay: lr = initial_lr * e^(-decay * iteration)
- Cosine annealing: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(iteration/max_iterations * pi))
'''

class Optimizer_GD_with_Decay:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1.0, decay=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases
        
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


'''
Momentum optimization improves upon standard gradient descent by adding a "velocity" term:

1. Instead of just moving in the direction of the current gradient, momentum accumulates past gradients 
2. This creates inertia, helping the optimizer move through flat regions and small local minima
3. It dampens oscillations in steep directions, allowing faster convergence in ravine-like surfaces
4. Effectively, momentum acts as a low-pass filter on noisy gradients

The update rule combines the current gradient with the previous velocity:
- v(t) = β * v(t-1) + (1-β) * gradient
- parameters -= learning_rate * v(t)

Where:
- β (typically 0.9) is the momentum coefficient that determines how much past gradients influence movement
- Higher β values (closer to 1) give more weight to past gradients
- When β=0, momentum reduces to standard gradient descent

Momentum often enables faster convergence and better generalization than standard SGD.
'''
class Optimizer_with_Monentum:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


'''
AdaGrad (Adaptive Gradient) addresses the challenge of choosing learning rates by adapting them per-parameter:

1. Unlike SGD, AdaGrad automatically adjusts learning rates for each parameter based on historical gradients
2. Parameters that receive large gradients get smaller learning rates, while rarely updated parameters receive larger updates
3. This makes it particularly effective for sparse features or when dealing with varied feature frequencies
4. AdaGrad maintains a cache of squared gradients that accumulates over time
'''
# Adagrad optimizer
class Optimizer_Adagrad:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        #Epsilon (typically 1e-7) prevents division by zero and improves numerical stability
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        # These caches are used to store the squared gradients
        # for each parameter (weights and biases)
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

'''
RMSprop (Root Mean Square Propagation) addresses AdaGrad's aggressive learning rate decay with exponential moving averages:

1. It improves upon AdaGrad by using a moving average of squared gradients rather than accumulating all past gradients
2. The decay factor (rho, typically 0.9) controls how much history influences the current update, also called cache decay rate.
3. This prevents the learning rate from becoming vanishingly small over time, allowing for continued learning
4. RMSprop effectively divides the learning rate by a moving average of recent gradient magnitudes
'''
class Optimizer_RMSprop:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                             (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                           (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


'''
Adam (Adaptive Moment Estimation) combines the best of RMSprop and momentum for robust optimization:

1. It maintains both first-moment (mean) and second-moment (uncentered variance) of gradients
2. The first moment acts like momentum, accelerating in consistent directions
3. The second moment adapts learning rates per-parameter like RMSprop
4. Bias correction terms fix initialization bias when exponential moving averages start at zero

Key implementation details:
- Beta1 (typically 0.9) controls the exponential decay rate for the first moment
- Beta2 (typically 0.999) controls the exponential decay rate for the second moment 
- Momentum cache tracks the moving average of gradients
- RMSprop-like cache tracks the moving average of squared gradients
- Both caches are bias-corrected to account for their initialization at zero
'''

class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1



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
#optimizer = Optimizer_GD()
#optimizer = Optimizer_GD_with_Decay(learning_rate=1.0, decay=1e-3)
#optimizer = Optimizer_with_Monentum(decay=1e-3, momentum=0.9)
#optimizer = Optimizer_Adagrad(decay=1e-4)

#Higher rho values (closer to 1) create a longer-term memory of past gradients
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5,rho=0.999)

optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

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
                f'loss: {loss:.3f}' +
                f'lr: {optimizer.current_learning_rate}')
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


# ============== OPTIMIZER_SUMMARY ==============
'''
The code implements various gradient descent optimizers for training a neural network on a spiral dataset.

- Gradient Descent (GD): The foundation of neural network optimization, updating parameters directly based on gradients. 
                        Simple but limited by fixed learning rates and susceptibility to local minima.

- GD with Decay: Addresses the fixed learning rate problem by gradually reducing learning rates over time, 
                allowing faster initial progress and finer adjustments as training proceeds.

- Momentum: Adds a velocity term that accumulates past gradients, helping overcome oscillations 
            in steep gradients and push through flat regions and shallow local minima.

- AdaGrad: Adapts learning rates per-parameter based on historical gradient information, giving smaller updates to 
            frequently updated parameters and larger updates to infrequent ones.

- RMSprop: Improves upon AdaGrad by using exponential moving averages of squared gradients instead of accumulating all past gradients, 
            preventing learning rates from becoming too small over time.

- Adam: Combines the best of momentum and RMSprop by maintaining both first-moment (velocity) and second-moment (adaptive learning rates) estimates,
        with bias correction for more stable updates.

The implementation demonstrates how each optimizer builds upon the previous one's strengths while addressing its limitations.
The spiral dataset classification example effectively showcases how advanced optimizers like Adam typically deliver faster convergence and better
performance than simpler approaches.

This progression of optimizers represents the evolution of gradient-based optimization techniques in deep learning, 
each addressing specific challenges in neural network training such as learning rate selection, navigating complex loss landscapes, 
and balancing speed with stability.
    '''