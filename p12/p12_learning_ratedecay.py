import numpy as np
from create_data import spiral_data
from p11.p11_gradientdescent import (
    Layer_Dense, 
    Activation_ReLU, 
    Activation_Softmax, 
    Loss, 
    Loss_CategoricalCrossentropy,
    Activation_Softmax_Loss_CategoricalCrossentropy
)
np.random.seed(0)

'''
Here we will implement the Optimizer gradient descent class/method with learning rate decay.
We will use the same data as in the previous example. We will also use the same model and loss function.
We will also use the same optimizer class, but we will add a learning rate decay to it.
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


# Create dataset
X, y = spiral_data(points=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_GD_with_Decay(learning_rate=1.0, decay=1e-3)

# Train in loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    
    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}, ' +
                f'lr: {optimizer.current_learning_rate:.3f}')
    
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

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