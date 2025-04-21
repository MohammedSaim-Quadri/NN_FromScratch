# Notes related to p11 file

## Training the Model with Optimizers
After defining various optimizers (SGD, SGD with decay, Momentum, Adagrad, RMSprop, Adam), the next step is to use them during training. In this code, we're using the Adam optimizer, known for combining the benefits of both RMSprop and Momentum optimizers.

```python
optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)
```

## Model Architecture Recap:
```bash
Layer_Dense(2, 64) – Input layer: 2D features → 64 neurons.

Activation_ReLU() – ReLU activation after the first dense layer.

Layer_Dense(64, 3) – Output layer: 64 neurons → 3 class logits.

Activation_Softmax_Loss_CategoricalCrossentropy() – Combines softmax activation + cross-entropy loss for multi-class classification.
```

## Training Loop (10,001 Epochs)
```python
for epoch in range(10001):
```
Each Epoch involves:

    1. Forward Pass:
        - Data passes through dense1, ReLU, dense2, and finally the softmax + loss layer.
        - This gives us predictions and loss.

    2. Accuracy Calculation:
        - We compare the predicted class indices with true labels to compute accuracy.
    
    3. Printing Progress Every 100 Epochs:
        - Prints epoch number, accuracy, loss, and current learning rate (especially useful when decay is applied).

    4. Backward Pass (Backpropagation):
        - Gradients are computed from the loss layer backward through the network.
        - These gradients are stored in the layers (dweights, dbiases).

    5. Parameter Update:
        - The optimizer (Adam in this case) uses gradients to update weights and biases of both layers.

```python

optimizer.update_params(dense1)
optimizer.update_params(dense2)
```

| Final accuracy with Adam optimizer:

```python
# epoch: 10000, acc: 0.973, loss: 0.082
```

## Validation
After training, we evaluate the model using new spiral data (np.random.seed(1) ensures a different test set):

``` python
X_test, y_test = spiral_data(points=100, classes=3)
```

Then we perform another full forward pass (no backward or parameter update this time), to:

- Calculate predictions
- Evaluate the test loss
- Compute test accuracy

## Key Observations
- Learning rate decay gradually reduces the learning rate to fine-tune learning in later epochs.
- Adam optimizer performs the best in this setup, showing quick convergence and high accuracy.

- This setup mimics real training loops and demonstrates practical usage of optimizers from scratch, reinforcing understanding of what frameworks like TensorFlow or PyTorch do behind the scenes.



**Test data helps evaluate a model’s ability to generalize.**

**A model that performs well on training but poorly on test data is likely overfitting — it has memorized patterns (and noise) instead of learning generalizable features.**

## Understanding Hyperparameters
Hyperparameters are the knobs you tune before training starts — they stay constant during training.

### Common hyperparameters:
- Number of layers / neurons per layer
- Activation functions (ReLU, Sigmoid, Tanh, etc.)
- Learning rate (α), epochs
- Optimizer type and parameters (e.g., β₁ and β₂ in Adam)

These impact how well the model learns and generalizes.

## Role of Validation vs Test Data
- Test Set:
- - Used only after training is complete.
- - Helps evaluate final model performance on completely unseen data.

- Validation Set:
- - Used during training to tune hyperparameters.
- - Enables choosing the best model configuration before evaluating on test data.

## Data Splitting Strategies
- Case 1: When You Have Sufficient Data
        Split into:

        - Training set

        - Validation set

        - Test set

        Workflow:

        - Train with the training set.

        - Use validation set to fine-tune hyperparameters.

        - Once the model is tuned, evaluate performance on the test set.

- Case 2: Limited Data Scenario
        Use only:

        - Training (e.g., 80%)

        - Test (e.g., 20%)

🌀 Validation is done via K-Fold Cross Validation within the training set.

## K-Fold Cross Validation (k=5 Example)
Used when there’s no separate validation set due to limited data.

### Steps:

- Split the training data into k folds (e.g., A, B, C, D, E).
- Run k training-validation cycles:
- Cycle 1: Train on B+C+D+E, Validate on A
- Cycle 2: Train on A+C+D+E, Validate on B

- ...

- Collect validation error from each fold → E1, E2, ..., E5.

🧮 Final Validation Error:

$$[E_{mean} = \frac{E_1 + E_2 + E_3+E_4+E_5}{5}]$$
​
**Repeat for each hyperparameter configuration to find the best one.**

## What is Data Leakage?
- Definition: Data leakage happens when future information (often from the test set) unintentionally influences model training.

- Result: Inflated accuracy during development but poor performance on truly unseen data.

- Example: If you include a feature like Customer Cancellation Call in your training data, it can signal the final outcome (i.e., the customer won't purchase), giving away future information.

- Fix: Ensure your model only trains on features that are available at prediction time, not post-outcome indicators.

✅ 7. Summary Table

|Data Type	|Purpose |Used For|
|:-----------:|:--------:|:--------:|
|Training	|Train the model |Optimize weights & biases|
|Validation	|Tune hyperparameters |Choose best model configuration|
|Testing	|Final evaluation on unseen data |Assess generalization performance|
|Leakage	|Must be avoided |Preserve true model evaluation|
