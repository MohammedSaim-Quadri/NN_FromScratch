''' 
batches allow us to perform multiple operations in parallel,
in this case it would be to process multiple input at a time.
another thing batches help with in neural networks is with generalization.
'''
import numpy as np

# inputs = [1, 2, 3, 2.5] # a single sample

# multiple samples comprising a batch
# shape : (3,4)
inputs = [[1, 2, 3, 2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5, 2.7, 3.3,-0.8]]

#list of lists - weights
#shape : (3,4)
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# similarly for biases
biases = [2, 3, 0.5]

# standard way 
'''
the matrix in this case would need to be transposed i.e (R,C) -> (C,R)
inorder to satisfy the rule of matrix multiplication
which requires that in matrix multiplication,the number of columns of first opperand -
be equal to the number of rows of second operand, i.e matrix1(R1,c1) * matrix2(R2,c2) - 
where c1 = R2, so in our case that would be inputs(3,4) and weights'T(4,3)
'''

output = np.dot(inputs,np.array(weights).T) + biases
print(output)
'''
[[ 4.8    1.21   2.385]
 [ 8.9   -1.81   0.2  ]
 [ 1.41   1.051  0.026]]
'''
#######################################################################
# now adding another layer to the entire calculation
#list of lists - weights
#shape : (3,4)
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# similarly for biases
biases = [2, 3, 0.5]

#list of lists - weights
#shape : (3,4)
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

# similarly for biases
biases2 = [-1, 2, -0.5]

# outputs of layer 1 
layer1_outputs = np.dot(inputs,np.array(weights).T) + biases
# output shape: (3,3)

# become the inputs for layer 2
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)
'''
[[ 0.5031  -1.04185 -2.03875]
 [ 0.2434  -2.7332  -5.7633 ]
 [-0.99314  1.41254 -0.35655]]
'''

"""
but this way of doing things not really efficient, as the number of layers increase - 
the number of lines of code will also increase drastically, so instead we start using OOP to create -
class/objects that we can then recall each time a layer is needed, also in the actual NN the weights and biases - 
are not manually set rather they are randomly generated
"""
##################################################################################
import numpy as np
np.random.seed(0)

#renamed input matrix
# shape : (3,4)
X = [[1, 2, 3, 2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5, 2.7, 3.3,-0.8]]

# defining the hidden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # 0.10 is multiplied to keep the weights in range of -1 to 1
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # weights that will be size of input multiplied by number of neurons
        self.biases = np.zeros((1, n_neurons)) # by default biases are initialized to 0 unless necessary
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# using objects to create the layers
layer1 = Layer_Dense(4,5) #(R1,c1)
layer2 = Layer_Dense(5,2) # (R2=c1, c2), because the output of layer1 is input for layer2

layer1.forward(X)
print(layer1.output)
'''
[[ 0.10758131  1.03983522  0.24462411  0.31821498  0.18851053]
 [-0.08349796  0.70846411  0.00293357  0.44701525  0.36360538]
 [-0.50763245  0.55688422  0.07987797 -0.34889573  0.04553042]]
'''
layer2.forward(layer1.output)
print(layer2.output)
'''
[[ 0.148296   -0.08397602]
 [ 0.14100315 -0.01340469]
 [ 0.20124979 -0.07290616]]
'''