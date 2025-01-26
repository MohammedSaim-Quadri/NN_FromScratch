# raw python code for the first part of neuron
# every neuron has its own unique weight associated with it
# which gets multiplied by the input from that particular neuron
# this is done for the number of neurons present, and at the end a bias term is added


# here we are assuming 3 input neurons whose weighted sum input is going
# to one particular neuron in the next layer( Hidden Layer 1 ), hence one o/p and one bias
# i.e each neuron has unique bias term associated with, just as the weights

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2] + bias
print(output)
