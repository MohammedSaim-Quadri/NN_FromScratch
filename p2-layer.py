# now if we want to try and code for 3 neurons having 4 input neurons,
# for eg: we have three o/p neurons who take inputs from 4 i/p neurons in the hidden layer,
# w/o the activation function of o/p neuron for now,
# it could look somethings like this:

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3] + bias1,
            inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3] + bias2,
            inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3] + bias3]
print(output)
#[ 4.8    1.21   2.385]