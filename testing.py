from NeuralNetwork import NeuralNetwork
import numpy as np

def sigmoid(x):
    return 1/(1+np.power(np.e, -x))
def x(x):
    return x
nn = NeuralNetwork(2, (2, 1), 2, sigmoid)
nn.predict([10, 4])
print(nn.outputs)
print(nn.weights)


nn._trainFromOneSample(np.array([7., 4.]), np.array([[1.],
       [0.]]), 1)
print(nn.weightGradient)