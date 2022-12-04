from NeuralNetwork import NeuralNetwork
import numpy as np

def sigmoid(x):
    return 1/(1+np.power(np.e, -x))
def x(x):
    return x
nn = NeuralNetwork(2, (2, 1), 2, x)
nn.predict([10, 4])
print(nn.outputs)
print(nn.weights)

i = np.array([[7.],
       [7.]])
w = np.array([[0.2, 0.2],
       [0.2, 0.2]])
print(np.dot(w, i))
