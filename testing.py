from NeuralNetwork import NeuralNetwork
import numpy as np

def sigmoid(x):
    return 1/(1+np.power(np.e, -x))
def x(x):
    return x
nn = NeuralNetwork(3, (5, 1), 3, sigmoid)
nn.predict(np.array([7., 4., 2.]))
print(nn.outputs)

for i in range(5000):
    nn.train([np.array([7., 4., 2.]), np.array([2., 9., 6.]),np.array([0., 2., 10.])], [np.array([[1.],
        [0.], [0.]]),np.array([[0.],
        [1.], [0.]]),np.array([[0.],
        [0.], [1.]])])
# print("weights", nn.weights)
# print("WEight gradients", nn.weightGradient)
nn.predict(np.array([2., 9., 6.]))
print(nn.outputs)