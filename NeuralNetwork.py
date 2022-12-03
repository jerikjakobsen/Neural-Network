import numpy as np

class NeuralNetwork:

    # Hidden Layers should be supplied as (x, y)
    # Where x is the number of nodes in each layer 
    # And y is the number of layers
    def __init__(self, inputSize, hiddenLayersSize, outputSize, activation) -> None:
        hiddenNodeCount = hiddenLayersSize[0]
        self.weights = []
        self.weights.append(np.zeros((hiddenNodeCount, inputSize))) # Set up weight Matrix for input layer to first hidden layer
        self.weights.extend([np.zeros((hiddenNodeCount, hiddenNodeCount)) for i in range(hiddenLayersSize-1)]) # Set up weight matrices for hidden layers
        self.weights.append(np.zeros((outputSize, hiddenNodeCount))) # Set up Weight Matrix for output layer
        self.inputSize = inputSize
        self.hiddenLayersSize = hiddenLayersSize
        self.outputSize = outputSize
        self.activation = activation
        self.outputs = []
        self.outputs.append(np.zeros((inputSize,1))) # Set up output vector first layer to second layer
        self.outputs.extend([np.zeros((hiddenLayersSize, 1)) for i in range(hiddenLayersSize-1)]) # Set up output layers for hidden layers
        self.outputs.append(np.zeros((outputSize, 1))) # Set up vector for the output layer

    # Input should be the same size as the inputNodes provided in initialization

    def _layerOutput(input, weights, activation):
        vectorizedActivation = np.vectorize(activation)
        return vectorizedActivation(np.dot(weights, input))

    def predict(self, input):
        if input != self.inputSize:
            print("Error: Input provided does not match input layer size")
            return
        for layer in range(self.hiddenLayersSize + 1):
            if layer == 1:
                self.outputs[0] = np.array(input).transpose()
            else:
                self.outputs[layer] = _layerOutput(self.outputs[layer], self.weights[layer], self.activation)
    
