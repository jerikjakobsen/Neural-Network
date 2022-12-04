import numpy as np

class NeuralNetwork:

    # Hidden Layers should be supplied as (x, y)
    # Where x is the number of nodes in each layer 
    # And y is the number of layers
    def __init__(self, inputSize, hiddenLayersSize, outputSize, activation) -> None:
        hiddenNodeCount = hiddenLayersSize[0]
        self.weights = []
        self.weights.append(np.full((hiddenNodeCount, inputSize), 0.5)) # Set up weight Matrix for input layer to first hidden layer
        self.weights.extend([np.full((hiddenNodeCount, hiddenNodeCount), 0.1) for i in range(hiddenNodeCount-2)]) # Set up weight matrices for hidden layers
        self.weights.append(np.full((outputSize, hiddenNodeCount), 0.2)) # Set up Weight Matrix for output layer
        self.inputSize = inputSize
        self.hiddenLayersSize = hiddenLayersSize
        self.outputSize = outputSize
        self.activation = activation
        self.outputs = []
        self.outputs.append(np.zeros((inputSize,1))) # Set up output vector first layer to second layer
        self.outputs.extend([np.zeros((hiddenNodeCount, 1)) for i in range(hiddenNodeCount-1)]) # Set up output layers for hidden layers
        self.outputs.append(np.zeros((outputSize, 1))) # Set up vector for the output layer

    # Input should be the same size as the inputNodes provided in initialization

    def _layerOutput(self, input, weights, activation):
        vectorizedActivation = np.vectorize(activation)
        return vectorizedActivation(np.dot(weights, input))

    def predict(self, input):
        if len(input) != self.inputSize:
            print("Error: Input provided does not match input layer size")
            return
        for layer in range(self.hiddenLayersSize[1] + 2):
            if layer == 0:
                self.outputs[0] = np.array(input).transpose()
            else:
                self.outputs[layer] = self._layerOutput(self.outputs[layer-1], self.weights[layer-1], self.activation) 

    def _trainFromOneSample(self, sampleX, sampleY, trainSize):
        self.predict(sampleX)
        AOEInitial = []
        for i in range(self.outputs[-1]):
            nodeOutput = self.outputs[-1][i]
            err = - (sampleY[i] - nodeOutput)
            AOEInitial.append[err]
        self._BackProp(AOEInitial, self.hiddenLayersSize + 1, trainSize)

    def _BackProp(self, AOE, layer, trainSize):
        if layer == 0:
            return
        Deltas = []
        for j in range(len(AOE)):
            affectInErr = AOE[j]
            outForNodeJ = self.outputs[layer, j]
            delta = affectInErr * outForNodeJ * (1-outForNodeJ)
            Deltas.append(delta)
        
        AOENext = []
        for j in range(self.layerSize(layer-1)):
            for i in range(self.layerSize(layer)):
                weightAdjustment = self.outputs[layer-1, j] * Deltas[j]
                self.weightGradient[layer, i, j] += weightAdjustment / trainSize
                AOENext[j] += Deltas[j] * self.weights[layer, i, j]
        self._BackProp(AOENext, layer-1, trainSize)


    def layerSize(self, layer):
        if layer == 0:
            return self.inputSize
        elif layer == self.hiddenLayersSize + 1:
            return self.outputSize
        else:
            return self.hiddenLayersSize
