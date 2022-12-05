import numpy as np
from numpy.random import rand

class NeuralNetwork:

    # Hidden Layers should be supplied as (x, y)
    # Where x is the number of nodes in each layer 
    # And y is the number of layers
    def __init__(self, inputSize, hiddenLayersSize, outputSize, activation) -> None:
        self.hiddenNodeCount = hiddenLayersSize[0]
        self.lastLayerIndex = hiddenLayersSize[1] + 1
        self.numOfLayers = hiddenLayersSize[1] + 2
        self.weightGradient = []
        self.weights = []
        self.weights.append(rand(self.hiddenNodeCount, inputSize)) # Set up weight Matrix for input layer to first hidden layer
        self.weights.extend([rand(self.hiddenNodeCount, self.hiddenNodeCount) for i in range(hiddenLayersSize[1]-1)]) # Set up weight matrices for hidden layers
        self.weights.append(rand(outputSize, self.hiddenNodeCount)) # Set up Weight Matrix for output layer
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.outputs = []
        self.outputs.append(np.zeros((inputSize,1))) # Set up output vector first layer to second layer
        self.outputs.extend([np.zeros((self.hiddenNodeCount, 1)) for i in range(self.numOfLayers-2)]) # Set up output layers for hidden layers
        self.outputs.append(np.zeros((outputSize, 1))) # Set up vector for the output layer

    # Input should be the same size as the inputNodes provided in initialization

    def _layerOutput(self, input, weights, activation):
        vectorizedActivation = np.vectorize(activation)
        return vectorizedActivation(np.dot(weights, input))

    def predict(self, input):
        if len(input) != self.inputSize:
            print("Error: Input provided does not match input layer size")
            return
        for layer in range(self.numOfLayers):
            if layer == 0:
                self.outputs[0] = np.array(input).transpose()
            else:
                self.outputs[layer] = self._layerOutput(self.outputs[layer-1], self.weights[layer-1], self.activation) 

    def _trainFromOneSample(self, sampleX, sampleY, trainSize):
        self.predict(sampleX)
        AOEInitial = []
        for i in range(self.outputSize):
            nodeOutput = self.outputs[self.lastLayerIndex][i]
            err = - (sampleY[i] - nodeOutput)
            AOEInitial.append(err)
        self._BackProp(AOEInitial, self.lastLayerIndex, trainSize)

    def train(self, xSamples, ySamples):
        self.weightGradient = []
        self.weightGradient.append(np.zeros((self.hiddenNodeCount, self.inputSize))) # Set up weight Matrix for input layer to first hidden layer
        self.weightGradient.extend([np.zeros((self.hiddenNodeCount, self.hiddenNodeCount)) for i in range(self.numOfLayers-3)]) # Set up weight matrices for hidden layers
        self.weightGradient.append(np.zeros((self.outputSize, self.hiddenNodeCount))) # Set up Weight Matrix for output layer
        if len(xSamples) != len(ySamples):
            print("Length of X does not match length of Y")

        for sampleIndex in range(len(xSamples)):
            xSample = xSamples[sampleIndex]
            ySample = ySamples[sampleIndex]
            self._trainFromOneSample(xSample, ySample, len(xSample))

        for i in range(len(self.weights)):
            self.weights[i] -= self.weightGradient[i]


    def _BackProp(self, AOE, layer, trainSize):
        if layer == 0:
            return
        Deltas = []
        for i in range(len(AOE)):
            affectInErr = AOE[i]
            outForNodeI = self.outputs[layer][i]
            delta = affectInErr * outForNodeI * (1-outForNodeI)
            Deltas.append(delta)

        AOENext = np.zeros(self.layerSize(layer-1))
        for j in range(self.layerSize(layer-1)):
            for i in range(self.layerSize(layer)):
                weightAdjustment = self.outputs[layer-1][j] * Deltas[i]
                self.weightGradient[layer-1][i, j] += weightAdjustment / trainSize
                AOENext[j] += Deltas[i] * self.weights[layer-1][i, j]
        self._BackProp(AOENext, layer-1, trainSize)



    def layerSize(self, layer):
        if layer == 0:
            return self.inputSize
        elif layer == self.lastLayerIndex:
            return self.outputSize
        else:
            return self.hiddenNodeCount
