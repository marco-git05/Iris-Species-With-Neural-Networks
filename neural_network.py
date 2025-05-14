from enum import Enum
import numpy as np

class NeuralNetwork:
    learning_rate = 0.01
    input = None
    target = None
    weights = None
    input = None
    layers = None

    def __init__(self, input, target, hidden):
        self.input = np.hstack([input, np.ones((input.shape[0], 1))])
        self.approx = np.zeros_like(target)
        self.target = target
        self.weights = []
        self.layers = [self.input]

        layer_structure = [self.input.shape[1]] + [h + 1 for h in hidden[:-1]] + [hidden[-1], target.shape[1]]

        for i in range(len(layer_structure) - 1):
            weight_matrix = np.random.rand(layer_structure[i], layer_structure[i + 1])
            self.weights.append(weight_matrix)

    def feed_forward(obj):
        pass

    def backpropigation(obj):
        pass

    def train(obj):
        pass

    def activation_function():
        pass

    def activation_derivative():
        pass


    