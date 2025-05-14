import numpy as np

class NeuralNetwork:
    learning_rate = None
    input = None
    target = None
    weights = None
    input = None
    layers = None
    iterations = None

    def __init__(self, input, target, hidden, learning_rate, iterations):
        self.input = np.hstack([input, np.ones((input.shape[0], 1))])
        self.target = target
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.approx = np.zeros_like(target)
        self.weights = []
        self.layers = [self.input]

        layer_structure = [self.input.shape[1]] + [h + 1 for h in hidden[:-1]] + [hidden[-1], target.shape[1]]

        for i in range(len(layer_structure) - 1):
            weight_matrix = np.random.rand(layer_structure[i], layer_structure[i + 1])
            self.weights.append(weight_matrix)

    def feed_forward(self):
        for i in range(self.weights)
            

    def backpropigation(self):
        error = 2 * (self.target - self.approx)
        for i in range(self.weights)

    def train(self):
        for i in range(self.iterations):
            self.feed_forward()
            self.backpropigation

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid_function(z)
        return s * (1 - s)


    