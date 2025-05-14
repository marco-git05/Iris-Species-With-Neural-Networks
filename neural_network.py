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

    def sigmoid_function(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid_function(z)
        return s * (1 - s)

    def softmax(z):
        return np.exp(z) / sum(np.exp(z))

    def feed_forward(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.sigmoid_function(Z1)
        Z2 = W2.dot(X) + b2
        A2 = self.softmax(Z2)

    def backpropagation(self,  W1, b1, W2, b2, X):
        dW2 = -2 * (self.target - self.approx - b1) * np.transpose(X)
        db2 = -2 * (self.target - self.approx - b1)
        dW1 = -2 * (self.target - self.approx - b2) * np.transpose(X)
        db1 =  -2 * (self.target - self.approx - b2)

        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1 
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2 

    def train(self):
        for i in range(self.iterations):
            self.feed_forward()
            self.backpropigation