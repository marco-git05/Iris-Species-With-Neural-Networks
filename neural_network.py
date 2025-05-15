import numpy as np

class NeuralNetwork:

    def __init__(self, input, target, hidden, learning_rate, iterations):
        self.X = input
        self.Y = target
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.W1 = np.random.rand(10, 784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid_function(z)
        return s * (1 - s)

    def softmax(z):
        return np.exp(z) / sum(np.exp(z))

    def feed_forward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = self.sigmoid_function(self.Z1)
        self.Z2 = np.dot(self.W2, self.X) + self.b2
        self.fA2 = self.softmax(self.Z2)

    def backpropagation(self):
        self.dW2 = -2 * (self.target - self.approx - self.b1) * np.transpose(self.X)
        self.db2 = -2 * (self.target - self.approx - self.b1)
        self.dW1 = -2 * (self.target - self.approx - self.b2) * np.transpose(self.X)
        self.db1 =  -2 * (self.target - self.approx - self.b2)

        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.b1 = self.b1 - self.learning_rate * self.db1 
        self.W2 = self.W2 - self.learning_rate * self.dW2
        self.b2 = self.b2 - self.learning_rate * self.db2 

    def train(self):
        for i in range(self.iterations):
            self.feed_forward()
            self.backpropigation()

    def predict(self, X):
        pass